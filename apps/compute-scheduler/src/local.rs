use std::{
    fs::{self, File, OpenOptions},
    io::{self, ErrorKind, Read},
    os::{
        fd::AsRawFd,
        unix::fs::{FileTypeExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    },
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use platform_node_link::{local_node::LocalNode, receive, send};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::Value;
use tokio::{
    net::{UnixListener, UnixStream},
    signal::unix::{signal, SignalKind},
    task::JoinSet,
    time::{interval, timeout, MissedTickBehavior},
};
use uuid::Uuid;

use crate::{
    admission::Journal,
    gateway,
    node::{Hub, RemoteConfig},
    queue::{valid_name, NodeConfig, Scheduler, WebIdentity},
};

const REQUEST_LIMIT: usize = 16 * 1024;
const RESPONSE_LIMIT: usize = 1024 * 1024;
const EXCHANGE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    state_dir: PathBuf,
    node: Option<LocalNode>,
    remote: Option<RemoteConfig>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Request {
    Status,
    Submit {
        id: Uuid,
        plan: String,
        prompt: String,
    },
    SubmitWeb {
        id: Uuid,
        plan: String,
        prompt: String,
        browser_endpoint_id: String,
    },
    Task {
        id: Uuid,
    },
    Gateway {
        operation: gateway::Operation,
    },
}

pub fn run(args: &[String]) -> Result<(), String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async {
        match args {
            [command, config] if command == "gateway" => gateway::serve(Path::new(config)).await,
            [command, config] if command == "serve" => {
                let mut input = Vec::new();
                File::open(config)
                    .and_then(|file| file.take(65_537).read_to_end(&mut input))
                    .map_err(|error| format!("读取调度配置失败（{error}）"))?;
                if input.len() > 65_536 {
                    return Err("调度配置超过大小限制。".into());
                }
                let config = serde_json::from_slice(&input)
                    .map_err(|error| format!("调度配置无效（{error}）"))?;
                serve(config).await
            }
            [command, state_dir, args @ ..] => {
                let parse_id =
                    |value: &str| Uuid::parse_str(value).map_err(|_| "任务 UUID 无效。".to_owned());
                let request = match (command.as_str(), args) {
                    ("status", []) => Request::Status,
                    ("task", [id]) => Request::Task { id: parse_id(id)? },
                    ("submit" | "submit-web", [plan, rest @ ..]) => {
                        let (browser, id) = if command == "submit-web" {
                            let (endpoint, id) =
                                rest.split_first().ok_or("缺少浏览器 Endpoint ID。")?;
                            (Some(endpoint.clone()), id)
                        } else {
                            (None, rest)
                        };
                        if id.len() > 1 {
                            return Err("提交任务参数过多。".into());
                        }
                        let mut prompt = String::new();
                        io::stdin()
                            .take(2049)
                            .read_to_string(&mut prompt)
                            .map_err(|_| "提示输入无效，请提供 UTF-8 文本。".to_owned())?;
                        let id = id
                            .first()
                            .map(|id| parse_id(id))
                            .transpose()?
                            .unwrap_or_else(Uuid::new_v4);
                        if let Some(browser_endpoint_id) = browser {
                            Request::SubmitWeb {
                                id,
                                plan: plan.clone(),
                                prompt,
                                browser_endpoint_id,
                            }
                        } else {
                            Request::Submit {
                                id,
                                plan: plan.clone(),
                                prompt,
                            }
                        }
                    }
                    _ => return Err("命令参数无效，请执行 compute-scheduler --help。".into()),
                };
                let value = request_local(Path::new(state_dir), request).await?;
                print_value(&value)
            }
            _ => Err("命令参数无效，请执行 compute-scheduler --help。".into()),
        }
    })
}

pub fn read_json_input<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, String> {
    let mut bytes = Vec::new();
    File::open(path)
        .and_then(|file| file.take(REQUEST_LIMIT as u64 + 1).read_to_end(&mut bytes))
        .map_err(|error| format!("读取网关配置失败（{error}）"))?;
    if bytes.len() > REQUEST_LIMIT {
        return Err("网关配置超过大小限制。".into());
    }
    serde_json::from_slice(&bytes).map_err(|error| format!("网关配置无效（{error}）"))
}

fn print_value(value: &Value) -> Result<(), String> {
    println!(
        "{}",
        serde_json::to_string_pretty(value).map_err(|error| error.to_string())?
    );
    Ok(())
}

async fn serve(config: Config) -> Result<(), String> {
    let mut routes = Vec::new();
    if let Some(node) = &config.node {
        if !valid_name(&node.id)
            || node.plans.is_empty()
            || node.plans.iter().any(|plan| !valid_name(plan))
            || !node.binary.is_absolute()
            || !node.config_home.is_absolute()
            || !node.data_dir.is_absolute()
            || config.state_dir == node.data_dir
        {
            return Err("请配置独立调度状态目录、节点名称、绝对路径和模型计划。".into());
        }
        routes.push(NodeConfig {
            id: node.id.clone(),
            plans: node.plans.clone(),
            max_dispatched: usize::MAX,
        });
    }
    let remote = config
        .remote
        .map(|remote| remote.prepare(&mut routes))
        .transpose()?;
    if routes.is_empty() {
        return Err("请至少配置一个本机或远程节点。".into());
    }
    let path = socket_path(&config.state_dir)?;
    let _lock =
        lock(&config.state_dir.join("scheduler.lock")).map_err(|error| error.to_string())?;
    match socket_metadata(&path) {
        Ok(_) => fs::remove_file(&path).map_err(|error| error.to_string())?,
        Err(error) if error.kind() == ErrorKind::NotFound => (),
        Err(error) => return Err(error.to_string()),
    }
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|error| error.to_string())?;
    let mut terminate = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
    let listener = UnixListener::bind(&path).map_err(|error| error.to_string())?;
    let _socket = SocketGuard::new(path)?;
    fs::set_permissions(&_socket.path, fs::Permissions::from_mode(0o600))
        .map_err(|error| error.to_string())?;

    let node = config.node.map(Arc::new);
    let admissions = Journal::open(&config.state_dir)
        .map_err(|error| format!("打开网页任务准入记录失败（{error}）"))?;
    let scheduler = Arc::new(Mutex::new(
        Scheduler::new(routes).with_admissions(admissions),
    ));
    let hub = Arc::new(Mutex::new(Hub::new(Arc::clone(&scheduler))));
    let mut jobs = JoinSet::new();
    if let Some(endpoint) = remote {
        let listener = endpoint.bind().await?;
        jobs.spawn(endpoint.serve(listener, Arc::clone(&hub)));
    }
    if let Some(inventory_node) = node.clone() {
        let inventory_state = Arc::clone(&scheduler);
        jobs.spawn(async move {
            loop {
                let sampled_at = Instant::now();
                let result = inventory_node.inventory().await;
                inventory_state
                    .lock()
                    .unwrap()
                    .update(0, result, sampled_at);
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
    }
    let mut clients = JoinSet::new();
    let mut tick = interval(Duration::from_millis(200));
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    println!("调度器已启动。");
    loop {
        tokio::select! {
            _ = interrupt.recv() => break,
            _ = terminate.recv() => break,
            _ = tick.tick() => {
                let now = Instant::now();
                hub.lock().unwrap().expire(now);
                let jobs_to_start = scheduler.lock().unwrap().dispatch(now);
                for job in jobs_to_start {
                    if let Some(node) = node.as_ref().filter(|_| job.node == 0) {
                        let node = Arc::clone(node);
                        let scheduler = Arc::clone(&scheduler);
                        jobs.spawn(async move {
                            let result = node.generate(&job.plan, job.id, &job.prompt, None).await;
                            scheduler.lock().unwrap().complete(job.id, result, Instant::now());
                            Ok(())
                        });
                    } else {
                        hub.lock().unwrap().assign(job, now);
                    }
                }
            },
            accepted = listener.accept(), if clients.len() < 16 => {
                let (stream, _) = accepted.map_err(|error| error.to_string())?;
                clients.spawn(handle(stream, Arc::clone(&scheduler)));
            }
            Some(_) = clients.join_next() => (),
            Some(result) = jobs.join_next() => result.map_err(|error| format!("调度后台任务异常（{error}）"))??,
        }
    }
    Ok(())
}

async fn handle(mut stream: UnixStream, scheduler: Arc<Mutex<Scheduler>>) {
    let _ = timeout(EXCHANGE_TIMEOUT, async {
        check_peer(&stream)?;
        let request = receive(&mut stream, REQUEST_LIMIT).await?;
        let now = Instant::now();
        let result = {
            let mut scheduler = scheduler.lock().unwrap();
            match request {
                Request::Status => Ok(scheduler.status(now)),
                Request::Task { id } => scheduler.task(id),
                Request::Submit { id, plan, prompt } => scheduler.submit(id, plan, prompt, now),
                Request::SubmitWeb {
                    id,
                    plan,
                    prompt,
                    browser_endpoint_id,
                } => scheduler
                    .submit_web(
                        id,
                        plan,
                        prompt,
                        WebIdentity {
                            endpoint_id: browser_endpoint_id,
                            platform_model: None,
                        },
                        now,
                    )
                    .map_err(|error| error.to_string()),
                Request::Gateway { operation } => {
                    serde_json::to_value(gateway::execute(&mut scheduler, operation, now))
                        .map_err(|_| "无法编码网关回执。".into())
                }
            }
        };
        send(&mut stream, &result, RESPONSE_LIMIT).await
    })
    .await;
}

pub async fn request_local(state_dir: &Path, request: Request) -> Result<Value, String> {
    let path = socket_path(state_dir)?;
    socket_metadata(&path).map_err(|error| format!("无法访问调度服务（{error}）"))?;
    timeout(EXCHANGE_TIMEOUT, async {
        let mut stream = UnixStream::connect(&path).await?;
        check_peer(&stream)?;
        send(&mut stream, &request, REQUEST_LIMIT).await?;
        receive(&mut stream, RESPONSE_LIMIT).await
    })
    .await
    .map_err(|_| "调度服务响应超时。".to_owned())?
    .map_err(|error| format!("调度服务通信失败（{error}）"))?
}

fn uid() -> u32 {
    // SAFETY: geteuid has no preconditions.
    unsafe { libc::geteuid() }
}

fn check_peer(stream: &UnixStream) -> io::Result<()> {
    if stream.peer_cred()?.uid() != uid() {
        return Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "调度进程身份不匹配",
        ));
    }
    Ok(())
}

fn socket_path(dir: &Path) -> Result<PathBuf, String> {
    if !dir.is_absolute() || dir.canonicalize().ok().as_deref() != Some(dir) {
        return Err("调度状态目录必须是已创建、无符号链接的绝对路径。".into());
    }
    for ancestor in dir.ancestors() {
        let meta = fs::symlink_metadata(ancestor).map_err(|error| error.to_string())?;
        let root_sticky = meta.uid() == 0 && meta.mode() & 0o1000 != 0;
        if !meta.is_dir()
            || ![uid(), 0].contains(&meta.uid())
            || (meta.mode() & 0o022 != 0 && !root_sticky)
            || (ancestor == dir && (meta.uid() != uid() || meta.mode() & 0o077 != 0))
        {
            return Err("调度状态目录必须由当前用户所有且权限为 0700，父目录不得公开写入。".into());
        }
    }
    Ok(dir.join("control.sock"))
}

fn lock(path: &Path) -> io::Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path)?;
    let meta = file.metadata()?;
    if !meta.is_file() || meta.uid() != uid() || meta.mode() & 0o077 != 0 || meta.nlink() != 1 {
        return Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "调度锁文件不安全",
        ));
    }
    // SAFETY: the open file owns this descriptor until the service exits.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
        return Err(io::Error::new(
            ErrorKind::AddrInUse,
            "该目录已有调度服务持有运行锁",
        ));
    }
    Ok(file)
}

fn socket_metadata(path: &Path) -> io::Result<fs::Metadata> {
    let metadata = fs::symlink_metadata(path)?;
    if !metadata.file_type().is_socket() || metadata.uid() != uid() {
        return Err(io::Error::new(
            ErrorKind::PermissionDenied,
            "调度 socket 类型或所有者不匹配",
        ));
    }
    Ok(metadata)
}

struct SocketGuard {
    path: PathBuf,
    identity: (u64, u64),
}

impl SocketGuard {
    fn new(path: PathBuf) -> Result<Self, String> {
        let meta = socket_metadata(&path).map_err(|error| error.to_string())?;
        Ok(Self {
            path,
            identity: (meta.dev(), meta.ino()),
        })
    }
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        if socket_metadata(&self.path).is_ok_and(|meta| (meta.dev(), meta.ino()) == self.identity) {
            let _ = fs::remove_file(&self.path);
        }
    }
}
