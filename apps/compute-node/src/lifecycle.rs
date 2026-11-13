use std::{
    fs,
    os::unix::fs::{MetadataExt, PermissionsExt},
    path::PathBuf,
    sync::Arc,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{UnixListener, UnixStream},
    sync::{watch, Semaphore},
    time::{sleep, timeout},
};

use crate::{
    model_pool::{ModelPool, ModelRequest},
    private_fs,
    state::{self, Installation, Paths, RuntimeRecord},
};

const REQUEST_FRAME_LIMIT: u32 = 8 * 1024;
const RESPONSE_FRAME_LIMIT: u32 = 512 * 1024;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Status {
    pub schema_version: u32,
    pub initialized: bool,
    pub local_status: String,
    pub platform_tasks_ready: bool,
    pub platform_registration: String,
    pub paths: Paths,
    pub installation: Option<Installation>,
    pub runtime: Option<RuntimeRecord>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    schema_version: u32,
    command: String,
    request_id: String,
    installation_id: String,
    generation: Option<u64>,
    run_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model: Option<ModelRequest>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    schema_version: u32,
    request_id: String,
    status: Option<Status>,
    stop_accepted: bool,
    error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_result: Option<serde_json::Value>,
}

pub(super) fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())
}

pub fn status(paths: Paths) -> Result<Status, String> {
    let identity = state::installation(&paths)?;
    let mut result = Status {
        schema_version: 1,
        initialized: identity.is_some(),
        local_status: "uninitialized".into(),
        platform_tasks_ready: false,
        platform_registration: "not_implemented".into(),
        paths,
        installation: identity,
        runtime: None,
    };
    let Some(identity) = &result.installation else {
        return Ok(result);
    };
    state::validate_existing_paths(&result.paths)?;
    result.runtime = state::runtime_record(&result.paths, identity)?;
    let counter = state::generation_counter(&result.paths, identity)?;
    result.local_status = if result
        .runtime
        .as_ref()
        .is_some_and(|value| value.state == "running")
        || counter.last_generation > result.runtime.as_ref().map_or(0, |value| value.generation)
    {
        "unreachable".into()
    } else {
        "stopped".into()
    };
    match fs::symlink_metadata(&result.paths.socket) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(result),
        Err(error) => return Err(error.to_string()),
        Ok(_) => state::socket_metadata(&result.paths.socket)?,
    }
    let request = Request {
        schema_version: 1,
        command: "status".into(),
        request_id: uuid::Uuid::new_v4().to_string(),
        installation_id: identity.installation_id.clone(),
        generation: None,
        run_id: None,
        model: None,
    };
    if let Ok(reply) =
        runtime()?.block_on(exchange(&result.paths, &request, Duration::from_secs(1)))
    {
        if let Some(view) = reply.status {
            if view
                .installation
                .as_ref()
                .map(|value| &value.installation_id)
                == Some(&request.installation_id)
                && view
                    .runtime
                    .as_ref()
                    .is_some_and(|value| value.state == "running")
                && view.paths.state == result.paths.state
                && !view.platform_tasks_ready
            {
                return Ok(view);
            }
        }
    }
    Ok(result)
}

pub fn stop(paths: Paths) -> Result<Status, String> {
    let current = status(paths.clone())?;
    if current.local_status != "running" {
        if current.local_status == "stopped" {
            return Ok(current);
        }
        return Err("没有确认可响应的本机节点；未向PID发送停止信号，请先检查 status。".into());
    }
    let record = current.runtime.as_ref().ok_or("缺少当前启动代次。")?;
    let request = Request {
        schema_version: 1,
        command: "stop".into(),
        request_id: uuid::Uuid::new_v4().to_string(),
        installation_id: record.installation_id.clone(),
        generation: Some(record.generation),
        run_id: Some(record.run_id.clone()),
        model: None,
    };
    let execution = runtime()?;
    // A missing acknowledgement is resolved by observing this same run. Never
    // replay the mutation or target a process that replaced it.
    let acknowledgement = execution.block_on(exchange(&paths, &request, Duration::from_secs(1)));
    if acknowledgement
        .as_ref()
        .is_ok_and(|reply| !reply.stop_accepted)
    {
        return Err("停止请求未被当前代次接受。".into());
    }
    for _ in 0..50 {
        let view = status(paths.clone())?;
        if let Some(observed) = &view.runtime {
            if observed.run_id != record.run_id || observed.generation != record.generation {
                return Err("节点启动代次已变化，未停止新进程；请重新检查状态。".into());
            }
            if view.local_status == "stopped" && observed.state == "stopped" {
                return Ok(view);
            }
        }
        execution.block_on(sleep_for_poll());
    }
    Err("停止结果尚未确认，请保留状态和日志检查；未重发停止请求。".into())
}

pub(super) fn model_request(
    paths: Paths,
    model: ModelRequest,
    expected_run: Option<uuid::Uuid>,
) -> Result<serde_json::Value, String> {
    let view = status(paths.clone())?;
    if view.local_status != "running" {
        return Err("请先启动本机节点 compute-node serve。".into());
    }
    let current = view.runtime.ok_or("缺少节点运行代次。")?;
    if expected_run.is_some_and(|run| run.to_string() != current.run_id) {
        return Err("节点启动代次已变化，本次任务未执行。".into());
    }
    let request = Request {
        schema_version: 1,
        command: "model".into(),
        request_id: uuid::Uuid::new_v4().to_string(),
        installation_id: current.installation_id,
        generation: Some(current.generation),
        run_id: Some(current.run_id),
        model: Some(model),
    };
    runtime()?
        .block_on(exchange(&paths, &request, Duration::from_secs(900)))?
        .model_result
        .ok_or("节点未返回模型操作结果。".into())
}

async fn sleep_for_poll() {
    sleep(Duration::from_millis(100)).await;
}

async fn read_frame<T: for<'de> Deserialize<'de>>(
    stream: &mut UnixStream,
    max: u32,
) -> Result<T, String> {
    let size = stream.read_u32().await.map_err(|error| error.to_string())?;
    if size == 0 || size > max {
        return Err("本机控制消息大小无效。".into());
    }
    let mut data = vec![0; size as usize];
    stream
        .read_exact(&mut data)
        .await
        .map_err(|error| error.to_string())?;
    serde_json::from_slice(&data).map_err(|_| "本机控制消息格式无效。".into())
}

async fn write_frame<T: Serialize>(
    stream: &mut UnixStream,
    value: &T,
    limit: u32,
) -> Result<(), String> {
    let bytes = serde_json::to_vec(value).map_err(|_| "本机控制消息编码失败。".to_string())?;
    if bytes.is_empty() || bytes.len() > limit as usize {
        return Err("本机控制消息超过限制。".into());
    }
    stream
        .write_u32(bytes.len() as u32)
        .await
        .map_err(|error| error.to_string())?;
    stream
        .write_all(&bytes)
        .await
        .map_err(|error| error.to_string())?;
    stream.flush().await.map_err(|error| error.to_string())
}

async fn exchange(paths: &Paths, request: &Request, budget: Duration) -> Result<Response, String> {
    timeout(budget, async {
        state::socket_metadata(&paths.socket)?;
        let mut stream = UnixStream::connect(&paths.socket)
            .await
            .map_err(|error| error.to_string())?;
        if stream.peer_cred().map_err(|error| error.to_string())?.uid() != private_fs::uid() {
            return Err("本机节点进程身份不匹配。".into());
        }
        write_frame(&mut stream, request, REQUEST_FRAME_LIMIT).await?;
        let reply: Response = read_frame(&mut stream, RESPONSE_FRAME_LIMIT).await?;
        if reply.schema_version != 1 || reply.request_id != request.request_id {
            return Err("本机节点回执不匹配。".into());
        }
        if let Some(error) = &reply.error {
            return Err(error.clone());
        }
        Ok(reply)
    })
    .await
    .map_err(|_| "本机节点响应超时。".to_string())?
}

struct SocketGuard {
    path: PathBuf,
    dev: u64,
    ino: u64,
}
impl Drop for SocketGuard {
    fn drop(&mut self) {
        if let Ok(metadata) = fs::symlink_metadata(&self.path) {
            if metadata.dev() == self.dev && metadata.ino() == self.ino {
                let _ = fs::remove_file(&self.path);
            }
        }
    }
}

pub fn serve(paths: Paths) -> Result<(), String> {
    let identity = state::require_installation(&paths)?;
    state::validate_existing_paths(&paths)?;
    let _lock = private_fs::lock(&paths.runtime.join("node.lock"), false)?;
    let pool = Arc::new(ModelPool::new(
        paths.clone(),
        identity.installation_id.clone(),
    )?);
    let previous = state::runtime_record(&paths, &identity)?;
    let (generation, previous_run_unclean) =
        state::reserve_generation(&paths, &identity, previous.as_ref())?;
    runtime()?.block_on(serve_inner(
        paths,
        identity,
        previous_run_unclean,
        generation,
        pool,
    ))
}

async fn serve_inner(
    paths: Paths,
    identity: Installation,
    previous_run_unclean: bool,
    generation: u64,
    pool: Arc<ModelPool>,
) -> Result<(), String> {
    let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|error| error.to_string())?;
    let mut interrupt = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())
        .map_err(|error| error.to_string())?;
    match fs::symlink_metadata(&paths.socket) {
        Ok(_) => {
            state::socket_metadata(&paths.socket)?;
            fs::remove_file(&paths.socket).map_err(|error| error.to_string())?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.to_string()),
    }
    let listener = UnixListener::bind(&paths.socket).map_err(|error| error.to_string())?;
    fs::set_permissions(&paths.socket, fs::Permissions::from_mode(0o600))
        .map_err(|error| error.to_string())?;
    let metadata = fs::symlink_metadata(&paths.socket).map_err(|error| error.to_string())?;
    let _socket = SocketGuard {
        path: paths.socket.clone(),
        dev: metadata.dev(),
        ino: metadata.ino(),
    };
    let mut record = RuntimeRecord {
        schema_version: 1,
        installation_id: identity.installation_id.clone(),
        generation,
        run_id: uuid::Uuid::new_v4().to_string(),
        pid: std::process::id(),
        state: "running".into(),
        started_at_unix_ms: state::now_ms()?,
        stopped_at_unix_ms: None,
        previous_run_unclean,
    };
    state::save_runtime(&paths, &record)?;
    let view = Status {
        schema_version: 1,
        initialized: true,
        local_status: "running".into(),
        platform_tasks_ready: false,
        platform_registration: "not_implemented".into(),
        paths: paths.clone(),
        installation: Some(identity),
        runtime: Some(record.clone()),
    };
    let (stop_sender, mut stop_receiver) = watch::channel(false);
    // Leave control capacity even if all eight four-slot instances are full.
    let permits = Arc::new(Semaphore::new(40));
    let maintenance = {
        let pool = pool.clone();
        let mut stop = stop_sender.subscribe();
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(1));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tokio::select! {
                    _ = stop.changed() => break,
                    _ = tick.tick() => {
                        pool.refresh_observations().await;
                        pool.reap_idle().await;
                    },
                }
            }
        })
    };
    let result = async {
        private_fs::append_event(
            &paths.logs.join("node.log"),
            &serde_json::json!({"event":"local_started","runtime":record}),
        )?;
        println!(
            "本机节点已启动：代次 {}。跨机联调用 compute-node connect；Ctrl+C 正常停止。",
            generation
        );
        loop {
            tokio::select! {
                _ = terminate.recv() => break,
                _ = interrupt.recv() => break,
                _ = stop_receiver.changed() => break,
                connection = listener.accept() => {
                    let (stream, _) = connection.map_err(|error| error.to_string())?;
                    let Ok(permit) = permits.clone().try_acquire_owned() else { continue; };
                    let snapshot = view.clone();
                    let sender = stop_sender.clone();
                    let pool = pool.clone();
                    tokio::spawn(async move {
                        let _permit = permit;
                        let _ = handle(stream, snapshot, sender, pool).await;
                    });
                }
            }
        }
        Ok::<(), String>(())
    }
    .await;
    stop_sender.send_replace(true);
    let maintenance_result = maintenance.await.map_err(|error| error.to_string());
    pool.stop_all().await?;
    maintenance_result?;
    record.state = "stopped".into();
    record.stopped_at_unix_ms = Some(state::now_ms()?);
    state::save_runtime(&paths, &record)?;
    drop(listener);
    private_fs::append_event(
        &paths.logs.join("node.log"),
        &serde_json::json!({"event":"local_stopped","runtime":record}),
    )?;
    result
}

async fn handle(
    mut stream: UnixStream,
    view: Status,
    stop: watch::Sender<bool>,
    pool: Arc<ModelPool>,
) -> Result<(), String> {
    if stream.peer_cred().map_err(|error| error.to_string())?.uid() != private_fs::uid() {
        return Err("本机调用者身份不匹配。".into());
    }
    let request: Request = timeout(
        Duration::from_secs(1),
        read_frame(&mut stream, REQUEST_FRAME_LIMIT),
    )
    .await
    .map_err(|_| "本机控制请求超时。")??;
    let runtime = view.runtime.as_ref().ok_or("缺少本机启动代次。")?;
    let identity_ok = request.schema_version == 1
        && uuid::Uuid::parse_str(&request.request_id).is_ok()
        && request.installation_id == runtime.installation_id;
    let fenced = identity_ok
        && request.generation == Some(runtime.generation)
        && request.run_id.as_ref() == Some(&runtime.run_id);
    let stop_ok = fenced && request.command == "stop" && request.model.is_none();
    let status_ok = identity_ok && request.command == "status" && request.model.is_none();
    let mut model_result = None;
    let error = match (fenced, request.command.as_str(), request.model) {
        (true, "model", Some(model)) => match pool.execute(model).await {
            Ok(value) => {
                model_result = Some(value);
                None
            }
            Err(error) => Some(error),
        },
        _ if stop_ok || status_ok => None,
        _ => Some("请求身份、启动代次或本机命令不匹配。".into()),
    };
    let response = Response {
        schema_version: 1,
        request_id: request.request_id,
        status: status_ok.then_some(view),
        stop_accepted: stop_ok,
        error,
        model_result,
    };
    let written = timeout(
        Duration::from_secs(1),
        write_frame(&mut stream, &response, RESPONSE_FRAME_LIMIT),
    )
    .await;
    if stop_ok {
        let _ = stop.send(true);
    }
    written.map_err(|_| "本机控制响应超时。".to_owned())?
}
