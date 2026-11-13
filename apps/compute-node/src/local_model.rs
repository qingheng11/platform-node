//! Operator-provisioned, single-model local acceptance. The platform installer
//! and task transport will supply their own verified inputs to the runner.
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    net::{Ipv4Addr, TcpListener},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};

use reqwest::{Client, RequestBuilder, StatusCode};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use tokio::{
    process::{Child, Command},
    sync::watch,
    time::{sleep, timeout},
};

use crate::{
    lifecycle, private_fs,
    resource_plan::{self, Placement, ResourceMode},
    state::{self, Paths},
};

pub(super) const MIB: u64 = 1024 * 1024;
const ALIAS: &str = "compute-local-check";
const RUNNER_NAME: &str = "llama-server";
const START_TIMEOUT: Duration = Duration::from_secs(300);
const MAX_RESPONSE: usize = 256 * 1024;
const SLOT_CLEANUP_TIMEOUT: Duration = Duration::from_secs(5);
// Non-streaming generation returns its first response only after decoding.
const GENERATION_TIMEOUT: Duration = Duration::from_secs(600);

fn default_max_gpus() -> u8 {
    8
}

fn one_slot() -> u8 {
    1
}

fn default_output_tokens() -> u32 {
    64
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelSpec {
    schema_version: u32,
    model_id: String,
    variant_id: String,
    runtime_id: String,
    model_file: String,
    model_sha256: String,
    pub runtime_files: BTreeMap<String, String>,
    context_tokens: u32,
    #[serde(default = "one_slot")]
    pub parallel_slots: u8,
    #[serde(default)]
    pub idle_unload_seconds: Option<u64>,
    #[serde(default = "default_output_tokens")]
    output_tokens: u32,
    gpu_layers: u32,
    pub cpu_threads: u32,
    #[serde(default)]
    pub vram_mib: u32,
    #[serde(default)]
    pub memory_mib: u32,
    #[serde(default)]
    pub resource_mode: ResourceMode,
    #[serde(default = "default_max_gpus")]
    pub max_gpus: u8,
    #[serde(default)]
    pub gpu_uuid: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Instance {
    pub instance_id: String,
    pub installation_id: String,
    pub model_id: String,
    pub variant_id: String,
    runtime_id: String,
    model_sha256: String,
    pub gpu_uuid: String,
    #[serde(default)]
    pub placement: Option<Placement>,
    pub reserved_vram_bytes: u64,
    pub reserved_memory_bytes: u64,
    pub runner_pid: Option<u32>,
    ready_at_unix_ms: Option<u64>,
    pub process_group_exit_confirmed: bool,
    pub phase: Phase,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Starting,
    Ready,
    Stopped,
    Quarantined,
}

pub(super) struct Assets {
    pub model: PathBuf,
    pub runtime: PathBuf,
}

#[derive(Serialize)]
pub(super) struct Answer {
    text: String,
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

pub(super) enum TaskResult {
    Finished(Result<Answer, String>),
    Cancelled { execution_started: bool },
}

impl TaskResult {
    pub fn into_answer(self) -> Result<Answer, String> {
        match self {
            Self::Finished(answer) => answer,
            Self::Cancelled {
                execution_started: true,
            } => Err("请求已取消，生成已停止。".into()),
            Self::Cancelled {
                execution_started: false,
            } => Err("请求已取消，本次生成未执行。".into()),
        }
    }
}

pub(super) fn valid_component(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && !value.starts_with('.')
        && value
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'.' | b'_' | b'-'))
}

impl ModelSpec {
    pub(super) fn memory_arguments_without_common(&self) -> Vec<String> {
        // One policy for estimator and server: fixed batch, KV dtype and FA.
        [
            "--batch-size",
            "512",
            "--ubatch-size",
            "128",
            "--cache-type-k",
            "f16",
            "--cache-type-v",
            "f16",
            "--flash-attn",
            "off",
        ]
        .into_iter()
        .map(str::to_owned)
        .collect()
    }

    pub(super) fn memory_arguments(&self) -> Vec<String> {
        let mut args = self.memory_arguments_without_common();
        args.extend([
            "--ctx-size".into(),
            (self.context_tokens * u32::from(self.parallel_slots)).to_string(),
            "--parallel".into(),
            self.parallel_slots.to_string(),
            "--n-gpu-layers".into(),
            self.gpu_layers.to_string(),
            "--threads".into(),
            self.cpu_threads.to_string(),
        ]);
        args
    }

    pub(super) fn verify(&self, paths: &Paths) -> Result<Assets, String> {
        if self.schema_version != 1
            || ![
                &self.model_id,
                &self.variant_id,
                &self.runtime_id,
                &self.model_file,
            ]
            .into_iter()
            .all(|value| valid_component(value))
            || !self.model_file.ends_with(".gguf")
            || !(512..=32768).contains(&self.context_tokens)
            || !matches!(self.parallel_slots, 1 | 4)
            || !matches!(self.idle_unload_seconds, None | Some(600))
            || !(1..=2048).contains(&self.output_tokens)
            || self.output_tokens >= self.context_tokens
            || !(1..=999).contains(&self.gpu_layers)
            || !(1..=128).contains(&self.cpu_threads)
            || (self.resource_mode == ResourceMode::Manual
                && (self.vram_mib == 0 || self.memory_mib == 0))
            || !(1..=8).contains(&self.max_gpus)
            || self.runtime_files.len() > 128
            || !self.runtime_files.contains_key(RUNNER_NAME)
        {
            return Err("本机模型计划的版本、路径、文件清单或资源参数无效。".into());
        }
        let model = paths.models.join(&self.model_file);
        let runtime = paths.environments.join(&self.runtime_id);
        if runtime.as_os_str().as_encoded_bytes().contains(&b':') {
            return Err("运行库目录不能包含冒号。".into());
        }
        private_fs::check_directory(&runtime)?;
        let entries = fs::read_dir(&runtime).map_err(|e| e.to_string())?;
        let mut count = 0;
        for entry in entries {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name();
            let name = name
                .to_str()
                .filter(|name| valid_component(name))
                .ok_or("环境文件名无效。")?;
            let digest = self
                .runtime_files
                .get(name)
                .ok_or("环境目录含计划以外的文件。")?;
            let magic =
                (name == RUNNER_NAME || name == "llama-fit-params").then_some(&b"\x7fELF"[..]);
            verify_file(&entry.path(), digest, magic)?;
            count += 1;
        }
        if count != self.runtime_files.len() {
            return Err("环境清单有文件缺失。".into());
        }
        let executable =
            fs::symlink_metadata(runtime.join(RUNNER_NAME)).map_err(|e| e.to_string())?;
        if executable.permissions().mode() & 0o100 == 0 {
            return Err("llama-server 未设置当前用户执行权限。".into());
        }
        verify_file(&model, &self.model_sha256, Some(b"GGUF"))?;
        Ok(Assets { model, runtime })
    }
}

fn verify_file(path: &Path, expected: &str, magic: Option<&[u8]>) -> Result<(), String> {
    if expected.len() != 64 || !expected.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("模型计划中的 SHA256 无效。".into());
    }
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
        .open(path)
        .map_err(|e| format!("打开制品失败 {}：{e}", path.display()))?;
    let meta = file.metadata().map_err(|e| e.to_string())?;
    if !meta.is_file()
        || meta.uid() != private_fs::uid()
        || meta.nlink() != 1
        || meta.mode() & 0o7022 != 0
    {
        return Err(format!(
            "制品必须为当前用户所有的普通文件，且无链接或不安全权限：{}",
            path.display()
        ));
    }
    let mut digest = Sha256::new();
    if let Some(magic) = magic {
        let mut header = [0; 4];
        file.read_exact(&mut header).map_err(|e| e.to_string())?;
        if header.as_slice() != magic {
            return Err(format!("制品格式不符：{}", path.display()));
        }
        digest.update(header);
    }
    let mut bytes = vec![0; 256 * 1024];
    loop {
        let count = file.read(&mut bytes).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        digest.update(&bytes[..count]);
    }
    if !format!("{:x}", digest.finalize()).eq_ignore_ascii_case(expected) {
        return Err(format!("制品 SHA256 校验失败：{}", path.display()));
    }
    Ok(())
}

pub(super) fn check(paths: Paths, spec_path: &Path, output_json: bool) -> Result<(), String> {
    let identity = state::require_installation(&paths)?;
    state::validate_existing_paths(&paths)?;
    let _lock = private_fs::lock(&paths.runtime.join("node.lock"), false)?;
    crate::model_pool::check_previous(&paths, &identity.installation_id)?;
    let journal = paths.state.join("local-model-check.json");
    let spec = private_fs::read_json::<ModelSpec>(spec_path)?.ok_or("找不到本机模型计划。")?;
    let assets = spec.verify(&paths)?;
    let (hardware, reading) = resource_plan::sample()?;
    let placement = resource_plan::plan(&paths, &spec, &assets, &hardware, &reading, &[])?;
    let (fresh, reading) = resource_plan::sample()?;
    resource_plan::recheck(&placement, &fresh, &reading, &[])?;
    let mut instance = new_instance(&spec, identity.installation_id, placement);
    let answer =
        lifecycle::runtime()?.block_on(run(&paths, &journal, &spec, &assets, &mut instance))?;
    if output_json {
        println!(
            "{}",
            json!({"instance":instance,"answer":answer,"reservation_released":true})
        );
    } else {
        println!("{}", answer.text);
        println!("模型进程组已退出，本次资源预留已释放。");
    }
    Ok(())
}

pub(super) fn new_instance(
    spec: &ModelSpec,
    installation_id: String,
    placement: Placement,
) -> Instance {
    Instance {
        instance_id: uuid::Uuid::new_v4().to_string(),
        installation_id,
        model_id: spec.model_id.clone(),
        variant_id: spec.variant_id.clone(),
        runtime_id: spec.runtime_id.clone(),
        model_sha256: spec.model_sha256.clone(),
        gpu_uuid: placement.gpus[0].gpu_uuid.clone(),
        reserved_vram_bytes: placement.total_reserved(),
        reserved_memory_bytes: placement.host_reserved_bytes,
        placement: Some(placement),
        runner_pid: None,
        ready_at_unix_ms: None,
        process_group_exit_confirmed: false,
        phase: Phase::Starting,
    }
}

struct KeyFile(PathBuf);
impl Drop for KeyFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

struct SlotDirectory(PathBuf);
impl Drop for SlotDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir(&self.0);
    }
}

async fn run(
    paths: &Paths,
    journal: &Path,
    spec: &ModelSpec,
    assets: &Assets,
    instance: &mut Instance,
) -> Result<Answer, String> {
    use tokio::signal::unix::{signal, SignalKind};
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|e| e.to_string())?;
    let mut terminate = signal(SignalKind::terminate()).map_err(|e| e.to_string())?;
    let mut runner = Runner::spawn(paths, journal, spec, assets, instance)?;
    let execution = async {
        runner.ready(journal, instance).await?;
        runner.answer("Reply with the word OK.").await
    };
    let answer = tokio::select! {
        biased;
        _ = interrupt.recv() => Err("本机模型检查已由 Ctrl+C 停止。".into()),
        _ = terminate.recv() => Err("本机模型检查已收到停止信号。".into()),
        result = execution => result,
    };
    runner.stop(journal, instance).await?;
    answer
}

pub(super) struct Runner {
    child: Child,
    pid: u32,
    port: u16,
    key: String,
    _key_file: KeyFile,
    _slot_directory: Option<SlotDirectory>,
    client: Client,
}

#[derive(Clone)]
pub(super) struct RunnerConnection {
    client: Client,
    url: String,
    key: String,
}

impl Runner {
    pub fn spawn(
        paths: &Paths,
        journal: &Path,
        spec: &ModelSpec,
        assets: &Assets,
        instance: &mut Instance,
    ) -> Result<Self, String> {
        let port_reservation =
            TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).map_err(|e| e.to_string())?;
        let port = port_reservation
            .local_addr()
            .map_err(|e| e.to_string())?
            .port();
        let key = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        let key_path = paths
            .runtime
            .join(format!("model-key-{}", instance.instance_id));
        let mut key_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&key_path)
            .map_err(|e| e.to_string())?;
        let key_path = KeyFile(key_path);
        key_file
            .write_all(key.as_bytes())
            .map_err(|e| e.to_string())?;
        drop(key_file);
        let client = runner_client_builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .retry(reqwest::retry::never())
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| e.to_string())?;
        let log = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(
                paths
                    .logs
                    .join(format!("model-{}.log", instance.instance_id)),
            )
            .map_err(|e| e.to_string())?;
        let stdout = log.try_clone().map_err(|e| e.to_string())?;
        let mut command = Command::new(assets.runtime.join(RUNNER_NAME));
        command
            .env_clear()
            .env("PATH", "/usr/bin:/bin")
            .env("LANG", "C.UTF-8")
            .env("HOME", &paths.runtime)
            .env(
                "CUDA_VISIBLE_DEVICES",
                instance
                    .placement
                    .as_ref()
                    .map(Placement::visible_devices)
                    .unwrap_or_else(|| instance.gpu_uuid.clone()),
            )
            .env("LD_LIBRARY_PATH", &assets.runtime)
            .env("CUDA_CACHE_DISABLE", "1")
            .current_dir(&assets.runtime)
            .arg("--model")
            .arg(&assets.model)
            .args([
                "--host",
                "127.0.0.1",
                "--port",
                &port.to_string(),
                "--alias",
                ALIAS,
                "--parallel",
                &spec.parallel_slots.to_string(),
                "--no-webui",
                "--ctx-size",
                &(spec.context_tokens * u32::from(spec.parallel_slots)).to_string(),
                "--n-gpu-layers",
                &spec.gpu_layers.to_string(),
                "--threads",
                &spec.cpu_threads.to_string(),
                "--api-key-file",
            ])
            .arg(&key_path.0)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(log))
            .process_group(0)
            .kill_on_drop(true);
        if let Some(placement) = &instance.placement {
            command.args(placement.arguments());
        }
        command.args(spec.memory_arguments_without_common());
        let slot_directory = if spec.parallel_slots > 1 {
            let path = paths
                .runtime
                .join(format!("model-slots-{}", instance.instance_id));
            // b11669 gates erase on this option too. A private read-only
            // directory enables erase without saving request KV to disk.
            fs::DirBuilder::new()
                .mode(0o500)
                .create(&path)
                .map_err(|e| e.to_string())?;
            command.arg("--slot-save-path").arg(&path);
            Some(SlotDirectory(path))
        } else {
            None
        };
        let parent = std::process::id() as libc::pid_t;
        // SAFETY: the child hook uses only async-signal-safe system calls. The
        // parent check closes the race between fork and arming the death signal.
        unsafe {
            command.pre_exec(move || {
                if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) == -1 {
                    return Err(io::Error::last_os_error());
                }
                if libc::getppid() != parent {
                    return Err(io::Error::from_raw_os_error(libc::EINTR));
                }
                Ok(())
            });
        }
        private_fs::write_json(journal, instance)?;
        drop(port_reservation);
        let child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                instance.phase = Phase::Stopped;
                instance.process_group_exit_confirmed = true;
                instance.reserved_vram_bytes = 0;
                instance.reserved_memory_bytes = 0;
                if let Some(placement) = &mut instance.placement {
                    placement.clear_reservations();
                }
                private_fs::write_json(journal, instance)?;
                return Err(format!("模型进程未能启动：{error}"));
            }
        };
        let pid = child.id().expect("spawned child has not been reaped");
        instance.runner_pid = Some(pid);

        Ok(Self {
            child,
            pid,
            port,
            key,
            _key_file: key_path,
            _slot_directory: slot_directory,
            client,
        })
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub async fn ready(&mut self, journal: &Path, instance: &mut Instance) -> Result<(), String> {
        private_fs::write_json(journal, instance)?;
        let url = self.url();
        wait_ready(
            &mut self.child,
            self.pid,
            self.port,
            &self.client,
            &url,
            &self.key,
        )
        .await?;
        instance.phase = Phase::Ready;
        instance.ready_at_unix_ms = Some(state::now_ms()?);
        private_fs::write_json(journal, instance)
    }

    pub async fn answer(&mut self, prompt: &str) -> Result<Answer, String> {
        if !self.owns_endpoint()? {
            return Err("生成前无法确认模型进程仍持有监听端口。".into());
        }
        let response = self.connection().submit(prompt, None, 64).await?;
        parse_answer(response).await
    }

    pub fn connection(&self) -> RunnerConnection {
        RunnerConnection {
            client: self.client.clone(),
            url: self.url(),
            key: self.key.clone(),
        }
    }

    pub fn owns_endpoint(&mut self) -> Result<bool, String> {
        Ok(self.child.try_wait().map_err(|e| e.to_string())?.is_none()
            && owns_listener(self.pid, self.port))
    }

    pub async fn stop(mut self, journal: &Path, instance: &mut Instance) -> Result<(), String> {
        let cleanup = stop_owned_group(&mut self.child, self.pid).await;
        instance.process_group_exit_confirmed = cleanup.is_ok();
        instance.phase = if cleanup.is_ok() {
            Phase::Stopped
        } else {
            Phase::Quarantined
        };
        if cleanup.is_ok() {
            instance.reserved_vram_bytes = 0;
            instance.reserved_memory_bytes = 0;
            if let Some(placement) = &mut instance.placement {
                placement.clear_reservations();
            }
        }
        private_fs::write_json(journal, instance)?;
        cleanup
    }
}

impl RunnerConnection {
    async fn slots(&self) -> Result<Vec<Value>, String> {
        response_json(
            self.client
                .get(format!("{}/slots", self.url))
                .bearer_auth(&self.key)
                .timeout(SLOT_CLEANUP_TIMEOUT),
            MAX_RESPONSE,
        )
        .await?
        .as_array()
        .cloned()
        .ok_or_else(|| "模型运行器未返回槽位列表。".into())
    }

    pub async fn verify_slots(&self, spec: &ModelSpec) -> Result<(), String> {
        let slots = self.slots().await?;
        let count = u64::from(spec.parallel_slots);
        if slots.len() == count as usize
            && (0..count).all(|id| {
                slots.iter().any(|slot| {
                    slot["id"].as_u64() == Some(id)
                        && slot["n_ctx"].as_u64() == Some(u64::from(spec.context_tokens))
                        && slot["is_processing"].as_bool() == Some(false)
                })
            })
        {
            Ok(())
        } else {
            Err("模型运行器的实际槽位数或每路上下文与计划不一致。".into())
        }
    }

    pub async fn run_task(
        &self,
        spec: &ModelSpec,
        slot_id: u8,
        prompt: &str,
        cancel: &mut watch::Receiver<bool>,
    ) -> Result<TaskResult, String> {
        if *cancel.borrow() {
            return Ok(TaskResult::Cancelled {
                execution_started: false,
            });
        }
        if spec.parallel_slots == 1 {
            let response = self.submit(prompt, None, spec.output_tokens).await?;
            return Ok(TaskResult::Finished(Ok(parse_answer(response).await?)));
        }
        let reply = {
            let response = self.submit(prompt, Some(slot_id), spec.output_tokens);
            tokio::pin!(response);
            tokio::select! {
                biased;
                _ = async { let _ = cancel.wait_for(|requested| *requested).await; } => {
                    // Before dropping HTTP, prove that llama-server received this
                    // submission or finished it. An idle snapshot alone could
                    // precede a late submission.
                    tokio::select! {
                        result = &mut response => Some(result?),
                        result = timeout(SLOT_CLEANUP_TIMEOUT, self.wait_active(slot_id)) => {
                            result.map_err(|_| "取消前无法确认模型请求已接收。")??;
                            None
                        }
                    }
                }
                result = &mut response => Some(result?),
            }
        };
        let answer = match reply {
            Some(response) => TaskResult::Finished(parse_answer(response).await),
            None => TaskResult::Cancelled {
                execution_started: true,
            },
        };
        // b11669 defers erase until this slot is inactive. The erase reply
        // confirms KV cleanup after the completion or HTTP disconnect.
        let cleared = response_json(
            self.client
                .post(format!("{}/slots/{slot_id}?action=erase", self.url))
                .bearer_auth(&self.key)
                .timeout(SLOT_CLEANUP_TIMEOUT),
            16_384,
        )
        .await?;
        if cleared["id_slot"].as_u64() != Some(u64::from(slot_id))
            || cleared["n_erased"].as_u64().is_none()
        {
            return Err("模型运行器未确认指定槽位已清理。".into());
        }
        Ok(answer)
    }

    async fn wait_active(&self, id: u8) -> Result<(), String> {
        loop {
            let slots = self.slots().await?;
            let slot = slots
                .iter()
                .find(|slot| slot["id"].as_u64() == Some(u64::from(id)))
                .ok_or("模型运行器缺少指定槽位。")?;
            match slot["is_processing"].as_bool() {
                Some(true) => return Ok(()),
                Some(false) => sleep(Duration::from_millis(30)).await,
                None => return Err("模型运行器槽位状态无效。".into()),
            }
        }
    }

    async fn submit(
        &self,
        prompt: &str,
        slot_id: Option<u8>,
        max_tokens: u32,
    ) -> Result<reqwest::Response, String> {
        self.submit_with_timeout(prompt, slot_id, max_tokens, GENERATION_TIMEOUT)
            .await
    }

    async fn submit_with_timeout(
        &self,
        prompt: &str,
        slot_id: Option<u8>,
        max_tokens: u32,
        budget: Duration,
    ) -> Result<reqwest::Response, String> {
        let mut body = json!({
            "model": ALIAS,
            "messages": [{"role":"user","content":prompt}],
            "max_tokens": max_tokens, "temperature": 0, "stream": false, "cache_prompt": false,
            "chat_template_kwargs": {"enable_thinking": false}
        });
        if let Some(id) = slot_id {
            body["id_slot"] = json!(id);
        }
        self.client
            .post(format!("{}/v1/chat/completions", self.url))
            .header(reqwest::header::CONNECTION, "close")
            .bearer_auth(&self.key)
            .json(&body)
            .timeout(budget)
            .send()
            .await
            .map_err(|error| {
                let code = if error.is_timeout() {
                    "model_inference_timeout"
                } else if error.is_connect() {
                    "model_runner_connect_failed"
                } else {
                    "model_inference_transport_error"
                };
                format!(
                    "模型 HTTP 请求失败：{code}；推理时限 {} 秒；{}；请核对对应实例运行日志。",
                    budget.as_secs(),
                    error.without_url()
                )
            })
    }
}

fn runner_client_builder() -> reqwest::ClientBuilder {
    // iroh enables reqwest's rustls-no-provider feature in this binary.
    // Reqwest initializes its TLS backend even for a loopback HTTP client.
    static PROVIDER: std::sync::Once = std::sync::Once::new();
    PROVIDER.call_once(|| {
        let _ = rustls::crypto::ring::default_provider().install_default();
    });
    Client::builder()
}

async fn parse_answer(response: reqwest::Response) -> Result<Answer, String> {
    let response = read_response_json(response, MAX_RESPONSE).await?;
    let text = response["choices"][0]["message"]["content"]
        .as_str()
        .filter(|text| !text.trim().is_empty())
        .ok_or("模型未返回可用文本。")?;
    Ok(Answer {
        text: text.to_owned(),
        input_tokens: response["usage"]["prompt_tokens"].as_u64(),
        output_tokens: response["usage"]["completion_tokens"].as_u64(),
    })
}

fn owns_listener(pid: u32, port: u16) -> bool {
    let Ok(table) = fs::read_to_string("/proc/net/tcp") else {
        return false;
    };
    let address = format!("0100007F:{port:04X}");
    let inodes: HashSet<_> = table
        .lines()
        .skip(1)
        .filter_map(|row| {
            let columns: Vec<_> = row.split_whitespace().collect();
            (columns.get(1) == Some(&address.as_str()) && columns.get(3) == Some(&"0A"))
                .then(|| columns.get(9))
                .flatten()
                .copied()
        })
        .collect();
    if inodes.is_empty() {
        return false;
    }
    let Ok(descriptors) = fs::read_dir(format!("/proc/{pid}/fd")) else {
        return false;
    };
    descriptors
        .flatten()
        .filter_map(|entry| fs::read_link(entry.path()).ok())
        .any(|path| {
            path.to_str()
                .and_then(|value| value.strip_prefix("socket:["))
                .and_then(|value| value.strip_suffix(']'))
                .is_some_and(|inode| inodes.contains(inode))
        })
}

async fn wait_ready(
    child: &mut Child,
    pid: u32,
    port: u16,
    client: &Client,
    url: &str,
    key: &str,
) -> Result<(), String> {
    timeout(START_TIMEOUT, async {
        let mut successes = 0;
        loop {
            if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
                return Err(format!("模型就绪前退出（{status}），请检查本机模型日志。"));
            }
            if owns_listener(pid, port)
                && probe_ready(client, url, key).await
                && owns_listener(pid, port)
            {
                successes += 1;
                if successes == 2 {
                    return Ok(());
                }
            } else {
                successes = 0;
            }
            sleep(Duration::from_millis(250)).await;
        }
    })
    .await
    .map_err(|_| "模型健康检查超时，请检查本机模型日志。".to_owned())?
}

async fn probe_ready(client: &Client, url: &str, key: &str) -> bool {
    let health = response_json(
        client
            .get(format!("{url}/health"))
            .timeout(Duration::from_secs(1)),
        16_384,
    )
    .await;
    if !health.is_ok_and(|value| value["status"] == "ok") {
        return false;
    }
    response_json(
        client
            .get(format!("{url}/v1/models"))
            .bearer_auth(key)
            .timeout(Duration::from_secs(1)),
        16_384,
    )
    .await
    .is_ok_and(|value| {
        value["data"]
            .as_array()
            .is_some_and(|models| models.iter().any(|m| m["id"] == ALIAS))
    })
}

async fn response_json(request: RequestBuilder, limit: usize) -> Result<Value, String> {
    let response = request
        .send()
        .await
        .map_err(|e| format!("模型 HTTP 请求失败：{}", e.without_url()))?;
    read_response_json(response, limit).await
}

async fn read_response_json(
    mut response: reqwest::Response,
    limit: usize,
) -> Result<Value, String> {
    if response.status() != StatusCode::OK {
        return Err(format!("模型 HTTP 状态：{}", response.status()));
    }
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| e.without_url().to_string())?
    {
        if chunk.len() > limit.saturating_sub(body.len()) {
            return Err("模型 HTTP 响应超过本机限制。".into());
        }
        body.extend_from_slice(&chunk);
    }
    serde_json::from_slice(&body).map_err(|_| "模型 HTTP 响应 JSON 无效。".into())
}

async fn stop_owned_group(child: &mut Child, pid: u32) -> Result<(), String> {
    let group = -(pid as libc::pid_t);
    if child.try_wait().map_err(|e| e.to_string())?.is_none() {
        // The live Child owns the unreaped group leader, so its PID cannot
        // be recycled between this check and signaling its process group.
        send_group_signal(group, libc::SIGTERM)?;
        if timeout(Duration::from_secs(2), child.wait()).await.is_err() {
            send_group_signal(group, libc::SIGKILL)?;
            timeout(Duration::from_secs(2), child.wait())
                .await
                .map_err(|_| "无法确认模型进程退出，保留资源预留。".to_owned())?
                .map_err(|e| e.to_string())?;
        } else {
            child
                .try_wait()
                .map_err(|e| e.to_string())?
                .ok_or("无法确认模型进程退出。")?;
        }
    }
    // SAFETY: signal zero probes existence; it does not signal a reused PID.
    if unsafe { libc::kill(group, 0) } == -1
        && io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
    {
        Ok(())
    } else {
        Err("模型进程组仍存在或无法核实，保留资源预留。".into())
    }
}

fn send_group_signal(group: libc::pid_t, signal: libc::c_int) -> Result<(), String> {
    // SAFETY: group is the negative PID of our still-unreaped Child.
    if unsafe { libc::kill(group, signal) } == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error.to_string())
    }
}

#[cfg(test)]
mod generation_timeout_tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn delayed_engine(delay: Duration) -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let job = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 8192];
            let _ = stream.read(&mut request).await.unwrap();
            sleep(delay).await;
            let body = r#"{"choices":[{"message":{"content":"OK"}}],"usage":{"prompt_tokens":2,"completion_tokens":1}}"#;
            let response=format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len());
            let _ = stream.write_all(response.as_bytes()).await;
        });
        (format!("http://{address}"), job)
    }

    fn connection(url: String) -> RunnerConnection {
        RunnerConnection {
            url,
            key: "private-test-key".into(),
            client: runner_client_builder()
                .no_proxy()
                .retry(reqwest::retry::never())
                .timeout(Duration::from_millis(20))
                .build()
                .unwrap(),
        }
    }

    #[test]
    fn slow_generation_overrides_short_control_client_timeout() {
        lifecycle::runtime().unwrap().block_on(async {
            let (url, job) = delayed_engine(Duration::from_millis(100)).await;
            let response = connection(url)
                .submit_with_timeout("hello", Some(0), 8, Duration::from_secs(2))
                .await
                .unwrap();
            let answer = parse_answer(response).await.unwrap();
            assert_eq!(answer.text, "OK");
            assert_eq!(answer.output_tokens, Some(1));
            job.await.unwrap();
        });
    }

    #[test]
    fn generation_still_has_a_bounded_deadline_and_reports_timeout_without_secrets() {
        lifecycle::runtime().unwrap().block_on(async {
            let (url, job) = delayed_engine(Duration::from_millis(200)).await;
            let error = connection(url)
                .submit_with_timeout("hello", Some(0), 8, Duration::from_millis(50))
                .await
                .unwrap_err();
            assert!(error.contains("model_inference_timeout"), "{error}");
            assert!(!error.contains("private-test-key") && !error.contains("http://"));
            job.await.unwrap();
        });
    }
}
