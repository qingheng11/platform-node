use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    io::Read,
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{local_model, private_fs, state::Paths, web_result};
use nvml_wrapper::Nvml;
use platform_node_link::{
    local_node::{Answer, Failure, Gpu, Inventory, LocalNode, Output},
    receive, send, tls, web, Heartbeat, Reply, Report, FRAME_LIMIT, HEARTBEAT, IO_TIMEOUT,
    MAX_IN_FLIGHT, VERSION,
};
use serde::Deserialize;
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::TcpStream,
    signal::unix::{signal, SignalKind},
    sync::watch,
    task::JoinSet,
    time::{sleep, timeout},
};

const MAX_INVENTORY_AGE: Duration = Duration::from_secs(18);
type InventorySample = Option<(Instant, Inventory)>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    node_id: String,
    scheduler: SocketAddr,
    server_name: String,
    ca_certificate: PathBuf,
    certificate: PathBuf,
    private_key: PathBuf,
    experience_relay_url: Option<String>,
    experience_relay_token_file: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RelayCredential {
    token: String,
}

pub(super) fn gpu_inventory() -> Result<Vec<Gpu>, String> {
    // Scheduling only needs NVML device identity and memory. The detailed
    // daemon diagnostic also scans system facts and per-process memory.
    let nvml = Nvml::init().map_err(|error| format!("GPU 库存无法初始化 NVML（{error}）。"))?;
    let count = nvml
        .device_count()
        .map_err(|error| format!("读取 GPU 数量失败（{error}）。"))?;
    let mut gpus = Vec::new();
    for index in 0..count {
        let Ok(device) = nvml.device_by_index(index) else {
            continue;
        };
        let memory = device.memory_info().ok().filter(|memory| memory.total > 0);
        gpus.push(Gpu {
            index,
            name: device
                .name()
                .unwrap_or_else(|_| format!("NVIDIA GPU {index}")),
            uuid: device.uuid().ok(),
            free_bytes: memory.as_ref().map(|memory| memory.free.min(memory.total)),
            total_bytes: memory.map(|memory| memory.total),
        });
    }
    if count != 0 && gpus.iter().all(|gpu| gpu.free_bytes.is_none()) {
        return Err("NVML 未返回可用的逐卡显存信息。".into());
    }
    Ok(gpus)
}

pub fn run(config_file: PathBuf, paths: Paths) -> Result<(), String> {
    let mut bytes = Vec::new();
    File::open(config_file)
        .and_then(|file| file.take(65_537).read_to_end(&mut bytes))
        .map_err(|error| format!("读取节点联调配置失败（{error}）"))?;
    if bytes.len() > 65_536 {
        return Err("节点联调配置超过大小限制。".into());
    }
    let config: Config =
        serde_json::from_slice(&bytes).map_err(|error| format!("节点联调配置无效（{error}）"))?;
    if !local_model::valid_component(&config.node_id) {
        return Err("节点联调 ID 无效。".into());
    }
    let _lock = private_fs::lock(&paths.runtime.join("remote-link.lock"), false)?;
    let server_name = tls::ServerName::try_from(config.server_name.clone())
        .map_err(|_| "调度器证书名称无效。")?;
    let connector = tls::client(
        &config.ca_certificate,
        &config.certificate,
        &config.private_key,
    )?;
    let experience_key = config
        .experience_relay_url
        .as_ref()
        .map(|_| {
            let installation = crate::state::require_installation(&paths)?;
            web_result::identity(&paths.state, &installation.installation_id)
        })
        .transpose()?;
    let node = LocalNode {
        id: config.node_id,
        binary: std::env::current_exe().map_err(|error| error.to_string())?,
        config_home: dirs::config_dir().ok_or("无法确定节点配置根。")?,
        data_dir: paths.state,
        plans: Vec::new(),
    };
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async {
        let mut interrupt = signal(SignalKind::interrupt()).map_err(|error| error.to_string())?;
        let mut terminate = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
        let (inventory, samples) = watch::channel(None);
        let relay_token = config.experience_relay_token_file.as_deref().map(|path| -> Result<String, String> {
            if !path.is_absolute() { return Err("Relay 准入凭据文件必须使用绝对路径。".into()); }
            private_fs::read_json::<RelayCredential>(path)?
                .map(|credential| credential.token).ok_or_else(|| "无法读取 Relay 准入凭据文件。".into())
        }).transpose()?;
        let experience = match config.experience_relay_url.zip(experience_key) {
            Some((url, key)) => {
                let service = web_result::Service::bind(&url, relay_token, key).await?;
                println!("{}", serde_json::json!({
                    "event": "scheduler_iroh_endpoint_created",
                    "node_id": node.id,
                    "endpoint_id": service.target().endpoint_id,
                    "relay_url": service.target().relay_url,
                }));
                Some(Arc::new(service))
            }
            None if relay_token.is_some() => return Err("Relay 准入凭据需要对应的 Relay 地址。".into()),
            None => None,
        };
        let result = tokio::select! {
            _ = interrupt.recv() => Ok(()),
            _ = terminate.recv() => Ok(()),
            result = maintain(config.scheduler, server_name, connector, &node, &samples, experience.as_deref()) => result,
            _ = sample_inventory(&node, inventory) => Err("节点库存采集意外结束。".into()),
            result = async {
                match &experience {
                    Some(service) => service.listen().await,
                    None => std::future::pending().await,
                }
            } => result,
        };
        if let Some(service) = experience { service.close().await; }
        result
    })
}

async fn sample_inventory(node: &LocalNode, samples: watch::Sender<InventorySample>) {
    let mut last_error = None;
    loop {
        let sampled_at = Instant::now();
        let inventory = match node.scheduling_inventory().await {
            Ok(inventory) => {
                last_error = None;
                Some((sampled_at, inventory))
            }
            Err(error) => {
                if last_error.as_ref() != Some(&error) {
                    eprintln!("远程联调库存采集失败：{error}");
                    last_error = Some(error);
                }
                None
            }
        };
        samples.send_replace(inventory);
        sleep(HEARTBEAT).await;
    }
}

async fn maintain(
    address: SocketAddr,
    name: tls::ServerName<'static>,
    connector: tls::TlsConnector,
    node: &LocalNode,
    samples: &watch::Receiver<InventorySample>,
    experience: Option<&web_result::Service>,
) -> Result<(), String> {
    let mut last_error = None;
    loop {
        let mut jobs = JoinSet::new();
        let connection = timeout(IO_TIMEOUT, async {
            let stream = TcpStream::connect(address).await?;
            stream.set_nodelay(true)?;
            connector.connect(name.clone(), stream).await
        })
        .await;
        let error = match connection {
            Ok(Ok(mut stream)) => {
                exchange(
                    &mut stream,
                    node,
                    samples,
                    &mut jobs,
                    &mut last_error,
                    experience,
                )
                .await
            }
            Ok(Err(error)) => format!("连接调度器失败（{error}）"),
            Err(_) => "连接调度器超时。".into(),
        };
        if let Some(service) = experience {
            service.state(None, false);
        }
        if last_error.as_ref() != Some(&error) {
            eprintln!("{error}");
            last_error = Some(error);
        }
        // The previous connection cannot confirm these outcomes. Let local
        // work settle before accepting jobs on a new connection.
        while jobs.join_next().await.is_some() {}
        sleep(HEARTBEAT).await;
    }
}

async fn exchange(
    stream: &mut (impl AsyncRead + AsyncWrite + Unpin),
    node: &LocalNode,
    samples: &watch::Receiver<InventorySample>,
    jobs: &mut JoinSet<Report>,
    last_error: &mut Option<String>,
    experience: Option<&web_result::Service>,
) -> String {
    let mut accepted = HashMap::<uuid::Uuid, (uuid::Uuid, watch::Sender<bool>)>::new();
    let mut grants = HashMap::new();
    let mut finished = VecDeque::new();
    let mut registered = false;
    loop {
        while let Some(report) = jobs.try_join_next() {
            match report {
                Ok(report) => finished.push_back(report),
                Err(_) => return "本机任务进程异常，远程会话已结束。".into(),
            }
        }
        let inventory = samples
            .borrow()
            .as_ref()
            .filter(|(sampled_at, _)| sampled_at.elapsed() < MAX_INVENTORY_AGE)
            .map(|(_, inventory)| inventory.clone());
        let run = inventory.as_ref().map(|inventory| inventory.run.clone());
        let mut result = finished.pop_front();
        if let (Some(service), Some(report)) = (experience, result.as_mut()) {
            if let (Some((run, grant)), Ok(output)) =
                (grants.remove(&report.attempt), &mut report.result)
            {
                let direct = Output {
                    task_id: output.task_id,
                    instance_id: output.instance_id,
                    answer: Answer {
                        text: std::mem::take(&mut output.answer.text),
                        input_tokens: output.answer.input_tokens,
                        output_tokens: output.answer.output_tokens,
                    },
                };
                // The scheduler reveals the grant only after recording success.
                // Storing first keeps the body available if its receipt is lost.
                service.save(&run, report.attempt, grant, direct);
            }
        }
        let settled = result.as_ref().map(|result| result.attempt);
        let endpoint =
            experience.and_then(|service| run.as_ref().and_then(|run| service.registration(run)));
        let advertised = endpoint.is_some();
        let heartbeat = Heartbeat {
            version: VERSION,
            inventory,
            available_slots: MAX_IN_FLIGHT - accepted.len() + usize::from(settled.is_some()),
            result,
            endpoint,
        };
        let reply: Reply = match timeout(IO_TIMEOUT, async {
            send(stream, &heartbeat, FRAME_LIMIT).await?;
            receive(stream, FRAME_LIMIT).await
        })
        .await
        {
            Ok(Ok(reply)) => reply,
            Ok(Err(error)) => return format!("调度连接结束（{error}）"),
            Err(_) => return "调度心跳未收到回执。".into(),
        };
        if reply.version != VERSION
            || reply.node_id != node.id
            || reply.cancel_attempts.len() > MAX_IN_FLIGHT
            || accepted
                .len()
                .saturating_sub(usize::from(settled.is_some()))
                + reply.assignments.len()
                > MAX_IN_FLIGHT
        {
            return "调度回执的节点身份、版本或任务数无效。".into();
        }
        if let Some(settled) = settled {
            accepted.remove(&settled);
        }
        for attempt in reply.cancel_attempts {
            if let Some((_, cancellation)) = accepted.get(&attempt) {
                cancellation.send_replace(true);
            }
        }
        if let Some(service) = experience {
            service.state(run.as_ref(), advertised);
        }
        if !registered {
            println!("远程调度已登记为 {}，正在上报心跳。", node.id);
            *last_error = None;
            registered = true;
        }
        for assignment in reply.assignments {
            let id = assignment.id;
            let attempt = assignment.attempt;
            if accepted.contains_key(&attempt) || accepted.values().any(|(task, _)| *task == id) {
                return "远程调度回执含重复的活动任务。".into();
            }
            let (cancel, cancellation) = watch::channel(false);
            accepted.insert(attempt, (id, cancel));
            if run.as_ref() != Some(&assignment.run)
                || !local_model::valid_component(&assignment.plan)
                || assignment.prompt.trim().is_empty()
                || assignment.prompt.len() > 2048
                || assignment.web.as_ref().is_some_and(|grant| {
                    !advertised
                        || grant.ticket.is_nil()
                        || web::endpoint_bytes(&grant.browser_endpoint_id).is_none()
                })
            {
                finished.push_back(Report {
                    attempt,
                    result: Err(Failure::Retry("节点库存已变化，任务未启动。".into())),
                });
                continue;
            }
            if let Some(grant) = assignment.web {
                grants.insert(attempt, (assignment.run.clone(), grant));
            }
            let node = node.clone();
            jobs.spawn(async move {
                let result = node
                    .generate_cancellable(
                        &assignment.plan,
                        id,
                        &assignment.prompt,
                        assignment.run.run_id,
                        cancellation,
                    )
                    .await;
                Report { attempt, result }
            });
        }
        sleep(HEARTBEAT).await;
    }
}

#[cfg(test)]
pub(super) async fn test_connect(
    config_file: PathBuf,
    node: &LocalNode,
    service: &web_result::Service,
) -> Result<(), String> {
    let config: Config = serde_json::from_slice(&std::fs::read(config_file).unwrap()).unwrap();
    let name = tls::ServerName::try_from(config.server_name).unwrap();
    let tls = tls::client(
        &config.ca_certificate,
        &config.certificate,
        &config.private_key,
    )?;
    let (inventory, samples) = watch::channel(None);
    tokio::select! {
        result = maintain(config.scheduler, name, tls, node, &samples, Some(service)) => result,
        _ = sample_inventory(node, inventory) => Err("test inventory ended".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, os::unix::fs::PermissionsExt};

    #[tokio::test]
    async fn slow_gpu_query_keeps_heartbeats_alive_and_eventually_reports_inventory() {
        let root = tempfile::tempdir().unwrap();
        let binary = root.path().join("test-node");
        fs::write(&binary, r#"#!/bin/sh
case "$1" in
  status) sleep 0.1; echo '{"local_status":"running","runtime":{"installation_id":"11111111-1111-4111-8111-111111111111","generation":1,"run_id":"22222222-2222-4222-8222-222222222222"}}';;
  model-list) echo '{"instances":[]}';;
  gpu) [ "$2" = inventory ] || exit 1; sleep 5.2; echo '{"gpus":[{"index":0,"name":"Tesla V100","vram_free_bytes":1024}]}';;
  *) exit 1;;
esac
"#).unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o700)).unwrap();
        let node = LocalNode {
            id: "v100-test".into(),
            binary,
            config_home: root.path().into(),
            data_dir: root.path().into(),
            plans: Vec::new(),
        };
        let (inventory, samples) = watch::channel(None);
        let (mut agent, mut scheduler) = tokio::io::duplex(8192);
        let mut jobs = JoinSet::new();
        let mut last_error = None;
        let check = async {
            for _ in 0..3 {
                let heartbeat: Heartbeat = receive(&mut scheduler, FRAME_LIMIT).await.unwrap();
                assert!(heartbeat.inventory.is_none());
                send(
                    &mut scheduler,
                    &Reply {
                        version: VERSION,
                        node_id: node.id.clone(),
                        assignments: Vec::new(),
                        cancel_attempts: Vec::new(),
                    },
                    FRAME_LIMIT,
                )
                .await
                .unwrap();
            }
            let heartbeat: Heartbeat = receive(&mut scheduler, FRAME_LIMIT).await.unwrap();
            assert_eq!(heartbeat.inventory.unwrap().gpus[0].name, "Tesla V100");
        };
        tokio::select! {
            result = timeout(Duration::from_secs(9), check) => result.unwrap(),
            error = exchange(&mut agent, &node, &samples, &mut jobs, &mut last_error, None) => panic!("{error}"),
            _ = sample_inventory(&node, inventory) => panic!("采集任务提前结束"),
        }
    }
}
