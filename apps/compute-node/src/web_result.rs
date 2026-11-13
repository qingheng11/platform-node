use std::{
    collections::HashMap,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};

use iroh::{
    endpoint::presets::Minimal, Endpoint, RelayConfig, RelayMap, RelayMode, RelayUrl, SecretKey,
    Watcher,
};
use platform_node_link::{
    local_node::{Output, Run},
    receive, web, IO_TIMEOUT,
};
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncWriteExt, task::JoinSet, time::timeout};
use uuid::Uuid;

const MAX_RESULTS: usize = 256;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    installation_id: String,
    secret: [u8; 32],
}

pub(super) fn identity(state: &Path, installation_id: &str) -> Result<SecretKey, String> {
    let path = state.join("scheduled-iroh-identity.json");
    if let Some(identity) = crate::private_fs::read_json::<Identity>(&path)? {
        if identity.installation_id != installation_id {
            return Err("网页体验 iroh 端点身份与本机安装 ID 不匹配。".into());
        }
        return Ok(SecretKey::from_bytes(&identity.secret));
    }
    let key = SecretKey::generate();
    crate::private_fs::write_json(
        &path,
        &Identity {
            installation_id: installation_id.to_owned(),
            secret: key.to_bytes(),
        },
    )?;
    Ok(key)
}

pub(super) struct Service {
    endpoint: Endpoint,
    relay: Option<RelayUrl>,
    target: web::Target,
    cache: Mutex<Cache>,
}

#[derive(Default)]
struct Cache {
    run: Option<Run>,
    available: bool,
    results: HashMap<Uuid, Record>,
}

struct Record {
    task: Uuid,
    grant: web::Grant,
    expires: Instant,
    response: Vec<u8>,
}

impl Service {
    pub async fn bind(relay: &str, token: Option<String>, key: SecretKey) -> Result<Self, String> {
        if !web::valid_relay(relay) {
            return Err("网页体验 Relay 必须是 HTTPS 地址。".into());
        }
        let relay_url = relay.to_owned();
        let relay: RelayUrl = relay.parse().map_err(|_| "网页体验 Relay 地址无效。")?;
        let mut relay_config = RelayConfig::from(relay.clone());
        if let Some(token) = token {
            if token.is_empty()
                || token.len() > 4096
                || !token.bytes().all(|byte| byte.is_ascii_graphic())
            {
                return Err("网页体验 Relay 准入凭据格式无效。".into());
            }
            relay_config = relay_config.with_auth_token(token);
        }
        let endpoint = Endpoint::builder(Minimal)
            .secret_key(key)
            .clear_ip_transports()
            .relay_mode(RelayMode::Custom(RelayMap::from_iter([relay_config])))
            .alpns(vec![web::ALPN.as_bytes().to_vec()])
            .bind()
            .await
            .map_err(|error| format!("网页体验 iroh 端点启动失败（{error}）"))?;
        let target = web::Target {
            endpoint_id: endpoint.id().to_string(),
            relay_url,
        };
        Ok(Self {
            endpoint,
            relay: Some(relay),
            target,
            cache: Mutex::default(),
        })
    }

    pub fn target(&self) -> &web::Target {
        &self.target
    }

    fn relay_connected(&self) -> bool {
        self.relay.as_ref().is_none_or(|relay| {
            self.endpoint
                .home_relay_status()
                .get()
                .iter()
                .any(|status| status.url() == relay && status.is_connected())
        })
    }

    pub fn registration(&self, run: &Run) -> Option<web::Registration> {
        if !self.relay_connected() {
            return None;
        }
        let signature = self
            .endpoint
            .secret_key()
            .sign(&self.target.signing_bytes(run));
        Some(web::Registration {
            target: self.target.clone(),
            signature: signature.to_bytes().to_vec(),
        })
    }

    pub fn state(&self, run: Option<&Run>, available: bool) {
        let mut cache = self.cache.lock().unwrap();
        if let Some(run) = run.filter(|run| cache.run.as_ref() != Some(*run)) {
            cache.run = Some(run.clone());
            cache.results.clear();
        }
        cache.available = available;
        cache
            .results
            .retain(|_, record| record.expires > Instant::now());
    }

    pub fn save(&self, run: &Run, attempt: Uuid, grant: web::Grant, output: Output) {
        let mut cache = self.cache.lock().unwrap();
        if cache.run.as_ref() != Some(run) {
            return;
        }
        let task = output.task_id;
        let response = serde_json::to_vec(&web::Response::Result {
            attempt_id: attempt,
            output,
        })
        .ok()
        .filter(|response| response.len() <= web::FRAME_MAX)
        .unwrap_or_else(|| error("result_too_large"));
        if cache.results.len() >= MAX_RESULTS {
            if let Some(oldest) = cache
                .results
                .iter()
                .min_by_key(|(_, record)| record.expires)
                .map(|(id, _)| *id)
            {
                cache.results.remove(&oldest);
            }
        }
        cache.results.insert(
            attempt,
            Record {
                task,
                grant,
                response,
                expires: Instant::now() + web::RESULT_TTL,
            },
        );
    }

    fn response(&self, attach: web::Attach, peer: &str) -> Vec<u8> {
        let cache = self.cache.lock().unwrap();
        if cache.available && self.relay_connected() {
            if let Some(record) = cache.results.get(&attach.attempt_id).filter(|record| {
                record.expires > Instant::now()
                    && record.task == attach.task_id
                    && record.grant.browser_endpoint_id == peer
                    && record.grant.ticket == attach.ticket
            }) {
                return record.response.clone();
            }
        }
        error("result_unavailable")
    }

    pub async fn listen(self: &Arc<Self>) -> Result<(), String> {
        let mut connections = JoinSet::new();
        // Each connection consumes one stream. Reconnection with the same peer and
        // unexpired task grant re-reads the stored result without running the model.
        loop {
            tokio::select! {
                incoming = self.endpoint.accept() => {
                    let incoming = incoming.ok_or("网页体验 iroh 端点已关闭。")?;
                    if connections.len() >= 32 { incoming.refuse(); continue; }
                    let service = Arc::clone(self);
                    connections.spawn(async move {
                        let Ok(Ok(connection)) = timeout(IO_TIMEOUT, incoming).await else { return; };
                        let _ = timeout(IO_TIMEOUT, async {
                            let (mut send, mut recv) = connection.accept_bi().await.ok()?;
                            let attach = receive(&mut recv, web::FRAME_MAX).await.ok()?;
                            let response = service.response(attach, &connection.remote_id().to_string());
                            send.write_u32(response.len() as u32).await.ok()?;
                            send.write_all(&response).await.ok()?;
                            send.finish().ok()?;
                            let _ = connection.closed().await;
                            Some(())
                        }).await;
                    });
                }
                _ = connections.join_next(), if !connections.is_empty() => {}
            }
        }
    }

    pub async fn close(&self) {
        self.endpoint.close().await;
    }
}

fn error(code: &str) -> Vec<u8> {
    serde_json::to_vec(&web::Response::Error { code: code.into() }).expect("static error frame")
}

#[cfg(test)]
mod tests {
    use super::*;
    use iroh::EndpointAddr;
    use platform_node_link::send;
    use serde_json::Value;
    use std::{path::PathBuf, process::Stdio, time::Duration};
    use tokio::process::Command;

    #[test]
    fn endpoint_identity_is_stable_and_bound_to_installation() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        std::fs::set_permissions(directory.path(), std::fs::Permissions::from_mode(0o700)).unwrap();
        let installation = Uuid::new_v4().to_string();
        let original = identity(directory.path(), &installation).unwrap();
        let reopened = identity(directory.path(), &installation).unwrap();
        assert_eq!(original.public(), reopened.public());
        assert!(identity(directory.path(), &Uuid::new_v4().to_string()).is_err());
    }

    async fn loopback_endpoint() -> Endpoint {
        Endpoint::builder(Minimal)
            .clear_ip_transports()
            .clear_relay_transports()
            .bind_addr("127.0.0.1:0")
            .unwrap()
            .alpns(vec![web::ALPN.as_bytes().to_vec()])
            .bind()
            .await
            .unwrap()
    }

    async fn browser_endpoint(relay: &str, direct: bool) -> Endpoint {
        if direct {
            return loopback_endpoint().await;
        }
        let browser = Service::bind(relay, None, SecretKey::generate())
            .await
            .unwrap();
        if timeout(Duration::from_secs(18), browser.endpoint.online())
            .await
            .is_err()
        {
            for status in browser.endpoint.home_relay_status().get() {
                eprintln!(
                    "Relay diagnostic: denied={:?}, error={:?}",
                    status.auth_denied_reason(),
                    status.last_error()
                );
            }
            panic!("browser Relay did not become ready");
        }
        browser.endpoint
    }

    #[test]
    #[ignore = "requires a test scheduler, a model node and access to the configured Relay"]
    fn scheduled_iroh_answer() {
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(async {
            let relay = std::env::var("COMPUTE_WEB_TEST_RELAY").unwrap();
            let scheduler = PathBuf::from(std::env::var_os("COMPUTE_WEB_TEST_SCHEDULER").unwrap());
            let state = PathBuf::from(std::env::var_os("COMPUTE_WEB_TEST_STATE").unwrap());
            let plan = std::env::var("COMPUTE_WEB_TEST_PLAN").unwrap();
            let direct = std::env::var_os("COMPUTE_WEB_TEST_DIRECT").is_some();
            let browser = browser_endpoint(&relay, direct).await;
            let simulated_route = if direct {
                use platform_node_link::local_node::LocalNode;
                let endpoint = loopback_endpoint().await;
                let addr = endpoint.addr();
                let service = Arc::new(Service { target: web::Target { endpoint_id: endpoint.id().to_string(), relay_url: relay.clone() },
                    endpoint, relay: None, cache: Mutex::default() });
                let node = LocalNode { id: "gpu-test".into(), binary: PathBuf::from(std::env::var_os("COMPUTE_WEB_TEST_NODE").unwrap()),
                    config_home: dirs::config_dir().unwrap(), data_dir: PathBuf::from(std::env::var_os("COMPUTE_WEB_TEST_DATA").unwrap()),
                    plans: Vec::new() };
                let config = PathBuf::from(std::env::var_os("COMPUTE_WEB_TEST_NODE_CONFIG").unwrap());
                tokio::spawn(async move {
                    tokio::select! {
                        result = crate::remote::test_connect(config, &node, &service) => panic!("{result:?}"),
                        result = service.listen() => panic!("{result:?}"),
                    }
                });
                Some(addr)
            } else { None };
            let task = Uuid::new_v4();
            let mut submit = Command::new(&scheduler).arg("submit-web").arg(&state)
                .args([&plan, &browser.id().to_string(), &task.to_string()])
                .stdin(Stdio::piped()).stdout(Stdio::piped()).spawn().unwrap();
            submit.stdin.take().unwrap().write_all("请只回答 OK。".as_bytes()).await.unwrap();
            assert!(submit.wait_with_output().await.unwrap().status.success());

            let route = timeout(Duration::from_secs(110), async {
                loop {
                    let output = Command::new(&scheduler).arg("task").arg(&state).arg(task.to_string()).output().await.unwrap();
                    assert!(output.status.success());
                    let row: Value = serde_json::from_slice(&output.stdout).unwrap();
                    assert!(row["state"] != "failed" && row["state"] != "outcome_unknown");
                    if row["result_route"].is_object() {
                        assert!(row["output"]["answer"].get("text").is_none());
                        break row["result_route"].clone();
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
            }).await.unwrap();
            let selected: web::Target = serde_json::from_value(route["target"].clone()).unwrap();
            assert_eq!(route["alpn"], web::ALPN);
            let target = match simulated_route {
                Some(addr) => { assert_eq!(addr.id.to_string(), selected.endpoint_id); addr }
                None => EndpointAddr::new(selected.endpoint_id.parse().unwrap()).with_relay_url(selected.relay_url.parse().unwrap()),
            };
            let attempt = serde_json::from_value(route["attempt_id"].clone()).unwrap();
            let ticket = serde_json::from_value(route["ticket"].clone()).unwrap();
            let read = |endpoint: Endpoint, ticket: Uuid| {
                let target = target.clone();
                async move {
                    timeout(Duration::from_secs(15), async {
                        let connection = endpoint.connect(target, web::ALPN.as_bytes()).await.unwrap();
                        let (mut tx, mut rx) = connection.open_bi().await.unwrap();
                        send(&mut tx, &web::Attach { task_id: task, attempt_id: attempt, ticket }, web::FRAME_MAX).await.unwrap();
                        tx.finish().unwrap();
                        let response: web::Response = receive(&mut rx, web::FRAME_MAX).await.unwrap();
                        connection.close(0u32.into(), b"read complete");
                        response
                    }).await.unwrap()
                }
            };
            let wrong_peer = browser_endpoint(&relay, direct).await;
            assert!(matches!(read(wrong_peer.clone(), ticket).await, web::Response::Error { .. }));
            assert!(matches!(read(browser.clone(), Uuid::new_v4()).await, web::Response::Error { .. }));
            let answer = read(browser.clone(), ticket).await;
            let again = read(browser.clone(), ticket).await;
            let web::Response::Result { attempt_id, output } = answer else { panic!("valid peer did not receive the result") };
            assert_eq!((attempt_id, output.task_id), (attempt, task));
            assert!(output.answer.text.contains("OK"));
            assert!(matches!(again, web::Response::Result { output: repeated, .. } if repeated.instance_id == output.instance_id && repeated.answer.text == output.answer.text));
            println!("scheduler selected the iroh result peer; direct_loopback={direct}; wrong peer and ticket rejected; repeated read returned the same result; usage={:?}/{:?}", output.answer.input_tokens, output.answer.output_tokens);
            wrong_peer.close().await;
            browser.close().await;
        });
    }
}
