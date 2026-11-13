use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    net::SocketAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use http_body_util::{BodyExt, Full, Limited};
use hyper::{
    body::{Bytes, Incoming},
    header,
    server::conn::http1,
    service::service_fn,
    Method, Request, Response, StatusCode,
};
use hyper_util::rt::{TokioIo, TokioTimer};
use platform_node_link::tls;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{
    net::{TcpListener, TcpStream},
    signal::unix::{signal, SignalKind},
    task::JoinSet,
    time::timeout,
};
use uuid::Uuid;

use crate::{
    local::{self, Request as LocalRequest},
    queue::{valid_name, Scheduler, TaskError, WebIdentity},
};

const BODY_LIMIT: usize = 16 * 1024;
const CONNECTION_LIMIT: usize = 32;
const CONNECTION_TIMEOUT: Duration = Duration::from_secs(12);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    listen: SocketAddr,
    state_dir: PathBuf,
    ca_certificate: PathBuf,
    certificate: PathBuf,
    private_key: PathBuf,
    backend_certificate: PathBuf,
    models: Vec<ModelRoute>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelRoute {
    model_id: String,
    model_variant_id: String,
    plan: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Submit {
    task_id: Uuid,
    model_id: String,
    model_variant_id: String,
    prompt: String,
    browser_endpoint_id: String,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operation {
    Health,
    Submit {
        id: Uuid,
        plan: String,
        model_id: String,
        model_variant_id: String,
        prompt: String,
        browser_endpoint_id: String,
    },
    Task {
        id: Uuid,
    },
    Cancel {
        id: Uuid,
    },
}

#[derive(Serialize, Deserialize)]
pub struct Reply {
    status: u16,
    body: Value,
}

impl Reply {
    fn error(status: u16, code: &str, message: &str) -> Self {
        Self {
            status,
            body: json!({"error":{"code":code,"message":message}}),
        }
    }

    fn task_error(error: TaskError) -> Self {
        let (status, code) = match error {
            TaskError::Invalid(_) => (400, "invalid_request"),
            TaskError::Conflict => (409, "idempotency_conflict"),
            TaskError::Model => (409, "model_not_available"),
            TaskError::Full => (429, "queue_full"),
            TaskError::NotFound => (404, "task_not_found"),
            TaskError::Unavailable => (503, "scheduler_unavailable"),
        };
        Self::error(status, code, &error.to_string())
    }

    fn response(self) -> Response<Full<Bytes>> {
        Response::builder()
            .status(StatusCode::from_u16(self.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR))
            .header(header::CONTENT_TYPE, "application/json")
            .header(header::CACHE_CONTROL, "no-store")
            .body(Full::new(Bytes::from(
                serde_json::to_vec(&self.body).expect("JSON value"),
            )))
            .expect("valid HTTP response")
    }
}

pub fn execute(scheduler: &mut Scheduler, operation: Operation, now: Instant) -> Reply {
    let result = match operation {
        Operation::Health => {
            return Reply {
                status: 200,
                body: json!({"status":"up"}),
            }
        }
        Operation::Submit {
            id,
            plan,
            model_id,
            model_variant_id,
            prompt,
            browser_endpoint_id,
        } => scheduler.submit_web(
            id,
            plan,
            prompt,
            WebIdentity {
                endpoint_id: browser_endpoint_id,
                platform_model: Some((model_id, model_variant_id)),
            },
            now,
        ),
        Operation::Task { id } => {
            return match scheduler.web_task(id) {
                Ok(body) => Reply { status: 200, body },
                Err(error) => Reply::task_error(error),
            }
        }
        Operation::Cancel { id } => scheduler.cancel_web(id, now),
    };
    match result {
        Ok(body) => Reply {
            status: if matches!(
                body["state"].as_str(),
                Some("queued" | "dispatched" | "cancelling")
            ) {
                202
            } else {
                200
            },
            body,
        },
        Err(error) => Reply::task_error(error),
    }
}

struct Gateway {
    state: PathBuf,
    tls: tls::TlsAcceptor,
    allowed_peer: Vec<u8>,
    routes: HashMap<(String, String), String>,
}

pub async fn serve(config_file: &Path) -> Result<(), String> {
    let config: Config = local::read_json_input(config_file)?;
    if !config.listen.ip().is_loopback()
        || !config.state_dir.is_absolute()
        || config.models.is_empty()
        || config.models.len() > 32
    {
        return Err("网关必须监听回环地址，配置绝对状态目录和 1 至 32 个模型映射。".into());
    }
    let mut routes = HashMap::new();
    let mut plans = HashSet::new();
    for model in config.models {
        if !valid_name(&model.model_id)
            || !valid_name(&model.model_variant_id)
            || !valid_name(&model.plan)
            || !plans.insert(model.plan.clone())
            || routes
                .insert((model.model_id, model.model_variant_id), model.plan)
                .is_some()
        {
            return Err("网关模型映射无效或重复。".into());
        }
    }
    let peer = tls::certificates(&config.backend_certificate)?;
    let tls = tls::server_with_alpn(
        &config.ca_certificate,
        &config.certificate,
        &config.private_key,
        b"http/1.1",
    )?;
    let gateway = Arc::new(Gateway {
        state: config.state_dir,
        tls,
        allowed_peer: peer[0].as_ref().to_vec(),
        routes,
    });
    let listener = TcpListener::bind(config.listen)
        .await
        .map_err(|error| format!("网关监听失败（{error}）"))?;
    let mut interrupt = signal(SignalKind::interrupt()).map_err(|error| error.to_string())?;
    let mut terminate = signal(SignalKind::terminate()).map_err(|error| error.to_string())?;
    let mut workers = JoinSet::new();
    println!(
        "调度 HTTPS 网关监听 {}",
        listener.local_addr().map_err(|error| error.to_string())?
    );
    loop {
        tokio::select! {
            _ = interrupt.recv() => return Ok(()),
            _ = terminate.recv() => return Ok(()),
            accepted = listener.accept(), if workers.len() < CONNECTION_LIMIT => {
                let (stream, _) = accepted.map_err(|error| error.to_string())?;
                let gateway = Arc::clone(&gateway);
                workers.spawn(async move { let _ = timeout(CONNECTION_TIMEOUT, gateway.connection(stream)).await; });
            }
            _ = workers.join_next(), if !workers.is_empty() => (),
        }
    }
}

impl Gateway {
    async fn connection(self: Arc<Self>, stream: TcpStream) {
        let Ok(stream) = self.tls.accept(stream).await else {
            return;
        };
        if !stream
            .get_ref()
            .1
            .peer_certificates()
            .and_then(|certs| certs.first())
            .is_some_and(|cert| cert.as_ref() == self.allowed_peer)
        {
            return;
        }
        let service = service_fn(|request| async {
            Ok::<_, Infallible>(self.route(request).await.response())
        });
        let _ = http1::Builder::new()
            .timer(TokioTimer::new())
            .header_read_timeout(Duration::from_secs(3))
            .max_headers(32)
            .keep_alive(false)
            .serve_connection(TokioIo::new(stream), service)
            .await;
    }

    async fn route(&self, request: Request<Incoming>) -> Reply {
        let path = request.uri().path().to_owned();
        if request.uri().query().is_some() {
            return Reply::error(400, "invalid_request", "此接口不接收 URL 查询参数。");
        }
        let operation = match (request.method(), path.as_str()) {
            (&Method::GET, "/healthz") => Operation::Health,
            (&Method::POST, "/v1/tasks") => {
                if !request
                    .headers()
                    .get(header::CONTENT_TYPE)
                    .and_then(|value| value.to_str().ok())
                    .is_some_and(|value| {
                        value.split(';').next().is_some_and(|part| {
                            part.trim().eq_ignore_ascii_case("application/json")
                        })
                    })
                {
                    return Reply::error(
                        415,
                        "unsupported_media_type",
                        "请求正文需要 application/json。",
                    );
                }
                let Ok(bytes) = Limited::new(request.into_body(), BODY_LIMIT)
                    .collect()
                    .await
                else {
                    return Reply::error(413, "request_too_large", "提交正文超限或读取未完成。");
                };
                let Ok(body) = serde_json::from_slice::<Submit>(&bytes.to_bytes()) else {
                    return Reply::error(400, "invalid_request", "提交字段或 JSON 格式无效。");
                };
                let model = (body.model_id, body.model_variant_id);
                let Some(plan) = self.routes.get(&model) else {
                    return Reply::task_error(TaskError::Model);
                };
                Operation::Submit {
                    id: body.task_id,
                    plan: plan.clone(),
                    model_id: model.0,
                    model_variant_id: model.1,
                    prompt: body.prompt,
                    browser_endpoint_id: body.browser_endpoint_id,
                }
            }
            (method, path) => {
                let Some(suffix) = path.strip_prefix("/v1/tasks/") else {
                    return Reply::error(404, "route_not_found", "未找到该接口。");
                };
                let (id, cancel) = suffix
                    .strip_suffix("/cancel")
                    .map_or((suffix, false), |id| (id, true));
                let Ok(id) = Uuid::parse_str(id) else {
                    return Reply::error(400, "invalid_request", "任务 UUID 无效。");
                };
                match (method, cancel) {
                    (&Method::GET, false) => Operation::Task { id },
                    (&Method::POST, true) => Operation::Cancel { id },
                    _ => return Reply::error(405, "method_not_allowed", "请求方法不匹配。"),
                }
            }
        };
        match local::request_local(&self.state, LocalRequest::Gateway { operation })
            .await
            .ok()
            .and_then(|value| serde_json::from_value(value).ok())
        {
            Some(reply) => reply,
            None => Reply::task_error(TaskError::Unavailable),
        }
    }
}
