use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::Instant,
};

use platform_node_link::{
    local_node::{Failure, Run},
    receive, send, tls, web, Assignment, Heartbeat, Reply, FRAME_LIMIT, IO_TIMEOUT, MAX_IN_FLIGHT,
    TASK_DEADLINE, VERSION,
};
use serde::Deserialize;
use tokio::{
    net::{TcpListener, TcpStream},
    task::JoinSet,
    time::timeout,
};
use uuid::Uuid;

use crate::queue::{valid_name, Dispatch, NodeConfig, Scheduler};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RemoteConfig {
    listen: SocketAddr,
    ca_certificate: PathBuf,
    certificate: PathBuf,
    private_key: PathBuf,
    nodes: Vec<AllowedNode>,
    experience_relay_url: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AllowedNode {
    id: String,
    certificate: PathBuf,
    plans: Vec<String>,
    installation_id: Option<Uuid>,
}

pub struct Endpoint {
    listen: SocketAddr,
    acceptor: tls::TlsAcceptor,
    peers: HashMap<Vec<u8>, Peer>,
    relay: Option<String>,
}

#[derive(Clone, Copy)]
struct Peer {
    index: usize,
    installation_id: Option<Uuid>,
}

pub struct Hub {
    scheduler: Arc<Mutex<Scheduler>>,
    sessions: HashMap<usize, Session>,
}

#[derive(Default)]
struct Session {
    pending: Vec<Assignment>,
    active: HashMap<Uuid, Active>,
}

struct Active {
    task: Uuid,
    run: Run,
    deadline: Instant,
    web: bool,
}

impl RemoteConfig {
    pub fn prepare(self, routes: &mut Vec<NodeConfig>) -> Result<Endpoint, String> {
        if self.nodes.is_empty() || self.nodes.len() > 16 {
            return Err("远程联调需要配置 1 至 16 个节点。".into());
        }
        if self
            .experience_relay_url
            .as_deref()
            .is_some_and(|url| !web::valid_relay(url))
        {
            return Err("网页体验 Relay 必须是固定 HTTPS 地址。".into());
        }
        let acceptor = tls::server(&self.ca_certificate, &self.certificate, &self.private_key)?;
        let mut peers = HashMap::new();
        let mut ids: HashSet<_> = routes.iter().map(|route| route.id.clone()).collect();
        let mut installations = HashSet::new();
        for node in self.nodes {
            if !valid_name(&node.id)
                || !ids.insert(node.id.clone())
                || node.plans.is_empty()
                || node.plans.iter().any(|plan| !valid_name(plan))
                || node
                    .installation_id
                    .is_some_and(|id| id.is_nil() || !installations.insert(id))
            {
                return Err("远程节点 ID 重复、无效或缺少有效模型计划。".into());
            }
            let chain = tls::certificates(&node.certificate)?;
            if peers
                .insert(
                    chain[0].as_ref().to_vec(),
                    Peer {
                        index: routes.len(),
                        installation_id: node.installation_id,
                    },
                )
                .is_some()
            {
                return Err("每个远程节点必须使用不同的联调证书。".into());
            }
            routes.push(NodeConfig {
                id: node.id,
                plans: node.plans,
                max_dispatched: MAX_IN_FLIGHT,
            });
        }
        Ok(Endpoint {
            listen: self.listen,
            acceptor,
            peers,
            relay: self.experience_relay_url,
        })
    }
}

impl Endpoint {
    pub async fn bind(&self) -> Result<TcpListener, String> {
        TcpListener::bind(self.listen)
            .await
            .map_err(|error| format!("远程联调监听失败（{error}）"))
    }

    pub async fn serve(self, listener: TcpListener, hub: Arc<Mutex<Hub>>) -> Result<(), String> {
        println!(
            "远程联调监听 {}",
            listener.local_addr().map_err(|error| error.to_string())?
        );
        let endpoint = Arc::new(self);
        let mut connections = JoinSet::new();
        loop {
            tokio::select! {
                result = listener.accept(), if connections.len() < 32 => {
                    let (stream, _) = result.map_err(|error| error.to_string())?;
                    let endpoint = Arc::clone(&endpoint);
                    let hub = Arc::clone(&hub);
                    connections.spawn(async move {
                        let _ = endpoint.handle(stream, hub).await;
                    });
                }
                Some(_) = connections.join_next() => (),
            }
        }
    }

    async fn handle(&self, stream: TcpStream, hub: Arc<Mutex<Hub>>) -> Result<(), String> {
        stream
            .set_nodelay(true)
            .map_err(|error| error.to_string())?;
        let mut stream = timeout(IO_TIMEOUT, self.acceptor.accept(stream))
            .await
            .map_err(|_| "远程证书握手超时。")?
            .map_err(|error| error.to_string())?;
        let peer = stream
            .get_ref()
            .1
            .peer_certificates()
            .and_then(|chain| chain.first())
            .and_then(|cert| self.peers.get(cert.as_ref()))
            .copied()
            .ok_or("远程节点证书未登记。")?;
        let _connection = Connection::open(Arc::clone(&hub), peer.index)?;
        loop {
            let message: Heartbeat = timeout(IO_TIMEOUT, receive(&mut stream, FRAME_LIMIT))
                .await
                .map_err(|_| "远程节点心跳超时。")?
                .map_err(|error| error.to_string())?;
            let reply = hub.lock().unwrap().heartbeat(
                peer,
                self.relay.as_deref(),
                message,
                Instant::now(),
            )?;
            timeout(IO_TIMEOUT, send(&mut stream, &reply, FRAME_LIMIT))
                .await
                .map_err(|_| "远程节点回执发送超时。")?
                .map_err(|error| error.to_string())?;
        }
    }
}

impl Hub {
    pub fn new(scheduler: Arc<Mutex<Scheduler>>) -> Self {
        Self {
            scheduler,
            sessions: HashMap::new(),
        }
    }

    pub fn assign(&mut self, job: Dispatch, now: Instant) {
        let mut scheduler = self.scheduler.lock().unwrap();
        if let (Some(session), Some(run)) = (
            self.sessions.get_mut(&job.node),
            scheduler.run(job.node, now).cloned(),
        ) {
            let attempt = job.attempt;
            session.active.insert(
                attempt,
                Active {
                    task: job.id,
                    run: run.clone(),
                    deadline: now + TASK_DEADLINE,
                    web: job.web.is_some(),
                },
            );
            session.pending.push(Assignment {
                id: job.id,
                attempt,
                run,
                plan: job.plan,
                prompt: job.prompt,
                web: job.web,
            });
        } else {
            scheduler.complete(
                job.id,
                Err(Failure::Retry("远程节点已离线，任务未发送。".into())),
                now,
            );
        }
    }

    pub fn expire(&mut self, now: Instant) {
        let mut scheduler = self.scheduler.lock().unwrap();
        for session in self.sessions.values_mut() {
            session.discard(
                &mut scheduler,
                now,
                |active| active.deadline <= now,
                "远程任务回执超时。",
            );
        }
    }

    fn heartbeat(
        &mut self,
        peer: Peer,
        relay: Option<&str>,
        heartbeat: Heartbeat,
        now: Instant,
    ) -> Result<Reply, String> {
        if heartbeat.version != VERSION || heartbeat.available_slots > MAX_IN_FLIGHT {
            return Err("远程联调协议版本或可用任务数无效。".into());
        }
        let session = self
            .sessions
            .get_mut(&peer.index)
            .ok_or("远程节点会话已失效。")?;
        let mut scheduler = self.scheduler.lock().unwrap();
        let mut endpoint = None;
        if let Some(inventory) = &heartbeat.inventory {
            scheduler.check_remote_run(peer.index, &inventory.run, now)?;
            if peer
                .installation_id
                .is_some_and(|id| id != inventory.run.installation_id)
            {
                return Err("节点证书与登记的安装身份不匹配。".into());
            }
            if let (Some(registration), Some(relay), Some(_)) =
                (heartbeat.endpoint, relay, peer.installation_id)
            {
                if !registration.verify(&inventory.run, relay) {
                    return Err("iroh 端点签名、运行身份或 Relay 不匹配。".into());
                }
                endpoint = Some(registration.target);
            }
            session.discard(
                &mut scheduler,
                now,
                |active| active.run != inventory.run,
                "节点启动代次已变化，未确认原任务结果。",
            );
        }
        if let Some(report) = heartbeat.result {
            if let Some(active) = session.active.get(&report.attempt) {
                if session
                    .pending
                    .iter()
                    .any(|assignment| assignment.attempt == report.attempt)
                    || report.result.as_ref().is_ok_and(|output| {
                        output.task_id != active.task
                            || (active.web && !output.answer.text.is_empty())
                    })
                {
                    return Err("远程任务回执与派发记录不匹配。".into());
                }
                scheduler.complete(active.task, report.result, now);
                session.active.remove(&report.attempt);
            }
        }
        session.pending.retain(|assignment| {
            if !scheduler.cancellation_pending(assignment.id) {
                return true;
            }
            session.active.remove(&assignment.attempt);
            scheduler.complete(
                assignment.id,
                Err(Failure::Cancelled {
                    execution_started: false,
                }),
                now,
            );
            false
        });
        let assignments = if let Some(inventory) = heartbeat.inventory {
            if endpoint.is_none() {
                session.pending.retain(|assignment| {
                    if assignment.web.is_none() {
                        return true;
                    }
                    session.active.remove(&assignment.attempt);
                    scheduler.complete(
                        assignment.id,
                        Err(Failure::Retry("节点 iroh 端点未就绪，任务未发送。".into())),
                        now,
                    );
                    false
                });
            }
            scheduler.update_remote(peer.index, inventory, endpoint, now);
            let count = heartbeat.available_slots.min(session.pending.len());
            session.pending.drain(..count).collect()
        } else {
            scheduler.update(
                peer.index,
                Err("远程节点已登记，本机运行或库存未就绪。".into()),
                now,
            );
            Vec::new()
        };
        Ok(Reply {
            version: VERSION,
            node_id: scheduler.node_id(peer.index).into(),
            assignments,
            cancel_attempts: session
                .active
                .iter()
                .filter_map(|(attempt, active)| {
                    scheduler
                        .cancellation_pending(active.task)
                        .then_some(*attempt)
                })
                .collect(),
        })
    }
}

impl Session {
    fn discard(
        &mut self,
        scheduler: &mut Scheduler,
        now: Instant,
        mut matches: impl FnMut(&Active) -> bool,
        reason: &str,
    ) {
        let pending = &self.pending;
        self.active.retain(|_, active| {
            if !matches(active) {
                return true;
            }
            let failure = if pending
                .iter()
                .any(|assignment| assignment.id == active.task)
            {
                Failure::Retry("节点在任务送出前失去可用状态。".into())
            } else {
                Failure::Unknown(reason.into())
            };
            scheduler.complete(active.task, Err(failure), now);
            false
        });
        self.pending
            .retain(|assignment| self.active.contains_key(&assignment.attempt));
    }
}

struct Connection {
    hub: Arc<Mutex<Hub>>,
    peer: usize,
}

impl Connection {
    fn open(hub: Arc<Mutex<Hub>>, peer: usize) -> Result<Self, String> {
        {
            let mut state = hub.lock().unwrap();
            if state.sessions.contains_key(&peer) {
                return Err("该远程节点已存在活动会话。".into());
            }
            state.sessions.insert(peer, Session::default());
        }
        Ok(Self { hub, peer })
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        let mut hub = self.hub.lock().unwrap();
        if let Some(mut session) = hub.sessions.remove(&self.peer) {
            let now = Instant::now();
            let mut scheduler = hub.scheduler.lock().unwrap();
            session.discard(
                &mut scheduler,
                now,
                |_| true,
                "远程节点断开，未确认任务结果。",
            );
            scheduler.update(self.peer, Err("远程节点已断开。".into()), now);
        }
    }
}
