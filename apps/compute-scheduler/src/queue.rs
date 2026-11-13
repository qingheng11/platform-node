use std::{
    collections::{HashMap, VecDeque},
    time::{Duration, Instant},
};

use serde::Serialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::admission::{request_digest, Admission, Journal};
use platform_node_link::{
    local_node::{Failure, Inventory, Output, Run},
    web,
};

const INVENTORY_TTL: Duration = Duration::from_secs(12);
const QUEUE_TTL: Duration = Duration::from_secs(120);
const RESULT_TTL: Duration = Duration::from_secs(600);
const RETRY_DELAY: Duration = Duration::from_secs(3);
const MAX_QUEUED: usize = 64;
const MAX_TASKS: usize = 256;

pub struct Scheduler {
    nodes: Vec<Node>,
    tasks: HashMap<Uuid, Task>,
    queued: VecDeque<Uuid>,
    admissions: Option<Journal>,
}

pub struct Node {
    config: NodeConfig,
    snapshot: Option<(Instant, Inventory)>,
    endpoint: Option<web::Target>,
    last_remote_run: Option<Run>,
    last_error: Option<String>,
    retry_after: HashMap<String, Instant>,
}

pub struct NodeConfig {
    pub id: String,
    pub plans: Vec<String>,
    pub max_dispatched: usize,
}

struct Task {
    plan: String,
    prompt: String,
    queued_at: Instant,
    retry_detail: Option<String>,
    finished: Option<Instant>,
    state: State,
    web: Option<WebTask>,
    cancel_requested: bool,
}

struct WebTask {
    browser_endpoint_id: String,
    request_sha256: String,
    route: Option<WebRoute>,
}

pub struct WebIdentity {
    pub endpoint_id: String,
    pub platform_model: Option<(String, String)>,
}

#[derive(Serialize)]
struct WebRoute {
    target: web::Target,
    runtime_id: Uuid,
    generation: u64,
    attempt_id: Uuid,
    ticket: Uuid,
    alpn: &'static str,
}

#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum State {
    Queued,
    Dispatched { node: usize, cold: bool },
    Succeeded { node: usize, output: Output },
    Failed { error: String },
    Cancelled { execution_started: bool },
    OutcomeUnknown { node: usize, error: String },
}

#[derive(Debug)]
pub enum TaskError {
    Invalid(&'static str),
    Conflict,
    Model,
    Full,
    NotFound,
    Unavailable,
}

impl std::fmt::Display for TaskError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Invalid(message) => message,
            Self::Conflict => "任务编号已绑定到不同的请求。",
            Self::Model => "调度器没有为该模型计划配置节点。",
            Self::Full => "调度队列或结果缓存已满，请稍后提交。",
            Self::NotFound => "未找到该网页任务。",
            Self::Unavailable => "任务记录暂不可用。",
        })
    }
}

pub struct Dispatch {
    pub id: Uuid,
    pub node: usize,
    pub plan: String,
    pub prompt: String,
    pub attempt: Uuid,
    pub web: Option<web::Grant>,
}

pub fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && !value.starts_with('.')
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
}

pub fn valid_prompt(prompt: &str) -> bool {
    !prompt.trim().is_empty() && prompt.len() <= 2048
}

impl Scheduler {
    pub fn new(configs: impl IntoIterator<Item = NodeConfig>) -> Self {
        Self {
            nodes: configs
                .into_iter()
                .map(|config| Node {
                    config,
                    snapshot: None,
                    endpoint: None,
                    last_remote_run: None,
                    last_error: None,
                    retry_after: HashMap::new(),
                })
                .collect(),
            tasks: HashMap::new(),
            queued: VecDeque::new(),
            admissions: None,
        }
    }

    pub fn with_admissions(mut self, journal: Journal) -> Self {
        self.admissions = Some(journal);
        self
    }

    pub fn run(&self, node: usize, now: Instant) -> Option<&Run> {
        self.nodes[node].fresh(now).map(|inventory| &inventory.run)
    }

    pub fn node_id(&self, node: usize) -> &str {
        &self.nodes[node].config.id
    }

    pub fn has_other_installation(&self, node: usize, installation: Uuid, now: Instant) -> bool {
        self.nodes.iter().enumerate().any(|(index, candidate)| {
            index != node
                && candidate
                    .fresh(now)
                    .is_some_and(|inventory| inventory.run.installation_id == installation)
        })
    }

    pub fn update(&mut self, node: usize, result: Result<Inventory, String>, sampled_at: Instant) {
        let node = &mut self.nodes[node];
        node.endpoint = None;
        match result {
            Ok(inventory) => {
                node.snapshot = Some((sampled_at, inventory));
                node.last_error = None;
            }
            Err(error) => {
                node.snapshot = None;
                node.last_error = Some(error);
            }
        }
    }

    pub fn check_remote_run(&self, node: usize, run: &Run, now: Instant) -> Result<(), String> {
        if run.run_id.is_nil()
            || run.installation_id.is_nil()
            || run.generation == 0
            || self.has_other_installation(node, run.installation_id, now)
            || self.nodes[node]
                .last_remote_run
                .as_ref()
                .is_some_and(|last| {
                    last.installation_id != run.installation_id
                        || run.generation < last.generation
                        || (run.generation == last.generation && run.run_id != last.run_id)
                })
        {
            return Err("节点安装身份冲突，或启动代次已过期。".into());
        }
        Ok(())
    }

    pub fn update_remote(
        &mut self,
        node: usize,
        inventory: Inventory,
        endpoint: Option<web::Target>,
        now: Instant,
    ) {
        self.nodes[node].last_remote_run = Some(inventory.run.clone());
        self.update(node, Ok(inventory), now);
        self.nodes[node].endpoint = endpoint;
    }

    pub fn submit(
        &mut self,
        id: Uuid,
        plan: String,
        prompt: String,
        now: Instant,
    ) -> Result<Value, String> {
        self.submit_task(id, plan, prompt, None, now)
            .map_err(|error| error.to_string())
    }

    pub fn submit_web(
        &mut self,
        id: Uuid,
        plan: String,
        prompt: String,
        identity: WebIdentity,
        now: Instant,
    ) -> Result<Value, TaskError> {
        if web::endpoint_bytes(&identity.endpoint_id).is_none() {
            return Err(TaskError::Invalid(
                "浏览器 iroh 端点 ID 必须为 64 位小写十六进制。",
            ));
        }
        let browser = &identity.endpoint_id;
        let request_sha256 = match &identity.platform_model {
            Some((model, variant)) => {
                request_digest(&["gateway", model, variant, &prompt, browser])
            }
            None => request_digest(&["local", &plan, &prompt, browser]),
        };
        self.submit_task(
            id,
            plan,
            prompt,
            Some(WebTask {
                browser_endpoint_id: identity.endpoint_id,
                request_sha256,
                route: None,
            }),
            now,
        )
    }

    fn submit_task(
        &mut self,
        id: Uuid,
        plan: String,
        prompt: String,
        web: Option<WebTask>,
        now: Instant,
    ) -> Result<Value, TaskError> {
        if id.is_nil() || !valid_prompt(&prompt) {
            return Err(TaskError::Invalid(
                "任务 UUID 必须有效，提示必须非空且不超过 2048 字节。",
            ));
        }
        if let Some(existing) = self.tasks.get(&id) {
            let same_request = match (&existing.web, &web) {
                (Some(previous), Some(request)) => {
                    previous.request_sha256 == request.request_sha256
                }
                (None, None) => existing.plan == plan && existing.prompt == prompt,
                _ => false,
            };
            if !same_request {
                return Err(TaskError::Conflict);
            }
            return self.task(id).map_err(|_| TaskError::Unavailable);
        }
        let previous = self.admission(id)?;
        let digest = web.as_ref().map(|request| &request.request_sha256);
        if let Some(previous) = previous {
            if digest != Some(&previous.request_sha256) {
                return Err(TaskError::Conflict);
            }
            return Ok(Self::archived_task(id, previous));
        }
        if !self
            .nodes
            .iter()
            .any(|node| node.config.plans.contains(&plan))
        {
            return Err(TaskError::Model);
        }
        if self.queued.len() >= MAX_QUEUED || self.tasks.len() >= MAX_TASKS {
            return Err(TaskError::Full);
        }
        if let (Some(journal), Some(request_sha256)) = (&self.admissions, digest) {
            journal
                .write(
                    id,
                    &Admission {
                        request_sha256: request_sha256.clone(),
                        terminal: None,
                    },
                )
                .map_err(|_| TaskError::Unavailable)?;
        }
        self.tasks.insert(
            id,
            Task {
                plan,
                prompt,
                queued_at: now,
                retry_detail: None,
                finished: None,
                state: State::Queued,
                cancel_requested: false,
                web,
            },
        );
        self.queued.push_back(id);
        self.task(id).map_err(|_| TaskError::Unavailable)
    }

    fn admission(&self, id: Uuid) -> Result<Option<Admission>, TaskError> {
        self.admissions
            .as_ref()
            .map(|journal| journal.read(id))
            .transpose()
            .map(Option::flatten)
            .map_err(|_| TaskError::Unavailable)
    }

    fn archived_task(id: Uuid, record: Admission) -> Value {
        record.terminal.unwrap_or_else(
            || json!({"task_id":id, "state":"outcome_unknown", "execution_state":"unknown"}),
        )
    }

    pub fn web_task(&self, id: Uuid) -> Result<Value, TaskError> {
        match self.tasks.get(&id) {
            Some(task) if task.web.is_some() => self.task(id).map_err(|_| TaskError::Unavailable),
            Some(_) => Err(TaskError::NotFound),
            None => self
                .admission(id)?
                .map(|entry| Self::archived_task(id, entry))
                .ok_or(TaskError::NotFound),
        }
    }

    pub fn cancel_web(&mut self, id: Uuid, now: Instant) -> Result<Value, TaskError> {
        if let Some(task) = self.tasks.get_mut(&id).filter(|task| task.web.is_some()) {
            match task.state {
                State::Queued => {
                    self.queued.retain(|queued| *queued != id);
                    task.state = State::Cancelled {
                        execution_started: false,
                    };
                    task.finished = Some(now);
                    self.record_terminal(id);
                }
                State::Dispatched { .. } => task.cancel_requested = true,
                _ => (),
            }
        }
        self.web_task(id)
    }

    pub fn cancellation_pending(&self, id: Uuid) -> bool {
        self.tasks.get(&id).is_some_and(|task| {
            task.cancel_requested && matches!(task.state, State::Dispatched { .. })
        })
    }

    fn record_terminal(&self, id: Uuid) {
        let Some(journal) = &self.admissions else {
            return;
        };
        if self.tasks.get(&id).is_none_or(|task| task.web.is_none()) {
            return;
        }
        let stored = (|| {
            let mut entry = journal
                .read(id)?
                .ok_or_else(|| std::io::Error::other("missing admission"))?;
            let mut value = self.task(id).map_err(std::io::Error::other)?;
            if value.get("result_route").is_some() {
                value["result_route"] = Value::Null;
            }
            entry.terminal = Some(value);
            journal.write(id, &entry)
        })();
        if stored.is_err() {
            eprintln!("无法保存网页任务终态，保留待对账记录。task_id={id}");
        }
    }

    pub fn task(&self, id: Uuid) -> Result<Value, String> {
        let task = self.tasks.get(&id).ok_or("未找到该任务。")?;
        let mut value = serde_json::to_value(&task.state).map_err(|error| error.to_string())?;
        if let Some(node) = value["node"].as_u64() {
            value["node"] = json!(self.nodes[node as usize].config.id);
        }
        value["task_id"] = json!(id);
        value["plan"] = json!(task.plan);
        if self.cancellation_pending(id) {
            value["state"] = json!("cancelling");
        }
        value["execution_state"] = json!(match task.state {
            State::Succeeded { .. }
            | State::Cancelled {
                execution_started: true,
            } => "executed",
            State::Queued
            | State::Failed { .. }
            | State::Cancelled {
                execution_started: false,
            } => "not_executed",
            State::Dispatched { .. } | State::OutcomeUnknown { .. } => "unknown",
        });
        if let Some(error) = &task.retry_detail {
            value["last_retry"] = json!(error);
        }
        if let (Some(web_task), State::Succeeded { node, .. }) = (&task.web, &task.state) {
            let now = Instant::now();
            let route = web_task.route.as_ref().filter(|route| {
                task.finished.is_some_and(|finished| {
                    now.saturating_duration_since(finished) < web::RESULT_TTL
                }) && self.run(*node, now).is_some_and(|run| {
                    run.run_id == route.runtime_id && run.generation == route.generation
                }) && self.nodes[*node].endpoint.as_ref() == Some(&route.target)
            });
            value["output"]["answer"]
                .as_object_mut()
                .ok_or("节点回答用量格式无效。")?
                .remove("text");
            value["result_route"] = json!(route);
        }
        Ok(value)
    }

    pub fn status(&self, now: Instant) -> Value {
        let nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|node| {
                json!({
                    "id": node.config.id,
                    "plans": node.config.plans,
                    "inventory": node.fresh(now),
                    "iroh_endpoint": node.fresh(now).and(node.endpoint.as_ref()),
                    "last_error": node.last_error,
                })
            })
            .collect();
        json!({
            "nodes": nodes,
            "queued": self.queued.len(),
            "dispatched": self.tasks.values().filter(|task| matches!(task.state, State::Dispatched { .. })).count(),
        })
    }

    pub fn dispatch(&mut self, now: Instant) -> Vec<Dispatch> {
        self.tasks.retain(|_, task| {
            task.finished
                .is_none_or(|finished| now.duration_since(finished) < RESULT_TTL)
        });
        let mut dispatches = Vec::new();
        for _ in 0..self.queued.len() {
            let id = self.queued.pop_front().expect("queued task");
            let task = &self.tasks[&id];
            if now.duration_since(task.queued_at) >= QUEUE_TTL {
                let task = self.tasks.get_mut(&id).unwrap();
                task.state = State::Failed {
                    error: "等待可用节点或容量超时。".into(),
                };
                task.finished = Some(now);
                self.record_terminal(id);
            } else if let Some((node, cold)) = self.choose_node(&task.plan, now, task.web.is_some())
            {
                let attempt = Uuid::new_v4();
                let target = self.nodes[node].endpoint.clone();
                let run = self
                    .run(node, now)
                    .expect("selected node has fresh inventory")
                    .clone();
                let task = self.tasks.get_mut(&id).unwrap();
                let web = task.web.as_mut().map(|task| {
                    let ticket = Uuid::new_v4();
                    task.route = Some(WebRoute {
                        target: target.expect("selected web node has an endpoint"),
                        runtime_id: run.run_id,
                        generation: run.generation,
                        attempt_id: attempt,
                        ticket,
                        alpn: web::ALPN,
                    });
                    web::Grant {
                        browser_endpoint_id: task.browser_endpoint_id.clone(),
                        ticket,
                    }
                });
                task.state = State::Dispatched { node, cold };
                task.retry_detail = None;
                dispatches.push(Dispatch {
                    id,
                    node,
                    plan: task.plan.clone(),
                    prompt: task.prompt.clone(),
                    attempt,
                    web,
                });
            } else {
                self.queued.push_back(id);
            }
        }
        dispatches
    }

    fn choose_node(&self, plan: &str, now: Instant, require_web: bool) -> Option<(usize, bool)> {
        let mut warm = None;
        let mut cold = None;
        for (index, node) in self.nodes.iter().enumerate() {
            if (require_web && node.endpoint.is_none())
                || !node.config.plans.iter().any(|candidate| candidate == plan)
                || node
                    .retry_after
                    .get(plan)
                    .is_some_and(|retry_at| *retry_at > now)
            {
                continue;
            }
            let Some(snapshot) = node.fresh(now) else {
                continue;
            };
            let dispatched = self
                .tasks
                .values()
                .filter(
                    |task| matches!(task.state, State::Dispatched { node, .. } if node == index),
                )
                .count();
            if dispatched >= node.config.max_dispatched {
                continue;
            }
            if let Some(model) = snapshot.models.iter().find(|model| model.plan == plan) {
                if model.instance.phase != "ready"
                    || model.unloading
                    || model.calibration_error.is_some()
                {
                    continue;
                }
                let unobserved = self
                    .tasks
                    .iter()
                    .filter(|(id, active)| {
                        matches!(active.state, State::Dispatched { node, .. } if node == index)
                            && active.plan == plan
                            && !model.active_task_ids.contains(id)
                    })
                    .count();
                let free = model
                    .slots_total
                    .saturating_sub(model.slots_used.saturating_add(unobserved));
                if free > 0 && warm.is_none_or(|(_, best)| free > best) {
                    warm = Some((index, free));
                }
                continue;
            }
            let cold_in_progress = self.tasks.values().any(|active| {
                matches!(active.state, State::Dispatched { node, cold: true } if node == index)
                    && !snapshot
                        .models
                        .iter()
                        .any(|model| model.plan == active.plan && model.instance.phase == "ready")
            });
            if cold_in_progress {
                continue;
            }
            // Raw headroom only ranks candidates. The node performs admission,
            // including its per-GPU reservation ledger and model estimate.
            let headroom = snapshot.gpus.iter().filter_map(|gpu| gpu.free_bytes).max();
            if let Some(headroom) = headroom.filter(|bytes| *bytes > 0) {
                if cold.is_none_or(|(_, best)| headroom > best) {
                    cold = Some((index, headroom));
                }
            }
        }
        warm.map(|(index, _)| (index, false))
            .or_else(|| cold.map(|(index, _)| (index, true)))
    }

    pub fn complete(&mut self, id: Uuid, result: Result<Output, Failure>, now: Instant) {
        let Some(task) = self.tasks.get_mut(&id) else {
            return;
        };
        let State::Dispatched { node, .. } = task.state else {
            return;
        };
        task.state = match result {
            Ok(output) => State::Succeeded { node, output },
            Err(Failure::Cancelled { execution_started }) => State::Cancelled { execution_started },
            Err(Failure::Retry(_) | Failure::Rejected(_)) if task.cancel_requested => {
                State::Cancelled {
                    execution_started: false,
                }
            }
            Err(Failure::Retry(error))
                if now.duration_since(task.queued_at) < QUEUE_TTL
                    && self.queued.len() < MAX_QUEUED =>
            {
                self.nodes[node]
                    .retry_after
                    .insert(task.plan.clone(), now + RETRY_DELAY);
                task.retry_detail = Some(error);
                if let Some(web) = &mut task.web {
                    web.route = None;
                }
                self.queued.push_back(id);
                State::Queued
            }
            Err(Failure::Retry(error) | Failure::Rejected(error)) => State::Failed { error },
            Err(Failure::Unknown(error)) => State::OutcomeUnknown { node, error },
        };
        if !matches!(task.state, State::Queued) {
            task.finished = Some(now);
            self.record_terminal(id);
        }
    }
}

impl Node {
    fn fresh(&self, now: Instant) -> Option<&Inventory> {
        self.snapshot
            .as_ref()
            .filter(|(sampled_at, _)| now.duration_since(*sampled_at) < INVENTORY_TTL)
            .map(|(_, snapshot)| snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use platform_node_link::local_node::{Gpu, Instance, Model};

    fn node(id: &str) -> NodeConfig {
        NodeConfig {
            id: id.into(),
            plans: vec!["qwen".into(), "other".into()],
            max_dispatched: usize::MAX,
        }
    }

    fn inventory(free: u64, models: Vec<Model>) -> Inventory {
        Inventory {
            run: Run {
                installation_id: Uuid::new_v4(),
                generation: 1,
                run_id: Uuid::new_v4(),
            },
            models,
            gpus: vec![Gpu {
                index: 0,
                name: "test".into(),
                uuid: None,
                free_bytes: Some(free),
                total_bytes: None,
            }],
        }
    }

    fn model(slots_used: usize, active_task_ids: Vec<Uuid>) -> Model {
        Model {
            plan: "qwen".into(),
            instance: Instance {
                instance_id: Uuid::new_v4(),
                phase: "ready".into(),
            },
            active_task_ids,
            slots_used,
            slots_total: 4,
            unloading: false,
            calibration_error: None,
        }
    }

    fn enqueue(scheduler: &mut Scheduler, plan: &str, now: Instant) -> Uuid {
        let id = Uuid::new_v4();
        scheduler
            .submit(id, plan.into(), "hello".into(), now)
            .unwrap();
        id
    }

    #[test]
    fn warm_beats_cold_and_ready_slots_are_claimed_without_double_counting() {
        let mut scheduler = Scheduler::new([node("cold"), node("warm")]);
        let now = Instant::now();
        scheduler.update(0, Ok(inventory(100, vec![])), now);
        scheduler.update(1, Ok(inventory(2, vec![model(2, vec![])])), now);
        let first = enqueue(&mut scheduler, "qwen", now);
        let second = enqueue(&mut scheduler, "qwen", now);
        let third = enqueue(&mut scheduler, "qwen", now);
        let dispatched = scheduler.dispatch(now);
        assert_eq!(
            dispatched
                .iter()
                .map(|job| (job.id, job.node))
                .collect::<Vec<_>>(),
            [(first, 1), (second, 1), (third, 0)]
        );

        scheduler.update(0, Err("offline".into()), now);
        scheduler.update(
            1,
            Ok(inventory(1, vec![model(3, vec![first, second])])),
            now,
        );
        let fourth = enqueue(&mut scheduler, "qwen", now);
        let fifth = enqueue(&mut scheduler, "qwen", now);
        let dispatched = scheduler.dispatch(now);
        assert_eq!(dispatched.len(), 1);
        assert_eq!((dispatched[0].id, dispatched[0].node), (fourth, 1));
        assert_eq!(scheduler.task(fifth).unwrap()["state"], "queued");
    }

    #[test]
    fn cold_admission_is_serialized_and_stale_inventory_expires() {
        let mut scheduler = Scheduler::new([node("test")]);
        let now = Instant::now();
        scheduler.update(0, Ok(inventory(10, vec![])), now);
        let first = enqueue(&mut scheduler, "qwen", now);
        let second = enqueue(&mut scheduler, "other", now);
        assert_eq!(scheduler.dispatch(now)[0].id, first);
        assert!(scheduler.dispatch(now).is_empty());
        scheduler.complete(first, Err(Failure::Rejected("missing".into())), now);
        assert!(scheduler.dispatch(now + INVENTORY_TTL).is_empty());
        assert_eq!(scheduler.task(second).unwrap()["state"], "queued");
        assert!(scheduler.dispatch(now + QUEUE_TTL).is_empty());
        assert_eq!(scheduler.task(second).unwrap()["state"], "failed");
    }

    #[test]
    fn capacity_refusal_can_move_to_another_node_but_unknown_result_is_not_replayed() {
        let mut scheduler = Scheduler::new([node("largest"), node("alternate")]);
        let now = Instant::now();
        scheduler.update(0, Ok(inventory(100, vec![])), now);
        scheduler.update(1, Ok(inventory(20, vec![])), now);
        let id = enqueue(&mut scheduler, "qwen", now);
        assert_eq!(scheduler.dispatch(now)[0].node, 0);
        scheduler.complete(id, Err(Failure::Retry("capacity".into())), now);
        let next = scheduler.dispatch(now);
        assert_eq!(next.len(), 1);
        assert_eq!((next[0].id, next[0].node), (id, 1));
        scheduler.complete(id, Err(Failure::Unknown("lost".into())), now);
        assert!(scheduler.dispatch(now + RETRY_DELAY).is_empty());
        assert_eq!(scheduler.task(id).unwrap()["state"], "outcome_unknown");
        assert_eq!(
            scheduler
                .submit(id, "qwen".into(), "hello".into(), now)
                .unwrap()["state"],
            "outcome_unknown"
        );
        assert!(scheduler
            .submit(id, "qwen".into(), "other prompt".into(), now)
            .is_err());
    }

    #[test]
    fn web_route_follows_selected_runtime_and_requires_a_live_endpoint() {
        use platform_node_link::local_node::Answer;
        let now = Instant::now();
        let mut scheduler = Scheduler::new([node("gpu-1"), node("v100-101")]);
        let target = |digit: &str| web::Target {
            endpoint_id: digit.repeat(64),
            relay_url: "https://relay.example.invalid/".into(),
        };
        let warm = inventory(20, vec![model(0, vec![])]);
        scheduler.update_remote(0, warm.clone(), None, now);
        let cold = inventory(100, vec![]);
        scheduler.update_remote(1, cold.clone(), Some(target("b")), now);
        let submit = |scheduler: &mut Scheduler| {
            let id = Uuid::new_v4();
            scheduler
                .submit_web(id, "qwen".into(), "hello".into(), web_identity("c"), now)
                .unwrap();
            id
        };
        let first = submit(&mut scheduler);
        assert_eq!(scheduler.dispatch(now)[0].node, 1);
        scheduler.complete(first, Err(Failure::Retry("capacity".into())), now);
        assert!(scheduler.dispatch(now).is_empty());

        scheduler.update_remote(0, warm.clone(), Some(target("a")), now);
        let dispatch = scheduler.dispatch(now).pop().unwrap();
        assert_eq!((dispatch.id, dispatch.node), (first, 0));
        assert!(scheduler.task(first).unwrap().get("result_route").is_none());
        scheduler.complete(
            first,
            Ok(Output {
                task_id: first,
                instance_id: Uuid::new_v4(),
                answer: Answer {
                    text: String::new(),
                    input_tokens: Some(3),
                    output_tokens: Some(1),
                },
            }),
            now,
        );
        let completed = scheduler.task(first).unwrap();
        assert_eq!(
            completed["result_route"]["attempt_id"],
            json!(dispatch.attempt)
        );
        assert_eq!(
            completed["result_route"]["target"]["endpoint_id"],
            "a".repeat(64)
        );
        assert_eq!(
            completed["result_route"]["runtime_id"],
            json!(warm.run.run_id)
        );
        assert!(completed["output"]["answer"].get("text").is_none());
        assert!(scheduler
            .submit_web(first, "qwen".into(), "hello".into(), web_identity("d"), now)
            .is_err());

        let mut restarted = warm.clone();
        restarted.run.generation += 1;
        restarted.run.run_id = Uuid::new_v4();
        scheduler.update_remote(0, restarted, Some(target("d")), now);
        assert!(scheduler.check_remote_run(0, &warm.run, now).is_err());
        assert!(scheduler.task(first).unwrap()["result_route"].is_null());
        scheduler.update(0, Err("offline".into()), now);
        let later = now + RETRY_DELAY;
        scheduler.update_remote(1, cold, Some(target("b")), later);
        let second = submit(&mut scheduler);
        let next = scheduler.dispatch(later).pop().unwrap();
        assert_eq!((next.id, next.node), (second, 1));
    }

    fn web_identity(digit: &str) -> WebIdentity {
        WebIdentity {
            endpoint_id: digit.repeat(64),
            platform_model: None,
        }
    }

    #[test]
    fn dispatched_cancel_waits_for_execution_evidence() {
        let now = Instant::now();
        for (outcome, state, execution) in [
            (
                Failure::Cancelled {
                    execution_started: false,
                },
                "cancelled",
                "not_executed",
            ),
            (
                Failure::Cancelled {
                    execution_started: true,
                },
                "cancelled",
                "executed",
            ),
            (
                Failure::Unknown("node disconnected".into()),
                "outcome_unknown",
                "unknown",
            ),
        ] {
            let mut scheduler = Scheduler::new([node("v100-101")]);
            scheduler.update_remote(
                0,
                inventory(100, vec![]),
                Some(web::Target {
                    endpoint_id: "a".repeat(64),
                    relay_url: "https://relay.example.invalid/".into(),
                }),
                now,
            );
            let id = Uuid::new_v4();
            scheduler
                .submit_web(id, "qwen".into(), "hello".into(), web_identity("c"), now)
                .unwrap();
            assert_eq!(scheduler.dispatch(now)[0].id, id);
            let pending = scheduler.cancel_web(id, now).unwrap();
            assert_eq!(pending["state"], "cancelling");
            assert_eq!(pending["execution_state"], "unknown");
            scheduler.complete(id, Err(outcome), now);
            let terminal = scheduler.cancel_web(id, now).unwrap();
            assert_eq!(terminal["state"], state);
            assert_eq!(terminal["execution_state"], execution);
            assert!(scheduler.dispatch(now).is_empty());
        }
    }
}
