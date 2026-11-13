//! Local instance admission. A future platform transport supplies the same
//! model selection and a task-specific execution permit.
use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use hardware_probe::HardwareSnapshotV1;
use nvml_wrapper::{enums::device::UsedGpuMemory, Nvml};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{
    sync::{watch, Mutex as AsyncMutex},
    task::JoinSet,
    time::Instant,
};

use crate::{
    local_model::{self, Assets, Instance, ModelSpec, Phase, Runner, RunnerConnection},
    private_fs,
    resource_plan::{self, Placement, Sample},
    state::Paths,
};

const MAX_INSTANCES: usize = 8;
const MAX_TASK_IDS: usize = 4096;

#[derive(Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum ModelRequest {
    List,
    GpuList,
    Estimate {
        plan: String,
    },
    Infer {
        plan: String,
        task_id: String,
        prompt: String,
    },
    Cancel {
        task_id: String,
    },
    Unload {
        plan: String,
    },
}

pub(super) struct ModelPool {
    paths: Paths,
    installation_id: String,
    registry: Mutex<Registry>,
    admission: AsyncMutex<()>,
    shutdown: watch::Sender<bool>,
}

#[derive(Default)]
struct Registry {
    entries: BTreeMap<String, Entry>,
    seen_tasks: HashSet<String>,
}

struct Entry {
    slot: Arc<Slot>,
    view: Instance,
    active: BTreeMap<u8, ActiveTask>,
    idle_since: Option<Instant>,
    unloading: bool,
    calibration_error: Option<String>,
}

struct ActiveTask {
    id: String,
    cancel: watch::Sender<bool>,
}

struct Slot {
    journal: PathBuf,
    spec: ModelSpec,
    assets: Assets,
    runtime: AsyncMutex<InstanceRuntime>,
    closing: watch::Sender<bool>,
}

struct InstanceRuntime {
    instance: Instance,
    runner: Option<Runner>,
}

pub(super) fn check_previous(paths: &Paths, installation: &str) -> Result<(), String> {
    let smoke = paths.state.join("local-model-check.json");
    let check = |path: PathBuf| -> Result<(), String> {
        if let Some(instance) = private_fs::read_json::<Instance>(&path)? {
            if instance.installation_id != installation
                || instance.phase != Phase::Stopped
                || !instance.process_group_exit_confirmed
            {
                return Err(format!(
                    "上一模型实例退出未确认，保留隔离记录：{}",
                    path.display()
                ));
            }
        }
        Ok(())
    };
    check(smoke)?;
    let journal_dir = paths.state.join("model-instances");
    match std::fs::symlink_metadata(&journal_dir) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.to_string()),
        Ok(_) => private_fs::check_directory(&journal_dir)?,
    }
    for entry in std::fs::read_dir(journal_dir).map_err(|e| e.to_string())? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            check(path)?;
        }
    }
    Ok(())
}

impl ModelPool {
    pub fn new(paths: Paths, installation_id: String) -> Result<Self, String> {
        check_previous(&paths, &installation_id)?;
        private_fs::ensure_directory(&paths.state.join("model-plans"))?;
        private_fs::ensure_directory(&paths.state.join("model-instances"))?;
        Ok(Self {
            paths,
            installation_id,
            registry: Mutex::new(Registry::default()),
            admission: AsyncMutex::new(()),
            shutdown: watch::channel(false).0,
        })
    }

    fn registry(&self) -> MutexGuard<'_, Registry> {
        self.registry.lock().expect("model registry poisoned")
    }

    pub async fn execute(&self, request: ModelRequest) -> Result<Value, String> {
        match request {
            ModelRequest::List => {
                self.reap_idle().await;
                let entries: Vec<_> = self
                    .registry()
                    .entries
                    .iter()
                    .map(|(plan, entry)| {
                        json!({
                            "plan": plan,
                            "instance": entry.view,
                            "active_task_id": (entry.active.len() == 1)
                                .then(|| &entry.active.values().next().expect("one task").id),
                            "active_task_ids": entry.active.values().map(|task| &task.id).collect::<Vec<_>>(),
                            "slots_used": entry.active.len(),
                            "slots_total": entry.slot.spec.parallel_slots,
                            "unloading": entry.unloading,
                            "calibration_error":entry.calibration_error,
                        })
                    })
                    .collect();
                Ok(json!({"instances": entries}))
            }
            ModelRequest::GpuList => self.gpu_list().await,
            ModelRequest::Estimate { plan } => {
                if !local_model::valid_component(&plan) {
                    return Err("模型计划名无效。".into());
                }
                let _guard = self.admission.lock().await;
                if let Some(entry) = self.registry().entries.get(&plan) {
                    return Ok(
                        json!({"plan":plan,"placement":entry.view.placement,"already_loaded":true}),
                    );
                }
                let (_, _, placement, hardware, reading) = self.prepare(&plan).await?;
                let instances: Vec<_> = self
                    .registry()
                    .entries
                    .values()
                    .map(|entry| entry.view.clone())
                    .collect();
                resource_plan::recheck(&placement, &hardware, &reading, &instances)?;
                Ok(
                    json!({"plan":plan,"placement":placement,"loaded":false,"reserved":false,"sampled_at_unix_ms":hardware.captured_at_unix_ms}),
                )
            }
            ModelRequest::Cancel { task_id } => {
                let registry = self.registry();
                let active = registry
                    .entries
                    .values()
                    .flat_map(|entry| entry.active.values())
                    .find(|task| task.id == task_id)
                    .ok_or("当前节点没有对应的活动任务。")?;
                active.cancel.send_replace(true);
                Ok(json!({"task_id":task_id,"cancel_requested":true}))
            }
            ModelRequest::Unload { plan } => {
                let slot = {
                    let mut registry = self.registry();
                    let entry = registry
                        .entries
                        .get_mut(&plan)
                        .ok_or("本机未加载该计划。")?;
                    if !entry.active.is_empty()
                        || entry.unloading
                        || entry.view.phase == Phase::Quarantined
                    {
                        return Err("实例有活动任务、正在卸载或需要清理核对。".into());
                    }
                    entry.unloading = true;
                    entry.slot.clone()
                };
                self.unload(&plan, &slot).await?;
                Ok(json!({"plan":plan,"unloaded":true}))
            }
            ModelRequest::Infer {
                plan,
                task_id,
                prompt,
            } => self.infer(plan, task_id, prompt).await,
        }
    }

    async fn infer(&self, plan: String, task_id: String, prompt: String) -> Result<Value, String> {
        if !local_model::valid_component(&plan)
            || !uuid::Uuid::parse_str(&task_id).is_ok_and(|id| id.to_string() == task_id)
            || prompt.trim().is_empty()
            || prompt.len() > 2048
        {
            return Err("模型计划名、任务 UUID 或提示内容无效。".into());
        }
        let (slot, slot_id, mut cancel) = {
            // Serialize estimate -> fresh sample -> atomic group reservation.
            // HTTP generation remains outside this guard.
            let _guard = self.admission.lock().await;
            let prepared = if self.registry().entries.contains_key(&plan) {
                None
            } else {
                Some(self.prepare(&plan).await?)
            };
            self.admit(&plan, &task_id, prepared)?
        };
        let mut shutdown = self.shutdown.subscribe();
        let mut closing = slot.closing.subscribe();
        let mut legacy_cancel = cancel.clone();
        let result = tokio::select! {
            biased;
            _ = requested(&mut shutdown) => Err("节点正在停止。".into()),
            _ = requested(&mut closing) => Err("实例正在停止。".into()),
            _ = requested(&mut legacy_cancel), if slot.spec.parallel_slots == 1 => {
                Err("请求已取消，单路实例正在停止。".into())
            }
            result = async {
                let (runner, instance_id) = self.ready_runner(&plan, &slot).await?;
                if *slot.closing.borrow() || *self.shutdown.borrow() {
                    return Err("实例正在停止。".into());
                }
                let answer = runner.run_task(&slot.spec, slot_id, &prompt, &mut cancel).await?;
                Ok((instance_id, answer))
            } => result,
        };
        match result {
            Ok((instance_id, answer)) => {
                let mut registry = self.registry();
                if let Some(entry) = registry.entries.get_mut(&plan) {
                    if Arc::ptr_eq(&entry.slot, &slot)
                        && entry
                            .active
                            .get(&slot_id)
                            .is_some_and(|task| task.id == task_id)
                    {
                        entry.active.remove(&slot_id);
                        if entry.active.is_empty() {
                            entry.idle_since = Some(Instant::now());
                        }
                    }
                }
                let answer = answer.into_answer()?;
                Ok(json!({"task_id":task_id,"instance_id":instance_id,"answer":answer}))
            }
            Err(error) => {
                let should_stop = {
                    let mut registry = self.registry();
                    registry.entries.get_mut(&plan).is_some_and(|entry| {
                        if !Arc::ptr_eq(&entry.slot, &slot) || entry.unloading {
                            return false;
                        }
                        entry.unloading = true;
                        slot.closing.send_replace(true);
                        true
                    })
                };
                if should_stop {
                    self.unload(&plan, &slot).await?;
                }
                Err(error)
            }
        }
    }

    async fn prepare(
        &self,
        plan: &str,
    ) -> Result<(ModelSpec, Assets, Placement, HardwareSnapshotV1, Sample), String> {
        let paths = self.paths.clone();
        let plan_path = paths.state.join("model-plans").join(format!("{plan}.json"));
        let instances: Vec<_> = self
            .registry()
            .entries
            .values()
            .map(|entry| entry.view.clone())
            .collect();
        tokio::task::spawn_blocking(move || {
            let spec = private_fs::read_json::<ModelSpec>(&plan_path)?
                .ok_or("节点未配置该本机模型计划。")?;
            let assets = spec.verify(&paths)?;
            let (hardware, mut reading) = resource_plan::sample()?;
            for instance in &instances {
                if let Some(pid) = instance.runner_pid {
                    if let Some(bytes) = resource_plan::host_private_bytes(pid) {
                        reading.host_process_bytes.insert(pid, bytes);
                    }
                }
            }
            let placement =
                resource_plan::plan(&paths, &spec, &assets, &hardware, &reading, &instances)?;
            let (fresh, mut fresh_reading) = resource_plan::sample()?;
            for instance in &instances {
                if let Some(pid) = instance.runner_pid {
                    if let Some(bytes) = resource_plan::host_private_bytes(pid) {
                        fresh_reading.host_process_bytes.insert(pid, bytes);
                    }
                }
            }
            Ok((spec, assets, placement, fresh, fresh_reading))
        })
        .await
        .map_err(|e| e.to_string())?
    }

    fn admit(
        &self,
        plan: &str,
        task_id: &str,
        prepared: Option<(ModelSpec, Assets, Placement, HardwareSnapshotV1, Sample)>,
    ) -> Result<(Arc<Slot>, u8, watch::Receiver<bool>), String> {
        let mut registry = self.registry();
        if *self.shutdown.borrow() {
            return Err("节点正在停止。".into());
        }
        if registry.seen_tasks.contains(task_id) {
            return Err("该任务已提交，不能重复生成。".into());
        }
        if registry.seen_tasks.len() >= MAX_TASK_IDS {
            return Err("本机任务记录达到上限，请维护后重启节点。".into());
        }
        if !registry.entries.contains_key(plan) {
            if registry.entries.len() >= MAX_INSTANCES {
                return Err("本机实例数达到上限。".into());
            }
            let (spec, assets, placement, hardware, reading) =
                prepared.ok_or("实例状态已改变，请重新查询节点。")?;
            let instances: Vec<_> = registry
                .entries
                .values()
                .map(|entry| entry.view.clone())
                .collect();
            resource_plan::recheck(&placement, &hardware, &reading, &instances)?;
            let committed_threads: u32 = registry
                .entries
                .values()
                .map(|entry| entry.slot.spec.cpu_threads)
                .sum();
            if hardware.cpu.logical_cores.saturating_sub(committed_threads) < spec.cpu_threads {
                return Err("CPU 线程预算不足。".into());
            }
            let view = local_model::new_instance(&spec, self.installation_id.clone(), placement);
            let slot = Arc::new(Slot {
                journal: self
                    .paths
                    .state
                    .join("model-instances")
                    .join(format!("{plan}.json")),
                spec,
                assets,
                runtime: AsyncMutex::new(InstanceRuntime {
                    instance: view.clone(),
                    runner: None,
                }),
                closing: watch::channel(false).0,
            });
            registry.entries.insert(
                plan.to_owned(),
                Entry {
                    slot,
                    view,
                    active: BTreeMap::new(),
                    idle_since: None,
                    unloading: false,
                    calibration_error: None,
                },
            );
        }
        let entry = registry
            .entries
            .get_mut(plan)
            .expect("instance is registered");
        if entry.unloading
            || entry.calibration_error.is_some()
            || entry.view.phase == Phase::Quarantined
        {
            return Err("实例正忙或处于隔离状态，请稍后重新调度。".into());
        }
        let slot_id = (0..entry.slot.spec.parallel_slots)
            .find(|id| !entry.active.contains_key(id))
            .ok_or("实例并发槽位已满，请稍后重新调度。")?;
        let (sender, receiver) = watch::channel(false);
        entry.active.insert(
            slot_id,
            ActiveTask {
                id: task_id.to_owned(),
                cancel: sender,
            },
        );
        entry.idle_since = None;
        let slot = entry.slot.clone();
        registry.seen_tasks.insert(task_id.to_owned());
        Ok((slot, slot_id, receiver))
    }

    async fn ready_runner(
        &self,
        plan: &str,
        slot: &Slot,
    ) -> Result<(RunnerConnection, String), String> {
        let mut state = slot.runtime.lock().await;
        let runtime = &mut *state;
        if *slot.closing.borrow() {
            return Err("实例正在停止。".into());
        }
        if runtime.instance.phase == Phase::Starting && runtime.runner.is_none() {
            runtime.runner = Some(Runner::spawn(
                &self.paths,
                &slot.journal,
                &slot.spec,
                &slot.assets,
                &mut runtime.instance,
            )?);
            let runner = runtime.runner.as_mut().expect("runner spawned");
            runner.ready(&slot.journal, &mut runtime.instance).await?;
            if slot.spec.parallel_slots > 1 {
                runner.connection().verify_slots(&slot.spec).await?;
            }
            if let Some(entry) = self.registry().entries.get_mut(plan) {
                entry.view = runtime.instance.clone();
            }
        }
        let runner = runtime.runner.as_mut().ok_or("模型运行器尚未就绪。")?;
        if runtime.instance.phase != Phase::Ready || !runner.owns_endpoint()? {
            return Err("生成前无法确认模型进程仍持有监听端口。".into());
        }
        Ok((runner.connection(), runtime.instance.instance_id.clone()))
    }

    async fn stop_locked(
        &self,
        plan: &str,
        slot: &Slot,
        runtime: &mut InstanceRuntime,
    ) -> Result<(), String> {
        slot.closing.send_replace(true);
        let result = if let Some(runner) = runtime.runner.take() {
            runner.stop(&slot.journal, &mut runtime.instance).await
        } else if runtime.instance.phase == Phase::Quarantined {
            Err("无法确认实例退出，保留隔离。".into())
        } else {
            runtime.instance.phase = Phase::Stopped;
            runtime.instance.process_group_exit_confirmed = true;
            runtime.instance.reserved_vram_bytes = 0;
            runtime.instance.reserved_memory_bytes = 0;
            if let Some(placement) = &mut runtime.instance.placement {
                placement.clear_reservations();
            }
            private_fs::write_json(&slot.journal, &runtime.instance)
        };
        let mut registry = self.registry();
        if result.is_ok() {
            if registry
                .entries
                .get(plan)
                .is_some_and(|entry| std::ptr::eq(entry.slot.as_ref(), slot))
            {
                registry.entries.remove(plan);
            }
        } else {
            runtime.instance.phase = Phase::Quarantined;
            if let Some(entry) = registry.entries.get_mut(plan) {
                entry.view = runtime.instance.clone();
                entry.active.clear();
                entry.unloading = false;
            }
        }
        result
    }

    async fn unload(&self, plan: &str, slot: &Slot) -> Result<(), String> {
        let mut runtime = slot.runtime.lock().await;
        // Another stop may have completed while we waited for the same guard.
        if !self
            .registry()
            .entries
            .get(plan)
            .is_some_and(|entry| std::ptr::eq(entry.slot.as_ref(), slot))
        {
            return Ok(());
        }
        self.stop_locked(plan, slot, &mut runtime).await
    }

    pub async fn refresh_observations(&self) {
        let pids: Vec<_> = self
            .registry()
            .entries
            .values()
            .filter_map(|entry| entry.view.runner_pid)
            .collect();
        if pids.is_empty() {
            return;
        }
        let Ok(Ok((_, reading))) = tokio::task::spawn_blocking(move || {
            let (hardware, mut reading) = resource_plan::sample()?;
            for pid in pids {
                if let Some(bytes) = resource_plan::host_private_bytes(pid) {
                    reading.host_process_bytes.insert(pid, bytes);
                }
            }
            Ok::<_, String>((hardware, reading))
        })
        .await
        else {
            return;
        };
        let mut registry = self.registry();
        for entry in registry.entries.values_mut() {
            if entry.view.phase != Phase::Ready || entry.unloading {
                continue;
            }
            if let Err(error) = resource_plan::observe(&self.paths, &mut entry.view, &reading)
                .and_then(|()| private_fs::write_json(&entry.slot.journal, &entry.view))
            {
                entry.calibration_error = Some(error);
            }
        }
    }

    async fn gpu_list(&self) -> Result<Value, String> {
        let (hardware, gpu_processes) = tokio::task::spawn_blocking(|| {
            let snapshot = hardware_probe::probe_hardware().map_err(|error| error.to_string())?;
            let nvml = Nvml::init().ok();
            let processes: BTreeMap<_, _> = snapshot
                .gpus
                .iter()
                .filter_map(|gpu| {
                    let uuid = gpu.uuid.as_deref()?;
                    let device = nvml.as_ref()?.device_by_uuid(uuid).ok()?;
                    Some((uuid.to_owned(), device.running_compute_processes().ok()?))
                })
                .collect();
            Ok::<_, String>((snapshot, processes))
        })
        .await
        .map_err(|error| error.to_string())??;
        let registry = self.registry();
        let mut unobserved = Vec::new();
        let gpus: Vec<_> = hardware
            .gpus
            .iter()
            .map(|gpu| {
                let uuid = gpu.uuid.as_deref();
                let processes = uuid.and_then(|uuid| gpu_processes.get(uuid));
                let instances: Vec<_> = registry
                    .entries
                    .values()
                    .filter(|entry| entry.uses_gpu(uuid))
                    .map(|entry| entry.gpu_view(uuid, processes.map(Vec::as_slice)))
                    .collect();
                json!({
                    "index": gpu.index, "uuid": uuid, "name": gpu.name,
                    "vram_total_bytes": (gpu.vram_total_bytes != 0).then_some(gpu.vram_total_bytes),
                    "vram_free_bytes": (gpu.vram_total_bytes != 0).then_some(gpu.vram_free_bytes),
                    "vram_used_bytes": (gpu.vram_total_bytes != 0)
                        .then(|| gpu.vram_total_bytes.saturating_sub(gpu.vram_free_bytes)),
                    "instances": instances,
                })
            })
            .collect();
        for entry in registry.entries.values() {
            if !hardware
                .gpus
                .iter()
                .any(|gpu| entry.uses_gpu(gpu.uuid.as_deref()))
            {
                unobserved.push(entry.gpu_view(None, None));
            }
        }
        Ok(json!({
            "captured_at_unix_ms": hardware.captured_at_unix_ms,
            "gpus": gpus,
            "instances_without_gpu_sample": unobserved
        }))
    }

    pub async fn reap_idle(&self) {
        let candidates: Vec<_> = self
            .registry()
            .entries
            .iter()
            .filter(|(_, entry)| {
                entry.view.phase == Phase::Ready && entry.active.is_empty() && !entry.unloading
            })
            .map(|(name, entry)| (name.clone(), entry.slot.clone()))
            .collect();
        for (name, slot) in candidates {
            let Ok(mut runtime) = slot.runtime.try_lock() else {
                continue;
            };
            let alive = runtime
                .runner
                .as_mut()
                .is_some_and(|runner| runner.owns_endpoint().unwrap_or(false));
            {
                let mut registry = self.registry();
                let Some(entry) = registry.entries.get_mut(&name) else {
                    continue;
                };
                if !entry.active.is_empty()
                    || entry.unloading
                    || !Arc::ptr_eq(&entry.slot, &slot)
                    || (alive && !entry.idle_expired())
                {
                    continue;
                }
                entry.unloading = true;
            }
            // stop_locked retains an uncertain exit as quarantined inventory.
            let _ = self.stop_locked(&name, &slot, &mut runtime).await;
        }
    }

    pub async fn stop_all(self: &Arc<Self>) -> Result<(), String> {
        self.shutdown.send_replace(true);
        let slots: Vec<_> = self
            .registry()
            .entries
            .iter()
            .map(|(name, entry)| (name.clone(), entry.slot.clone()))
            .collect();
        let mut jobs = JoinSet::new();
        for (name, slot) in slots {
            let pool = self.clone();
            jobs.spawn(async move { pool.unload(&name, &slot).await });
        }
        let mut result = Ok(());
        while let Some(job) = jobs.join_next().await {
            if let Err(error) = job.map_err(|e| e.to_string()).and_then(|value| value) {
                result = Err(error);
            }
        }
        result
    }
}

impl Entry {
    fn idle_expired(&self) -> bool {
        self.idle_since
            .zip(self.slot.spec.idle_unload_seconds)
            .is_some_and(|(since, seconds)| since.elapsed() >= Duration::from_secs(seconds))
    }

    fn uses_gpu(&self, uuid: Option<&str>) -> bool {
        let Some(uuid) = uuid else {
            return false;
        };
        self.view
            .placement
            .as_ref()
            .map(|placement| placement.gpus.iter().any(|gpu| gpu.gpu_uuid == uuid))
            .unwrap_or(self.view.gpu_uuid == uuid)
    }

    fn gpu_view(
        &self,
        gpu_uuid: Option<&str>,
        processes: Option<&[nvml_wrapper::struct_wrappers::device::ProcessInfo]>,
    ) -> Value {
        let used = self.view.runner_pid.and_then(|pid| {
            processes?
                .iter()
                .find(|process| process.pid == pid)
                .and_then(|process| match process.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => Some(bytes),
                    UsedGpuMemory::Unavailable => None,
                })
        });
        let state = if self.unloading {
            "unloading"
        } else {
            match self.view.phase {
                Phase::Starting => "starting",
                Phase::Ready if self.active.is_empty() => "idle",
                Phase::Ready => "in_use",
                Phase::Quarantined => "quarantined",
                Phase::Stopped => "stopped",
            }
        };
        let per_gpu = self.view.placement.as_ref().and_then(|placement| {
            placement
                .gpus
                .iter()
                .find(|gpu| Some(gpu.gpu_uuid.as_str()) == gpu_uuid)
        });
        json!({
            "gpu_uuid": gpu_uuid,
            "gpu_group":self.view.placement.as_ref().map(|p|p.gpus.iter().map(|g|&g.gpu_uuid).collect::<Vec<_>>()),
            "estimated_bytes":per_gpu.map(|g|g.estimated_bytes),
            "observed_peak_bytes":per_gpu.and_then(|g|g.observed_peak_bytes),
            "resource_source":self.view.placement.as_ref().map(|p|&p.source),
            "calibration_error":self.calibration_error,
            "model_id": self.view.model_id, "variant_id": self.view.variant_id,
            "instance_id": self.view.instance_id,
            "vram_used_bytes": used, "vram_reserved_bytes": per_gpu.map(|g|g.reserved_bytes).unwrap_or(self.view.reserved_vram_bytes),
            "state": state, "slots_used": self.active.len(), "slots_total": self.slot.spec.parallel_slots,
        })
    }
}

async fn requested(signal: &mut watch::Receiver<bool>) {
    let _ = signal.wait_for(|value| *value).await;
}
