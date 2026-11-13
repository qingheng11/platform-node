//! Local capacity planning. Cached demand never substitutes for fresh admission.
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    os::unix::{fs::OpenOptionsExt, process::CommandExt},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use hardware_probe::HardwareSnapshotV1;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};

use crate::{
    local_model::{Assets, Instance, ModelSpec, MIB},
    private_fs,
    state::Paths,
};

const POLICY: &str = "node-layer-f16-v1";
const GPU_MARGIN: u64 = 512 * MIB;
const HOST_MARGIN: u64 = 1024 * MIB;
const FIT_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct GpuBudget {
    pub gpu_uuid: String,
    pub estimated_bytes: u64,
    pub reserved_bytes: u64,
    pub observed_peak_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Placement {
    pub gpus: Vec<GpuBudget>,
    pub split_mode: String,
    pub tensor_split: Vec<u32>,
    pub host_estimated_bytes: u64,
    pub host_reserved_bytes: u64,
    pub host_observed_peak_bytes: Option<u64>,
    pub cache_key: Option<String>,
    pub source: String,
}

impl Placement {
    pub fn visible_devices(&self) -> String {
        self.gpus
            .iter()
            .map(|gpu| gpu.gpu_uuid.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    pub fn arguments(&self) -> Vec<String> {
        let mut args = vec![
            "--split-mode".into(),
            self.split_mode.clone(),
            "--fit".into(),
            "off".into(),
        ];
        if self.gpus.len() > 1 {
            args.extend([
                "--tensor-split".into(),
                self.tensor_split
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ]);
        }
        args
    }

    pub fn clear_reservations(&mut self) {
        for gpu in &mut self.gpus {
            gpu.reserved_bytes = 0;
        }
        self.host_reserved_bytes = 0;
    }

    pub fn total_reserved(&self) -> u64 {
        self.gpus.iter().map(|gpu| gpu.reserved_bytes).sum()
    }
}

#[derive(Default)]
pub(super) struct Sample {
    // A missing per-process reading stays unknown (not zero).
    pub gpu_process_bytes: BTreeMap<(String, u32), u64>,
    pub host_process_bytes: BTreeMap<u32, u64>,
}

pub(super) fn sample() -> Result<(HardwareSnapshotV1, Sample), String> {
    use nvml_wrapper::{enums::device::UsedGpuMemory, Nvml};
    let hardware = hardware_probe::probe_hardware().map_err(|error| error.to_string())?;
    let mut reading = Sample::default();
    if let Ok(nvml) = Nvml::init() {
        for gpu in &hardware.gpus {
            let Some(uuid) = &gpu.uuid else {
                continue;
            };
            let Ok(device) = nvml.device_by_uuid(uuid.as_str()) else {
                continue;
            };
            let Ok(processes) = device.running_compute_processes() else {
                continue;
            };
            for process in processes {
                if let UsedGpuMemory::Used(bytes) = process.used_gpu_memory {
                    reading
                        .gpu_process_bytes
                        .insert((uuid.clone(), process.pid), bytes);
                }
            }
        }
    }
    Ok((hardware, reading))
}

pub(super) fn host_private_bytes(pid: u32) -> Option<u64> {
    // RssAnon avoids crediting reclaimable mmap/file-cache pages against MemAvailable.
    let text = fs::read_to_string(format!("/proc/{pid}/status")).ok()?;
    let line = text.lines().find(|line| line.starts_with("RssAnon:"))?;
    line.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

fn budgets(instance: &Instance) -> Vec<(String, u64)> {
    match &instance.placement {
        Some(placement) => placement
            .gpus
            .iter()
            .map(|gpu| (gpu.gpu_uuid.clone(), gpu.reserved_bytes))
            .collect(),
        None => vec![(instance.gpu_uuid.clone(), instance.reserved_vram_bytes)],
    }
}

pub(super) fn available(
    hardware: &HardwareSnapshotV1,
    reading: &Sample,
    instances: &[Instance],
) -> (BTreeMap<String, u64>, u64) {
    let mut remaining: BTreeMap<_, _> = hardware
        .gpus
        .iter()
        .filter_map(|gpu| {
            let id = gpu.uuid.as_deref()?;
            uuid::Uuid::parse_str(id.strip_prefix("GPU-")?).ok()?;
            (gpu.vram_total_bytes > 0).then(|| (id.to_owned(), gpu.vram_free_bytes))
        })
        .collect();
    let mut host = hardware.memory.available_bytes;
    for instance in instances {
        for (uuid, reserved) in budgets(instance) {
            let observed = instance
                .runner_pid
                .and_then(|pid| reading.gpu_process_bytes.get(&(uuid.clone(), pid)))
                .copied()
                .unwrap_or(0);
            if let Some(free) = remaining.get_mut(&uuid) {
                *free = free.saturating_sub(reserved.saturating_sub(observed));
            }
        }
        let observed = instance
            .runner_pid
            .and_then(|pid| reading.host_process_bytes.get(&pid))
            .copied()
            .unwrap_or(0);
        host = host.saturating_sub(instance.reserved_memory_bytes.saturating_sub(observed));
    }
    (remaining, host)
}

pub(super) fn recheck(
    placement: &Placement,
    hardware: &HardwareSnapshotV1,
    reading: &Sample,
    instances: &[Instance],
) -> Result<(), String> {
    let (gpus, host) = available(hardware, reading, instances);
    if placement.gpus.is_empty()
        || placement.gpus.iter().any(|gpu| {
            gpus.get(&gpu.gpu_uuid)
                .is_none_or(|free| *free < gpu.reserved_bytes)
        })
        || host < placement.host_reserved_bytes
    {
        return Err("resource_capacity_changed：资源已变化，整组 GPU 未预留，请重新评估。".into());
    }
    Ok(())
}

pub(super) fn plan(
    paths: &Paths,
    spec: &ModelSpec,
    assets: &Assets,
    hardware: &HardwareSnapshotV1,
    reading: &Sample,
    instances: &[Instance],
) -> Result<Placement, String> {
    if hardware.cpu.logical_cores < spec.cpu_threads {
        return Err("resource_capacity_unavailable：CPU 线程预算不足。".into());
    }
    let (free, host) = available(hardware, reading, instances);
    let mut cards: Vec<_> = free
        .into_iter()
        .filter(|(id, _)| spec.gpu_uuid.as_ref().is_none_or(|target| target == id))
        .collect();
    cards.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if spec.resource_mode == ResourceMode::Manual {
        let (uuid, _) = cards
            .into_iter()
            .find(|(_, free)| *free >= u64::from(spec.vram_mib) * MIB)
            .ok_or("resource_capacity_unavailable：没有满足手动预算的 GPU。")?;
        let result = Placement {
            gpus: vec![GpuBudget {
                gpu_uuid: uuid,
                estimated_bytes: u64::from(spec.vram_mib) * MIB,
                reserved_bytes: u64::from(spec.vram_mib) * MIB,
                observed_peak_bytes: None,
            }],
            split_mode: "none".into(),
            tensor_split: vec![1],
            host_estimated_bytes: u64::from(spec.memory_mib) * MIB,
            host_reserved_bytes: u64::from(spec.memory_mib) * MIB,
            host_observed_peak_bytes: None,
            cache_key: None,
            source: "manual".into(),
        };
        if result.host_reserved_bytes > host {
            return Err("resource_capacity_unavailable：主存预算不足。".into());
        }
        return Ok(result);
    }
    if !spec.runtime_files.contains_key("llama-fit-params") {
        return Err(
            "resource_estimator_missing：自动评估需要清单内已校验的 llama-fit-params。".into(),
        );
    }
    let tool = assets.runtime.join("llama-fit-params");
    use std::os::unix::fs::PermissionsExt;
    if fs::metadata(&tool)
        .map_err(|e| format!("resource_estimator_missing：{e}"))?
        .permissions()
        .mode()
        & 0o100
        == 0
    {
        return Err("resource_estimator_not_executable：llama-fit-params 没有执行权限。".into());
    }
    private_fs::ensure_directory(&paths.state.join("resource-profiles"))?;
    select(&cards, host, spec.max_gpus as usize, |selected| {
        let split = proportions(selected);
        let key = cache_key(spec, hardware, selected, &split)?;
        let cache = paths
            .state
            .join("resource-profiles")
            .join(format!("{key}.json"));
        if let Some(mut cached) = private_fs::read_json::<Placement>(&cache)? {
            validate_cache(&cached, &key, selected, &split)?;
            cached.source = "cache".into();
            return Ok(cached);
        }
        let output = run_fitter(paths, spec, assets, selected, &split)?;
        let (gpu_bytes, host_bytes) = parse_estimate(&output, selected.len())?;
        let result = Placement {
            gpus: selected
                .iter()
                .zip(gpu_bytes)
                .map(|((uuid, _), bytes)| GpuBudget {
                    gpu_uuid: uuid.clone(),
                    estimated_bytes: bytes,
                    reserved_bytes: margin(bytes, GPU_MARGIN),
                    observed_peak_bytes: None,
                })
                .collect(),
            split_mode: if selected.len() == 1 { "none" } else { "layer" }.into(),
            tensor_split: split,
            host_estimated_bytes: host_bytes,
            host_reserved_bytes: margin(host_bytes, HOST_MARGIN),
            host_observed_peak_bytes: None,
            cache_key: Some(key),
            source: "estimated".into(),
        };
        private_fs::write_json(&cache, &result)?;
        Ok(result)
    })
}

#[derive(Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum ResourceMode {
    #[default]
    Manual,
    Auto,
}

fn margin(bytes: u64, floor: u64) -> u64 {
    bytes.saturating_add((bytes / 8).max(floor))
}

fn proportions(cards: &[(String, u64)]) -> Vec<u32> {
    if cards.len() == 1 {
        return vec![1];
    }
    let total: u64 = cards
        .iter()
        .map(|(_, free)| free.saturating_sub(GPU_MARGIN))
        .sum();
    cards
        .iter()
        .map(|(_, free)| {
            ((u128::from(free.saturating_sub(GPU_MARGIN)) * 100 / u128::from(total.max(1))) as u32)
                .max(1)
        })
        .collect()
}

fn select(
    cards: &[(String, u64)],
    host: u64,
    max_gpus: usize,
    mut estimate: impl FnMut(&[(String, u64)]) -> Result<Placement, String>,
) -> Result<Placement, String> {
    let started = Instant::now();
    // Try every singleton, then the groups with the most free memory. Bounded
    // search avoids exponential work on large servers; no cross-host splitting.
    let fits = |placement: &Placement, group: &[(String, u64)]| {
        placement.host_reserved_bytes <= host
            && placement.gpus.len() == group.len()
            && placement
                .gpus
                .iter()
                .zip(group)
                .all(|(budget, (uuid, free))| {
                    budget.gpu_uuid == *uuid && budget.reserved_bytes <= *free
                })
    };
    for card in cards {
        if started.elapsed() > Duration::from_secs(180) {
            return Err("resource_planning_timeout：评估超时，未启动模型。".into());
        }
        let group = std::slice::from_ref(card);
        let result = estimate(group)?;
        if fits(&result, group) {
            return Ok(result);
        }
    }
    for count in 2..=max_gpus.min(cards.len()) {
        if started.elapsed() > Duration::from_secs(180) {
            return Err("resource_planning_timeout：评估超时，未启动模型。".into());
        }
        let group = &cards[..count];
        let result = estimate(group)?;
        if fits(&result, group) {
            return Ok(result);
        }
    }
    Err("resource_capacity_unavailable：单卡及允许的 GPU 组均不能容纳指定上下文和并发；未降低配置或启动模型。".into())
}

fn cache_key(
    spec: &ModelSpec,
    hardware: &HardwareSnapshotV1,
    cards: &[(String, u64)],
    split: &[u32],
) -> Result<String, String> {
    let gpus: Vec<_> = cards.iter().map(|(uuid,_)| hardware.gpus.iter().find(|gpu| gpu.uuid.as_ref()==Some(uuid)).map(|gpu| {
        json!({"uuid":uuid,"name":gpu.name,"total":gpu.vram_total_bytes,"driver":gpu.driver_version,"cuda":gpu.cuda_driver_version})
    }).ok_or("资源评估 GPU 身份已变化。")).collect::<Result<_,_>>()?;
    let key = json!({"policy":POLICY,"spec":spec,"gpu":gpus,"split":split,"cpu":hardware.cpu.model,"cores":hardware.cpu.logical_cores});
    Ok(format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(&key).map_err(|e| e.to_string())?)
    ))
}

fn validate_cache(
    p: &Placement,
    key: &str,
    cards: &[(String, u64)],
    split: &[u32],
) -> Result<(), String> {
    if p.cache_key.as_deref() != Some(key)
        || p.tensor_split != split
        || p.gpus.len() != cards.len()
        || p.split_mode != if cards.len() == 1 { "none" } else { "layer" }
        || p.host_reserved_bytes
            < margin(
                p.host_estimated_bytes
                    .max(p.host_observed_peak_bytes.unwrap_or(0)),
                HOST_MARGIN,
            )
        || p.gpus.iter().zip(cards).any(|(g, (id, _))| {
            g.gpu_uuid != *id
                || g.estimated_bytes == 0
                || g.reserved_bytes
                    < margin(
                        g.estimated_bytes.max(g.observed_peak_bytes.unwrap_or(0)),
                        GPU_MARGIN,
                    )
        })
    {
        return Err("resource_cache_invalid：资源缓存无效，保留文件供检查。".into());
    }
    Ok(())
}

fn run_fitter(
    paths: &Paths,
    spec: &ModelSpec,
    assets: &Assets,
    cards: &[(String, u64)],
    split: &[u32],
) -> Result<String, String> {
    let log_path = paths
        .logs
        .join(format!("resource-estimate-{}.log", uuid::Uuid::new_v4()));
    let log = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(&log_path)
        .map_err(|e| e.to_string())?;
    let mut command = Command::new(assets.runtime.join("llama-fit-params"));
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("LANG", "C.UTF-8")
        .env("HOME", &paths.runtime)
        .env(
            "CUDA_VISIBLE_DEVICES",
            cards
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>()
                .join(","),
        )
        .env("CUDA_CACHE_DISABLE", "1")
        .env("LD_LIBRARY_PATH", &assets.runtime)
        .current_dir(&assets.runtime)
        .arg("--model")
        .arg(&assets.model)
        .args(spec.memory_arguments())
        .args([
            "--fit",
            "off",
            "-fitp",
            "on",
            "--split-mode",
            if cards.len() == 1 { "none" } else { "layer" },
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?))
        .stderr(Stdio::from(log));
    if cards.len() > 1 {
        command.args([
            "--tensor-split",
            &split
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        ]);
    }
    command.process_group(0);
    let parent = std::process::id() as libc::pid_t;
    // SAFETY: only async-signal-safe syscalls are made between fork and exec.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL, 0, 0, 0) == -1
                || libc::getppid() != parent
            {
                return Err(std::io::Error::other("estimator parent changed"));
            }
            Ok(())
        });
    }
    let mut child = command
        .spawn()
        .map_err(|e| format!("resource_estimator_start_failed：{e}"))?;
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().map_err(|e| e.to_string())? {
            break status;
        }
        if started.elapsed() > FIT_TIMEOUT
            || fs::metadata(&log_path).map_err(|e| e.to_string())?.len() > 2 * 1024 * 1024
        {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!(
                "resource_estimator_timeout_or_output_limit：请检查 {}",
                log_path.display()
            ));
        }
        thread::sleep(Duration::from_millis(25));
    };
    if !status.success() {
        return Err(format!(
            "resource_estimator_failed：不是显存容量结论，请检查 {}",
            log_path.display()
        ));
    }
    let output = fs::read_to_string(&log_path).map_err(|e| e.to_string())?;
    if output.len() > 2 * 1024 * 1024 {
        return Err("resource_estimator_output_limit".into());
    }
    Ok(output)
}

fn parse_estimate(output: &str, count: usize) -> Result<(Vec<u64>, u64), String> {
    let mut gpu = BTreeMap::new();
    let mut host = None;
    for line in output.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields
            .first()
            .is_none_or(|name| !name.starts_with("CUDA") && *name != "Host")
        {
            continue;
        }
        if fields.len() != 4 {
            return Err("resource_estimator_format_invalid：无法识别逐设备资源数据。".into());
        }
        let values = fields[1..]
            .iter()
            .try_fold(0u64, |sum, word| sum.checked_add(word.parse::<u64>().ok()?))
            .and_then(|mib| mib.checked_mul(MIB))
            .ok_or("resource_estimator_format_invalid：资源数值无效。")?;
        if fields[0] == "Host" {
            if host.replace(values).is_some() {
                return Err("resource_estimator_format_invalid：重复主存数据。".into());
            }
        } else {
            let index = fields[0]
                .strip_prefix("CUDA")
                .and_then(|text| text.parse::<usize>().ok())
                .filter(|index| *index < count)
                .ok_or("resource_estimator_format_invalid：GPU 索引越界。")?;
            if values == 0 || gpu.insert(index, values).is_some() {
                return Err("resource_estimator_format_invalid：重复或空 GPU 数据。".into());
            }
        }
    }
    let host = host.ok_or("resource_estimator_format_invalid：缺少主存数据。")?;
    let gpu = (0..count)
        .map(|index| {
            gpu.get(&index)
                .copied()
                .ok_or_else(|| "resource_estimator_format_invalid：缺少 GPU 数据。".to_owned())
        })
        .collect::<Result<_, _>>()?;
    Ok((gpu, host))
}

pub(super) fn observe(
    paths: &Paths,
    instance: &mut Instance,
    reading: &Sample,
) -> Result<(), String> {
    let Some(pid) = instance.runner_pid else {
        return Ok(());
    };
    let Some(placement) = &mut instance.placement else {
        return Ok(());
    };
    let mut changed = false;
    for gpu in &mut placement.gpus {
        if let Some(used) = reading.gpu_process_bytes.get(&(gpu.gpu_uuid.clone(), pid)) {
            if *used > gpu.observed_peak_bytes.unwrap_or(0) {
                gpu.observed_peak_bytes = Some(*used);
                changed = true;
                gpu.reserved_bytes = gpu.reserved_bytes.max(margin(*used, GPU_MARGIN));
            }
        }
    }
    if let Some(used) = reading.host_process_bytes.get(&pid) {
        if *used > placement.host_observed_peak_bytes.unwrap_or(0) {
            placement.host_observed_peak_bytes = Some(*used);
            changed = true;
            placement.host_reserved_bytes = placement
                .host_reserved_bytes
                .max(margin(*used, HOST_MARGIN));
        }
    }
    instance.reserved_vram_bytes = placement.total_reserved();
    instance.reserved_memory_bytes = placement.host_reserved_bytes;
    if changed {
        if let Some(key) = &placement.cache_key {
            let path = paths
                .state
                .join("resource-profiles")
                .join(format!("{key}.json"));
            // Preserve the largest demand across instances reusing one profile.
            let mut saved =
                private_fs::read_json::<Placement>(&path)?.ok_or("资源校准缓存丢失。")?;
            for (target, current) in saved.gpus.iter_mut().zip(&placement.gpus) {
                target.observed_peak_bytes =
                    target.observed_peak_bytes.max(current.observed_peak_bytes);
                target.reserved_bytes = target.reserved_bytes.max(current.reserved_bytes);
            }
            saved.host_observed_peak_bytes = saved
                .host_observed_peak_bytes
                .max(placement.host_observed_peak_bytes);
            saved.host_reserved_bytes =
                saved.host_reserved_bytes.max(placement.host_reserved_bytes);
            private_fs::write_json(&path, &saved)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_model;

    fn id(index: usize) -> String {
        format!("GPU-00000000-0000-4000-8000-{index:012}")
    }

    fn hardware() -> HardwareSnapshotV1 {
        serde_json::from_value(json!({
            "schema_version":1,"captured_at_unix_ms":1000,
            "os":{"family":"linux","version":"22.04","arch":"x86_64"},
            "cpu":{"vendor":"intel","vendor_id":"GenuineIntel","model":"test","physical_cores":8,
                "logical_cores":16,"frequency_mhz":2000,"features":[]},
            "memory":{"total_bytes":64*1024*MIB,"available_bytes":50*1024*MIB},
            "gpus":(0..3).map(|index|json!({"id":index.to_string(),"uuid":id(index),"index":index,
                "vendor":"nvidia","name":"V100 fixture","vram_total_bytes":16384*MIB,"vram_free_bytes":15000*MIB,
                "utilization_percent":0,"temperature_celsius":45,"driver_version":"580","cuda_driver_version":"13"})).collect::<Vec<_>>(),
            "warnings":[]
        })).unwrap()
    }

    fn spec() -> ModelSpec {
        serde_json::from_value(
            json!({"schema_version":1,"model_id":"test-27b","variant_id":"Q8_0",
            "runtime_id":"test-runtime","model_file":"test.gguf","model_sha256":"a".repeat(64),
            "runtime_files":{"llama-server":"b".repeat(64),"llama-fit-params":"c".repeat(64)},
            "context_tokens":2048,"parallel_slots":4,"output_tokens":800,"gpu_layers":999,
            "cpu_threads":4,"resource_mode":"auto","max_gpus":3}),
        )
        .unwrap()
    }

    fn demand(cards: &[(String, u64)], amount: u64) -> Placement {
        Placement {
            gpus: cards
                .iter()
                .map(|(uuid, _)| GpuBudget {
                    gpu_uuid: uuid.clone(),
                    estimated_bytes: amount,
                    reserved_bytes: margin(amount, GPU_MARGIN),
                    observed_peak_bytes: None,
                })
                .collect(),
            split_mode: if cards.len() == 1 { "none" } else { "layer" }.into(),
            tensor_split: proportions(cards),
            host_estimated_bytes: MIB,
            host_reserved_bytes: HOST_MARGIN + MIB,
            host_observed_peak_bytes: None,
            cache_key: None,
            source: "estimated".into(),
        }
    }

    #[test]
    fn parses_each_device_not_the_sum_and_rejects_ambiguous_data() {
        let (gpus, host) = parse_estimate(
            "some diagnostic\nCUDA1 10000 100 20\nCUDA0 9000 200 30\nHost 1 2 3\n",
            2,
        )
        .unwrap();
        assert_eq!(gpus, vec![9230 * MIB, 10120 * MIB]);
        assert_eq!(host, 6 * MIB);
        for bad in [
            "CUDA0 1 2 3\nHost 1 2 3",
            "CUDA0 1 2 3\nCUDA0 4 5 6\nHost 1 2 3",
            "CUDA2 1 2 3\nHost 1 2 3",
            "CUDA0 -1 2 3\nHost 1 2 3",
            "CUDA0 1 2 3 4\nHost 1 2 3",
            "CUDA0 18446744073709551615 2 3\nHost 1 2 3",
        ] {
            assert!(parse_estimate(bad, 2).is_err(), "{bad}");
        }
    }

    #[test]
    fn automatically_expands_to_two_cards_and_checks_each_card() {
        let cards = vec![
            (id(0), 15000 * MIB),
            (id(1), 14000 * MIB),
            (id(2), 13000 * MIB),
        ];
        let selected = select(&cards, 4 * 1024 * MIB, 3, |group| {
            Ok(demand(
                group,
                if group.len() == 1 {
                    24000 * MIB
                } else {
                    12000 * MIB
                },
            ))
        })
        .unwrap();
        assert_eq!(selected.gpus.len(), 2);
        assert_eq!(selected.visible_devices(), format!("{},{}", id(0), id(1)));
        let uneven = vec![(id(0), 15000 * MIB), (id(1), 9000 * MIB)];
        assert!(select(&uneven, 4 * 1024 * MIB, 2, |group| Ok(demand(
            group,
            if group.len() == 1 {
                24000 * MIB
            } else {
                12000 * MIB
            }
        )))
        .is_err());
    }

    #[test]
    fn small_model_stays_single_card_and_respects_maximum_and_host_budget() {
        let cards = vec![(id(0), 15000 * MIB), (id(1), 14000 * MIB)];
        assert_eq!(
            select(&cards, 4 * 1024 * MIB, 2, |group| Ok(demand(
                group,
                1000 * MIB
            )))
            .unwrap()
            .gpus
            .len(),
            1
        );
        assert!(select(&cards, 4 * 1024 * MIB, 1, |group| Ok(demand(
            group,
            24000 * MIB
        )))
        .is_err());
        assert!(select(&cards, 100 * MIB, 2, |group| Ok(demand(group, 1000 * MIB))).is_err());
    }

    #[test]
    fn admission_credits_actual_allocations_once_unknowns_conservatively() {
        let hw = hardware();
        let plan = demand(&[(id(0), 15000 * MIB), (id(1), 15000 * MIB)], 1000 * MIB);
        let mut instance = local_model::new_instance(&spec(), "test".into(), plan.clone());
        instance.runner_pid = Some(99);
        let mut reading = Sample::default();
        reading.gpu_process_bytes.insert((id(0), 99), 800 * MIB);
        let (free, _) = available(&hw, &reading, &[instance]);
        assert_eq!(
            free[&id(0)],
            15000 * MIB - (plan.gpus[0].reserved_bytes - 800 * MIB)
        );
        assert_eq!(free[&id(1)], 15000 * MIB - plan.gpus[1].reserved_bytes);
        assert_eq!(free[&id(2)], 15000 * MIB);
    }

    #[test]
    fn changed_or_missing_second_card_rejects_whole_group() {
        let mut hw = hardware();
        let plan = demand(&[(id(0), 15000 * MIB), (id(1), 15000 * MIB)], 9000 * MIB);
        assert!(recheck(&plan, &hw, &Sample::default(), &[]).is_ok());
        hw.gpus[1].vram_free_bytes = 1;
        assert!(recheck(&plan, &hw, &Sample::default(), &[]).is_err());
        hw.gpus.remove(1);
        assert!(recheck(&plan, &hw, &Sample::default(), &[]).is_err());
    }

    #[test]
    fn cache_binds_config_runtime_and_gpu_order_but_not_free_memory() {
        let hw = hardware();
        let model = spec();
        let mut cards = vec![(id(0), 15000 * MIB), (id(1), 15000 * MIB)];
        let key = cache_key(&model, &hw, &cards, &[50, 50]).unwrap();
        cards[0].1 = 14000 * MIB;
        assert_eq!(key, cache_key(&model, &hw, &cards, &[50, 50]).unwrap());
        cards.reverse();
        assert_ne!(key, cache_key(&model, &hw, &cards, &[50, 50]).unwrap());
        cards.reverse();
        let mut changed = model.clone();
        changed.parallel_slots = 1;
        assert_ne!(key, cache_key(&changed, &hw, &cards, &[50, 50]).unwrap());
        changed = model.clone();
        changed
            .runtime_files
            .insert("llama-server".into(), "d".repeat(64));
        assert_ne!(key, cache_key(&changed, &hw, &cards, &[50, 50]).unwrap());
        assert_ne!(key, cache_key(&model, &hw, &cards, &[40, 60]).unwrap());
    }

    #[test]
    fn estimates_and_observations_never_shrink_reservation_and_all_cards_release() {
        let mut plan = demand(&[(id(0), 15000 * MIB), (id(1), 15000 * MIB)], 1000 * MIB);
        let mut instance = local_model::new_instance(&spec(), "test".into(), plan.clone());
        instance.runner_pid = Some(99);
        let mut reading = Sample::default();
        reading.gpu_process_bytes.insert((id(1), 99), 3000 * MIB);
        // No cache writes for this in-memory/manual fixture.
        let paths: Paths = serde_json::from_value(
            json!({"state":"/unused","models":"/unused","environments":"/unused",
            "logs":"/unused","runtime":"/unused","config":"/unused","socket":"/unused"}),
        )
        .unwrap();
        observe(&paths, &mut instance, &reading).unwrap();
        plan = instance.placement.clone().unwrap();
        assert_eq!(plan.gpus[0].observed_peak_bytes, None);
        assert_eq!(plan.gpus[1].observed_peak_bytes, Some(3000 * MIB));
        assert!(plan.gpus[1].reserved_bytes > 3000 * MIB);
        reading.gpu_process_bytes.insert((id(1), 99), 2000 * MIB);
        observe(&paths, &mut instance, &reading).unwrap();
        assert_eq!(
            instance.placement.as_ref().unwrap().gpus[1].reserved_bytes,
            plan.gpus[1].reserved_bytes
        );
        plan.clear_reservations();
        assert_eq!(plan.total_reserved(), 0);
        assert_eq!(plan.host_reserved_bytes, 0);
    }

    #[test]
    fn launch_arguments_keep_estimator_and_runner_memory_configuration_identical() {
        let model = spec();
        let args = model.memory_arguments();
        assert_eq!(
            args[args.iter().position(|v| v == "--ctx-size").unwrap() + 1],
            "8192"
        );
        assert_eq!(
            args[args.iter().position(|v| v == "--parallel").unwrap() + 1],
            "4"
        );
        assert!(args.windows(2).any(|pair| pair == ["--flash-attn", "off"]));
        let multi = demand(&[(id(2), 1000 * MIB), (id(0), 2000 * MIB)], 500 * MIB);
        assert_eq!(multi.visible_devices(), format!("{},{}", id(2), id(0)));
        assert!(multi
            .arguments()
            .windows(2)
            .any(|pair| pair == ["--split-mode", "layer"]));
        assert!(!multi.arguments().iter().any(|arg| arg == "tensor"));
    }

    #[test]
    fn malformed_cache_cannot_bypass_per_card_or_observed_peak_limits() {
        let hw = hardware();
        let cards = vec![(id(0), 15000 * MIB), (id(1), 15000 * MIB)];
        let split = proportions(&cards);
        let key = cache_key(&spec(), &hw, &cards, &split).unwrap();
        let mut good = demand(&cards, 1000 * MIB);
        good.cache_key = Some(key.clone());
        assert!(validate_cache(&good, &key, &cards, &split).is_ok());
        let mut bad = good.clone();
        bad.gpus.reverse();
        assert!(validate_cache(&bad, &key, &cards, &split).is_err());
        bad = good.clone();
        bad.gpus[1].reserved_bytes = 1;
        assert!(validate_cache(&bad, &key, &cards, &split).is_err());
        bad = good.clone();
        bad.host_observed_peak_bytes = Some(9000 * MIB);
        assert!(validate_cache(&bad, &key, &cards, &split).is_err());
        bad = good;
        bad.cache_key = Some("0".repeat(64));
        assert!(validate_cache(&bad, &key, &cards, &split).is_err());
    }

    #[test]
    fn persisted_calibration_retains_high_water_and_unknown_gpu_values() {
        let temp = tempfile::tempdir().unwrap();
        let state = temp.path().join("state");
        private_fs::ensure_directory(&state).unwrap();
        private_fs::ensure_directory(&state.join("resource-profiles")).unwrap();
        let paths: Paths = serde_json::from_value(
            json!({"state":state,"models":temp.path(),"environments":temp.path(),
            "logs":temp.path(),"runtime":temp.path(),"config":temp.path(),"socket":temp.path()}),
        )
        .unwrap();
        let cards = vec![(id(0), 15000 * MIB), (id(1), 15000 * MIB)];
        let mut placement = demand(&cards, 1000 * MIB);
        let key = cache_key(&spec(), &hardware(), &cards, &proportions(&cards)).unwrap();
        placement.cache_key = Some(key.clone());
        let file = state.join("resource-profiles").join(format!("{key}.json"));
        private_fs::write_json(&file, &placement).unwrap();
        let mut instance = local_model::new_instance(&spec(), "fixture".into(), placement);
        instance.runner_pid = Some(99);
        let mut reading = Sample::default();
        reading.gpu_process_bytes.insert((id(1), 99), 4000 * MIB);
        observe(&paths, &mut instance, &reading).unwrap();
        let saved: Placement = private_fs::read_json(&file).unwrap().unwrap();
        assert_eq!(saved.gpus[0].observed_peak_bytes, None);
        assert_eq!(saved.gpus[1].observed_peak_bytes, Some(4000 * MIB));
        assert!(saved.gpus[1].reserved_bytes > 4000 * MIB);
        reading.gpu_process_bytes.insert((id(1), 99), 1000 * MIB);
        observe(&paths, &mut instance, &reading).unwrap();
        let second: Placement = private_fs::read_json(&file).unwrap().unwrap();
        assert_eq!(saved.gpus[1].reserved_bytes, second.gpus[1].reserved_bytes);
    }
}
