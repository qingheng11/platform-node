use hardware_probe::{probe_hardware, HardwareSnapshotV1};
use serde::Serialize;

#[derive(Serialize)]
struct Capabilities {
    hardware_diagnostics: bool,
    local_initialization: bool,
    local_lifecycle: bool,
    local_model_check: bool,
    local_model_pool: bool,
    local_resource_estimation: bool,
    local_gpu_group_placement: bool,
    node_registration: bool,
    platform_tasks: bool,
    model_management: bool,
    multi_model_scheduling: bool,
    idle_unload: bool,
}

#[derive(Serialize)]
struct Report {
    schema_version: u32,
    edition: &'static str,
    version: &'static str,
    stage: &'static str,
    capabilities: Capabilities,
    planned_directories: crate::state::Paths,
    hardware: HardwareSnapshotV1,
}

pub(super) fn run(data_dir: Option<std::path::PathBuf>, json: bool) -> Result<(), String> {
    let paths = crate::state::resolve(data_dir)?;
    let hardware = probe_hardware().map_err(|error| format!("硬件诊断失败：{error}"))?;
    let report = Report {
        schema_version: 1,
        edition: "platform_node",
        version: env!("CARGO_PKG_VERSION"),
        stage: "local_resource_planning",
        capabilities: Capabilities {
            hardware_diagnostics: true,
            local_initialization: true,
            local_lifecycle: true,
            local_model_check: true,
            local_model_pool: true,
            local_resource_estimation: true,
            local_gpu_group_placement: true,
            node_registration: false,
            platform_tasks: false,
            model_management: false,
            multi_model_scheduling: false,
            idle_unload: false,
        },
        planned_directories: paths,
        hardware,
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&report)
                .map_err(|error| format!("诊断结果序列化失败：{error}"))?
        );
    } else {
        print_report(&report);
    }
    Ok(())
}

fn print_report(report: &Report) {
    let hardware = &report.hardware;
    println!("平台节点 {} · 本机诊断", report.version);
    println!(
        "系统：{} {} · {}",
        hardware.os.family, hardware.os.version, hardware.os.arch
    );
    println!("CPU：{}", hardware.cpu.model);
    println!(
        "内存：可用 {} / 总计 {} MiB",
        hardware.memory.available_bytes / 1_048_576,
        hardware.memory.total_bytes / 1_048_576
    );
    for gpu in &hardware.gpus {
        println!(
            "GPU {}：{} · 可用显存 {} / 总计 {} MiB",
            gpu.index,
            gpu.name,
            gpu.vram_free_bytes / 1_048_576,
            gpu.vram_total_bytes / 1_048_576
        );
    }
    if hardware.gpus.is_empty() {
        println!("GPU：当前未取得 NVIDIA GPU 信息。");
    }
    for warning in &hardware.warnings {
        println!("探测提示：{:?}", warning.code);
    }
    println!(
        "数据目录（本次诊断不创建）：{}",
        report.planned_directories.state.display()
    );
    println!("当前阶段：本机多模型、资源估算及 GPU 组加载；平台注册和远程任务尚未接入。");
}
