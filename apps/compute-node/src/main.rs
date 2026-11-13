#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod doctor;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod lifecycle;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod local_model;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod model_pool;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod private_fs;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod remote;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod resource_plan;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod state;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod web_result;

use std::{ffi::OsString, path::PathBuf, process::ExitCode};

const HELP: &str = "平台节点 · 开发基础版

用法：compute-node <命令>

  init [--data-dir 绝对路径] [--json]    初始化本机节点，保存独立安装身份
  serve [--data-dir 绝对路径]           前台运行本机节点控制服务，Ctrl+C停止
  connect 联调配置 [--data-dir 绝对路径] 向远程调度器登记和领取任务
  status [--data-dir 绝对路径] [--json] 查询本机节点真实响应和启动代次
  stop [--data-dir 绝对路径] [--json]   停止已确认的当前本机节点进程
  doctor [--data-dir 绝对路径] [--json] 读取硬件和节点目录，不创建状态
  model-check 计划绝对路径 [--data-dir 绝对路径] [--json]
                    校验并加载一个本机模型，执行一轮固定问答后停止
  model-estimate 计划名 [--json]           自动评估逐卡资源，不加载模型
  model-run 计划名 任务UUID 提示 [--json]  在常驻节点上加载或复用实例
  model-run-stdin 计划名 任务UUID [--json] 从标准输入读取提示，用于本机调度
  model-list [--json]                    查看本机实例、GPU 和活动任务
  gpu list [--json]                      查看逐卡显存、模型实例及并发槽位
  gpu inventory                          读取调度使用的逐卡显存 JSON
  model-cancel 任务UUID [--json]           请求停止指定任务
  model-unload 计划名 [--json]             卸载指定的空闲实例
  --version         显示平台程序版本
  --help            显示帮助

当前支持 Linux x86_64 的本机多模型、自动资源评估及 GPU 组加载。
默认使用独立的节点配置和数据目录；COMPUTE_NODE_DATA_DIR可指定初始化数据根。
本机模型管理使用私有 Unix Socket。用 connect 连接远程调度器。";

enum Command {
    Help,
    Version,
    Local {
        action: String,
        data_dir: Option<PathBuf>,
        model_spec: Option<PathBuf>,
        model_args: Vec<String>,
        expected_run: Option<String>,
        json: bool,
    },
}

fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Command, String> {
    let arguments: Vec<_> = arguments.into_iter().collect();
    let arguments: Vec<_> = arguments.iter().map(|value| value.to_str()).collect();
    match arguments.as_slice() {
        [] | [Some("--help" | "-h")] => Ok(Command::Help),
        [Some("--version" | "-V")] => Ok(Command::Version),
        [Some(
            action @ ("init" | "serve" | "connect" | "status" | "stop" | "doctor" | "model-check"
            | "model-run" | "model-run-stdin" | "model-estimate" | "model-list"
            | "model-cancel" | "model-unload" | "gpu"),
        ), options @ ..] => {
            let mut data_dir = None;
            let mut json = false;
            let mut expected_run = None;
            let mut index = 0;
            let model_spec = if *action == "model-check" {
                let path = options
                    .first()
                    .and_then(|value| *value)
                    .map(PathBuf::from)
                    .filter(|path| path.is_absolute())
                    .ok_or("model-check 需要本机模型计划绝对路径。")?;
                index += 1;
                Some(path)
            } else {
                None
            };
            let model_count = match *action {
                "model-run" => 3,
                "model-run-stdin" => 2,
                "model-estimate" | "model-cancel" | "model-unload" | "gpu" | "connect" => 1,
                _ => 0,
            };
            let model_args = options
                .get(index..index + model_count)
                .and_then(|items| {
                    items
                        .iter()
                        .map(|item| item.map(str::to_owned))
                        .collect::<Option<Vec<_>>>()
                })
                .ok_or("模型命令参数不足或不是 UTF-8。")?;
            index += model_count;
            if *action == "gpu" && !matches!(model_args[0].as_str(), "list" | "inventory") {
                return Err("请使用 compute-node gpu list 或 gpu inventory。".into());
            }
            while index < options.len() {
                match options[index] {
                    Some("--json") if !json && !matches!(*action, "serve" | "connect") => {
                        json = true
                    }
                    Some("--expected-run")
                        if expected_run.is_none()
                            && matches!(*action, "model-run-stdin" | "model-cancel") =>
                    {
                        index += 1;
                        expected_run = Some(
                            options
                                .get(index)
                                .and_then(|value| *value)
                                .ok_or("--expected-run 需要节点运行 UUID。")?
                                .to_owned(),
                        );
                    }
                    Some("--data-dir") if data_dir.is_none() => {
                        index += 1;
                        let Some(Some(value)) = options.get(index) else {
                            return Err("--data-dir 缺少绝对路径。".into());
                        };
                        data_dir = Some(PathBuf::from(value));
                    }
                    _ => return Err("节点参数无效或重复，请执行 compute-node --help。".into()),
                }
                index += 1;
            }
            Ok(Command::Local {
                action: (*action).into(),
                data_dir,
                model_spec,
                model_args,
                expected_run,
                json,
            })
        }
        _ => Err("命令或参数无效，请执行 compute-node --help。".into()),
    }
}

fn execute() -> Result<(), String> {
    match parse(std::env::args_os().skip(1))? {
        Command::Help => println!("{HELP}"),
        Command::Version => println!("compute-node {}", env!("CARGO_PKG_VERSION")),
        Command::Local {
            action,
            data_dir,
            model_spec,
            model_args,
            expected_run,
            json,
        } => {
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            run_local(
                &action,
                data_dir,
                model_spec,
                model_args,
                expected_run,
                json,
            )?;
            #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
            {
                let _ = (action, data_dir, model_spec, model_args, expected_run, json);
                return Err("平台节点当前仅支持 Linux x86_64。".into());
            }
        }
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn run_local(
    action: &str,
    data_dir: Option<PathBuf>,
    model_spec: Option<PathBuf>,
    model_args: Vec<String>,
    expected_run: Option<String>,
    json: bool,
) -> Result<(), String> {
    if action == "doctor" {
        return doctor::run(data_dir, json);
    }
    if action == "init" {
        let (paths, installation) = state::initialize(data_dir)?;
        if json {
            println!(
                "{}",
                serde_json::json!({"schema_version":1,"installation":installation,"paths":paths,
                "platform_registration":"not_implemented","platform_tasks_ready":false})
            );
        } else {
            println!("本机节点初始化完成。");
            println!("安装身份：{}", installation.installation_id);
            println!("数据目录：{}", paths.state.display());
        }
        return Ok(());
    }
    let paths = state::resolve(data_dir)?;
    if action == "gpu" && model_args[0] == "inventory" {
        let gpus = remote::gpu_inventory()?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({ "gpus": gpus }))
                .map_err(|error| error.to_string())?
        );
        return Ok(());
    }
    if action == "connect" {
        state::require_installation(&paths)?;
        return remote::run(PathBuf::from(&model_args[0]), paths);
    }
    if let Some(spec) = model_spec {
        return local_model::check(paths, &spec, json);
    }
    use model_pool::ModelRequest;
    use std::io::Read;
    let model = match (action, model_args.as_slice()) {
        ("model-run", [plan, task_id, prompt]) => Some(ModelRequest::Infer {
            plan: plan.clone(),
            task_id: task_id.clone(),
            prompt: prompt.clone(),
        }),
        ("model-run-stdin", [plan, task_id]) => {
            let mut prompt = String::new();
            std::io::stdin()
                .take(2049)
                .read_to_string(&mut prompt)
                .map_err(|_| "提示输入无效，请提供 UTF-8 文本。")?;
            if prompt.trim().is_empty() || prompt.len() > 2048 {
                return Err("提示必须非空且不超过 2048 字节。".into());
            }
            Some(ModelRequest::Infer {
                plan: plan.clone(),
                task_id: task_id.clone(),
                prompt,
            })
        }
        ("model-list", []) => Some(ModelRequest::List),
        ("model-estimate", [plan]) => Some(ModelRequest::Estimate { plan: plan.clone() }),
        ("gpu", [_]) => Some(ModelRequest::GpuList),
        ("model-cancel", [task_id]) => Some(ModelRequest::Cancel {
            task_id: task_id.clone(),
        }),
        ("model-unload", [plan]) => Some(ModelRequest::Unload { plan: plan.clone() }),
        _ => None,
    };
    if let Some(request) = model {
        let expected_run = expected_run
            .as_deref()
            .map(uuid::Uuid::parse_str)
            .transpose()
            .map_err(|_| "--expected-run 需要节点运行 UUID。")?;
        let result = lifecycle::model_request(paths, request, expected_run)?;
        if !json {
            if action == "gpu" {
                print_gpus(&result);
                return Ok(());
            }
            if let Some(text) = result["answer"]["text"].as_str() {
                println!("{text}");
                return Ok(());
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    if action == "serve" {
        return lifecycle::serve(paths);
    }
    let view = if action == "stop" {
        lifecycle::stop(paths)?
    } else {
        lifecycle::status(paths)?
    };
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&view).map_err(|_| "状态编码失败。")?
        );
    } else {
        let status = match view.local_status.as_str() {
            "uninitialized" => "尚未初始化",
            "running" => "正在响应（本机）",
            "stopped" => "已停止",
            _ => "无法确认本机进程响应",
        };
        println!("本机节点状态：{status}");
        if let Some(identity) = view.installation {
            println!("安装身份：{}", identity.installation_id);
        }
        if let Some(record) = view.runtime {
            println!("启动代次：{} · PID {}", record.generation, record.pid);
        }
        println!("数据目录：{}", view.paths.state.display());
        println!("用 compute-node connect 连接调度器。");
    }
    Ok(())
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn print_gpus(view: &serde_json::Value) {
    fn mib(bytes: &serde_json::Value) -> String {
        bytes
            .as_u64()
            .map(|bytes| (bytes / local_model::MIB).to_string())
            .unwrap_or_else(|| "未知".into())
    }
    fn print_instances(instances: &serde_json::Value) {
        for instance in instances.as_array().into_iter().flatten() {
            let state = match instance["state"].as_str() {
                Some("starting") => "加载中",
                Some("idle") => "空闲",
                Some("in_use") => "使用中",
                Some("unloading") => "卸载中",
                Some("quarantined") => "隔离",
                _ => "已停止",
            };
            println!(
                "  {} / {} · {} · {} · 显存 {} MiB · 预留 {} MiB · 槽位 {}/{}",
                instance["model_id"].as_str().unwrap_or("?"),
                instance["variant_id"].as_str().unwrap_or("?"),
                instance["instance_id"].as_str().unwrap_or("?"),
                state,
                mib(&instance["vram_used_bytes"]),
                mib(&instance["vram_reserved_bytes"]),
                instance["slots_used"],
                instance["slots_total"],
            );
        }
    }
    if view["gpus"].as_array().is_some_and(Vec::is_empty) {
        println!("未取得本机 GPU 采样。");
    }
    for gpu in view["gpus"].as_array().into_iter().flatten() {
        println!(
            "GPU {} · {} · {} · 显存 {} / {} MiB",
            gpu["index"],
            gpu["name"].as_str().unwrap_or("?"),
            gpu["uuid"].as_str().unwrap_or("UUID 未知"),
            mib(&gpu["vram_used_bytes"]),
            mib(&gpu["vram_total_bytes"]),
        );
        print_instances(&gpu["instances"]);
    }
    if view["instances_without_gpu_sample"]
        .as_array()
        .is_some_and(|items| !items.is_empty())
    {
        println!("GPU 采样缺失的实例");
        print_instances(&view["instances_without_gpu_sample"]);
    }
}

fn main() -> ExitCode {
    match execute() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}
