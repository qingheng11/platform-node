use std::process::ExitCode;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod admission;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod gateway;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod local;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod node;
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod queue;

const HELP: &str = "全局调度服务 · 联调版

  compute-scheduler serve <配置文件>
  compute-scheduler gateway <独立 HTTPS 网关配置文件>
  compute-scheduler status <状态目录>
  compute-scheduler submit <状态目录> <计划名> [任务UUID] < 提示文件
  compute-scheduler submit-web <状态目录> <计划名> <浏览器Endpoint ID> [任务UUID] < 提示文件
  compute-scheduler task <状态目录> <任务UUID>
  compute-scheduler --version";

fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [] => println!("{HELP}"),
        [flag] if flag == "--help" || flag == "-h" => println!("{HELP}"),
        [flag] if flag == "--version" => {
            println!("compute-scheduler {}", env!("CARGO_PKG_VERSION"))
        }
        _ => {
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            let result = local::run(&args);
            #[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
            let result: Result<(), String> = Err("调度服务当前仅支持 Linux x86_64。".into());
            if let Err(error) = result {
                eprintln!("{error}");
                return ExitCode::FAILURE;
            }
        }
    }
    ExitCode::SUCCESS
}
