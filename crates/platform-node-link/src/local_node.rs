use std::{path::PathBuf, process::Stdio, time::Duration};

use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::{
    io::AsyncWriteExt,
    process::Command,
    sync::watch,
    time::{sleep, timeout},
};
use uuid::Uuid;

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalNode {
    pub id: String,
    pub binary: PathBuf,
    pub config_home: PathBuf,
    pub data_dir: PathBuf,
    pub plans: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Run {
    pub installation_id: Uuid,
    pub generation: u64,
    pub run_id: Uuid,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Instance {
    pub instance_id: Uuid,
    pub phase: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub plan: String,
    pub instance: Instance,
    pub active_task_ids: Vec<Uuid>,
    pub slots_used: usize,
    pub slots_total: usize,
    pub unloading: bool,
    pub calibration_error: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub uuid: Option<String>,
    #[serde(rename = "vram_free_bytes")]
    pub free_bytes: Option<u64>,
    #[serde(rename = "vram_total_bytes")]
    pub total_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Inventory {
    pub run: Run,
    pub models: Vec<Model>,
    pub gpus: Vec<Gpu>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Answer {
    pub text: String,
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Output {
    pub task_id: Uuid,
    pub instance_id: Uuid,
    pub answer: Answer,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", content = "error", rename_all = "snake_case")]
pub enum Failure {
    Retry(String),
    Rejected(String),
    Unknown(String),
    Cancelled { execution_started: bool },
}

impl LocalNode {
    pub async fn inventory(&self) -> Result<Inventory, String> {
        self.inventory_with_gpu_query("list", Duration::from_secs(5))
            .await
    }

    pub async fn scheduling_inventory(&self) -> Result<Inventory, String> {
        self.inventory_with_gpu_query("inventory", Duration::from_secs(12))
            .await
    }

    async fn inventory_with_gpu_query(
        &self,
        gpu_command: &str,
        gpu_timeout: Duration,
    ) -> Result<Inventory, String> {
        #[derive(Deserialize)]
        struct Status {
            local_status: String,
            runtime: Option<Run>,
        }
        #[derive(Deserialize)]
        struct Models {
            instances: Vec<Model>,
        }
        #[derive(Deserialize)]
        struct Gpus {
            gpus: Vec<Gpu>,
        }

        let read_run = async || {
            let status: Status = self.query(&["status"], Duration::from_secs(5)).await?;
            status
                .runtime
                .filter(|_| status.local_status == "running")
                .ok_or_else(|| "节点当前未运行。".to_owned())
        };
        let run = read_run().await?;
        let gpu_args = ["gpu", gpu_command];
        let (models, gpus): (Models, Gpus) = tokio::try_join!(
            self.query(&["model-list"], Duration::from_secs(5)),
            self.query(&gpu_args, gpu_timeout)
        )?;
        if read_run().await? != run {
            return Err("采集库存期间节点发生重启。".into());
        }
        Ok(Inventory {
            run,
            models: models.instances,
            gpus: gpus.gpus,
        })
    }

    async fn query<T: DeserializeOwned>(
        &self,
        args: &[&str],
        budget: Duration,
    ) -> Result<T, String> {
        let output = timeout(budget, self.command(args).output())
            .await
            .map_err(|_| format!("节点库存查询超时（{}）。", args.join(" ")))?
            .map_err(|error| format!("无法执行节点命令（{error}）"))?;
        if !output.status.success() {
            return Err(error_message(&output.stderr));
        }
        serde_json::from_slice(&output.stdout)
            .map_err(|_| format!("节点库存格式无效（{}）。", args.join(" ")))
    }

    pub async fn generate(
        &self,
        plan: &str,
        id: Uuid,
        prompt: &str,
        run: Option<Uuid>,
    ) -> Result<Output, Failure> {
        let output = timeout(Duration::from_secs(905), async {
            let mut command = self.command(&["model-run-stdin", plan, &id.to_string()]);
            if let Some(run) = run {
                command.arg("--expected-run").arg(run.to_string());
            }
            let mut child = command
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()?;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(prompt.as_bytes()).await?;
            }
            child.wait_with_output().await
        })
        .await
        .map_err(|_| Failure::Unknown("等待节点回执超时。".into()))?
        .map_err(|error| Failure::Unknown(format!("未取得节点回执（{error}）")))?;
        if !output.status.success() {
            return Err(classify_error(error_message(&output.stderr)));
        }
        let output: Output = serde_json::from_slice(&output.stdout)
            .map_err(|_| Failure::Unknown("节点结果格式无效。".into()))?;
        if output.task_id != id {
            return Err(Failure::Unknown("节点回执的任务编号不匹配。".into()));
        }
        Ok(output)
    }

    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(&self.binary);
        command
            .args(args)
            .arg("--data-dir")
            .arg(&self.data_dir)
            .arg("--json")
            .env("XDG_CONFIG_HOME", &self.config_home)
            .stdin(Stdio::null())
            .kill_on_drop(true);
        command
    }

    pub async fn generate_cancellable(
        &self,
        plan: &str,
        id: Uuid,
        prompt: &str,
        run: Uuid,
        mut cancel: watch::Receiver<bool>,
    ) -> Result<Output, Failure> {
        if *cancel.borrow() {
            return Err(Failure::Cancelled {
                execution_started: false,
            });
        }
        let result = self.generate(plan, id, prompt, Some(run));
        tokio::pin!(result);
        let cancelled = tokio::select! {
            answer = &mut result => return answer,
            requested = cancel.wait_for(|requested| *requested) => requested.is_ok(),
        };
        if !cancelled {
            return result.await;
        }
        // The task may still be in admission. Request cancellation until the
        // same runtime confirms receipt or the original job ends.
        let request_cancel = async {
            let id = id.to_string();
            let run = run.to_string();
            while self
                .query::<serde_json::Value>(
                    &["model-cancel", &id, "--expected-run", &run],
                    Duration::from_secs(3),
                )
                .await
                .is_err()
            {
                sleep(Duration::from_millis(250)).await;
            }
        };
        tokio::select! {
            answer = &mut result => answer,
            _ = request_cancel => result.await,
        }
    }
}

fn error_message(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let line = text.lines().find(|line| !line.trim().is_empty());
    line.unwrap_or("节点命令失败且未返回错误信息。")
        .chars()
        .take(240)
        .collect()
}

fn classify_error(message: String) -> Failure {
    match message.as_str() {
        "请求已取消，本次生成未执行。" => {
            return Failure::Cancelled {
                execution_started: false,
            }
        }
        "请求已取消，生成已停止。" => {
            return Failure::Cancelled {
                execution_started: true,
            }
        }
        _ => {}
    }
    if message.starts_with("resource_capacity_unavailable：")
        || message.starts_with("resource_capacity_changed：")
        || matches!(
            message.as_str(),
            "CPU 线程预算不足。"
                | "实例并发槽位已满，请稍后重新调度。"
                | "本机实例数达到上限。"
                | "请先启动本机节点 compute-node serve。"
                | "节点启动代次已变化，本次任务未执行。"
        )
    {
        Failure::Retry(message)
    } else if message == "节点未配置该本机模型计划。" {
        Failure::Rejected(message)
    } else {
        Failure::Unknown(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_only_for_known_pre_admission_errors() {
        assert!(matches!(
            classify_error("resource_capacity_changed：资源已变化".into()),
            Failure::Retry(_)
        ));
        assert!(matches!(
            classify_error("节点未配置该本机模型计划。".into()),
            Failure::Rejected(_)
        ));
        assert!(matches!(
            classify_error("模型 HTTP 请求失败：error sending request".into()),
            Failure::Unknown(_)
        ));
    }
}
