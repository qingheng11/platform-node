#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

pub mod local_node;
pub mod tls;
pub mod web;

use std::{io, time::Duration};

use local_node::{Failure, Inventory, Output, Run};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use uuid::Uuid;

pub const VERSION: u32 = 1;
pub const MAX_IN_FLIGHT: usize = 4;
pub const FRAME_LIMIT: usize = 1024 * 1024;
pub const IO_TIMEOUT: Duration = Duration::from_secs(8);
pub const HEARTBEAT: Duration = Duration::from_secs(2);
pub const TASK_DEADLINE: Duration = Duration::from_secs(930);

#[derive(Serialize, Deserialize)]
pub struct Heartbeat {
    pub version: u32,
    pub inventory: Option<Inventory>,
    pub available_slots: usize,
    pub result: Option<Report>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub endpoint: Option<web::Registration>,
}

#[derive(Serialize, Deserialize)]
pub struct Assignment {
    pub id: Uuid,
    pub attempt: Uuid,
    pub run: Run,
    pub plan: String,
    pub prompt: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub web: Option<web::Grant>,
}

#[derive(Serialize, Deserialize)]
pub struct Report {
    pub attempt: Uuid,
    pub result: Result<Output, Failure>,
}

#[derive(Serialize, Deserialize)]
pub struct Reply {
    pub version: u32,
    pub node_id: String,
    pub assignments: Vec<Assignment>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cancel_attempts: Vec<Uuid>,
}

pub async fn receive<T: DeserializeOwned>(
    stream: &mut (impl AsyncRead + Unpin),
    max: usize,
) -> io::Result<T> {
    let size = stream.read_u32().await? as usize;
    if size == 0 || size > max {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "控制消息大小无效",
        ));
    }
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes).await?;
    serde_json::from_slice(&bytes).map_err(Into::into)
}

pub async fn send<T: Serialize>(
    stream: &mut (impl AsyncWrite + Unpin),
    value: &T,
    max: usize,
) -> io::Result<()> {
    let bytes = serde_json::to_vec(value)?;
    if bytes.len() > max {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "控制消息超过限制",
        ));
    }
    stream.write_u32(bytes.len() as u32).await?;
    stream.write_all(&bytes).await?;
    stream.flush().await
}
