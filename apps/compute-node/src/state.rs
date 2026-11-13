use std::{
    path::{Component, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::private_fs;
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Paths {
    pub config: PathBuf,
    pub state: PathBuf,
    pub models: PathBuf,
    pub environments: PathBuf,
    pub logs: PathBuf,
    pub runtime: PathBuf,
    pub socket: PathBuf,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    schema_version: u32,
    data_dir: PathBuf,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Installation {
    pub schema_version: u32,
    pub role: String,
    pub installation_id: String,
    pub created_at_unix_ms: u64,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeRecord {
    pub schema_version: u32,
    pub installation_id: String,
    pub generation: u64,
    pub run_id: String,
    pub pid: u32,
    pub state: String,
    pub started_at_unix_ms: u64,
    pub stopped_at_unix_ms: Option<u64>,
    pub previous_run_unclean: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationCounter {
    schema_version: u32,
    installation_id: String,
    pub last_generation: u64,
}

fn absolute(path: PathBuf) -> Result<PathBuf, String> {
    if !path.is_absolute()
        || path
            .to_str()
            .is_none_or(|s| s.chars().any(char::is_control))
    {
        return Err("节点目录必须是没有控制字符的UTF-8绝对路径。".into());
    }
    if path
        .components()
        .any(|part| matches!(part, Component::ParentDir | Component::CurDir))
    {
        return Err("节点目录不能包含相对路径分量。".into());
    }
    Ok(path.components().collect())
}

fn config_directory() -> Result<PathBuf, String> {
    absolute(
        dirs::config_dir()
            .ok_or("无法确定节点配置目录。")?
            .join("compute-node"),
    )
}

pub fn resolve(requested: Option<PathBuf>) -> Result<Paths, String> {
    let config_dir = config_directory()?;
    let data_home = absolute(dirs::data_local_dir().ok_or("无法确定节点数据目录。")?)?;
    let config = config_dir.join("config.json");
    let saved: Option<Config> = if config_dir.exists() {
        private_fs::check_directory(&config_dir)?;
        private_fs::read_json(&config)?
    } else {
        None
    };
    if saved
        .as_ref()
        .is_some_and(|value| value.schema_version != 1)
    {
        return Err("节点配置版本不受支持。".into());
    }
    let requested =
        requested.or_else(|| std::env::var_os("COMPUTE_NODE_DATA_DIR").map(PathBuf::from));
    let state = absolute(requested.unwrap_or_else(|| {
        saved.as_ref().map_or_else(
            || data_home.join("compute-node"),
            |value| value.data_dir.clone(),
        )
    }))?;
    if saved.as_ref().is_some_and(|value| value.data_dir != state) {
        return Err("该节点已有数据目录；目录迁移尚未实现，未切换目录或重建身份。".into());
    }
    if state.starts_with(&config_dir) || config_dir.starts_with(&state) {
        return Err("节点数据目录不能与节点配置目录重叠。".into());
    }
    let paths = Paths {
        config,
        models: state.join("Content/Models"),
        environments: state.join("Content/Environments"),
        logs: state.join("logs"),
        runtime: state.join("run"),
        socket: state.join("run/control.sock"),
        state,
    };
    use std::os::unix::ffi::OsStrExt;
    if paths.socket.as_os_str().as_bytes().len() >= 108 {
        return Err("节点数据目录过长，无法建立本机控制连接。".into());
    }
    Ok(paths)
}

pub fn now_ms() -> Result<u64, String> {
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| "系统时间无效。")?;
    u64::try_from(time.as_millis()).map_err(|_| "系统时间超出支持范围。".into())
}

pub fn installation(paths: &Paths) -> Result<Option<Installation>, String> {
    if !paths.state.exists() {
        if paths.config.exists() {
            return Err("已配置节点的数据目录丢失，未自动重建身份。".into());
        }
        return Ok(None);
    }
    private_fs::check_directory(&paths.state)?;
    let installation: Option<Installation> =
        private_fs::read_json(&paths.state.join("installation.json"))?;
    if installation.is_none() && paths.config.exists() {
        return Err("已初始化节点的安装身份丢失，未自动重建身份。".into());
    }
    if let Some(value) = &installation {
        if value.schema_version != 1
            || value.role != "platform_node"
            || uuid::Uuid::parse_str(&value.installation_id).is_err()
        {
            return Err("平台安装身份无效；未自动重建身份。".into());
        }
    }
    Ok(installation)
}

pub fn initialize(requested: Option<PathBuf>) -> Result<(Paths, Installation), String> {
    let paths = resolve(requested.clone())?;
    let config_dir = paths.config.parent().ok_or("配置目录无效。")?;
    private_fs::ensure_directory(config_dir)?;
    let _guard = private_fs::lock(&config_dir.join("config.lock"), true)?;
    // Re-read configuration after obtaining the cross-process initializer lock.
    let paths = resolve(requested)?;
    for path in [
        &paths.state,
        &paths.models,
        &paths.environments,
        &paths.logs,
        &paths.runtime,
    ] {
        private_fs::ensure_directory(path)?;
    }
    let identity = match installation(&paths)? {
        Some(value) => value,
        None => {
            let value = Installation {
                schema_version: 1,
                role: "platform_node".into(),
                installation_id: uuid::Uuid::new_v4().to_string(),
                created_at_unix_ms: now_ms()?,
            };
            private_fs::write_json(&paths.state.join("installation.json"), &value)?;
            value
        }
    };
    let counter_path = paths.state.join("generation.json");
    if private_fs::read_json::<GenerationCounter>(&counter_path)?.is_none() {
        if paths.config.exists()
            || paths.state.join("runtime.json").exists()
            || paths.logs.join("node.log").exists()
        {
            return Err("已使用节点的启动代次记录丢失，未重置计数。".into());
        }
        private_fs::write_json(
            &counter_path,
            &GenerationCounter {
                schema_version: 1,
                installation_id: identity.installation_id.clone(),
                last_generation: 0,
            },
        )?;
    }
    generation_counter(&paths, &identity)?;
    private_fs::write_json(
        &paths.config,
        &Config {
            schema_version: 1,
            data_dir: paths.state.clone(),
        },
    )?;
    Ok((paths, identity))
}

pub fn generation_counter(
    paths: &Paths,
    identity: &Installation,
) -> Result<GenerationCounter, String> {
    let counter: GenerationCounter = private_fs::read_json(&paths.state.join("generation.json"))?
        .ok_or("节点启动代次记录缺失，未重置计数。")?;
    if counter.schema_version != 1
        || counter.installation_id != identity.installation_id
        || counter.last_generation > 9_007_199_254_740_991
    {
        return Err("节点启动代次记录无效，未自动覆盖。".into());
    }
    Ok(counter)
}

pub fn reserve_generation(
    paths: &Paths,
    identity: &Installation,
    previous: Option<&RuntimeRecord>,
) -> Result<(u64, bool), String> {
    let mut counter = generation_counter(paths, identity)?;
    if previous.is_some_and(|value| value.generation > counter.last_generation) {
        return Err("节点启动代次与运行记录不一致，未自动覆盖。".into());
    }
    let unclean = previous.is_some_and(|value| value.state == "running")
        || counter.last_generation > previous.map_or(0, |value| value.generation);
    counter.last_generation = counter
        .last_generation
        .checked_add(1)
        .filter(|value| *value <= 9_007_199_254_740_991)
        .ok_or("节点启动代次耗尽。")?;
    private_fs::write_json(&paths.state.join("generation.json"), &counter)?;
    Ok((counter.last_generation, unclean))
}

pub fn runtime_record(
    paths: &Paths,
    identity: &Installation,
) -> Result<Option<RuntimeRecord>, String> {
    let value: Option<RuntimeRecord> = private_fs::read_json(&paths.state.join("runtime.json"))?;
    if let Some(value) = &value {
        if value.schema_version != 1
            || value.installation_id != identity.installation_id
            || value.generation == 0
            || uuid::Uuid::parse_str(&value.run_id).is_err()
            || !matches!(value.state.as_str(), "running" | "stopped")
        {
            return Err("节点运行记录无效，未自动覆盖。".into());
        }
    }
    Ok(value)
}

pub fn save_runtime(paths: &Paths, value: &RuntimeRecord) -> Result<(), String> {
    private_fs::write_json(&paths.state.join("runtime.json"), value)
}

pub fn validate_existing_paths(paths: &Paths) -> Result<(), String> {
    for path in [
        &paths.state,
        &paths.models,
        &paths.environments,
        &paths.logs,
        &paths.runtime,
    ] {
        private_fs::check_directory(path)?;
    }
    Ok(())
}

pub fn require_installation(paths: &Paths) -> Result<Installation, String> {
    installation(paths)?.ok_or_else(|| "节点尚未初始化，请先执行 compute-node init。".into())
}

pub fn socket_metadata(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::{FileTypeExt, MetadataExt};
    let metadata = std::fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.file_type().is_socket()
        || metadata.uid() != private_fs::uid()
        || metadata.mode() & 0o077 != 0
    {
        return Err("本机控制路径不是当前用户的私有socket。".into());
    }
    Ok(())
}
