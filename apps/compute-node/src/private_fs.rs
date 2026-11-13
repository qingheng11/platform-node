//! Current-user filesystem boundary, based on daemon-ipc's Linux primitives.
//! No personal-client paths, IPC credentials or platform credentials are used.
use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Component, Path},
    thread,
    time::{Duration, Instant},
};

use serde::{de::DeserializeOwned, Serialize};

pub fn uid() -> u32 {
    // SAFETY: geteuid has no pointers or preconditions.
    unsafe { libc::geteuid() }
}

pub fn ensure_directory(path: &Path) -> Result<(), String> {
    if !path.is_absolute() {
        return Err("节点目录必须是绝对路径。".into());
    }
    let mut current = std::path::PathBuf::new();
    for part in path.components() {
        match part {
            Component::RootDir => current.push("/"),
            Component::Normal(name) => current.push(name),
            _ => return Err("节点目录不能包含相对路径分量。".into()),
        }
        let metadata = match fs::symlink_metadata(&current) {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match DirBuilder::new().mode(0o700).create(&current) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                    Err(error) => return Err(format!("创建节点目录失败：{error}")),
                }
                fs::symlink_metadata(&current).map_err(|error| error.to_string())?
            }
            Err(error) => return Err(format!("检查节点目录失败：{error}")),
        };
        let sticky_root = metadata.uid() == 0 && metadata.mode() & 0o1000 != 0;
        if !metadata.is_dir()
            || ![0, uid()].contains(&metadata.uid())
            || (metadata.mode() & 0o022 != 0 && !sticky_root)
        {
            return Err(format!("节点路径不安全或含符号链接：{}", current.display()));
        }
    }
    check_directory(path)
}

pub fn check_directory(path: &Path) -> Result<(), String> {
    let mut current = std::path::PathBuf::new();
    for part in path.components() {
        match part {
            Component::RootDir => current.push("/"),
            Component::Normal(name) => current.push(name),
            _ => return Err("节点目录必须为绝对路径且没有相对分量。".into()),
        }
        let metadata = fs::symlink_metadata(&current).map_err(|error| error.to_string())?;
        let sticky_root = metadata.uid() == 0 && metadata.mode() & 0o1000 != 0;
        if !metadata.is_dir()
            || ![0, uid()].contains(&metadata.uid())
            || (metadata.mode() & 0o022 != 0 && !sticky_root)
        {
            return Err("节点目录的父路径不安全或含符号链接。".into());
        }
    }
    let metadata = fs::symlink_metadata(path).map_err(|error| error.to_string())?;
    if !metadata.is_dir() || metadata.uid() != uid() || metadata.mode() & 0o077 != 0 {
        return Err(format!(
            "节点目录必须归当前用户所有且权限为0700：{}",
            path.display()
        ));
    }
    Ok(())
}

fn open_file(path: &Path, create: bool) -> Result<File, String> {
    let file = OpenOptions::new()
        .read(true)
        .write(create)
        .create(create)
        .truncate(false)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC | libc::O_NONBLOCK)
        .open(path)
        .map_err(|error| format!("打开节点状态文件失败：{error}"))?;
    let metadata = file.metadata().map_err(|error| error.to_string())?;
    if !metadata.is_file()
        || metadata.uid() != uid()
        || metadata.mode() & 0o077 != 0
        || metadata.nlink() != 1
    {
        return Err("节点状态文件必须是当前用户私有的普通文件，且不能有硬链接。".into());
    }
    Ok(file)
}

pub fn lock(path: &Path, wait: bool) -> Result<File, String> {
    check_directory(path.parent().ok_or("锁文件路径无效。")?)?;
    let file = open_file(path, true)?;
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        // SAFETY: this is a valid owned file descriptor; no pointers are passed.
        let result = unsafe {
            use std::os::fd::AsRawFd;
            libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB)
        };
        if result == 0 {
            return Ok(file);
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() != Some(libc::EWOULDBLOCK) {
            return Err(format!("锁定节点状态失败：{error}"));
        }
        if !wait || Instant::now() >= deadline {
            return Err("节点已在运行，或节点配置正被其他进程使用。".into());
        }
        thread::sleep(Duration::from_millis(20));
    }
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<Option<T>, String> {
    read_json_bounded(path, 65_536)
}

pub fn read_json_bounded<T: DeserializeOwned>(
    path: &Path,
    limit: u64,
) -> Result<Option<T>, String> {
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(format!("读取节点状态失败：{error}")),
        Ok(_) => {}
    }
    check_directory(path.parent().ok_or("状态路径无效。")?)?;
    let mut file = open_file(path, false)?;
    if file.metadata().map_err(|error| error.to_string())?.len() > limit {
        return Err("节点状态文件超过本机大小限制。".into());
    }
    let mut bytes = Vec::new();
    (&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > limit {
        return Err("节点状态文件超过本机大小限制。".into());
    }
    serde_json::from_slice(&bytes)
        .map(Some)
        .map_err(|_| "节点状态JSON损坏或版本不受支持；未重建身份。".into())
}

pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let parent = path.parent().ok_or("节点状态路径无效。")?;
    check_directory(parent)?;
    if fs::symlink_metadata(path).is_ok() {
        open_file(path, false)?;
    }
    let temporary = parent.join(format!(".state-{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&temporary)
            .map_err(|error| error.to_string())?;
        serde_json::to_writer_pretty(&mut file, value)
            .map_err(|_| "保存节点状态失败。".to_string())?;
        file.write_all(b"\n").map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        fs::rename(&temporary, path).map_err(|error| error.to_string())?;
        File::open(parent)
            .and_then(|file| file.sync_all())
            .map_err(|error| error.to_string())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn append_event(path: &Path, event: &impl Serialize) -> Result<(), String> {
    check_directory(path.parent().ok_or("日志路径无效。")?)?;
    let mut file = open_file(path, true)?;
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::End(0))
        .map_err(|error| error.to_string())?;
    serde_json::to_writer(&mut file, event).map_err(|_| "写入节点事件失败。".to_string())?;
    file.write_all(b"\n")
        .and_then(|_| file.sync_data())
        .map_err(|error| error.to_string())
}
