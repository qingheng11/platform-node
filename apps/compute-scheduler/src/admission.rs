use std::{
    fs::{self, DirBuilder, File, OpenOptions},
    io::{self, Read, Write},
    os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt},
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use uuid::Uuid;

pub struct Journal(PathBuf);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Admission {
    pub request_sha256: String,
    pub terminal: Option<Value>,
}

impl Journal {
    pub fn open(state: &Path) -> io::Result<Self> {
        let root = state.join("web-admissions");
        match DirBuilder::new().mode(0o700).create(&root) {
            Ok(()) => File::open(state)?.sync_all()?,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => (),
            Err(error) => return Err(error),
        }
        let meta = fs::symlink_metadata(&root)?;
        // SAFETY: geteuid has no arguments.
        if !meta.is_dir() || meta.uid() != unsafe { libc::geteuid() } || meta.mode() & 0o077 != 0 {
            return Err(io::Error::other("unsafe web admission directory"));
        }
        Ok(Self(root))
    }

    fn path(&self, id: Uuid) -> PathBuf {
        self.0.join(format!("{id}.json"))
    }

    pub fn read(&self, id: Uuid) -> io::Result<Option<Admission>> {
        let file = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(self.path(id))
        {
            Ok(file) => file,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error),
        };
        let meta = file.metadata()?;
        // SAFETY: geteuid has no arguments.
        if !meta.is_file()
            || meta.uid() != unsafe { libc::geteuid() }
            || meta.mode() & 0o077 != 0
            || meta.nlink() != 1
        {
            return Err(io::Error::other("unsafe web admission record"));
        }
        let mut data = Vec::new();
        file.take(16_385).read_to_end(&mut data)?;
        if data.len() > 16_384 {
            return Err(io::Error::other("web admission record too large"));
        }
        serde_json::from_slice(&data)
            .map(Some)
            .map_err(io::Error::other)
    }

    pub fn write(&self, id: Uuid, entry: &Admission) -> io::Result<()> {
        let bytes = serde_json::to_vec(entry)?;
        if bytes.len() > 16_384 {
            return Err(io::Error::other("web admission record too large"));
        }
        let pending = self.0.join(format!(".{}.tmp", Uuid::new_v4()));
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&pending)?;
        let result = (|| {
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&pending, self.path(id))?;
            File::open(&self.0)?.sync_all()
        })();
        if result.is_err() {
            let _ = fs::remove_file(&pending);
        }
        result
    }
}

pub fn request_digest(parts: &[&str]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update((part.len() as u64).to_be_bytes());
        digest.update(part.as_bytes());
    }
    format!("{:x}", digest.finalize())
}
