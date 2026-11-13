#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::{
    fs,
    io::{Read, Write},
    os::unix::{
        fs::{symlink, MetadataExt, PermissionsExt},
        net::UnixStream,
    },
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use serde_json::{json, Value};
use tempfile::TempDir;

struct Sandbox {
    root: TempDir,
}
struct Node(Child);

impl Drop for Node {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
}

impl Sandbox {
    fn new() -> Self {
        Self {
            root: TempDir::new().unwrap(),
        }
    }
    fn command(&self, args: &[&str]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_compute-node"));
        command
            .args(args)
            .env("HOME", self.root.path().join("home"))
            .env("XDG_DATA_HOME", self.root.path().join("data"))
            .env("XDG_CONFIG_HOME", self.root.path().join("config"))
            .env("XDG_RUNTIME_DIR", self.root.path().join("runtime"))
            .env_remove("COMPUTE_NODE_DATA_DIR");
        command
    }
    fn output(&self, args: &[&str]) -> Output {
        self.command(args).output().unwrap()
    }
    fn json(&self, args: &[&str]) -> Value {
        let output = self.output(args);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        serde_json::from_slice(&output.stdout).unwrap()
    }
    fn init(&self) -> Value {
        self.json(&["init", "--json"])
    }
    fn status(&self) -> Value {
        self.json(&["status", "--json"])
    }
    fn start(&self) -> Node {
        let mut node = Node(
            self.command(&["serve"])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        );
        let deadline = Instant::now() + Duration::from_secs(6);
        while Instant::now() < deadline {
            assert!(
                node.0.try_wait().unwrap().is_none(),
                "node exited during startup"
            );
            if self.status()["local_status"] == "running" {
                return node;
            }
            thread::sleep(Duration::from_millis(20));
        }
        panic!("node failed to become responsive")
    }
}

fn wait_success(node: &mut Node) {
    let deadline = Instant::now() + Duration::from_secs(6);
    while Instant::now() < deadline {
        if let Some(exit) = node.0.try_wait().unwrap() {
            assert!(exit.success());
            return;
        }
        thread::sleep(Duration::from_millis(20));
    }
    panic!("node failed to stop")
}

fn path(value: &Value, name: &str) -> PathBuf {
    PathBuf::from(value["paths"][name].as_str().unwrap())
}

fn control(socket: &Path, value: Value) -> Value {
    let mut stream = UnixStream::connect(socket).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream
        .set_write_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let bytes = serde_json::to_vec(&value).unwrap();
    stream
        .write_all(&(bytes.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(&bytes).unwrap();
    let mut header = [0; 4];
    stream.read_exact(&mut header).unwrap();
    let length = u32::from_be_bytes(header) as usize;
    assert!(length <= 16_384);
    let mut response = vec![0; length];
    stream.read_exact(&mut response).unwrap();
    serde_json::from_slice(&response).unwrap()
}

#[test]
fn initialization_is_persistent_private_and_supports_an_explicit_data_disk() {
    let sandbox = Sandbox::new();
    let custom = sandbox.root.path().join("volume/node");
    let first = sandbox.json(&["init", "--data-dir", custom.to_str().unwrap(), "--json"]);
    let second = sandbox.init();
    assert_eq!(first["installation"], second["installation"]);
    assert_eq!(path(&second, "state"), custom);
    assert_eq!(sandbox.status()["local_status"], "stopped");
    assert_eq!(sandbox.status()["platform_tasks_ready"], false);
    let doctor = sandbox.json(&["doctor", "--json"]);
    assert_eq!(
        doctor["planned_directories"]["state"],
        custom.to_string_lossy().as_ref()
    );
    for directory in ["state", "models", "environments", "logs", "runtime"] {
        assert_eq!(
            fs::metadata(path(&first, directory)).unwrap().mode() & 0o777,
            0o700
        );
    }
    for file in [
        path(&first, "state").join("installation.json"),
        path(&first, "config"),
    ] {
        assert_eq!(fs::metadata(file).unwrap().mode() & 0o777, 0o600);
    }
    let other = sandbox.root.path().join("other-node");
    assert!(!sandbox
        .output(&["init", "--data-dir", other.to_str().unwrap()])
        .status
        .success());
    assert!(!other.exists());
    assert_eq!(sandbox.init()["installation"], first["installation"]);
}

#[test]
fn concurrent_initializers_cannot_create_multiple_installation_identities() {
    let sandbox = Sandbox::new();
    let children: Vec<_> = (0..6)
        .map(|_| {
            sandbox
                .command(&["init", "--json"])
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .unwrap()
        })
        .collect();
    let identities: Vec<_> = children
        .into_iter()
        .map(|child| {
            let result = child.wait_with_output().unwrap();
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
            serde_json::from_slice::<Value>(&result.stdout).unwrap()["installation"]
                ["installation_id"]
                .clone()
        })
        .collect();
    assert!(identities.iter().all(|id| id == &identities[0]));
}

#[test]
fn single_instance_control_is_fenced_by_generation_and_run_identity() {
    let sandbox = Sandbox::new();
    let first = sandbox.init();
    let mut node = sandbox.start();
    let before = sandbox.status();
    assert_eq!(before["runtime"]["generation"], 1);
    let mut duplicate = Node(
        sandbox
            .command(&["serve"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if let Some(exit) = duplicate.0.try_wait().unwrap() {
            assert!(!exit.success());
            break;
        }
        assert!(
            Instant::now() < deadline,
            "second node did not reject its startup"
        );
        thread::sleep(Duration::from_millis(20));
    }
    assert_eq!(
        sandbox.status()["runtime"]["run_id"],
        before["runtime"]["run_id"]
    );
    assert_eq!(sandbox.json(&["stop", "--json"])["local_status"], "stopped");
    wait_success(&mut node);
    let mut next = sandbox.start();
    let after = sandbox.status();
    assert_eq!(after["installation"], first["installation"]);
    assert_eq!(after["runtime"]["generation"], 2);
    assert_ne!(after["runtime"]["run_id"], before["runtime"]["run_id"]);
    let rejected = control(
        &path(&first, "socket"),
        json!({"schema_version":1,"command":"stop",
        "request_id":uuid::Uuid::new_v4().to_string(), "installation_id":before["runtime"]["installation_id"],
        "generation":before["runtime"]["generation"], "run_id":before["runtime"]["run_id"]}),
    );
    assert_eq!(rejected["stop_accepted"], false);
    assert!(rejected["error"].is_string());
    assert_eq!(sandbox.status()["local_status"], "running");
    sandbox.json(&["stop", "--json"]);
    wait_success(&mut next);
    assert!(!path(&first, "socket").exists());
}

#[test]
fn crashes_are_unconfirmed_until_restart_and_signals_preserve_the_installation() {
    let sandbox = Sandbox::new();
    let identity = sandbox.init()["installation"].clone();
    let mut first = sandbox.start();
    first.0.kill().unwrap();
    first.0.wait().unwrap();
    let crashed = sandbox.status();
    assert_eq!(crashed["local_status"], "unreachable");
    assert_eq!(crashed["runtime"]["state"], "running");
    for (generation, signal) in [(2, libc::SIGTERM), (3, libc::SIGINT)] {
        let mut next = sandbox.start();
        let current = sandbox.status();
        assert_eq!(current["runtime"]["generation"], generation);
        assert_eq!(current["installation"], identity);
        assert_eq!(current["runtime"]["previous_run_unclean"], generation == 2);
        // SAFETY: this PID belongs to the live child owned by this test.
        assert_eq!(unsafe { libc::kill(next.0.id() as i32, signal) }, 0);
        wait_success(&mut next);
        assert_eq!(sandbox.status()["local_status"], "stopped");
    }
}

#[test]
fn config_paths_and_symbolic_links_cannot_be_used_as_node_storage() {
    let sandbox = Sandbox::new();
    let config_dir = sandbox.root.path().join("config/compute-node");
    fs::create_dir_all(&config_dir).unwrap();
    fs::set_permissions(&config_dir, fs::Permissions::from_mode(0o700)).unwrap();
    let sentinel = config_dir.join("sentinel");
    fs::write(&sentinel, b"synthetic-config-data").unwrap();
    assert!(!sandbox
        .output(&["init", "--data-dir", config_dir.to_str().unwrap()])
        .status
        .success());
    let external = sandbox.root.path().join("external");
    fs::create_dir(&external).unwrap();
    let alias = sandbox.root.path().join("alias");
    symlink(&external, &alias).unwrap();
    assert!(!sandbox
        .output(&["init", "--data-dir", alias.to_str().unwrap()])
        .status
        .success());
    assert_eq!(fs::read_dir(&external).unwrap().count(), 0);
    let initialized = sandbox.init();
    let identity = path(&initialized, "state").join("installation.json");
    fs::remove_file(&identity).unwrap();
    symlink(&sentinel, &identity).unwrap();
    assert!(!sandbox.output(&["init"]).status.success());
    assert_eq!(fs::read(&sentinel).unwrap(), b"synthetic-config-data");
}

#[test]
fn hardlinked_or_public_identity_files_are_rejected_without_repair() {
    let sandbox = Sandbox::new();
    let initialized = sandbox.init();
    let identity = path(&initialized, "state").join("installation.json");
    let original = fs::read(&identity).unwrap();
    let alias = sandbox.root.path().join("copied-identity");
    fs::hard_link(&identity, &alias).unwrap();
    assert!(!sandbox.output(&["init"]).status.success());
    assert_eq!(fs::read(&identity).unwrap(), original);
    fs::remove_file(&alias).unwrap();
    fs::set_permissions(&identity, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(!sandbox.output(&["init"]).status.success());
    assert!(!sandbox.output(&["status"]).status.success());
    assert_eq!(fs::metadata(&identity).unwrap().mode() & 0o777, 0o644);
}

#[test]
fn startup_does_not_delete_unrelated_files_at_the_control_path() {
    let sandbox = Sandbox::new();
    let initialized = sandbox.init();
    let socket = path(&initialized, "socket");
    fs::write(&socket, b"unrelated-file").unwrap();
    fs::set_permissions(&socket, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(!sandbox.output(&["serve"]).status.success());
    assert_eq!(fs::read(&socket).unwrap(), b"unrelated-file");
}

#[test]
fn malformed_and_wrong_identity_requests_do_not_stop_the_node() {
    let sandbox = Sandbox::new();
    let initialized = sandbox.init();
    let mut node = sandbox.start();
    let socket = path(&initialized, "socket");
    let reply = control(
        &socket,
        json!({"schema_version":1,"command":"stop", "request_id":uuid::Uuid::new_v4().to_string(),
        "installation_id":uuid::Uuid::new_v4().to_string(),"generation":1,"run_id":uuid::Uuid::new_v4().to_string()}),
    );
    assert_eq!(reply["stop_accepted"], false);
    let mut stream = UnixStream::connect(&socket).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    stream.write_all(&65_537u32.to_be_bytes()).unwrap();
    let mut data = [0; 1];
    assert!(!matches!(stream.read(&mut data), Ok(1)));
    assert_eq!(sandbox.status()["local_status"], "running");
    sandbox.json(&["stop", "--json"]);
    wait_success(&mut node);
}

#[test]
fn missing_or_corrupted_installation_data_cannot_silently_create_an_identity() {
    let sandbox = Sandbox::new();
    let initial = sandbox.init();
    let identity = path(&initial, "state").join("installation.json");
    let config = fs::read(path(&initial, "config")).unwrap();
    fs::remove_file(&identity).unwrap();
    assert!(!sandbox.output(&["init"]).status.success());
    assert!(!sandbox.output(&["status"]).status.success());
    assert!(!identity.exists());
    fs::write(&identity, b"incomplete-json").unwrap();
    fs::set_permissions(&identity, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(!sandbox.output(&["init"]).status.success());
    assert_eq!(fs::read(&identity).unwrap(), b"incomplete-json");
    assert_eq!(fs::read(path(&initial, "config")).unwrap(), config);
}

#[test]
fn losing_a_runtime_record_does_not_reset_generation_and_lost_counters_fail_closed() {
    let sandbox = Sandbox::new();
    let initial = sandbox.init();
    let mut first = sandbox.start();
    sandbox.json(&["stop", "--json"]);
    wait_success(&mut first);
    fs::remove_file(path(&initial, "state").join("runtime.json")).unwrap();
    assert_eq!(sandbox.status()["local_status"], "unreachable");
    let mut second = sandbox.start();
    assert_eq!(sandbox.status()["runtime"]["generation"], 2);
    assert_eq!(sandbox.status()["runtime"]["previous_run_unclean"], true);
    sandbox.json(&["stop", "--json"]);
    wait_success(&mut second);
    let counter = path(&initial, "state").join("generation.json");
    fs::remove_file(&counter).unwrap();
    assert!(!sandbox.output(&["init"]).status.success());
    assert!(!sandbox.output(&["serve"]).status.success());
    assert!(!counter.exists());
}

#[test]
fn losing_a_stop_acknowledgement_still_stops_only_the_requested_run() {
    let sandbox = Sandbox::new();
    let initial = sandbox.init();
    let mut node = sandbox.start();
    let current = sandbox.status();
    let request = json!({"schema_version":1,"command":"stop", "request_id":uuid::Uuid::new_v4().to_string(),
        "installation_id":current["runtime"]["installation_id"],"generation":current["runtime"]["generation"],
        "run_id":current["runtime"]["run_id"]});
    let bytes = serde_json::to_vec(&request).unwrap();
    let mut stream = UnixStream::connect(path(&initial, "socket")).unwrap();
    stream
        .write_all(&(bytes.len() as u32).to_be_bytes())
        .unwrap();
    stream.write_all(&bytes).unwrap();
    stream.shutdown(std::net::Shutdown::Both).unwrap();
    drop(stream);
    wait_success(&mut node);
    let stopped = sandbox.status();
    assert_eq!(stopped["local_status"], "stopped");
    assert_eq!(stopped["runtime"]["generation"], 1);
    assert_eq!(stopped["runtime"]["run_id"], current["runtime"]["run_id"]);
}
