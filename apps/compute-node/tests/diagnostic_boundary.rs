#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::{fs, path::Path, process::Command};

use serde_json::Value;
use tempfile::TempDir;

fn command(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_compute-node"));
    command
        .env("HOME", root.join("home"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CONFIG_HOME", root.join("config"))
        .env("XDG_RUNTIME_DIR", root.join("runtime"));
    command
}

fn snapshot(root: &Path) -> Vec<(std::path::PathBuf, Vec<u8>)> {
    fn visit(root: &Path, path: &Path, result: &mut Vec<(std::path::PathBuf, Vec<u8>)>) {
        let mut entries: Vec<_> = fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        entries.sort();
        for entry in entries {
            let content = if entry.is_dir() {
                Vec::new()
            } else {
                fs::read(&entry).unwrap()
            };
            result.push((entry.strip_prefix(root).unwrap().to_path_buf(), content));
            if entry.is_dir() {
                visit(root, &entry, result);
            }
        }
    }
    let mut result = Vec::new();
    visit(root, root, &mut result);
    result
}

#[test]
fn diagnostics_use_a_separate_namespace_without_changing_user_state() {
    let root = TempDir::new().unwrap();
    let personal = root.path().join("data/unrelated-app");
    fs::create_dir_all(&personal).unwrap();
    fs::write(personal.join("sentinel"), b"synthetic-user-state").unwrap();
    let before = snapshot(root.path());
    let output = command(root.path())
        .args(["doctor", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let result: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(result["edition"], "platform_node");
    assert_eq!(result["stage"], "local_resource_planning");
    assert_eq!(result["capabilities"]["local_resource_estimation"], true);
    assert_eq!(result["capabilities"]["local_gpu_group_placement"], true);
    let state = root.path().join("data/compute-node");
    assert_eq!(
        result["planned_directories"]["state"],
        state.to_string_lossy().as_ref()
    );
    assert_eq!(result["capabilities"]["hardware_diagnostics"], true);
    for capability in [
        "node_registration",
        "platform_tasks",
        "model_management",
        "multi_model_scheduling",
        "idle_unload",
    ] {
        assert_eq!(result["capabilities"][capability], false);
    }
    assert!(
        result["hardware"]["memory"]["total_bytes"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(snapshot(root.path()), before);
    assert!(!state.exists());
}

#[test]
fn unimplemented_runtime_and_login_commands_cannot_report_success() {
    let root = TempDir::new().unwrap();
    let before = snapshot(root.path());
    for args in [vec!["serve"], vec!["login"], vec!["run", "Qwen3-0.6B"]] {
        let output = command(root.path()).args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
    assert_eq!(snapshot(root.path()), before);
}
