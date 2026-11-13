#!/usr/bin/env python3
"""Compute service command launcher."""

import json
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile


PATH_BLOCK = """
# Compute service commands
case ":$PATH:" in
    *":$HOME/.local/bin:"*) ;;
    *) export PATH="$HOME/.local/bin:$PATH" ;;
esac
"""


def write(path, data, mode):
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        staged = Path(tmp.name)
        try:
            tmp.write(data)
            tmp.flush()
            staged.chmod(mode)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


def register(role, binary, scheduler_config="", node_config_home="", node_data_dir=""):
    home = Path.home()
    bindir = home / ".local/bin"
    config_dir = home / ".config/compute-service-cli"
    entry = bindir / role
    settings = config_dir / f"{role}.json"
    payload = Path(__file__).read_bytes()
    values = {"binary": binary}
    for key, value in (
        ("scheduler_config", scheduler_config),
        ("node_config_home", node_config_home),
        ("node_data_dir", node_data_dir),
    ):
        if value:
            values[key] = value
    if role not in {"compute-scheduler", "compute-node"} or any(not Path(v).is_absolute() for v in values.values()):
        raise ValueError("命令注册需要程序名称和绝对路径。")
    if not os.access(binary, os.X_OK) or not Path(binary).is_file():
        raise ValueError(f"找不到可执行程序 {binary}。")
    if entry.is_symlink() or (entry.exists() and (
        not entry.is_file() or b"Compute service command launcher" not in entry.read_bytes()[:160]
    )):
        raise ValueError(f"已有其他程序占用 {entry}。")
    if settings.is_symlink() or (settings.exists() and not settings.is_file()):
        raise ValueError(f"命令配置路径无效 {settings}。")
    if settings.exists() and len(values) == 1:
        previous = json.loads(settings.read_text(encoding="utf-8"))
        if previous.get("binary") == binary:
            values = previous
    login_files = [home / name for name in (".bash_profile", ".bash_login", ".profile")]
    profiles = [home / ".bashrc", next((p for p in login_files if p.exists()), login_files[-1])]
    profile_updates = []
    for profile in profiles:
        if profile.exists() and not profile.is_file():
            raise ValueError(f"Shell 配置路径无效 {profile}。")
        current = profile.read_bytes() if profile.exists() else b""
        if b"# Compute service commands" not in current:
            profile_updates.append(profile)
    bindir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    write(settings, json.dumps(values).encode() + b"\n", 0o600)
    write(entry, payload, 0o755)
    for profile in profile_updates:
        with profile.open("ab") as output:
            output.write(PATH_BLOCK.encode())
    print(f'已注册 {role}。重新登录 Bash 即可使用，或在当前终端执行 export PATH="$HOME/.local/bin:$PATH"。')


def service_environment(path, keys):
    """Read the absolute-path assignments documented by the service packages."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or key not in keys:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        if not Path(value).is_absolute() or any(c in value for c in "\\\"'"):
            raise ValueError(f"{path} 中的 {key} 需要使用包内文档所示的绝对路径格式。")
        result[key] = value
    return result


def scheduler_args(args, home, config):
    if not args or args[0] not in {"status", "task", "submit", "submit-web"}:
        return args
    if len(args) > 1 and Path(args[1]).is_absolute():
        return args  # Existing commands can still select their own state directory.
    state = os.environ.get("COMPUTE_SCHEDULER_STATE_DIR")
    if not state:
        if not config:
            config = os.environ.get("COMPUTE_SCHEDULER_CONFIG") or service_environment(
                home / ".config/compute-scheduler/service.env", {"COMPUTE_SCHEDULER_CONFIG"}
            ).get("COMPUTE_SCHEDULER_CONFIG", home / ".config/compute-scheduler/scheduler.json")
        state = json.loads(Path(config).read_text(encoding="utf-8"))["state_dir"]
    if not isinstance(state, str) or not Path(state).is_absolute():
        raise ValueError("调度器状态目录需要是绝对路径。")
    return [args[0], state, *args[1:]]


def main():
    role = Path(sys.argv[0]).name
    home = Path.home()
    settings = json.loads((home / ".config/compute-service-cli" / f"{role}.json").read_text(encoding="utf-8"))
    binary = settings["binary"]
    args = sys.argv[1:]
    if args and args[0] in {"-h", "--help", "--version"}:
        if role == "compute-scheduler" and args[0] != "--version":
            print("已安装的命令可省略状态目录，例如 compute-scheduler status 或 compute-scheduler task <任务UUID>。", flush=True)
        os.execv(binary, [binary, *args])
    if role == "compute-scheduler":
        args = scheduler_args(args, home, settings.get("scheduler_config"))
    else:
        service = service_environment(
            home / ".config/compute-node-service/service.env", {"XDG_CONFIG_HOME", "COMPUTE_NODE_DATA_DIR"}
        )
        config_home = os.environ.get("COMPUTE_NODE_CONFIG_HOME") or settings.get("node_config_home")
        os.environ["XDG_CONFIG_HOME"] = config_home or service.get(
            "XDG_CONFIG_HOME", str(home / ".config/compute-node-service"))
        os.environ.setdefault("COMPUTE_NODE_DATA_DIR", settings.get("node_data_dir") or service.get(
            "COMPUTE_NODE_DATA_DIR", str(home / ".local/state/compute-node")
        ))
    os.execv(binary, [binary, *args])


if __name__ == "__main__":
    try:
        if Path(sys.argv[0]).name == "cli.py" and sys.argv[1:2] == ["--register"]:
            register(*sys.argv[2:])
        else:
            main()
    except (OSError, ValueError, KeyError) as error:
        sys.exit(f"{Path(sys.argv[0]).name}：{error}")
