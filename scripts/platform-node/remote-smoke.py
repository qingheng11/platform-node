"""Exercise the remote scheduler over real mTLS and a disposable model node."""

import argparse
import json
import os
from pathlib import Path
import re
import socket
import ssl
import struct
import subprocess
import tempfile
import time
import uuid


def identifier():
    return str(uuid.uuid4())


def eventually(check, seconds=18):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.15)
    raise AssertionError("等待测试条件超时")


def exchange(stream, payload):
    data = json.dumps(payload).encode()
    stream.sendall(struct.pack("!I", len(data)) + data)
    with stream.makefile("rb") as handle:
        size, = struct.unpack("!I", handle.read(4))
        return json.loads(handle.read(size))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binaries", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--config-home", type=Path, required=True)
    parser.add_argument("--plan", required=True)
    options = parser.parse_args()
    binaries = options.binaries.resolve()
    node = binaries / "compute-node"
    scheduler = binaries / "compute-scheduler"
    data = options.data_dir.resolve()
    config_home = options.config_home.resolve()
    env = os.environ | {"XDG_CONFIG_HOME": str(config_home)}
    os.umask(0o077)

    def invoke(*args, prompt=None):
        result = subprocess.run(args, input=prompt, text=True, capture_output=True,
                                env=env, timeout=10)
        assert result.returncode == 0, (args, result.stderr)
        return json.loads(result.stdout)

    def node_command(*args):
        return [node, *args, "--data-dir", data]

    assert invoke(*node_command("status", "--json"))["local_status"] == "stopped"
    assert (data / "model-plans" / f"{options.plan}.json").is_file()

    with tempfile.TemporaryDirectory(prefix="compute-remote.") as directory:
        state = Path(directory)
        pki = state / "pki"
        subprocess.run(["bash", Path(__file__).with_name("remote-test-certs.sh"), pki,
                        "gpu-test", "drop-probe", "unlisted"], capture_output=True, check=True)

        def identity(name):
            return {"certificate": str(pki / f"{name}.pem"),
                    "private_key": str(pki / f"{name}.key"),
                    "ca_certificate": str(pki / "ca.pem")}

        server_config = state / "scheduler.json"
        server_config.write_text(json.dumps({
            "state_dir": directory,
            "remote": {"listen": "127.0.0.1:0", **identity("scheduler"), "nodes": [
                {"id": "gpu-test", "certificate": identity("gpu-test")["certificate"],
                 "plans": [options.plan]},
                {"id": "drop-probe", "certificate": identity("drop-probe")["certificate"],
                 "plans": ["drop-plan"]},
            ]},
        }))

        processes = {}

        def start(name, *args):
            with (state / f"{name}.log").open("w") as log:
                processes[name] = subprocess.Popen(args, env=env, stdout=log,
                                                     stderr=subprocess.STDOUT,
                                                     start_new_session=True)

        def call(command, *args, prompt=None):
            return invoke(scheduler, command, directory, *args, prompt=prompt)

        def submit(plan):
            task = identifier()
            assert call("submit", plan, task, prompt="请只回答 OK。")["state"] == "queued"
            return task

        def open_peer(peer):
            context = ssl.create_default_context(cafile=pki / "ca.pem")
            context.minimum_version = ssl.TLSVersion.TLSv1_3
            context.set_alpn_protocols(["compute-node-test/1"])
            context.load_cert_chain(pki / f"{peer}.pem", pki / f"{peer}.key")
            raw = socket.create_connection(("127.0.0.1", port), timeout=5)
            return context.wrap_socket(raw, server_hostname="compute-scheduler.test")

        def succeeded(task):
            result = call("task", task)
            assert result["state"] not in ("failed", "outcome_unknown"), result
            return result if result["state"] == "succeeded" else None

        try:
            start("scheduler", scheduler, "serve", server_config)
            match = eventually(lambda: re.search(r"127\.0\.0\.1:(\d+)",
                                                 (state / "scheduler.log").read_text()))
            port = int(match[1])
            assert all(entry["inventory"] is None for entry in call("status")["nodes"])

            fake = {"version": 1, "available_slots": 4, "result": None, "inventory": {
                "run": {"installation_id": identifier(), "generation": 1, "run_id": identifier()},
                "models": [], "gpus": [{"index": 0, "name": "test-only", "uuid": None,
                                       "vram_free_bytes": 100, "vram_total_bytes": 100}],
            }}
            try:
                with open_peer("unlisted") as stream:
                    exchange(stream, fake)
            except (OSError, struct.error):
                pass
            else:
                raise AssertionError("同 CA 签发但未登记的节点被接收")

            with open_peer("drop-probe") as stream:
                assert exchange(stream, fake)["node_id"] == "drop-probe"
                lost = submit("drop-plan")
                reply = eventually(lambda: exchange(stream, fake)["assignments"])
                assert reply[0]["id"] == lost
            eventually(lambda: call("task", lost)["state"] == "outcome_unknown")
            with open_peer("drop-probe") as stream:
                assert exchange(stream, fake)["assignments"] == []

            first_id = submit(options.plan)
            assert call("task", first_id)["state"] == "queued"
            start("node", *node_command("serve"))
            eventually(lambda: invoke(*node_command("status", "--json"))["local_status"] == "running")
            fenced = subprocess.run(node_command("model-run-stdin", options.plan, identifier(),
                                                "--expected-run", identifier()), env=env,
                                    input="请只回答 OK。", text=True, capture_output=True, timeout=10)
            assert fenced.returncode != 0 and "代次已变化" in fenced.stderr
            client_config = state / "node.json"
            client_config.write_text(json.dumps({
                "node_id": "gpu-test", "scheduler": f"127.0.0.1:{port}",
                "server_name": "compute-scheduler.test", **identity("gpu-test"),
            }))
            start("connect", *node_command("connect", client_config))
            first = eventually(lambda: succeeded(first_id), 100)
            second_id = submit(options.plan)
            second = eventually(lambda: succeeded(second_id), 100)
            assert first["node"] == second["node"] == "gpu-test"
            assert first["output"]["instance_id"] == second["output"]["instance_id"]
            assert all("OK" in item["output"]["answer"]["text"] for item in (first, second))
            assert call("submit", options.plan, second_id, prompt="请只回答 OK。") == second
            print(json.dumps({
                "mTLS_registration": True, "unlisted_cert_rejected": True,
                "disconnect_outcome_unknown_and_no_replay": True, "stale_run_rejected": True,
                "node": first["node"], "instance_reused": True,
                "answers": [item["output"]["answer"] for item in (first, second)],
            }, ensure_ascii=False))
        except Exception:
            for name in processes:
                print((state / f"{name}.log").read_text())
            raise
        finally:
            for process in reversed(tuple(processes.values())):
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=20)
        assert invoke(*node_command("status", "--json"))["local_status"] == "stopped"


if __name__ == "__main__":
    main()
