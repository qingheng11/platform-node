"""Exercise automatic placement, four slots and GPU-group cleanup on a local node."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from uuid import uuid4


def cleanup_owned_instance(current, unload, instance_id):
    target=current()
    if target is None:
        return "already_released"
    if target["instance"]["instance_id"] != instance_id:
        return "replacement_preserved"
    unload()
    return "unloaded"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--min-gpus", type=int, choices=range(1, 9), default=2)
    args = parser.parse_args()
    node = Path(__file__).resolve().with_name("node.sh")
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "plan": args.plan,
              "minimum_gpus": args.min_gpus, "passed": False, "tests": {}}
    report_file = Path.cwd() / ("gpu-group-check-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".json")
    jobs = []
    started_jobs = {}
    report["request_results"] = []
    owns_instance = False
    owned_instance_id = None

    def call(*arguments, allow_error=False):
        p = subprocess.run([str(node), *arguments, "--json"], capture_output=True, text=True, timeout=900)
        if p.returncode:
            if allow_error:
                return p
            raise RuntimeError(f"{arguments[0]}: {p.stderr.strip()}")
        return json.loads(p.stdout)

    def current():
        return next((item for item in call("model-list")["instances"] if item["plan"] == args.plan), None)

    def collect(task, child):
        item={"task_id":task}
        try:
            out,err=child.communicate(timeout=900)
            item.update(exit_code=child.returncode,elapsed_until_collected_seconds=round(time.monotonic()-started_jobs[task],3),stderr=err)
            if not child.returncode:
                item["response"]=json.loads(out)
        except subprocess.TimeoutExpired:
            item.update(error="test_client_wait_timeout",elapsed_until_collected_seconds=round(time.monotonic()-started_jobs[task],3))
            raise
        finally:
            report["request_results"].append(item)
        return item

    try:
        report["stage"]="preflight"
        report["node_status_before"]=call("status")
        if current() is not None:
            raise RuntimeError("请先确认目标计划无任务并执行 model-unload，再运行本验收，避免影响原实例")
        report["stage"]="estimate"
        estimate = call("model-estimate", args.plan)
        report["estimate"] = estimate
        assert not estimate["loaded"] and not estimate["reserved"] and current() is None
        assert len(estimate["placement"]["gpus"]) >= args.min_gpus, "自动选卡数量未达到本轮多卡要求"
        report["tests"]["estimate_without_loading"] = True
        report["stage"]="warmup"
        warm = call("model-run", args.plan, str(uuid4()), "请只回答 OK。")
        owned_instance_id=warm["instance_id"]
        owns_instance = True
        assert warm["answer"]["text"].strip()
        loaded = current()
        assert loaded["slots_total"] == 4, "此验收需要四槽位计划"
        instance = loaded["instance"]
        group = [gpu["gpu_uuid"] for gpu in instance["placement"]["gpus"]]
        assert len(group) >= args.min_gpus and instance["placement"]["source"] == "cache"
        report["instance"] = instance
        pid = instance["runner_pid"]
        environment = dict(item.split(b"=", 1) for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in item)
        assert environment[b"CUDA_VISIBLE_DEVICES"].decode().split(",") == group
        # Do not export environment or the private runner API key.
        report["stage"]="gpu_evidence"
        gpu_view = call("gpu", "list")
        report["gpu_view"] = gpu_view
        actual = []
        for gpu in gpu_view["gpus"]:
            for item in gpu["instances"]:
                if item["instance_id"] == instance["instance_id"]:
                    assert gpu["uuid"] in group
                    assert item["vram_reserved_bytes"] > 0
                    if item["vram_used_bytes"] is not None and item["vram_used_bytes"] > 0:
                        actual.append(gpu["uuid"])
        report["observed_on_gpus"] = actual
        if args.min_gpus > 1:
            assert set(actual) == set(group), "必须取得同一进程在每张卡上的真实显存证据"
        report["tests"]["placement_and_inference"] = True
        report["stage"]="four_slots"
        for i in range(4):
            task = str(uuid4())
            child = subprocess.Popen([str(node), "model-run", args.plan, task,
                f"List at least 750 consecutive integers starting at {i*1000}, separated by commas. No ellipsis.", "--json"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            jobs.append((task, child))
            started_jobs[task]=time.monotonic()
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            busy = current()
            if busy and busy["slots_used"] == 4:
                break
            time.sleep(0.025)
        else:
            raise AssertionError("未观测到四槽位同时占用")
        report["four_slots"] = busy
        fifth = call("model-run", args.plan, str(uuid4()), "只回答 OK", allow_error=True)
        assert isinstance(fifth, subprocess.CompletedProcess) and "槽位已满" in fifth.stderr
        report["stage"]="cancel_one"
        call("model-cancel", jobs[0][0])
        cancelled=collect(*jobs[0])
        assert cancelled["exit_code"] and "槽位已释放" in cancelled["stderr"],cancelled["stderr"]
        survivors = []
        failures = []
        report["stage"]="await_survivors"
        for task, child in jobs[1:]:
            outcome=collect(task,child)
            if outcome["exit_code"]:
                failures.append(outcome)
                continue
            result=outcome["response"]
            assert result["instance_id"] == instance["instance_id"]
            survivors.append(result)
        report["survivor_failures"]=failures
        assert not failures, "\n".join(item["stderr"].strip() for item in failures)
        report["survivors"] = survivors
        report["tests"]["four_slots_fifth_rejected_and_one_cancelled"] = True
        report["stage"]="unload"
        call("model-unload", args.plan)
        owns_instance = False
        assert current() is None and not Path(f"/proc/{pid}").exists()
        released = call("gpu", "list")
        assert not any(item["instance_id"] == instance["instance_id"] for gpu in released["gpus"] for item in gpu["instances"])
        report["after_unload"] = released
        report["tests"]["all_group_reservations_released"] = True
        report["stage"]="reload"
        reload_result = call("model-run", args.plan, str(uuid4()), "请只回答 OK。")
        owned_instance_id=reload_result["instance_id"]
        owns_instance = True
        assert reload_result["instance_id"] != instance["instance_id"]
        assert current()["instance"]["placement"]["source"] == "cache"
        report["reloaded"] = reload_result
        report["tests"]["cached_reload"] = True
        report["passed"] = True
        report["stage"]="complete"
    except BaseException as error:
        report["error"] = str(error)
        raise
    finally:
        try:
            report["inventory_before_cleanup"]=call("model-list")
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            report["inventory_error"]=str(error)
        for task, child in jobs:
            if child.poll() is None:
                try:
                    call("model-cancel", task)
                    child.communicate(timeout=60)
                except (RuntimeError, subprocess.TimeoutExpired):
                    child.terminate()
                    child.communicate(timeout=10)
        if owns_instance:
            try:
                report["cleanup"]=cleanup_owned_instance(current,lambda:call("model-unload",args.plan),owned_instance_id)
            except RuntimeError as error:
                report["cleanup_error"] = str(error)
                report["passed"] = False
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"验收报告 {report_file}", flush=True)
    if not report["passed"]:
        raise RuntimeError("实例清理未确认，请检查报告")
    print("自动估算、GPU 组放置、四路、单任务取消、整组卸载和缓存重载通过")


if __name__ == "__main__":
    os.umask(0o077)
    main()
