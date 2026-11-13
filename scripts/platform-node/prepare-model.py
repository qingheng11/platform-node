"""Copy trusted local artifacts into an isolated compute-node test package."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def runtime_file(name):
    return name in {"llama-server", "llama-fit-params"} or bool(
        re.fullmatch(r"lib[A-Za-z0-9_.-]+\.so(?:\.[A-Za-z0-9_.-]+)?", name)
    )


def prepare_runtime(source, destination, expected_sha256):
    if source.is_dir():
        for file in source.iterdir():
            if runtime_file(file.name):
                if not file.is_file():
                    raise ValueError(f"运行环境含无效文件 {file.name}")
                shutil.copyfile(file, destination / file.name)
    else:
        if not expected_sha256 or digest(source) != expected_sha256.lower():
            raise ValueError("运行环境 ZIP 需要提供匹配的 --runtime-sha256")
        with ZipFile(source) as archive:
            for name in archive.namelist():
                if runtime_file(name):
                    with archive.open(name) as incoming, (destination / name).open("xb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing, length=1024 * 1024)
    server = destination / "llama-server"
    if not server.is_file():
        raise ValueError("找不到 llama-server。请选包含该程序的运行环境目录或已校验的 ZIP")
    with server.open("rb") as stream:
        if stream.read(4) != b"\x7fELF":
            raise ValueError("llama-server 不是 Linux ELF")
    server.chmod(0o700)
    fitter=destination / "llama-fit-params"
    if fitter.exists():
        with fitter.open("rb") as stream:
            if stream.read(4)!=b"\x7fELF": raise ValueError("llama-fit-params 不是 Linux ELF")
        fitter.chmod(0o700)
    return {file.name: digest(file) for file in sorted(destination.iterdir())}


def prepare(args):
    for label in (args.plan, args.model_id, args.variant):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,47}", label):
            raise ValueError(f"计划名、模型 ID 和量化标识需使用字母、数字或 ._-，最长 48 字符。收到 {label!r}")
    if args.reuse_plan:
        reuse(args)
        return
    if args.model is None or args.runtime is None:
        raise ValueError("首次准备需要 --model 和 --runtime；复用已有文件可使用 --reuse-plan")
    args.context_tokens=args.context_tokens or 2048
    args.output_tokens=args.output_tokens or 800
    args.parallel_slots=args.parallel_slots or 4
    if not 512 <= args.context_tokens <= 32768 or not 1 <= args.output_tokens <= min(2048, args.context_tokens - 1):
        raise ValueError("每路上下文或输出 Token 上限无效")
    if args.vram_mib is not None and (not 1 <= args.vram_mib < 2**32 or not 1 <= args.memory_mib < 2**32):
        raise ValueError("显存和主存预算必须为正整数 MiB")
    model_source = args.model.resolve(strict=True)
    runtime_source = args.runtime.resolve(strict=True)
    if model_source.suffix.lower() != ".gguf" or not model_source.is_file():
        raise ValueError("请传入已有的 GGUF 模型文件")
    if re.search(r"-\d{5}-of-\d{5}\.gguf$",model_source.name,re.I):
        raise ValueError("本机准备工具当前只接受完整单文件 GGUF，不能只传分片第一文件")
    with model_source.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise ValueError("文件头不是 GGUF")
    state = json.loads(subprocess.check_output([str(ROOT / "node.sh"), "init", "--json"]))["paths"]
    plan_dir = Path(state["state"]) / "model-plans"
    models = Path(state["models"])
    runtimes = Path(state["environments"])
    for directory in (plan_dir, models, runtimes):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    runtime_id = f"local-{args.plan}"
    model_name = f"{args.plan}.gguf"
    targets = (models / model_name, runtimes / runtime_id, plan_dir / f"{args.plan}.json")
    with (ROOT / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if any(path.exists() or path.is_symlink() for path in targets):
            raise ValueError("同名计划或制品已存在。请使用已有计划，或给新计划另取一个名称")
        with tempfile.TemporaryDirectory(prefix="prepare-", dir=state["state"]) as staging:
            stage = Path(staging)
            stage_runtime = stage / "runtime"
            stage_runtime.mkdir(mode=0o700)
            stage_model = stage / model_name
            shutil.copyfile(model_source, stage_model)
            model_sha256 = digest(stage_model)
            if args.model_sha256 and model_sha256 != args.model_sha256.lower():
                raise ValueError("模型 SHA256 与输入值不一致")
            runtime_files = prepare_runtime(runtime_source, stage_runtime, args.runtime_sha256)
            plan = {
                "schema_version": 1, "model_id": args.model_id, "variant_id": args.variant,
                "model_file": model_name, "model_sha256": model_sha256,
                "runtime_id": runtime_id, "runtime_files": runtime_files,
                "context_tokens": args.context_tokens, "output_tokens": args.output_tokens,
                "parallel_slots": args.parallel_slots, "idle_unload_seconds": 600,
                "cpu_threads": 4, "gpu_layers": 999,
                "resource_mode": "manual" if args.vram_mib is not None else "auto",
                "max_gpus": args.max_gpus,
            }
            if args.vram_mib is not None:
                plan.update(vram_mib=args.vram_mib,memory_mib=args.memory_mib)
            elif "llama-fit-params" not in runtime_files:
                raise ValueError("自动评估需要运行环境包含 llama-fit-params")
            stage_plan = stage / "plan.json"
            stage_plan.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
            for source, target in zip((stage_model, stage_runtime, stage_plan), targets):
                source.rename(target)
    print(f"本机计划已生成 {targets[-1]}")
    print(f"模型 SHA256 {model_sha256}")



def reuse(args):
    if args.model is not None or args.runtime is not None or args.vram_mib is not None:
        raise ValueError("--reuse-plan 不与新制品或手动预算混用")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,47}",args.reuse_plan):
        raise ValueError("原计划名无效")
    state=json.loads(subprocess.check_output([str(ROOT/"node.sh"),"init","--json"]))["paths"]
    plans=Path(state["state"])/"model-plans"
    with (ROOT/".prepare.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        original=plans/f"{args.reuse_plan}.json"
        if original.is_symlink() or not original.is_file(): raise ValueError("原计划不存在或是符号链接")
        plan=json.loads(original.read_text())
        if "llama-fit-params" not in plan.get("runtime_files",{}):
            raise ValueError("旧计划未列出已校验的 llama-fit-params，需要用完整运行环境重新准备")
        # Previous test helpers copied the verified fitter but left mode 0600.
        runtime_id=plan.get("runtime_id","")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",runtime_id):
            raise ValueError("旧计划环境路径无效")
        runtime=Path(state["environments"])/runtime_id
        if runtime.resolve()!=runtime or not runtime.is_dir(): raise ValueError("旧环境目录不安全")
        tool=runtime/"llama-fit-params"
        if tool.is_symlink() or not tool.is_file() or digest(tool)!=plan["runtime_files"][tool.name]:
            raise ValueError("旧计划的资源估算工具缺失或摘要不符")
        tool.chmod(0o700)
        plan.update(resource_mode="auto",max_gpus=args.max_gpus)
        plan.pop("vram_mib",None);plan.pop("memory_mib",None)
        for arg,field in [(args.context_tokens,"context_tokens"),(args.output_tokens,"output_tokens"),(args.parallel_slots,"parallel_slots")]:
            if arg is not None: plan[field]=arg
        target=plans/f"{args.plan}.json"
        with target.open("x",encoding="utf-8") as stream: stream.write(json.dumps(plan,indent=2)+"\n")
    print(f"自动评估计划已生成 {target}；原计划和制品保留，未重新下载或复制")


if __name__ == "__main__":
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--runtime-sha256", help="Expected SHA-256 of a runtime ZIP")
    parser.add_argument("--model-sha256")
    parser.add_argument("--plan", default="qwen06")
    parser.add_argument("--model-id", default="Qwen3-0.6B")
    parser.add_argument("--variant", default="Q4_K_M")
    parser.add_argument("--vram-mib", type=int, help="仅用于显式手动预算；默认自动评估")
    parser.add_argument("--reuse-plan", help="用已有计划的制品创建自动评估计划，不再复制模型和环境")
    parser.add_argument("--max-gpus", type=int, choices=range(1,9), default=8)
    parser.add_argument("--memory-mib", default=4096, type=int)
    parser.add_argument("--context-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument("--parallel-slots", type=int, choices=(1, 4))
    try:
        prepare(parser.parse_args())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"{error}\n")
