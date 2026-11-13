"""Package locally built scheduler and node binaries for Linux x86_64."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import tempfile


HERE = Path(__file__).resolve().parent
TEMPLATES = HERE / "service-packages"


def checked_bytes(path):
    checksum, name = Path(str(path) + ".sha256").read_text(encoding="ascii").split(maxsplit=1)
    if name.strip().lstrip("*") != path.name:
        raise ValueError(f"checksum lists a different file: {path.name}")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != checksum.lower():
        raise ValueError(f"checksum mismatch: {path.name}")
    return content


def make_archive(target, package_dir, files):
    with target.open("xb") as output:
        with gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                directory = tarfile.TarInfo(package_dir)
                directory.type = tarfile.DIRTYPE
                directory.mode = 0o755
                archive.addfile(directory)
                for name, (content, mode) in sorted(files.items()):
                    member = tarfile.TarInfo(f"{package_dir}/{name}")
                    member.mode = mode
                    member.size = len(content)
                    archive.addfile(member, io.BytesIO(content))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheduler-binary", type=Path, required=True)
    parser.add_argument("--node-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", required=True, help="YYYYMMDD")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{8}", args.date):
        parser.error("--date must be YYYYMMDD")

    scheduler = checked_bytes(args.scheduler_binary)
    node = checked_bytes(args.node_binary)
    args.output.mkdir(parents=True, exist_ok=True)
    packages = (
        ("scheduler", scheduler, args.scheduler_binary.name),
        ("node", node, args.node_binary.name),
    )
    with tempfile.TemporaryDirectory(dir=args.output, prefix=".packages-") as scratch:
        staged = Path(scratch)
        for role, binary, source_name in packages:
            if binary[:6] != b"\x7fELF\x02\x01" or binary[18:20] != b"\x3e\x00":
                raise ValueError(f"{role} is not an x86_64 Linux ELF binary")
            program = f"compute-{role}"
            package_dir = program
            archive_name = f"{package_dir}-linux-x86_64-{args.date}.tar.gz"
            files = {program: (binary, 0o755), "install.sh": ((TEMPLATES / "install.sh").read_bytes(), 0o755)}
            files["cli.py"] = ((TEMPLATES / "cli.py").read_bytes(), 0o644)
            for path in (TEMPLATES / role).iterdir():
                name = f"systemd/{path.name}" if path.suffix == ".service" else path.name
                files[name] = (path.read_bytes(), 0o644)
            if role == "scheduler":
                files["remote-test-certs.sh"] = ((HERE / "remote-test-certs.sh").read_bytes(), 0o755)
            info = {
                "program": program,
                "date": args.date,
                "binary_sha256": hashlib.sha256(binary).hexdigest(),
                "source": source_name,
            }
            files["BUILD-INFO.json"] = ((json.dumps(info, indent=2) + "\n").encode(), 0o644)
            target = staged / archive_name
            make_archive(target, package_dir, files)
            checksum = hashlib.sha256(target.read_bytes()).hexdigest()
            (staged / (archive_name + ".sha256")).write_bytes(f"{checksum}  {archive_name}\n".encode("ascii"))
        for file in sorted(staged.iterdir()):
            if (args.output / file.name).exists():
                raise FileExistsError(args.output / file.name)
        for file in sorted(staged.iterdir()):
            file.rename(args.output / file.name)
            print(args.output / file.name)


if __name__ == "__main__":
    main()
