#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $EUID -eq 0 ]]; then
    printf '%s\n' '请用运行服务的普通 Linux 账号安装。' >&2
    exit 1
fi

source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -f "$source_dir/compute-scheduler" ]]; then
    role=compute-scheduler
    units=(compute-scheduler.service)
elif [[ -f "$source_dir/compute-node" ]]; then
    role=compute-node
    units=(compute-node.service compute-node-connect.service)
else
    printf '%s\n' '安装目录里找不到程序。' >&2
    exit 1
fi

command -v python3 >/dev/null || {
    printf '%s\n' '请安装 Python 3 后重试。' >&2
    exit 1
}

binary_dir="$HOME/.local/lib/$role"
unit_dir="$HOME/.config/systemd/user"
sources=("$source_dir/$role")
targets=("$binary_dir/$role")
modes=(755)
for unit in "${units[@]}"; do
    sources+=("$source_dir/systemd/$unit")
    targets+=("$unit_dir/$unit")
    modes+=(644)
done

for index in "${!targets[@]}"; do
    source_file=${sources[index]}
    target=${targets[index]}
    if [[ ! -f "$source_file" || -L "$source_file" ]]; then
        printf '包内文件缺失或无效 %s\n' "$source_file" >&2
        exit 1
    fi
    if [[ -L "$target" ]] || { [[ -e "$target" ]] &&
        { [[ ! -f "$target" ]] || ! cmp -s -- "$source_file" "$target"; }; }; then
        printf '安装位置已有不同的文件 %s。请先按升级步骤检查。\n' "$target" >&2
        exit 1
    fi
done

mkdir -p -- "$binary_dir" "$unit_dir"
for index in "${!targets[@]}"; do
    target=${targets[index]}
    if [[ ! -e "$target" ]]; then
        install -m "${modes[index]}" -- "${sources[index]}" "$target"
    fi
done
python3 "$source_dir/cli.py" --register "$role" "$binary_dir/$role"
printf '已安装 %s。配置及启动步骤见包内 README.txt。\n' "$role"
