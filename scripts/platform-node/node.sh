#!/usr/bin/env bash
# Standalone local test wrapper; does not install or contact the platform.
set -euo pipefail
root="$(cd -- "$(dirname -- "$0")" && pwd)"
if [[ -f "$root/../../Cargo.toml" ]]; then
    : "${COMPUTE_NODE_BINARY:?Set the built compute-node absolute path for source-tree tests}"
    : "${COMPUTE_NODE_DATA_DIR:?Set an independent absolute test data directory}"
    export XDG_CONFIG_HOME="${COMPUTE_NODE_TEST_CONFIG:?Set an independent absolute config directory}"
else
    export XDG_CONFIG_HOME="$root/config"
fi
exec "${COMPUTE_NODE_BINARY:-$root/compute-node}" "$@" --data-dir "${COMPUTE_NODE_DATA_DIR:-$root/data}"
