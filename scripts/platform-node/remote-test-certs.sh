#!/usr/bin/env bash
set -euo pipefail
if (( $# < 2 )); then
    printf '%s\n' '用法 remote-test-certs.sh <新的证书目录> <节点ID> [节点ID ...]' >&2
    exit 1
fi
directory=$1
shift
declare -A used=()
for node in "$@"; do
    if [[ ! $node =~ ^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$ || $node == ca || $node == scheduler ]]; then
        printf '%s\n' '节点 ID 无效。' >&2
        exit 1
    fi
    if [[ ${used[$node]+yes} ]]; then
        printf '%s\n' '节点 ID 重复。' >&2
        exit 1
    fi
    used[$node]=1
done
umask 077
mkdir -m 700 -- "$directory"
cd -- "$directory"
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes -days 7 \
    -subj '/CN=Compute remote lab CA' -keyout ca.key -out ca.pem \
    -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' \
    -addext 'keyUsage=critical,keyCertSign,cRLSign'

issue() {
    local name=$1 usage=$2
    openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:P-256 -nodes \
        -subj "/CN=$name" -keyout "$name.key" -out "$name.csr"
    openssl x509 -req -in "$name.csr" -CA ca.pem -CAkey ca.key -CAcreateserial \
        -days 7 -out "$name.pem" -extfile <(printf '%s\n' \
            'basicConstraints=critical,CA:FALSE' 'keyUsage=critical,digitalSignature' "$usage")
}

issue scheduler $'extendedKeyUsage=serverAuth\nsubjectAltName=DNS:compute-scheduler.test'
for node in "$@"; do
    issue "$node" 'extendedKeyUsage=clientAuth'
done
