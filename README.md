# Platform Node

Platform Node schedules text-inference jobs across Linux x86_64 GPU workers. The scheduler maintains a queue and places jobs on workers with a compatible model plan and available capacity. A compute node manages its local inference processes and reports GPU memory, loaded models, and active tasks.

| Program | Responsibilities |
| --- | --- |
| `compute-scheduler` | Node inventory, queue, dispatch, task status, and optional HTTPS gateway |
| `compute-node` | Hardware diagnostics, model lifecycle, inference, and remote scheduler connectivity |

Workers authenticate to the scheduler with mutual TLS. The optional gateway uses its own TLS credentials. Its result transport uses iroh and a configured relay.

## Build and test

Use Linux x86_64 with Rust, a C compiler, Python 3, and OpenSSL.

```sh
cargo test --workspace --locked
cargo build --release --locked -p compute-scheduler -p compute-node
./target/release/compute-scheduler --help
./target/release/compute-node doctor --json
```

Inference requires a compatible llama.cpp runtime, a GGUF model, and a node model plan. Consult the [scheduler setup](scripts/platform-node/service-packages/scheduler/README.txt) and [node setup](scripts/platform-node/service-packages/node/README.txt) for configuration and service commands. The examples use a local test endpoint.

## Package and install

Build on the oldest distribution you intend to support, then create one archive per binary.

```sh
(cd target/release && sha256sum compute-scheduler > compute-scheduler.sha256)
(cd target/release && sha256sum compute-node > compute-node.sha256)
python3 scripts/platform-node/package-services.py \
  --scheduler-binary target/release/compute-scheduler \
  --node-binary target/release/compute-node \
  --output dist --date "$(date +%Y%m%d)"
```

On the destination host, check the archive with its SHA-256 file, unpack it, and run `bash install.sh` as the service user. The installer registers the program in `~/.local/bin` and adds that directory to Bash's PATH. Configure and start the services using the package README, then run:

```sh
compute-scheduler status
compute-node status
compute-node gpu list
```

## License status

The extracted source currently carries no open-source license. Redistribution and licensing require permission from the applicable rights holders.
