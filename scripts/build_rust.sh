#!/usr/bin/env bash
set -euo pipefail

cargo build --release --manifest-path rust-services/recon-core/Cargo.toml

echo
echo "Built:"
echo "  rust-services/recon-core/target/release/mart_cli"
