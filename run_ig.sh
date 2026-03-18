#!/usr/bin/env bash
# Wrapper that runs run_ig_dataset.py using the uv-managed virtual environment.
#
# Usage:
#   ./run_ig.sh configs/pafa_config_embedding-signed.yaml
#   ./run_ig.sh configs/betalactamase_config_token-signed.yaml
#   ./run_ig.sh --list-datasets
#   ./run_ig.sh configs/pafa_config.yaml --validate-only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: uv venv not found. Run 'uv sync' inside $SCRIPT_DIR first." >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/scripts/run_ig_dataset.py" "$@"
