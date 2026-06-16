#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MODE=fixed_len TASK="${TASK:-all}" bash "${SCRIPT_DIR}/run.sh"
