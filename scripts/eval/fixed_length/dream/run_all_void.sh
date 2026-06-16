#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for block_length in 64 128 512; do
  echo "[info] running fixed_length family=dream mode=void length=512 block_length=${block_length}"
  MODE=void TASK="${TASK:-all}" LENGTH=512 BLOCK_LENGTH="${block_length}" bash "${SCRIPT_DIR}/run.sh"
done
