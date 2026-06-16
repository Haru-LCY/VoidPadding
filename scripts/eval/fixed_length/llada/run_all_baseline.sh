#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

for block_length in 64 128 512; do
  for mode in rainbow instruct; do
    echo "[info] running fixed_length family=llada mode=${mode} length=512 block_length=${block_length}"
    MODE="${mode}" TASK="${TASK:-all}" LENGTH=512 BLOCK_LENGTH="${block_length}" bash "${SCRIPT_DIR}/run.sh"
  done
done
