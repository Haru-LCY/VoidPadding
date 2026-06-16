#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  BASE_MODEL=/path/or/model-id \
  ADAPTER_PATH=/path/to/lora \
  OUTPUT_DIR=/path/to/merged \
  scripts/merge/merge_dream_lora.sh

Optional environment variables:
  TOKENIZER_PATH   Tokenizer path. Defaults to BASE_MODEL.
  TORCH_DTYPE      float32, float16, or bfloat16. Defaults to bfloat16.
  DEVICE           auto, cpu, cuda, or cuda:N. Defaults to auto.
  PYTHON           Python executable. Defaults to python.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${BASE_MODEL:?Set BASE_MODEL to a Dream base/instruct model path or Hugging Face model id.}"
: "${ADAPTER_PATH:?Set ADAPTER_PATH to a PEFT LoRA adapter directory.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to the destination directory for the merged model.}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
DEVICE="${DEVICE:-auto}"
PYTHON="${PYTHON:-python}"

"${PYTHON}" scripts/merge/merge_dream_lora.py \
  --base_model "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --tokenizer_path "${TOKENIZER_PATH}" \
  --torch_dtype "${TORCH_DTYPE}" \
  --device "${DEVICE}"
