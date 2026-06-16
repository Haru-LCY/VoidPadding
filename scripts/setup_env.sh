#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="${ENV_FILE:-environment.yaml}"
ENV_NAME="${ENV_NAME:-void}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available on PATH." >&2
  echo "Install Miniconda or source conda.sh before running this script." >&2
  exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: missing environment file: ${ENV_FILE}" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "[setup] Updating conda env: ${ENV_NAME}"
  conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "[setup] Creating conda env: ${ENV_NAME}"
  conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
fi

echo "[setup] Verifying core packages"
conda run -n "${ENV_NAME}" python - <<'PY'
import torch
import transformers
import peft

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
PY

echo
echo "Environment is ready."
echo "Activate it with: conda activate ${ENV_NAME}"
