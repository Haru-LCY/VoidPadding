#!/usr/bin/env bash
set -euo pipefail

MODEL_TYPE="llada_base"
PAD_NUM="1"
DEFAULT_RUN_NAME="llada8b_base_padvoid_normalmask_normal"

MODEL_PATH="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Base}"
: "${SFT_CACHE_DIR:?Set SFT_CACHE_DIR to a tokenized SFT dataset cache directory.}"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
cd "${REPO_ROOT}"

CONDA_ENV="${CONDA_ENV:-rainbow}"
if [[ -n "${CONDA_ROOT:-}" ]]; then
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy HF_ENDPOINT
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
echo "[env] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-}"
echo "[env] python=$(command -v python)"
echo "[env] accelerate=$(command -v accelerate)"
echo "[env] HF_HOME=${HF_HOME}"

NUM_PROCESSES="${NUM_PROCESSES:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-6}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
EPOCHS="${EPOCHS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
SAVE_AT_UPDATE_STEPS="${SAVE_AT_UPDATE_STEPS:-7000}"
STOP_AT_UPDATE_STEP="${STOP_AT_UPDATE_STEP:-}"
SEED="${SEED:-42}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
SWANLAB_MODE="${SWANLAB_MODE:-local}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-outputs/swanlab}"

accelerate_args=(--num_processes "${NUM_PROCESSES}" --mixed_precision "${MIXED_PRECISION}" --main_process_port 0)
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  accelerate_args+=(--multi_gpu)
fi

optional_args=()
if [[ -n "${RESUME_DIR:-}" ]]; then
  optional_args+=(--resume_dir "${RESUME_DIR}")
fi
if [[ -n "${CHECKPOINT_SUFFIX_EXTRA:-}" ]]; then
  optional_args+=(--checkpoint_suffix_extra "${CHECKPOINT_SUFFIX_EXTRA}")
fi
if [[ -n "${STOP_AT_UPDATE_STEP}" ]]; then
  optional_args+=(--stop_at_update_step "${STOP_AT_UPDATE_STEP}")
fi

accelerate launch "${accelerate_args[@]}" main.py \
  --method sft \
  --model_type "${MODEL_TYPE}" \
  --model_name "${MODEL_PATH}" \
  --sft_cache_dir "${SFT_CACHE_DIR}" \
  --learning_rate "${LEARNING_RATE}" \
  --batch_size "${BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --epochs "${EPOCHS}" \
  --save_interval "${SAVE_INTERVAL}" \
  --save_at_update_steps "${SAVE_AT_UPDATE_STEPS}" \
  --seed "${SEED}" \
  --run_name "${RUN_NAME}" \
  --pad_num "${PAD_NUM}" \
  --pad_strategy void \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --num_workers "${NUM_WORKERS}" \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --swanlab_mode "${SWANLAB_MODE}" \
  --swanlab_logdir "${SWANLAB_LOGDIR}" \
  "${optional_args[@]}"
