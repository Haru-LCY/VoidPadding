#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}/eval/llada"

TASK="${TASK:-all}"
MODE="${MODE:-void}"

if [[ "${TASK}" == "all" ]]; then
  for task in gsm8k humaneval math500 mbpp; do
    echo "[info] running llada fixed_length mode=${MODE} task=${task}"
    TASK="${task}" "${SELF}"
  done
  exit 0
fi

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-${HF_HOME}/modules}"
mkdir -p "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}"

RAINBOW_MODEL="${REPO_ROOT}/model/llada_instruct/sft_5e-05_lora_rank32_pad1_force_mask_eos_500k_step7000_merged"
case "${MODE}" in
  void)
    MODEL="${MODEL:-${MODEL_PATH:-${RAINBOW_MODEL}}}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/llada_void_fixed_length}"
    EXTRA_ARGS=",ban_tokens=void,cut=true"
    ;;
  rainbow)
    MODEL="${MODEL:-${MODEL_PATH:-${RAINBOW_MODEL}}}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/llada_rainbow_fixed_length}"
    EXTRA_ARGS=""
    ;;
  instruct)
    MODEL="${MODEL:-${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/llada_instruct_fixed_length}"
    EXTRA_ARGS=""
    ;;
  *) echo "Unsupported MODE=${MODE}; use void, rainbow, or instruct." >&2; exit 1 ;;
esac

TOKENIZER="${TOKENIZER:-${TOKENIZER_PATH:-${REPO_ROOT}/tokenizer_llada}}"
LENGTH="${LENGTH:-${GEN_LENGTH:-512}}"
STEPS="${STEPS:-${LENGTH}}"
BLOCK_LENGTH="${BLOCK_LENGTH:-${LENGTH}}"
BATCH_SIZE="${BATCH_SIZE:-1}"

case "${TASK}" in
  gsm8k) NUM_FEWSHOT="${NUM_FEWSHOT:-5}" ;;
  humaneval) NUM_FEWSHOT="${NUM_FEWSHOT:-0}" ;;
  math500) NUM_FEWSHOT="${NUM_FEWSHOT:-0}" ;;
  mbpp) NUM_FEWSHOT="${NUM_FEWSHOT:-3}" ;;
  *) echo "Unsupported TASK=${TASK}; use gsm8k, humaneval, math500, mbpp, or all." >&2; exit 1 ;;
esac

OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${TASK}-ns${NUM_FEWSHOT}-len${LENGTH}-b${BLOCK_LENGTH}}"
export EVAL_NFE_STATS_DIR="${EVAL_NFE_STATS_DIR:-${OUTPUT_PATH}/nfe_stats}"

MODEL_ARGS="model_path=${MODEL},tokenizer_path=${TOKENIZER},gen_length=${LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},show_speed=True${EXTRA_ARGS}"

CMD=(
  accelerate
  launch
  eval.py
  --model llada_dist
  --model_args "${MODEL_ARGS}"
  --tasks "${TASK}"
  --num_fewshot "${NUM_FEWSHOT}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${OUTPUT_PATH}"
  --confirm_run_unsafe_code
)

if [[ "${TASK}" == "math500" || "${TASK}" == "mbpp" ]]; then
  CMD+=(--include_path "${REPO_ROOT}/eval/common/tasks")
fi
if [[ "${TASK}" == "humaneval" || "${TASK}" == "math500" || "${TASK}" == "mbpp" ]]; then
  CMD+=(--log_samples)
fi
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  CMD+=(--limit "${EVAL_LIMIT}")
fi

echo "[info] command=${CMD[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
"${CMD[@]}"

if [[ "${TASK}" == "humaneval" && "${POSTPROCESS_CODE:-true}" == "true" ]]; then
  SAMPLE_FILE=$(find "${OUTPUT_PATH}" -type f -name "samples_${TASK}_*.jsonl" | sort | tail -n 1)
  python "${REPO_ROOT}/eval/common/postprocess_code.py" "${SAMPLE_FILE}"
elif [[ "${TASK}" == "mbpp" && "${POSTPROCESS_MBPP:-true}" == "true" ]]; then
  SAMPLE_FILE=$(find "${OUTPUT_PATH}" -type f -name "samples_${TASK}_*.jsonl" | sort | tail -n 1)
  python "${REPO_ROOT}/eval/common/postprocess_mbpp.py" "${SAMPLE_FILE}"
fi
