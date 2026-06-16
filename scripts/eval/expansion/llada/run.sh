#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../.." && pwd)
cd "${REPO_ROOT}/eval/llada"

TASK="${TASK:-all}"
MODE="${MODE:-fixed_len}"

if [[ "${TASK}" == "all" ]]; then
  for task in gsm8k humaneval math500 mbpp; do
    echo "[info] running llada expansion mode=${MODE} task=${task}"
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

TOKENIZER="${TOKENIZER:-${TOKENIZER_PATH:-${REPO_ROOT}/tokenizer_llada}}"
BATCH_SIZE="${BATCH_SIZE:-1}"

case "${TASK}" in
  gsm8k) NUM_FEWSHOT="${NUM_FEWSHOT:-5}" ;;
  humaneval) NUM_FEWSHOT="${NUM_FEWSHOT:-0}" ;;
  math500) NUM_FEWSHOT="${NUM_FEWSHOT:-0}" ;;
  mbpp) NUM_FEWSHOT="${NUM_FEWSHOT:-3}" ;;
  *) echo "Unsupported TASK=${TASK}; use gsm8k, humaneval, math500, mbpp, or all." >&2; exit 1 ;;
esac

RAINBOW_MODEL="${REPO_ROOT}/model/llada_instruct/sft_5e-05_lora_rank32_pad1_force_mask_eos_500k_step7000_merged"
MODEL_ARGS=()
case "${MODE}" in
  fixed_len)
    VARIANT="${VARIANT:-all}"
    case "${VARIANT}" in
      instruct|all) ;;
      void) ;;
      *) echo "Unsupported VARIANT=${VARIANT}; use instruct, void, or all." >&2; exit 1 ;;
    esac
    if [[ "${VARIANT}" == "all" ]]; then
      echo "[info] running llada expansion fixed_len variant=instruct task=${TASK}"
      VARIANT=instruct "${SELF}"
      echo "[info] running llada expansion fixed_len variant=void task=${TASK}"
      VARIANT=void "${SELF}"
      exit 0
    fi
    LENGTH="${LENGTH:-${GEN_LENGTH:-64}}"
    STEPS="${STEPS:-${LENGTH}}"
    BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
    if [[ "${VARIANT}" == "void" ]]; then
      MODEL="${MODEL:-${MODEL_PATH:-${RAINBOW_MODEL}}}"
      OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/llada_fixed_void_len64}"
      EXTRA_ARGS=",ban_tokens=void,cut=true"
    else
      MODEL="${MODEL:-${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}}"
      OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/llada_fixed_instruct_len64}"
      EXTRA_ARGS=""
    fi
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${TASK}-ns${NUM_FEWSHOT}-len${LENGTH}-b${BLOCK_LENGTH}}"
    MODEL_ARGS=("model_path=${MODEL},tokenizer_path=${TOKENIZER},gen_length=${LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},show_speed=True${EXTRA_ARGS}")
    ;;
  void_expand)
    MODEL="${MODEL:-${MODEL_PATH:-${RAINBOW_MODEL}}}"
    INIT_LENGTH="${INIT_LENGTH:-64}"
    MAX_LENGTH="${MAX_LENGTH:-512}"
    STEPS="${STEPS:-${INIT_LENGTH}}"
    BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
    VOID_EXPAND_WINDOW="${VOID_EXPAND_WINDOW:-16}"
    VOID_EXPAND_TAU_NONVOID="${VOID_EXPAND_TAU_NONVOID:-0.09}"
    VOID_EXPAND_TAU_GAP="${VOID_EXPAND_TAU_GAP:--0.45}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/llada_void_expand}"
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${TASK}-ns${NUM_FEWSHOT}-init${INIT_LENGTH}-max${MAX_LENGTH}-b${BLOCK_LENGTH}}"
    MODEL_ARGS=("model_path=${MODEL},tokenizer_path=${TOKENIZER},gen_length=${INIT_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},show_speed=True,ban_tokens=void,cut=true,void_expand=true,void_expand_max_length=${MAX_LENGTH},void_expand_window=${VOID_EXPAND_WINDOW},void_expand_tau_nonvoid=${VOID_EXPAND_TAU_NONVOID},void_expand_tau_gap=${VOID_EXPAND_TAU_GAP}")
    ;;
  daedal|rho_eos)
    MODEL="${MODEL:-${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}}"
    INIT_LENGTH="${INIT_LENGTH:-64}"
    BLOCK_LENGTH="${BLOCK_LENGTH:-32}"
    OUTPUT_ROOT="${OUTPUT_ROOT:-evals_results/llada_${MODE}}"
    OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_ROOT}/${TASK}-ns${NUM_FEWSHOT}-init${INIT_LENGTH}-b${BLOCK_LENGTH}}"
    MODEL_ARGS=("model_path=${MODEL},tokenizer_path=${TOKENIZER},generation_mode=${MODE},initial_gen_length=${INIT_LENGTH},block_length=${BLOCK_LENGTH},show_speed=True")
    ;;
  *) echo "Unsupported MODE=${MODE}; use fixed_len, void_expand, daedal, or rho_eos." >&2; exit 1 ;;
esac

export EVAL_NFE_STATS_DIR="${EVAL_NFE_STATS_DIR:-${OUTPUT_PATH}/nfe_stats}"

CMD=(
  accelerate
  launch
  eval.py
  --model llada_dist
  --model_args "${MODEL_ARGS[0]}"
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
