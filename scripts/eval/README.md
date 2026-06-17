# Eval

Dream-specific lm-eval code lives under `eval/dream`. LLaDA-specific lm-eval
code lives under `eval/llada`. Shared code benchmark helpers live under
`eval/common`.

The current Dream scripts cover four full benchmarks:

- GSM8K: `gsm8k`
- HumanEval: `humaneval`
- MATH-500: local `math500`
- MBPP: local `mbpp`

The current LLaDA scripts cover:

- GSM8K: `gsm8k`
- HumanEval: `humaneval`
- MATH-500: local `math500`
- MBPP: local `mbpp`

GSM8K and HumanEval use lm-eval built-in task definitions. Dream and LLaDA
MATH-500 both use the shared `eval/common/tasks/math500/math500.yaml` config
backed by `HuggingFaceH4/MATH-500`. Dream and LLaDA MBPP both use the shared
`eval/common/tasks/mbpp/mbpp.yaml` definition backed by
`Muennighoff/mbpp`, so they evaluate the same full JSONL. Hugging Face
`datasets` will download and cache public datasets automatically when needed.

## Prepare Datasets

Use a writable Hugging Face cache. This matters on clusters where the default
shared cache may be read-only.

```bash
cd llada_dream_sft/eval/dream
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
```

Pre-download the datasets used by the local tasks:

```bash
huggingface-cli download HuggingFaceH4/MATH-500 --repo-type dataset
huggingface-cli download Muennighoff/mbpp --repo-type dataset
```

After the datasets are cached, offline runs can use:

```bash
export HF_DATASETS_OFFLINE=1
```

## Run Eval

Fixed-length eval scripts live under `scripts/eval/fixed_length`:

- `llada/run.sh`: LLaDA fixed-length eval
- `llada/run_all_baseline.sh`: LLaDA `rainbow` and `instruct`
- `llada/run_all_void.sh`: LLaDA `void`
- `dream/run.sh`: Dream fixed-length eval
- `dream/run_all_baseline.sh`: Dream `rainbow` and `instruct`
- `dream/run_all_void.sh`: Dream `void`

Use `TASK=gsm8k|humaneval|math500|mbpp|all` and
`MODE=void|rainbow|instruct`.

- `MODE=void`: repository merged Rainbow model with `ban_tokens=void` and
  `cut=true`
- `MODE=rainbow`: repository merged Rainbow model, without
  `ban_tokens=void` and without `cut=true`
- `MODE=instruct`: original instruct model, without
  `ban_tokens=void` and without `cut=true`

Examples:

```bash
TASK=gsm8k MODE=void bash scripts/eval/fixed_length/llada/run.sh
TASK=all MODE=rainbow bash scripts/eval/fixed_length/dream/run.sh
bash scripts/eval/fixed_length/llada/run_all_baseline.sh
bash scripts/eval/fixed_length/llada/run_all_void.sh
bash scripts/eval/fixed_length/dream/run_all_baseline.sh
bash scripts/eval/fixed_length/dream/run_all_void.sh
```

Set `MODEL` to the model directory or Hugging Face model id. `TOKENIZER`
defaults to the repository tokenizer directory for that family.

Useful overrides:

```bash
TOKENIZER=/path/to/tokenizer
BATCH_SIZE=1
OUTPUT_ROOT=evals_results/baseline
NUM_FEWSHOT=0
BLOCK_LENGTH=512
```

Adaptive-length expansion scripts live under `scripts/eval/expansion`:

- `llada/run.sh`: `MODE=fixed_len|void_expand|daedal|rho_eos`
- `llada/run_all_baseline.sh`: LLaDA `fixed_len`, `daedal`, and `rho_eos`
- `llada/run_all_voidexpansion.sh`: LLaDA `void_expand`
- `dream/run.sh`: `MODE=fixed_len|void_expand`
- `dream/run_all_baseline.sh`: Dream `fixed_len`
- `dream/run_all_voidexpansion.sh`: Dream `void_expand`

Expansion uses `TASK=gsm8k|humaneval|math500|mbpp|all`. For
`MODE=fixed_len`, set `VARIANT=instruct|void|all`; the default `all` runs both.
Expansion fixed-length defaults are `genlen=64` and `block_length=32`. The void
fixed-length variant also sets `ban_tokens=void` and `cut=true`.

```bash
TASK=gsm8k MODE=fixed_len VARIANT=all bash scripts/eval/expansion/llada/run.sh
TASK=all MODE=void_expand bash scripts/eval/expansion/dream/run.sh
bash scripts/eval/expansion/llada/run_all_baseline.sh
bash scripts/eval/expansion/llada/run_all_voidexpansion.sh
bash scripts/eval/expansion/dream/run_all_baseline.sh
bash scripts/eval/expansion/dream/run_all_voidexpansion.sh
```

Default few-shot settings:

- GSM8K: `NUM_FEWSHOT=5`
- HumanEval: `NUM_FEWSHOT=0`
- MATH-500: `NUM_FEWSHOT=0`
- MBPP: `NUM_FEWSHOT=3`

## Code Benchmarks

HumanEval and MBPP execute generated Python during scoring. The scripts set:

```bash
HF_ALLOW_CODE_EVAL=1
HF_DATASETS_TRUST_REMOTE_CODE=true
```

They also pass `--confirm_run_unsafe_code` to lm-eval. Run these benchmarks only
in an environment where executing benchmark code is acceptable.

HumanEval runs `eval/common/postprocess_code.py` after lm-eval by default.
Dream and LLaDA MBPP use `eval/common/postprocess_mbpp.py`. The post-processed
samples are written next to the lm-eval samples with a `.cleaned` suffix. Set
`POSTPROCESS_CODE=false` for HumanEval or `POSTPROCESS_MBPP=false` for MBPP to
skip this step.

## NFE Stats

The eval scripts write true decoding NFE records to:

```bash
$OUTPUT_PATH/nfe_stats/rank_*.jsonl
```

Each record uses `nfe_scope=decoding_only`: early EOS/cut can make `nfe` smaller
than the configured budget, while expansion/probe calls such as `void_expand`
are recorded separately as `expansion_probe_nfe` and are not included in `nfe`.

## Notes

- MATH-500 uses the shared `eval/common/tasks/math500` task. Set
  `MATH500_DATA_FILE=/path/to/test.jsonl` to use a specific local JSONL file.
- MBPP is shared from `eval/common/tasks/mbpp/mbpp.yaml`. Set
  `MBPP_DATA_FILE=/path/to/mbpp.jsonl` to use a specific local JSONL file.
- If dataset loading fails with a lock-file or read-only-cache error, set
  `HF_HOME` and `HF_DATASETS_CACHE` to a writable directory and run the
  pre-download commands again.
- Shared tasks live under `eval/common/tasks`; both MATH-500 and MBPP scripts
  include that path for Dream and LLaDA.
