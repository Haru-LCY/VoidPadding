# Training Scripts

This directory contains LoRA SFT launch scripts for VoidPadding and baseline
padding strategies. Scripts are split by model family:

- `instruct/`: LLaDA-8B-Instruct and Dream-7B-Instruct experiments.
- `base/`: LLaDA-8B-Base and Dream-7B-Base appendix experiments.

All scripts should be run from the repository root.
Their default hyperparameters are the configurations used for the paper
experiments.

## Prepare Data

We use the same 0.5M-example instruction-tuning dataset as the official
RainbowPadding implementation. The data pool combines Tulu3 and SmolLM2. After
removing examples longer than 4096 tokens and multi-turn dialogues, 0.5M
examples are randomly sampled and used identically for all models in our
experiments.

For convenience and reproducibility, we recommend using the released tokenized
SFT dataset directly:

```bash
hf download akpon900/Tokenized_VoidPadding_Data \
  --repo-type dataset \
  --local-dir datasets
```

Expected local layout:

```text
datasets/
├── tokenized_sft_dataset_dream_500000/
└── tokenized_sft_dataset_llada_500000/
```

Set `SFT_CACHE_DIR` to the matching dataset before launching a script:

```bash
# LLaDA training
export SFT_CACHE_DIR="$PWD/datasets/tokenized_sft_dataset_llada_500000"

# Dream training
export SFT_CACHE_DIR="$PWD/datasets/tokenized_sft_dataset_dream_500000"
```

By default, scripts use public Hugging Face model ids. To train from a local
checkpoint, override `MODEL_PATH`:

```bash
export MODEL_PATH="/path/to/local/base/model"
```

## Instruct Models

Instruct scripts cover VoidPadding and RainbowPadding on the instruction-tuned
backbones.

VoidPadding:

```bash
bash scripts/train/instruct/voidpadding_finetune_llada8b_instruct.sh
bash scripts/train/instruct/voidpadding_finetune_dream7b_instruct.sh
```

RainbowPadding baseline:

```bash
bash scripts/train/instruct/rainbowpadding_finetune_llada8b_instruct.sh
bash scripts/train/instruct/rainbowpadding_finetune_dream7b_instruct.sh
```

Default base models:

| Script family | `MODEL_TYPE` | Default `MODEL_PATH` |
| ------------- | ------------ | -------------------- |
| LLaDA instruct | `llada_instruct` | `GSAI-ML/LLaDA-8B-Instruct` |
| Dream instruct | `dream_instruct` | `Dream-org/Dream-v0-Instruct-7B` |

## Base Models

Base scripts cover VoidPadding, RainbowPadding, and EOS-padding baselines on the
base backbones.

VoidPadding:

```bash
bash scripts/train/base/voidpadding_finetune_llada8b_base.sh
bash scripts/train/base/voidpadding_finetune_dream7b_base.sh
```

RainbowPadding baseline:

```bash
bash scripts/train/base/rainbowpadding_finetune_llada8b_base.sh
bash scripts/train/base/rainbowpadding_finetune_dream7b_base.sh
```

EOS-padding baseline:

```bash
bash scripts/train/base/eospadding_finetune_llada8b_base.sh
bash scripts/train/base/eospadding_finetune_dream7b_base.sh
```

Default base models:

| Script family | `MODEL_TYPE` | Default `MODEL_PATH` |
| ------------- | ------------ | -------------------- |
| LLaDA base | `llada_base` | `GSAI-ML/LLaDA-8B-Base` |
| Dream base | `dream_base` | `Dream-org/Dream-v0-Base-7B` |

## Paper Defaults

The launch scripts already set the paper training configuration by default.
The main defaults are:

| Setting | Default |
| ------- | ------- |
| `MIXED_PRECISION` | `bf16` |
| `LEARNING_RATE` | `5e-5` |
| `GRADIENT_ACCUMULATION_STEPS` | `4` |
| `SAVE_INTERVAL` | `1` |
| `SAVE_AT_UPDATE_STEPS` | `7000` |
| `SEED` | `42` |
| `LORA_RANK` | `32` |
| `LORA_ALPHA` | `64` |
| `NUM_WORKERS` | `4` |
| `PREFETCH_FACTOR` | `2` |
| `SWANLAB_MODE` | `local` |
| `SWANLAB_LOGDIR` | `outputs/swanlab` |

Model-family defaults:

| Family | `NUM_PROCESSES` | `BATCH_SIZE` |
| ------ | --------------- | ------------ |
| LLaDA | `2` | `6` |
| Dream | `3` | `4` |

Training-length defaults:

| Script group | `EPOCHS` | `STOP_AT_UPDATE_STEP` |
| ------------ | -------- | --------------------- |
| Instruct VoidPadding | `1` | `7000` |
| Instruct RainbowPadding | `1` | unset |
| Base VoidPadding | `2` | unset |
| Base EOS-padding | `2` | unset |
| Base RainbowPadding | `3` | unset |

Padding behavior:

| Strategy | `pad_strategy` | Behavior |
| -------- | -------------- | -------- |
| VoidPadding | `void` | pads with a single VOID token, resolved as pad0 |
| EOS-padding | `eos` | pads all post-response positions with EOS/EOT |
| RainbowPadding | `rainbow` | cycles through 7 Rainbow padding tokens |

All of these can be overridden with environment variables when launching a
script.

## Smoke-Test Overrides

For a short local run, override the paper defaults:

```bash
export NUM_PROCESSES=1
export MIXED_PRECISION=bf16
export LEARNING_RATE=5e-5
export BATCH_SIZE=1
export GRADIENT_ACCUMULATION_STEPS=1
export EPOCHS=1
export SAVE_AT_UPDATE_STEPS=7000
export STOP_AT_UPDATE_STEP=1
export SWANLAB_MODE=disabled
```

Other useful optional overrides:

```bash
export RUN_NAME="my_run_name"
export RESUME_DIR="/path/to/checkpoint"
export CHECKPOINT_SUFFIX_EXTRA="tag"
export LORA_RANK=32
export LORA_ALPHA=64
export HF_HOME="/path/to/hf/cache"
```

## Direct main.py Usage

The launch scripts are wrappers around `main.py`. A minimal direct run looks
like this:

```bash
accelerate launch --num_processes 1 --mixed_precision bf16 main.py \
  --method sft \
  --model_type llada_instruct \
  --model_name "GSAI-ML/LLaDA-8B-Instruct" \
  --sft_cache_dir "${SFT_CACHE_DIR}" \
  --pad_strategy void \
  --save_at_update_steps 7000
```

Supported `model_type` values:

- `llada_base`
- `llada_instruct`
- `dream_base`
- `dream_instruct`

Supported `pad_strategy` values:

- `void`
- `rainbow`
- `eos`
