# Checkpoints and Released Adapters

## Base Models

VoidPadding adapters are LoRA adapters and must be used with their original
base models.

Dream instruct model:

```bash
hf download Dream-org/Dream-v0-Instruct-7B \
  --local-dir pretrained/Dream-v0-Instruct-7B
```

LLaDA instruct model:

```bash
hf download GSAI-ML/LLaDA-8B-Instruct \
  --local-dir pretrained/LLaDA-8B-Instruct
```

You can pass either the Hugging Face model id or a local directory to merge and
eval scripts:

```bash
BASE_MODEL="Dream-org/Dream-v0-Instruct-7B"
BASE_MODEL="pretrained/Dream-v0-Instruct-7B"
```

For base-model training experiments:

```bash
hf download Dream-org/Dream-v0-Base-7B \
  --local-dir pretrained/Dream-v0-Base-7B

hf download GSAI-ML/LLaDA-8B-Base \
  --local-dir pretrained/LLaDA-8B-Base
```

## Released LoRA Adapters

Released adapter repositories:

```text
akpon900/dream-instruct-rainbow
akpon900/dream-instruct-void
akpon900/llada-instruct-rainbow
akpon900/llada-instruct-void
```

Download them with:

```bash
hf download akpon900/dream-instruct-void \
  --local-dir models/dream-instruct-void

hf download akpon900/dream-instruct-rainbow \
  --local-dir models/dream-instruct-rainbow

hf download akpon900/llada-instruct-void \
  --local-dir models/llada-instruct-void

hf download akpon900/llada-instruct-rainbow \
  --local-dir models/llada-instruct-rainbow
```

Expected adapter layout:

```text
models/
├── dream-instruct-void/
├── dream-instruct-rainbow/
├── llada-instruct-void/
└── llada-instruct-rainbow/
```

Each adapter repository contains:

- `README.md`
- `adapter_config.json`
- `adapter_model.safetensors`

## Merge Notes

Evaluation expects merged full-model checkpoints. The README Quick Start shows
the standard Dream and LLaDA merge commands.

Useful overrides:

```bash
export TORCH_DTYPE=bfloat16
export DEVICE=cuda          # auto, cpu, cuda, or cuda:N
export TOKENIZER_PATH=/path/to/tokenizer
export PYTHON=python
```
