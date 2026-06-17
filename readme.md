# VoidPadding

[![arXiv](https://img.shields.io/badge/arXiv-2606.17999-b31b1b.svg)](https://arxiv.org/abs/2606.17999)
[![Dataset](https://img.shields.io/badge/HF%20Dataset-Tokenized%20VoidPadding-yellow)](https://huggingface.co/datasets/akpon900/Tokenized_VoidPadding_Data)
[![Dream Void](https://img.shields.io/badge/HF%20Model-Dream%20Void-yellow)](https://huggingface.co/akpon900/dream-instruct-void)
[![LLaDA Void](https://img.shields.io/badge/HF%20Model-LLaDA%20Void-yellow)](https://huggingface.co/akpon900/llada-instruct-void)

VoidPadding lets `[VOID]` handle padding in masked diffusion language models so
that `[EOS]` can focus on semantic termination.

<p align="center">
  <img src="figures/voidpadding.png" alt="VoidPadding logo" width="360">
</p>

> **「假作真時真亦假，無為有處有還無。」 -《紅樓夢》太虛幻境**

Repeated padding can blur the boundary between emptiness and ending in MDLMs.
This ambiguity can cause `[EOS]` overflow under large-block decoding.
VoidPadding decouples the two signals:

- `[VOID]` represents padded emptiness during training.
- `[EOS]` marks semantic endings.
- `[VOID]` generation is banned at inference.

The result is more reliable termination, less wasted generation, and a learned
length signal for adaptive canvas expansion.

## Method

![VoidPadding overview](figures/void_padding_overview.png)

VoidPadding is evaluated on **LLaDA-8B-Instruct** and **Dream-7B-Instruct**.
This repository also includes scripts for appendix experiments on
**LLaDA-8B-Base** and **Dream-7B-Base**.

## Results

### Fixed-Length Evaluation

Generation length is fixed to 512 and block size is varied over 64, 128, and
512. The Avg. panel reports the mean over GSM8K, MATH500, HumanEval, and MBPP.
VoidPadding achieves block-averaged Avg. gains of +7.06/+3.59 on LLaDA and
+17.84/+6.95 on Dream over Original/RainbowPadding.

![LLaDA fixed-length robustness](figures/llada.png)

![Dream fixed-length robustness](figures/dream.png)

Average NFE reduction of VoidPadding relative to Original/RainbowPadding:

| Backbone | GSM8K | MATH500 | HumanEval |  MBPP |  Avg. |
| -------- | ----: | ------: | --------: | ----: | ----: |
| LLaDA    | 75.0% |   13.5% |     80.4% | 86.4% | 63.8% |
| Dream    | 69.3% |   27.5% |     41.3% | 84.6% | 55.7% |

### Short-Canvas Expansion

All expansion experiments use an initial response canvas length of 64 and a
decoding block size of 32.

For LLaDA, VoidExpansion improves the mean score over Fixed-64 by +11.96,
over rho-[EOS] by +5.84, and over DAEDAL by +0.57. It also reduces average NFE
from 228.82 to 172.10 relative to DAEDAL, with a 2.12x mean wall-clock speedup.

![LLaDA short-canvas expansion accuracy](figures/expansion_acc.png)

![LLaDA short-canvas expansion efficiency](figures/expansion_eff.png)

For Dream, VoidExpansion improves the mean score from 42.23 to 60.37.

| Method        | GSM8K | MATH500 | HumanEval |  MBPP |  Mean |
| ------------- | ----: | ------: | --------: | ----: | ----: |
| Fixed-64      | 53.22 |   22.80 |     43.90 | 49.00 | 42.23 |
| VoidExpansion | 79.08 |   44.00 |     60.98 | 57.40 | 60.37 |

## Quick Start

Create the environment:

```bash
bash scripts/setup_env.sh
conda activate void
```

Download released LoRA adapters:

```bash
hf download akpon900/llada-instruct-void \
  --local-dir models/llada-instruct-void

hf download akpon900/dream-instruct-void \
  --local-dir models/dream-instruct-void
```

Merge them with their base models:

```bash
BASE_MODEL="GSAI-ML/LLaDA-8B-Instruct" \
ADAPTER_PATH="models/llada-instruct-void" \
OUTPUT_DIR="outputs/merged/llada-instruct-void" \
bash scripts/merge/merge_llada_lora.sh

BASE_MODEL="Dream-org/Dream-v0-Instruct-7B" \
ADAPTER_PATH="models/dream-instruct-void" \
OUTPUT_DIR="outputs/merged/dream-instruct-void" \
bash scripts/merge/merge_dream_lora.sh
```

Run evaluation:

```bash
bash scripts/eval/fixed_length/llada/run_all_void.sh
bash scripts/eval/expansion/llada/run_all_voidexpansion.sh

bash scripts/eval/fixed_length/dream/run_all_void.sh
bash scripts/eval/expansion/dream/run_all_voidexpansion.sh
```

## Detailed Instructions

- [Checkpoints and released adapters](scripts/merge/checkpoints.md)
- [Training scripts and SFT data setup](scripts/train/README.md)
- [Evaluation scripts](scripts/eval/README.md)

## Acknowledgements

This repository builds upon and adapts components from:

- Evaluation code: [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
- Training code: [quasar529/rainbow-padding](https://github.com/quasar529/rainbow-padding)
- DAEDAL baseline: [Li-Jinsong/DAEDAL](https://github.com/Li-Jinsong/DAEDAL)
- rho-EOS baseline: [yjyddq/rho-EOS](https://github.com/yjyddq/rho-EOS)

We thank the authors for releasing their code.

## Citation

If you use this repository, please cite the VoidPadding paper.

```bibtex
@misc{voidpadding2026,
  title  = {VoidPadding: Let [VOID] Handle Padding in Masked Diffusion Language Models so that [EOS] Can Focus on Semantic Termination},
  author = {Chunyu Liu and Zhengyang Fan and Kaisen Yang and Alex Lamb},
  year   = {2026}
}
```
