## LLM FINNTUNNING REPO

A systematic benchmark of supervised fine-tuning, parameter-efficient fine-tuning, preference optimization, and reinforcement-learning methods for small language models under limited GPU resources.

## Results at a Glance

| # | Method | GSM8K Accuracy ↑ | Valid Format ↑ | Trainable Parameters | Peak VRAM ↓ | Training / Evaluation Time ↓ |
|---|---|---:|---:|---:|---:|---:|
| 1 | **Zero-shot Base Model** | **49.13%** | **83.09%** | 0 | — | 22.30 min |
| 2 | Full SFT | — | — | — | — | — |
| 3 | LoRA-SFT | — | — | — | — | — |
| 4 | QLoRA-SFT | — | — | — | — | — |
| 5 | DPO + LoRA | — | — | — | — | — |
| 6 | GRPO + LoRA | — | — | — | — | — |
| 7 | RLOO + LoRA | — | — | — | — | — |

> The table will be updated as each experiment is completed. All methods are evaluated using the same core reasoning benchmark so that improvements can be compared directly.

---

## 1. Introduction

Large Language Models can be adapted to downstream tasks using several different post-training strategies. Full supervised fine-tuning updates all model parameters, while parameter-efficient methods such as LoRA and QLoRA train only a small fraction of the model. Preference-based methods such as DPO optimize the model using preferred and rejected responses, while online reinforcement-learning methods such as GRPO and RLOO optimize the model from rewards obtained from newly generated responses during training.

The objective of this project is to build a **controlled benchmark of these methods on a small language model that can be trained and evaluated on Kaggle GPUs**.

The project focuses on the following question:

> **Which fine-tuning or post-training method provides the best reasoning performance while remaining efficient in terms of trainable parameters, GPU memory, training time, and model storage?**

The initial benchmark uses:

- **Base model:** `Qwen/Qwen3-0.6B-Base`
- **Primary dataset:** GSM8K
- **Primary metric:** Exact Match Accuracy
- **Environment:** Kaggle GPU
- **Task:** Mathematical reasoning with automatically verifiable final answers

Using one model and one dataset initially makes the comparison easier to interpret. The main variable changed between experiments is the training or post-training strategy.

### Initial Experiment Plan

The first version of the benchmark contains seven experiments:

1. Zero-shot base model
2. Full supervised fine-tuning
3. LoRA supervised fine-tuning
4. QLoRA supervised fine-tuning
5. DPO starting from the LoRA-SFT checkpoint
6. GRPO starting from the same LoRA-SFT checkpoint
7. RLOO starting from the same LoRA-SFT checkpoint

The experimental structure is:

```text
                         Qwen3-0.6B-Base
                                |
          +---------------------+---------------------+
          |                     |                     |
       Full SFT              LoRA-SFT             QLoRA-SFT
                                |
                       save common checkpoint
                                |
                    +-----------+-----------+
                    |           |           |
                   DPO         GRPO        RLOO
