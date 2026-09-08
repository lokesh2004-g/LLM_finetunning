# Qwen3-0.6B-Base Fine-Tuning and Post-Training Benchmark

A systematic benchmark of parameter-efficient supervised fine-tuning, preference optimization, and reinforcement-learning methods for small language models under limited compute resources.

## Results at a Glance

| # | Method                   | GSM8K Accuracy ↑ | Valid Format ↑ | Trainable Parameters ↓ | Training / Evaluation Time ↓ |
| - | ------------------------ | ---------------: | -------------: | ---------------------: | ---------------------------: |
| 1 | **Zero-shot Base Model** |       **49.13%** |     **83.09%** |                      0 |                    22.30 min |
| 2 | **LoRA-SFT (r=8)**       |       **52.84%** |     **97.57%** | **5,046,272 (~0.84%)** |           **61.46 min eval** |
| 3 | QLoRA-SFT                |                — |              — |                      — |                            — |
| 4 | DPO + LoRA               |                — |              — |                      — |                            — |
| 5 | GRPO + LoRA              |                — |              — |                      — |                            — |
| 6 | RLOO + LoRA              |                — |              — |                      — |                            — |

> The table will be updated as each experiment is completed. All methods are evaluated using the same reasoning benchmark so that their performance and training efficiency can be compared directly.

---

## 1. Introduction

Large Language Models can be adapted to downstream tasks using several different post-training strategies.

Parameter-efficient methods such as LoRA and QLoRA update only a small fraction of the model parameters while keeping most of the pretrained model frozen.

Preference-based methods such as DPO optimize the model using preferred and rejected responses, while online reinforcement-learning methods such as GRPO and RLOO optimize the model using rewards obtained from responses generated during training.

The objective of this project is to build a **controlled benchmark of these methods on a small language model that can be trained and evaluated on Kaggle**.

The project focuses on the following question:

> **Which fine-tuning or post-training method provides the best reasoning performance while remaining efficient in terms of trainable parameters and training time?**

The initial benchmark uses:

* **Base model:** `Qwen/Qwen3-0.6B-Base`
* **Primary dataset:** GSM8K
* **Primary metric:** Exact Match Accuracy
* **Environment:** Kaggle
* **Task:** Mathematical reasoning with automatically verifiable final answers

Using one model and one dataset initially makes the comparison easier to interpret. The main variable changed between experiments is the training or post-training strategy.

---

## 2. Initial Experiment Plan

The first version of the benchmark contains six experiments:

1. Zero-shot base model
2. LoRA supervised fine-tuning
3. QLoRA supervised fine-tuning
4. DPO starting from the LoRA-SFT checkpoint
5. GRPO starting from the same LoRA-SFT checkpoint
6. RLOO starting from the same LoRA-SFT checkpoint

The experimental structure is:

```text
                         Qwen3-0.6B-Base
                                |
                    +-----------+-----------+
                    |                       |
                 LoRA-SFT                QLoRA-SFT
                    |
            save common checkpoint
                    |
        +-----------+-----------+
        |           |           |
       DPO         GRPO        RLOO
```

---

## 3. Baseline Evaluation — Zero-Shot

Before applying any fine-tuning method, we evaluate the original **Qwen3-0.6B-Base** model on the complete GSM8K test set. No model parameters are updated during this experiment.

### Evaluation Setup

| Setting              | Value                  |
| -------------------- | ---------------------- |
| Model                | `Qwen/Qwen3-0.6B-Base` |
| Dataset              | GSM8K                  |
| Test Samples         | 1,319                  |
| Evaluation           | Zero-shot CoT          |
| Batch Size           | 32                     |
| Max New Tokens       | 512                    |
| Total Parameters     | 596,049,920            |
| Trainable Parameters | 0                      |

Each problem is solved using step-by-step generation, with the model instructed to return its final result as:

```text
Final Answer: <number>
```

### Zero-Shot Results

| Metric                     |          Result |
| -------------------------- | --------------: |
| GSM8K Exact Match Accuracy |      **49.13%** |
| Valid Final Answer Format  |      **83.09%** |
| Correct Answers            | **648 / 1,319** |
| Evaluation Time            |   **22.30 min** |

The zero-shot experiment establishes the reasoning capability of the original model before any task-specific training.

This result is used as the common baseline for measuring the improvement obtained from each fine-tuning and post-training method.

---

## 4. LoRA Supervised Fine-Tuning

The second experiment fine-tunes `Qwen/Qwen3-0.6B-Base` on GSM8K using **LoRA with rank 8**.

The original model weights remain frozen and only the low-rank adapter parameters are trained.

### Setup

| Setting               | Value                  |
| --------------------- | ---------------------- |
| Base Model            | `Qwen/Qwen3-0.6B-Base` |
| Dataset               | GSM8K                  |
| Method                | LoRA-SFT               |
| LoRA Rank (`r`)       | **8**                  |
| Trainable Parameters  | **5,046,272**          |
| Total Parameters      | **601,096,192**        |
| Trainable Percentage  | **0.8395% (~0.84%)**   |
| Test Samples          | 1,319                  |
| Evaluation Batch Size | 32                     |

LoRA adapters are applied to:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

During SFT, GSM8K questions are used as prompts and their step-by-step solutions are used as targets. Prompt tokens are masked from the loss so that optimization focuses on the generated solution tokens.

### LoRA-SFT Results

| Metric                  |   Zero-Shot |  LoRA-SFT (r=8) |        Change |
| ----------------------- | ----------: | --------------: | ------------: |
| GSM8K Accuracy          |      49.13% |      **52.84%** |  **+3.71 pp** |
| Valid Format            |      83.09% |      **97.57%** | **+14.48 pp** |
| Correct Answers         | 648 / 1,319 | **697 / 1,319** |       **+49** |
| Trainable Parameters    |           0 |   **5,046,272** |        ~0.84% |
| Evaluation Time         |   22.30 min |   **61.46 min** |             — |
| Average Time / Question |           — |     **2.796 s** |             — |

The rank-8 LoRA model improves GSM8K accuracy from:

```text
49.13% → 52.84%
```

corresponding to an absolute improvement of:

```text
+3.71 percentage points
```

while training only about:

```text
0.84% of the model parameters
```

The valid output-format rate also improves substantially:

```text
83.09% → 97.57%
```

The resulting LoRA-SFT checkpoint is used as the common starting point for DPO, GRPO, and RLOO experiments.

```text
                 LoRA-SFT (r=8)
                       |
             +---------+---------+
             |         |         |
            DPO       GRPO      RLOO
```

---

## 5. QLoRA Supervised Fine-Tuning

QLoRA will evaluate whether the memory requirements of LoRA fine-tuning can be reduced further by quantizing the frozen base model while training low-rank adapters.

The experiment will use the same GSM8K training and evaluation protocol so that its accuracy, trainable parameters, and computational efficiency can be directly compared with LoRA-SFT.

Results will be added after the experiment is completed.

---

## 6. DPO + LoRA

Direct Preference Optimization will start from the trained LoRA-SFT checkpoint.

Preference pairs containing preferred and rejected mathematical reasoning responses will be used to optimize the model toward better solution behavior without requiring a separate reward-model training stage.

Results will be added after the experiment is completed.

---

## 7. GRPO + LoRA

Group Relative Policy Optimization will start from the common LoRA-SFT checkpoint.

The model will generate multiple candidate responses for each GSM8K problem, and automatically verifiable rewards based on mathematical correctness and output formatting will be used for optimization.

Results will be added after the experiment is completed.

---

## 8. RLOO + LoRA

REINFORCE Leave-One-Out will provide another online reinforcement-learning comparison using the same LoRA-SFT starting checkpoint and reward formulation.

This experiment will allow GRPO and RLOO to be compared under a controlled model, dataset, and evaluation setup.

Results will be added after the experiment is completed.

---

## 9. Benchmark Objective

After completing all experiments, the final comparison will evaluate each method using:

* GSM8K Exact Match Accuracy
* Valid Answer Format
* Number of Trainable Parameters
* Training Time
* Evaluation Time
* Relative Improvement over the Base Model

The goal is not only to identify the method with the highest GSM8K accuracy, but also to understand the trade-off between **reasoning performance and training efficiency** for small language models under limited compute resources.
