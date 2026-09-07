# Qwen3-0.6B-Base Fine-Tuning and Post-Training Benchmark

A systematic benchmark of parameter-efficient supervised fine-tuning, preference optimization, and reinforcement-learning methods for small language models under limited compute resources.

## Results at a Glance

| # | Method                   | GSM8K Accuracy ↑ | Valid Format ↑ | Trainable Parameters ↓ | Training / Evaluation Time ↓ |
| - | ------------------------ | ---------------: | -------------: | ---------------------: | ---------------------------: |
| 1 | **Zero-shot Base Model** |       **49.13%** |     **83.09%** |                      0 |                    22.30 min |
| 2 | **LoRA-SFT**             |       **50.72%** |     **98.56%** |     **10.09M (~1.7%)** |                     3hours 15 min |
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

The second experiment applies **Low-Rank Adaptation (LoRA)** to `Qwen/Qwen3-0.6B-Base` using the GSM8K training set.

Instead of updating all parameters of the model, LoRA keeps the original model weights frozen and introduces small trainable low-rank matrices into selected transformer layers.

This reduces the number of trainable parameters while allowing the model to adapt to mathematical reasoning examples.

### Training Setup

| Setting                     | Value                  |
| --------------------------- | ---------------------- |
| Base Model                  | `Qwen/Qwen3-0.6B-Base` |
| Dataset                     | GSM8K                  |
| Fine-Tuning Method          | LoRA-SFT               |
| LoRA Rank (`r`)             | 16                     |
| LoRA Alpha                  | 32                     |
| LoRA Dropout                | 0.05                   |
| Learning Rate               | `2e-4`                 |
| Epochs                      | 3                      |
| Per-Device Batch Size       | 4                      |
| Gradient Accumulation Steps | 8                      |
| Effective Batch Size        | 32                     |
| Trainable Parameters        | ~10.09M                |
| Percentage of Model Trained | ~1.7%                  |

LoRA adapters are applied to both the attention and MLP projection layers:

```text
q_proj
k_proj
v_proj
o_proj
gate_proj
up_proj
down_proj
```

During supervised fine-tuning, the model receives the GSM8K question as input and the corresponding step-by-step solution as the training target.

Loss is calculated only on the generated solution tokens. Prompt tokens are masked from the loss so that optimization focuses on learning the reasoning response.

The expected response format remains:

```text
Final Answer: <number>
```

### LoRA-SFT Results

| Metric                     | Zero-Shot |    LoRA-SFT |         Change |
| -------------------------- | --------: | ----------: | -------------: |
| GSM8K Exact Match Accuracy |    49.13% |  **50.72%** |   **+1.59 pp** |
| Valid Final Answer Format  |    83.09% |  **98.56%** |  **+15.47 pp** |
| Trainable Parameters       |         0 | **~10.09M** | ~1.7% of model |

LoRA-SFT improves GSM8K exact-match accuracy from:

```text
49.13% → 50.72%
```

corresponding to an absolute improvement of:

```text
+1.59 percentage points
```

The improvement in valid answer formatting is substantially larger:

```text
83.09% → 98.56%
```

This shows that supervised fine-tuning strongly improves the model's ability to follow the required response format.

However, the improvement in exact mathematical accuracy is relatively small. The result suggests that LoRA-SFT adapts the model strongly to the GSM8K response structure while producing a more limited improvement in mathematical reasoning performance.

This LoRA-SFT checkpoint is used as the common starting point for the subsequent DPO, GRPO, and RLOO experiments.

```text
                  LoRA-SFT
                     |
          +----------+----------+
          |          |          |
         DPO        GRPO       RLOO
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
