# WWB Accuracy Validation — Gemma 4 INT4 Quantization

This document describes how to validate the accuracy of **INT4-quantized OpenVINO IR**
models against the **original HuggingFace (FP32/BF16)** model using the
[Who What Benchmark (WWB)](https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark)
from the `openvino.genai` repository.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Methodology](#methodology)
4. [Scripts](#scripts)
5. [How to Run](#how-to-run)
6. [Results](#results)
7. [Interpreting Metrics](#interpreting-metrics)
8. [FAQ](#faq)

---

## Overview

When exporting a model to OpenVINO IR with INT4 weight compression (e.g., `group_size=64`,
`ratio=1.0`, asymmetric), some accuracy degradation is expected.  The **Who What Benchmark
(WWB)** measures how similar the quantized model's answers are to the original model's
answers, providing a quantitative quality score.

### What is WWB?

WWB is an official accuracy-validation tool shipped with
[openvino.genai](https://github.com/openvinotoolkit/openvino.genai).
It works in three stages:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | **Ground Truth (GT)** | Generate reference answers from the **original unquantized** HuggingFace model (FP32 on CPU). These are the "correct" baseline. |
| 2 | **Target** | Generate answers from the **INT4 OpenVINO** model (via `openvino.genai` VLMPipeline on GPU). This is the model under test. |
| 3 | **Score** | Compare GT vs Target using sentence-embedding cosine similarity and token-divergence metrics. |

---

## Prerequisites

### Hardware

- Intel platform with integrated GPU (tested on Intel Panther Lake, 12 Xe EUs)
- Minimum 32 GB RAM (64 GB+ recommended for 8B+ models)

### Software

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.12+ | |
| openvino | 2026.2.0+ | Nightly or release |
| openvino-genai | 2026.2.0+ | Built from [PR #3644](https://github.com/openvinotoolkit/openvino.genai/pull/3644) for Gemma 4 support |
| transformers | 5.6.0.dev+ | Install from source (`pip install git+https://github.com/huggingface/transformers.git`) — released versions up to 5.3.0 do not support the `gemma4` model type |
| whowhatbench | (from openvino.genai) | Installed automatically when building PR #3644 |
| sentence-transformers | 5.x | For similarity scoring |
| torch | 2.x | For HF model inference on CPU |

### Models

| Model | Source | Size | Description |
|-------|--------|------|-------------|
| `gemma-4-E4B-it-hf/` | [google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) | 16 GB (BF16) | Original HF model (8B params, 4B active MoE) |
| `gemma-4-E4B-it-ov/` | Export via `optimum-cli` | ~4 GB (INT4) | OpenVINO IR, INT4 group_size=64 asymmetric |

Download the original HF model:

```powershell
huggingface-cli download google/gemma-4-E4B-it --local-dir ./gemma-4-E4B-it-hf
```

---

## Methodology

### Similarity Metric

- **Model**: `sentence-transformers/all-mpnet-base-v2`
  - This is the **official default** scorer used by WWB's `TextSimilarity` class
  - 768-dimensional embeddings trained on 1B+ sentence pairs
  - Standard in the NLP community for semantic textual similarity
- **Computation**: Cosine similarity between GT and Target sentence embeddings
- **Range**: 0.0 (completely different) to 1.0 (identical meaning)

### Token Divergence Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| **FDT** | First Divergent Token — position where GT and Target first differ | Higher ↑ |
| **FDT norm** | FDT / max(len(gt), len(target)) | Higher ↑ (best = 1.0) |
| **SDT** | Sum of Divergent Tokens — total number of token mismatches | Lower ↓ (best = 0) |
| **SDT norm** | SDT / max(len(gt), len(target)) | Lower ↓ (best = 0.0) |

### Quality Thresholds

| Similarity | Verdict |
|------------|---------|
| ≥ 0.97 | **Excellent** — virtually lossless quantization |
| ≥ 0.95 | **Good** — minor wording differences, same meaning |
| ≥ 0.90 | **Acceptable** — noticeable differences but semantically correct |
| < 0.90 | **Poor** — may need re-quantization with different settings |

---

## Scripts

This project includes two WWB scripts:

### `run_wwb_gemma4.py` — Custom Prompts (16 diverse topics)

Uses 16 handcrafted prompts covering science, coding, literature, economics,
biology, and technology.  Good for quick validation.

### `run_wwb_builtin.py` — Official WWB Built-in Prompts (27 who/what)

Uses WWB's built-in `TextEvaluator` with its default **27 English prompts**
(short factual who/what questions like "Who is Mark Twain?", "What is Python?").
This is the **standard evaluation** used across all openvino.genai model validations.

---

## How to Run

### Option A: Built-in 27 Prompts (Recommended)

```powershell
# Activate the venv with openvino-genai + whowhatbench
.\__envs_genai\Scripts\Activate.ps1

# Step 1 — Generate ground truth (CPU, slow: ~60 min for 27 prompts)
python run_wwb_builtin.py --step gt --hf-model ./gemma-4-E4B-it-hf

# Step 2 — Generate target answers (GPU, fast: ~5 min for 27 prompts)
python run_wwb_builtin.py --step target --ov-model ./gemma-4-E4B-it-ov

# Step 3 — Compute metrics
python run_wwb_builtin.py --step score --hf-model ./gemma-4-E4B-it-hf
```

**CLI arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--step` | (required) | `gt`, `target`, `score`, or `all` |
| `--hf-model` | — | Path to original HF model directory |
| `--ov-model` | — | Path to OpenVINO INT4 model directory |
| `--gt-csv` | `wwb_builtin_gt.csv` | Output CSV for ground truth |
| `--target-csv` | `wwb_builtin_target.csv` | Output CSV for target answers |
| `--max-new-tokens` | 128 | Max tokens per answer |
| `--num-samples` | all 27 | Limit number of prompts |
| `--device` | GPU | Device for OV inference |

### Option B: Custom 16 Prompts (Quick Check)

```powershell
python run_wwb_gemma4.py --step gt     --hf-model ./gemma-4-E4B-it-hf --ov-model ./gemma-4-E4B-it-ov
python run_wwb_gemma4.py --step target --hf-model ./gemma-4-E4B-it-hf --ov-model ./gemma-4-E4B-it-ov
python run_wwb_gemma4.py --step score  --hf-model ./gemma-4-E4B-it-hf --ov-model ./gemma-4-E4B-it-ov
```

---

## Results

### Test Configuration

| Item | Value |
|------|-------|
| Platform | Intel Panther Lake (12 Xe EUs) |
| RAM | 102.5 GB |
| Original model | `google/gemma-4-E4B-it` (8B params, BF16) |
| Quantized model | `gemma-4-E4B-it-ov` (INT4, group_size=64, asymmetric, ratio=1.0) |
| GT device | CPU (FP32) |
| Target device | GPU (INT4) |
| Scoring model | `sentence-transformers/all-mpnet-base-v2` |

### Custom 16 Prompts — `run_wwb_gemma4.py` (8 samples, max_new_tokens=512)

| Metric | Value |
|--------|-------|
| **Similarity** | **0.9191** |
| FDT | 5.6 |
| FDT norm | 0.0142 |
| SDT | 444.9 |
| SDT norm | 0.9646 |

Per-prompt similarity breakdown:

| # | Prompt | Similarity |
|---|--------|-----------|
| 1 | Explain quantum computing in simple terms | 0.8506 |
| 2 | Differences between Python and JavaScript | 0.9438 |
| 3 | Write a short poem about the ocean | 0.8693 |
| 4 | Describe the process of photosynthesis | 0.9266 |
| 5 | Benefits of regular exercise | 0.9421 |
| 6 | Explain how a neural network works | 0.9418 |
| 7 | What is the theory of relativity? | 0.9411 |
| 8 | Describe the water cycle in nature | 0.9375 |

**Verdict**: Acceptable INT4 quantization quality (similarity ≥ 0.90)

> Note: High SDT norm (0.96) indicates different wording but same semantic meaning.
> Worst cases are creative/abstract prompts (quantum computing, poetry) where
> multiple valid phrasings exist.

### Built-in 27 Prompts — `run_wwb_builtin.py` (all 27, max_new_tokens=128)

| Metric | Value |
|--------|-------|
| **Similarity** | **0.9451** |
| FDT | 6.1 |
| FDT norm | 0.0494 |
| SDT | 78.4 |
| SDT norm | 0.6128 |

Per-prompt similarity breakdown:

| # | Prompt | Similarity |
|---|--------|-----------|
| 1 | Who is Mark Twain? | 0.9697 |
| 2 | Who is William Shakespeare? | 0.9607 |
| 3 | Who is Agatha Christie? | 0.9271 |
| 4 | Who is Barbara Cartland? | 0.9676 |
| 5 | Who is Danielle Steel? | 0.9739 |
| 6 | Who is Harold Robbins? | 0.9561 |
| 7 | Who is Georges Simenon? | 0.8917 |
| 8 | Who is Enid Blyton? | 0.9704 |
| 9 | Who is Sidney Sheldon? | 0.9293 |
| 10 | Who is Akira Toriyama? | 0.9759 |
| 11 | Who is Leo Tolstoy? | 0.9200 |
| 12 | Who is Alexander Pushkin? | 0.9512 |
| 13 | Who is Stephen King? | 0.9716 |
| 14 | What is C++? | 0.9733 |
| 15 | What is Python? | 0.9794 |
| 16 | What is Java? | 0.9522 |
| 17 | What is JavaScript? | 0.9326 |
| 18 | What is Perl? | 0.9168 |
| 19 | What is OpenCV? | 0.9552 |
| 20 | Who is the most famous writer? | 0.8663 |
| 21 | Who is the most famous inventor? | 0.9133 |
| 22 | Who is the most famous mathematician? | 0.9445 |
| 23 | Who is the most famous composer? | 0.9081 |
| 24 | Who is the most famous programmer? | 0.8991 |
| 25 | Who is the most famous athlete? | 0.9434 |
| 26 | Who is the most famous ancient Greek scientist? | 0.9673 |
| 27 | What color will you get when you mix blue and yellow? | 1.0000 |

**Verdict**: Good overall INT4 quantization quality (similarity = 0.9451)

> **Key observations:**
> - 24 of 27 prompts score ≥ 0.90 (89%)
> - 15 of 27 prompts score ≥ 0.95 (56%)
> - Perfect score (1.0) on simple factual question (#27 blue+yellow=green)
> - Worst: "Who is the most famous writer?" (0.8663) — subjective questions
>   where both GT and INT4 produce valid but differently-worded responses
> - SDT norm 0.61 (moderate) means different wording but same semantics
> - Factual who/what questions (people, languages) score consistently high

---

## Interpreting Metrics

### Why is the similarity score not 1.0?

INT4 quantization compresses model weights from 16-bit floats to 4-bit integers.
This inevitably changes the probability distribution over tokens. The model often
says the same thing using different words. For example:

- **GT**: "Quantum computing uses qubits that can exist in superposition..."
- **INT4**: "Quantum computing leverages qubits that can be in multiple states..."

Both are correct but use different vocabulary, resulting in similarity < 1.0.

### Similarity vs Token Divergence

| Scenario | Similarity | SDT norm | Interpretation |
|----------|-----------|----------|----------------|
| High sim, low SDT | ≥ 0.97 | < 0.1 | Nearly identical output |
| High sim, high SDT | ≥ 0.90 | > 0.5 | Different words, same meaning |
| Low sim, low SDT | < 0.90 | < 0.3 | Minor factual error early on |
| Low sim, high SDT | < 0.90 | > 0.5 | Significantly different content |

### How many prompts are enough?

| Count | Use Case |
|-------|----------|
| 8–16 | Quick sanity check during development |
| 27 | Standard WWB evaluation (recommended) |
| 50+ | Rigorous validation for publication |
| 1000+ | Academic benchmark (MMLU, HellaSwag, etc.) |

---

## FAQ

**Q: Why `sentence-transformers/all-mpnet-base-v2`?**
A: It is the official default scorer in WWB's `TextSimilarity` class. It is the
highest-quality general English sentence embedding model in the `sentence-transformers`
library. For multilingual (e.g., Chinese), use `paraphrase-multilingual-MiniLM-L12-v2`.

**Q: Is `google/gemma-4-E4B-it` the correct baseline?**
A: Yes. It has 7,996,156,490 parameters (8B total, 4B active MoE), Apache-2.0 license,
and is not gated (no access request needed). The INT4 OV model was exported from this
exact checkpoint.

**Q: Can I test other Gemma 4 variants?**
A: Yes — replace `--hf-model` and `--ov-model` with the corresponding directories:

| Variant | HF Model ID | OV Dir |
|---------|-------------|--------|
| E2B | `google/gemma-4-E2B-it` | `gemma-4-E2B-it-ov/` |
| E4B | `google/gemma-4-E4B-it` | `gemma-4-E4B-it-ov/` |
| 26B-A4B | `google/gemma-4-26B-A4B-it` | `gemma-4-26B-A4B-it-ov/` |
| 31B | `google/gemma-4-31B-it` | `gemma-4-31B-it-ov/` |

**Q: Why does GT generation take so long?**
A: The original HF model runs on CPU in FP32 (16 GB for E4B). At ~3–4 tok/s on CPU,
generating 128 tokens × 27 prompts takes about 45–60 minutes. You only need to run
this once — the GT CSV can be reused for testing multiple INT4 configurations.
