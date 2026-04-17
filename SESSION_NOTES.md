# Session Notes — Gemma 4 OpenVINO GenAI Project

> **Date range:** April 2026  
> **System:** Intel Panther Lake, Arc B390 iGPU (96 EUs), Windows  
> **Memory:** LPDDR5 8533 MT/s — tested at 32 GB (full) and **16 GB** (BIOS iGPU 13 GB override)  
> **Python:** 3.12.0  
> **Repo:** https://github.com/jlee52tw/gemma4-openvino-genai  
> **Latest commit:** `1007c95` (as of April 16, 2026)

---

## 1. Environment Setup

### Venv: `__envs_genai`

Location: `C:\Users\Local_Admin\Documents\John_gemma4_31b\__envs_genai`  
(Also cloned at `C:\working\gemma4\gemma4-openvino-genai\`)

Key packages:
- **openvino** 2026.2.0.dev20260411 (nightly)
- **openvino-genai** 2026.2.0.0 — built from **PR#3644** branch `as/vlm_enable_1`
  - Source: `https://github.com/as-suvorov/openvino.genai.git`
  - Commit: `4f296476`
  - Cloned to: `openvino_genai_pr3644/`
  - Build command: `python -m pip install . --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release -v`
- **transformers** 5.6.0.dev0 (from source, needed for Gemma4 model class)
- **whowhatbench**, **sentence-transformers** 5.3.0, **torch** 2.11.0
- **psutil** (for memory measurement)

### Proxy (internal Intel network)
```
$env:http_proxy = "http://proxy-dmz.intel.com:912"
$env:https_proxy = "http://proxy-dmz.intel.com:912"
$env:no_proxy = ".intel.com,intel.com,localhost,127.0.0.1"
```

### Build tools
- Visual Studio 2022 Community (MSVC 14.44)
- CMake 3.23.5

---

## 2. Models

All models exported with `optimum-cli export openvino`:
```powershell
optimum-cli export openvino --model google/gemma-4-<variant>-it --weight-format int4 --group-size 64 --ratio 1.0 gemma-4-<variant>-it-ov
```

| Model | Variant | INT4 Disk Size | Notes |
|-------|---------|---------------|-------|
| gemma-4-E2B-it-ov | 2B equivalent | ~2 GB | Fastest |
| gemma-4-E4B-it-ov | 4B/8B equivalent | ~6 GB | Main test model |
| gemma-4-26B-A4B-it-ov | 26B MoE (4B active) | ~14 GB | MoE architecture |
| gemma-4-31B-it-ov | 31B dense | ~17 GB | Largest |

Model locations:
- `C:\Users\Local_Admin\Documents\John_gemma4_31b\gemma-4-*-it-ov\`
- `C:\working\models\gemma-4-*-it-ov\` (alternate location, same files)

HF original E4B (BF16): `gemma-4-E4B-it-hf/` — 7,996,156,490 params (16.0 GB)

---

## 3. GPU Benchmark Results (openvino.genai VLMPipeline)

All 4 models × 3 prompt types × 3 runs = **36/36 passed** on GPU.

| Model | Throughput (tok/s) | Device |
|-------|--------------------|--------|
| E2B | 34.7 | GPU |
| E4B | 24.0 | GPU |
| 26B-A4B | 5.80 | GPU |
| 31B | 5.06 | GPU |

Benchmark script: `benchmark.py` (multi-model, 3 prompt types, CSV output)

---

## 4. E4B Model Size Investigation (Key Finding)

### Why is E4B OV INT4 ~6 GB instead of expected 4-5 GB?

**Root cause: Gemma 4's unique per-layer embedding architecture**

File size breakdown (OV IR binaries):

| Component | File | Size | Precision | Shape |
|-----------|------|------|-----------|-------|
| Per-layer embeddings | `text_embeddings_per_layer_model.bin` | **2,689 MB** | INT8 (i8) + FP16 scales | `[262144, 10752]` = vocab × (256 × 42) |
| Language model (transformer) | `language_model.bin` | **2,685 MB** | INT4 (u4) group_size=64 | ~4.5B params |
| Main text embeddings | `text_embeddings_model.bin` | **641 MB** | INT8 (i8) + FP16 scales | `[262144, 2560]` = vocab × hidden |
| Vision encoder | `vision_embeddings_model.bin` | **162 MB** | INT8 (i8) + FP16 scales | SigLIP 16 layers |
| Tokenizer + other | | ~52 MB | | |
| **Total** | | **~6,229 MB (6.09 GB)** | | |

**Key architecture detail:** Gemma 4 has `embed_tokens_per_layer` — each of the 42 transformer layers has its own 256-dim embedding table mapping from the full 262K vocabulary:
- `vocab_size_per_layer_input` = 262,144
- `hidden_size_per_layer_input` = 256
- `num_hidden_layers` = 42
- **Total per-layer embedding params** = 262,144 × 256 × 42 = **2,818,572,288** (2.82B)

These 2.82B params are stored as **INT8** (not INT4) because embedding tables degrade at INT4. The `openvino_config.json` quantization config only targets `lm_model` (transformer layers) with INT4.

**Summary:**

| | Params | If all INT4 | Actual |
|---|---|---|---|
| Per-layer embeddings | 2.82B | 1.41 GB | **2.69 GB (INT8)** |
| Main embeddings | 0.67B | 0.34 GB | **0.64 GB (INT8)** |
| Transformer layers | ~4.5B | 2.25 GB | **2.69 GB (INT4+scales)** |
| Vision encoder | ~0.08B | 0.04 GB | **0.16 GB (INT8)** |
| **Total** | **~8.07B** | **~4.04 GB** | **~6.18 GB** |

**Conclusion:** 6 GB is correct and expected. The 43% of parameters that are embedding tables are kept at INT8 (not INT4), adding ~1.75 GB over an "all INT4" estimate. This is by design.

---

## 5. Memory Measurement (mmap vs no-mmap)

Added `--no-mmap` and `--show-memory` flags to both Python and C++ scripts.

### E4B GPU results

| Mode | RSS after load | RSS after gen | Peak |
|------|---------------|--------------|------|
| Default (mmap) | 7,091 MB | 7,170 MB | 12,466 MB |
| `--no-mmap` | 13,479 MB | 13,563 MB | 13,777 MB |

### Why `--no-mmap` uses more RSS

- **With mmap (default):** OS maps `.bin` files into virtual address space. Pages are file-backed — OS can evict and re-read from disk. After GPU upload, OS reclaims mmap pages → RSS drops.
- **Without mmap:** `np.fromfile()` loads into heap (anonymous pages). No backing file — OS won't reclaim. Heap buffers stay resident even after GPU upload.

### Implementation detail

**Updated 2026-04-17:** The `ENABLE_MMAP` property can be passed directly as a
**keyword argument** to the VLMPipeline constructor. This is far simpler than
the previous manual `np.fromfile()` workaround.

```python
# Python — pass ENABLE_MMAP as a kwarg (uppercase, string value)
pipe = ov_genai.VLMPipeline(str(model_dir), device, ENABLE_MMAP='NO')
```

```cpp
// C++ — pass via ov::AnyMap properties
ov::AnyMap props;
props["ENABLE_MMAP"] = false;
ov::genai::VLMPipeline pipe(model_dir, device, props);
```

> **Note:** `enable_mmap=False` (lowercase / bool) does NOT work in Python —
> it raises `Option not found: enable_mmap`. The correct form is
> `ENABLE_MMAP='NO'` (uppercase key, string `'NO'`).  Passing a dict as a
> positional argument also fails; it must be a `**kwarg`.

---

## 6. Audio Capability Research

### Architecture support

Gemma 4 E2B and E4B have an `audio_tower` in config.json:
- `model_type`: `gemma4_audio`
- `hidden_size`: 1024, `num_hidden_layers`: 12, `num_attention_heads`: 8
- `output_proj_dims`: 1536
- Estimated params: ~300M
- Audio limit: **30 seconds per clip** (750 tokens × 40ms/token)
- The 30s is a **per-request clip limit**, not overall context. Audio tokens (up to 750) consume part of the 8192-token text context window.

### NOT supported on OpenVINO

- `optimum-intel` export **skips** the audio tower — no `openvino_audio_embeddings_model.xml` produced
- VLMPipeline C++ code has **no audio processing path**
- Only text + vision work on OV currently

### Which models have audio

- E2B: yes (in architecture)
- E4B: yes (in architecture)
- 26B-A4B: no
- 31B: no

---

## 7. WWB Accuracy Validation

### Official 27-prompt evaluation (recommended)

Script: `run_wwb_builtin.py` — uses `whowhatbench.TextEvaluator` with built-in text prompts.

**Result: Similarity = 0.9451** (27 prompts, E4B, INT4 vs BF16 HF reference, max_new_tokens=128)

Steps:
```powershell
# Step 1: Generate ground truth (BF16 HF model) — takes ~45 min
python run_wwb_builtin.py --step gt --hf-model gemma-4-E4B-it-hf --max-new-tokens 128 --gt-csv wwb_builtin_gt.csv

# Step 2: Generate target (OV INT4 on GPU) — takes ~5 min
python run_wwb_builtin.py --step target --ov-model gemma-4-E4B-it-ov --gt-csv wwb_builtin_gt.csv --target-csv wwb_builtin_target.csv --max-new-tokens 128 --device GPU

# Step 3: Compare
python run_wwb_builtin.py --step compare --gt-csv wwb_builtin_gt.csv --target-csv wwb_builtin_target.csv
```

### Custom 16-prompt evaluation (for reference)

Script: `run_wwb_gemma4.py` — 16 curated prompts covering code, math, reasoning, etc.

**Result: Similarity = 0.9191** (8 samples, E4B)

---

## 8. OCR / Multimodal Tests

Tested on E4B with GPU:
- **English OCR**: Receipt image → extracted all text fields correctly, 24.25 tok/s
- **Chinese chart reasoning**: Bar chart with Chinese labels → correct data extraction + analysis, 22.99 tok/s

Results documented in `ocr-test-result.md`.

---

## 9. C++ Version

Location: `cpp/` subfolder

Files: `run_gemma4.cpp`, `load_image.cpp`, `load_image.hpp`, `CMakeLists.txt`

Build (requires setupvars):
```powershell
# Run create_release.ps1 first to set up openvino_genai_release/
.\create_release.ps1

# Then build
cd cpp
cmake -B build -DOpenVINOGenAI_DIR=..\openvino_genai_release\runtime\cmake
cmake --build build --config Release
```

Same features as Python: `--no-mmap`, `--show-memory`, `--prompt-file`, `--image`, PerfMetrics.

C++ memory measurement uses Windows `GetProcessMemoryInfo` API (`psapi.h`).

---

## 10. Project File Index

| File | Purpose |
|------|---------|
| `run_gemma4.py` | Main inference script (Python) with VLMPipeline |
| `benchmark.py` | Multi-model benchmark, 3 prompt types, CSV output |
| `cpp/run_gemma4.cpp` | C++ inference with VLMPipeline |
| `cpp/load_image.cpp/hpp` | stb_image-based image loader for C++ |
| `cpp/CMakeLists.txt` | CMake build for C++ |
| `run_wwb_builtin.py` | Official WWB 27-prompt evaluator |
| `run_wwb_gemma4.py` | Custom WWB 16-prompt evaluator |
| `wwb-accuracy-validation.md` | WWB results documentation (27-prompt only) |
| `ocr-test-result.md` | OCR multimodal test results |
| `create_release.ps1` | Creates openvino_genai_release/ for C++ builds |
| `requirements.txt` | Python dependencies |
| `requirements-export.txt` | Dependencies for model export |
| `README.md` | Full project documentation |

---

## 11. Commit History

```
1007c95 Add --no-mmap and --show-memory flags to Python and C++ scripts
920d7fa Simplify WWB doc: keep only official 27-prompt results, fix RAM to 96 GB
5d8e945 Add WWB built-in evaluator script and accuracy validation docs
8332608 Add WWB accuracy comparison script for INT4 quantization validation
549c9a2 Add Chinese chart reasoning test to OCR results (E4B, 22.99 tok/s)
ac7a46d Add OCR multimodal test: image + E4B result (24.25 tok/s)
24a8379 Simplify perf metrics: use .mean directly, remove fmt_ms/fmt_tok_s/iPOT
64e49cc Add create_release.ps1 and setupvars-based C++ workflow
72c31f4 README: fix C++ build instructions with actual paths, add DLL setup
e05794a Add C++ version of run_gemma4 (cpp/ subfolder with CMake build)
cbf96a3 run_gemma4: add --prompt-file to read prompt from a text file
32477cb run_gemma4: add openvino.genai PerfMetrics
0ecc5a9 README: Windows-only — remove Linux/macOS refs, PowerShell blocks
985d15a README: rewrite env setup with step-by-step deployment guide
ade91a5 README: add git clone step (1.1) and renumber sections
57e1fc9 README: expand build instructions with prerequisites, rebuild steps
c71c4b2 README: remove CPU example, clarify GPU (iGPU) is the target device
dc1f2ad Initial commit: Gemma 4 OpenVINO GenAI VLMPipeline example and benchmark
```

---

## 12. Known Issues & Gotchas

1. **Two repo copies exist** — `C:\Users\Local_Admin\Documents\John_gemma4_31b\gemma4-openvino-genai\` and `C:\working\gemma4\gemma4-openvino-genai\`. Keep them in sync with `git pull`.

2. **VLMPipeline `ENABLE_MMAP`** — Use `ENABLE_MMAP='NO'` as a **kwarg** (Python) or `ov::AnyMap` entry (C++). The lowercase `enable_mmap=False` does NOT work. See Section 5 for details.

3. **openvino-genai is from PR#3644** — Not yet merged to master. Must be built from `as-suvorov/openvino.genai.git` branch `as/vlm_enable_1`.

4. **transformers must be dev version** — Gemma4 model class (`Gemma4ForConditionalGeneration`) requires transformers >= 5.5.0.dev or 5.6.0.dev (from source).

5. **Audio not on OV** — Gemma 4 E2B/E4B architecture supports audio, but `optimum-intel` does not export the audio tower, and VLMPipeline has no audio path.

6. **GPU config** — Had to use `INFERENCE_PRECISION_HINT: f32` early on to fix f16/f32 type mismatch errors. Also fixed `AVAILABLE_DEVICE_MEM_SIZE` error.

7. **optimum-intel is too slow** — Early tests showed optimum-intel pipeline was much slower than openvino.genai VLMPipeline. The project uses openvino.genai exclusively.

---

## 13. Resuming on Another System

```powershell
# 1. Clone the repo
git clone https://github.com/jlee52tw/gemma4-openvino-genai.git
cd gemma4-openvino-genai

# 2. Create venv and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 3. Build openvino-genai from PR#3644 (until it's merged)
git clone --recursive --branch as/vlm_enable_1 https://github.com/as-suvorov/openvino.genai.git
cd openvino.genai
pip install . --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release -v

# 4. Export models (if not already available)
pip install -r requirements-export.txt
optimum-cli export openvino --model google/gemma-4-E4B-it --weight-format int4 --group-size 64 --ratio 1.0 gemma-4-E4B-it-ov

# 5. Run
python run_gemma4.py --model-dir gemma-4-E4B-it-ov --device GPU --prompt "Hello!" --show-memory
```

---

## 14. ASUS KPI Comparison — Gemma 4 E4B vs Llama 3.1 8B (16 GB Config)

### Background

ASUS requires Gemma-4-e4b-it performance to be competitive with Llama 3.1 8B
on **16 GB** client devices. They provided benchmark data on LNL U7 258V (32 GB)
and PTL 204 (16 GB). Key gaps to investigate:

1. **Memory:** Gemma4 E4B total peak ~10.4–11.6 GB vs Llama3.1 ~5.2–6.3 GB
   on 16 GB PTL 204. Target: total memory < 10 GB for 4K input/output.
   (Reference: llama.cpp+Vulkan+Q4_K_M on PTL 204 uses ~6 GB at 2K I/O.)
2. **Cache creation time:** Gemma4 = 23s, Llama3.1 = 14s on PTL 204. Target: < 20s.
3. **Prefill speed / Output TPS:** Gemma4 significantly slower than Llama3.1.
   (Reference: llama.cpp+Vulkan+Q4_K_M on PTL 204 Output TPS ~18 tok/s at 2K I/O.)

### ASUS Reference Data

#### LNL U7 258V — 32 GB 8533 MT/s

**Gemma-4-E4B-it (INT4) — OV 2026.2.X (Local Build) — VLMPipeline:**

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | Total peak (GB) | GPU peak (GB) |
|---:|---:|---:|---:|---:|---:|
| 467 | 63 | 382.1 | 19.2 | 11.0 | 6.9 |
| 1058 | 146 | 627.0 | 16.7 | 11.8 | 8.9 |
| 2075 | 221 | 718.3 | 13.2 | 12.0 | 11.3 |

Model=6 GB, Cache=6 GB, Cache create=35s, Cache peak=13.5 GB

**Llama 3.1-8B (INT4) — OV 2025.4.2 (Official) — Native API:**

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | Total peak (GB) | GPU peak (GB) |
|---:|---:|---:|---:|---:|---:|
| 462 | 92 | 833.9 | 22.2 | 5.2 | 5.0 |
| 1032 | 150 | 1314.6 | 21.9 | 5.4 | 5.2 |
| 2021 | 300 | 1016.0 | 21.1 | 6.1 | 6.0 |

Model=4.35 GB, Cache=4.35 GB, Cache create=21s, Cache peak=10.1 GB

#### PTL 204 — 16 GB 6800 MT/s (PRIMARY TARGET)

**Gemma-4-E4B-it (INT4) — OV 2026.2.X (Local Build) — VLMPipeline:**

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | Total peak (GB) | GPU peak (GB) |
|---:|---:|---:|---:|---:|---:|
| 467 | 63 | 258.4 | 13.5 | 10.4 | 6.6 |
| 1058 | 146 | 400.0 | 11.0 | 11.4 | 7.7 |
| 2075 | 221 | 327.9 | 8.3 | 11.6 | 7.9 |

Model=6 GB, Cache=6 GB, Cache create=23s, Cache peak=11.2 GB

**Llama 3.1-8B (INT4) — OV 2025.4.2 (Official) — Native API:**

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | Total peak (GB) | GPU peak (GB) |
|---:|---:|---:|---:|---:|---:|
| 462 | 92 | 677.4 | 19.1 | 5.2 | 4.7 |
| 1032 | 150 | 795.1 | 18.5 | 5.5 | 5.2 |
| 2021 | 300 | 662.4 | 17.4 | 6.3 | 6.0 |

Model=4.35 GB, Cache=4.35 GB, Cache create=14s, Cache peak=10.1 GB

### Gap Analysis (PTL 204 — 16 GB)

| KPI | Gemma4 E4B | Llama3.1 8B | Gap |
|---|---:|---:|---|
| Model size | 6.0 GB | 4.35 GB | +1.65 GB (embedding tables at INT8) |
| Cache create | 23s | 14s | +64% slower |
| Cache peak mem | 11.2 GB | 10.1 GB | +1.1 GB |
| Output TPS (~467 in) | 13.5 | 19.1 | −29% slower |
| Output TPS (~1058 in) | 11.0 | 18.5 | −41% slower |
| Output TPS (~2075 in) | 8.3 | 17.4 | −52% slower |
| Total peak (~467 in) | 10.4 GB | 5.2 GB | +5.2 GB (2× worse) |
| Total peak (~2075 in) | 11.6 GB | 6.3 GB | +5.3 GB |
| Prefill (~467 in) | 258.4 | 677.4 | −62% slower |

**Potential root causes:**
- E4B model is 6 GB vs 4.35 GB due to per-layer embeddings at INT8 (2.69 GB) — architectural
- VLMPipeline overhead vs Native LLMPipeline API
- OV 2026.2.X (dev) vs OV 2025.4.2 (stable) maturity
- GPU kernel optimization differences (Gemma4 is newer architecture)

### Our Test Plan (This System — 16 GB LPDDR5 8533 MT/s)

**Script:** `benchmark_asus_kpi.py`

**Step 1 — Baseline (mmap ON, 16 GB config):**
```powershell
python benchmark_asus_kpi.py `
    --model-dir <path-to-gemma-4-E4B-it-ov> `
    --device GPU `
    --scenarios 467 1058 2075 `
    --max-new-tokens 300 `
    --output-json results_16gb_mmap.json
```

**Step 2 — No-mmap (16 GB config):**
```powershell
python benchmark_asus_kpi.py `
    --model-dir <path-to-gemma-4-E4B-it-ov> `
    --device GPU `
    --scenarios 467 1058 2075 `
    --max-new-tokens 300 `
    --no-mmap `
    --output-json results_16gb_nommap.json
```

**Step 3 — Extended scenarios (4K I/O for ASUS request):**
```powershell
python benchmark_asus_kpi.py `
    --model-dir <path-to-gemma-4-E4B-it-ov> `
    --device GPU `
    --scenarios 467 1058 2075 4096 `
    --max-new-tokens 4096 `
    --output-json results_16gb_4k.json
```

**KPIs to collect & compare with ASUS data:**
- [ ] Model size on disk (GB)
- [ ] Cache creation time (s)
- [ ] Cache creation peak memory (GB)
- [ ] Per scenario: input tokens, output tokens, prefill speed, output TPS
- [ ] Per scenario: total peak memory, GPU peak memory
- [ ] mmap ON vs OFF memory comparison  
- [ ] TTFT and TPOT raw values

**Deliverable:** Data table matching ASUS format, gap analysis, internal ticket if needed.

### Our Results — 32 GB LPDDR5 8533 MT/s (Panther Lake B390 iGPU)

**System:** Intel Panther Lake, Arc B390 iGPU, 32 GB LPDDR5 8533 MT/s  
**OV:** 2026.2.0-21571 nightly  
**GenAI:** 2026.2.0.0-3058 (PR#3644 `as/vlm_enable_1`)  
**Model:** Gemma-4-E4B-it (INT4), 6.05 GB on disk  

#### Run 1 (manual np.fromfile workaround for no-mmap)

##### mmap ON (default)

| Cache time | Cache peak mem |
|---:|---:|
| 10.5s | 12.17 GB |

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1691.1 | 23.7 | 281.5 | 42.2 | 12.3 |
| 1067 | 300 | 1128.5 | 21.2 | 945.5 | 47.1 | 12.3 |
| 2084 | 300 | 1420.2 | 17.7 | 1467.4 | 56.4 | 12.3 |

##### mmap OFF (--no-mmap, manual np.fromfile)

| Cache time | Cache peak mem |
|---:|---:|
| 12.5s | 13.43 GB |

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1674.1 | 23.4 | 284.3 | 42.7 | 13.7 |
| 1067 | 300 | 1006.7 | 20.8 | 1059.9 | 48.0 | 13.9 |
| 2084 | 300 | 1494.8 | 17.4 | 1394.2 | 57.6 | 14.6 |

#### Run 2 (ENABLE_MMAP='NO' kwarg — correct approach)

##### mmap ON (default)

| Cache time | Cache peak mem |
|---:|---:|
| 10.1s | 12.18 GB |

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1747.1 | 23.5 | 272.4 | 42.5 | 12.2 |
| 1067 | 300 | 1200.6 | 21.0 | 888.7 | 47.6 | 12.2 |
| 2084 | 300 | 1897.0 | 17.5 | 1098.6 | 57.2 | 12.2 |

##### mmap OFF (--no-mmap, ENABLE_MMAP='NO')

| Cache time | Cache peak mem |
|---:|---:|
| 11.5s | 12.18 GB |

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1720.0 | 23.4 | 276.7 | 42.8 | 12.2 |
| 1067 | 300 | 1168.0 | 21.0 | 913.5 | 47.6 | 12.2 |
| 2084 | 300 | 1829.8 | 17.5 | 1138.9 | 57.3 | 12.2 |

#### Observations (32 GB)

- **Run 1 vs Run 2 mmap ON:** Very consistent — Output TPS within ±1%, TTFT within ±5%. Reproducible results.
- **Run 2 mmap OFF vs Run 1 mmap OFF:** Using `ENABLE_MMAP='NO'` shows **same peak memory as mmap ON** (12.2 GB vs 13.7-14.6 GB with old workaround). The old `np.fromfile()` approach created extra heap copies that inflated RSS. The `ENABLE_MMAP='NO'` kwarg lets the runtime handle it natively — no extra Python-side copies.
- **Run 2 mmap ON vs OFF:** Virtually identical performance and memory — suggests the GPU runtime copies weights off mmap pages during model compilation regardless, so `ENABLE_MMAP` has no practical effect on this platform at 32 GB.
- **Output TPS** substantially better than ASUS LNL U7 258V numbers (23.5 vs 19.2 at ~467 tokens). Likely due to B390 iGPU having more EUs.
- **Prefill speed** much higher (1747 vs 382 t/s) — likely iGPU compute advantage.
- **Cache creation:** 10.1s (mmap) / 11.5s (no-mmap) — both under ASUS target of 20s.
- **GPU peak memory reads as 0.0 GB** — Windows performance counter query not returning per-process GPU data.
- **Total peak ~12.2 GB** — on 16 GB config, should fit comfortably.

### Our Results — 16 GB LPDDR5 8533 MT/s (Panther Lake B390 iGPU)

**System:** Intel Panther Lake, Arc B390 iGPU, 16 GB (15.7 GB) LPDDR5 8533 MT/s  
**BIOS iGPU override:** 13 GB (OV reports GPU_DEVICE_TOTAL_MEM_SIZE: 8.5 GB)  
**OV:** 2026.2.0-21571 nightly  
**GenAI:** 2026.2.0.0-3058 (PR#3644 `as/vlm_enable_1`)  
**Model:** Gemma-4-E4B-it (INT4), 6.05 GB on disk  
**Model cache:** Pre-built on 32 GB, reused on 16 GB (~6.5 GB .blob + .cl_cache files)  

#### Per-Process Memory Profile (test_cache_memory.py — subprocess isolation)

| Config | Peak RSS (GB) | Load Time (s) |
|--------|---:|---:|
| No cache, mmap ON | 10.22 | 14.4 |
| No cache, mmap OFF | 11.35 | 17.4 |
| Cache, mmap ON | 12.60 | 7.4 |
| Cache, mmap OFF | 9.99 | 7.8 |

#### Config 1: No cache, mmap ON (default)

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1366.0 | 15.4 | 348.5 | 65.0 | 10.62 |
| 1067 | 300 | 860.5 | 13.9 | 1240.0 | 72.2 | 10.62 |
| 2084 | 300 | 1106.4 | 11.7 | 1883.6 | 85.3 | 10.62 |

#### Config 2: No cache, mmap OFF (`ENABLE_MMAP='NO'`)

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1384.0 | 15.2 | 343.9 | 65.9 | 10.87 |
| 1067 | 300 | 964.1 | 13.8 | 1106.7 | 72.4 | 10.87 |
| 2084 | 300 | 1263.8 | 11.8 | 1649.0 | 85.0 | 10.87 |

#### Config 3: Cache + mmap ON (`CACHE_DIR`)

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1337.1 | 14.6 | 356.0 | 68.3 | 11.19 |
| 1067 | 300 | 884.6 | 13.2 | 1206.2 | 75.6 | 11.19 |
| 2084 | 300 | 1091.7 | 11.5 | 1909.0 | 87.0 | 11.19 |

#### Config 4: Cache + mmap OFF (`CACHE_DIR` + `ENABLE_MMAP='NO'`) — Lowest Peak

| Input tokens | Output tokens | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Total peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1385.5 | 11.2 | 343.6 | 89.0 | 9.31 |
| 1067 | 300 | 845.3 | 10.2 | 1262.3 | 97.7 | 9.31 |
| 2084 | 300 | 1201.1 | 9.2 | 1735.1 | 108.1 | 9.31 |

#### Observations (16 GB vs 32 GB)

1. **~34% TPS regression** across the board from 32 GB to 16 GB:
   - 32 GB best: 23.5 t/s (mmap ON) → 16 GB best: 15.4 t/s (no-cache mmap ON)
   - Root cause: memory bandwidth contention — iGPU and CPU share LPDDR5

2. **Cache + mmap OFF achieves lowest peak (9.31 GB)** but worst TPS (11.2 t/s at ~467):
   - Loading 12.7 GB of combined cache+model files from disk without mmap causes I/O contention
   - On 16 GB, the memory savings don't translate to better performance

3. **No-cache configs give best TPS** on 16 GB:
   - mmap ON: 15.4 / 13.9 / 11.7 t/s → best overall performance
   - mmap OFF: 15.2 / 13.8 / 11.8 t/s → nearly identical
   - Peak memory: 10.6-10.9 GB — fits in 16 GB with ~5 GB headroom

4. **Cache + mmap ON is worst of both worlds on 16 GB:**
   - Highest peak (11.19 GB) due to mmap pages + cached blobs both resident
   - TPS slightly lower than no-cache (14.6 vs 15.4 at ~467)

5. **Comparison with ASUS PTL 204 data (16 GB 6800 MT/s):**

   | KPI | ASUS PTL 204 | Our PTL B390 (16 GB) | Delta |
   |---|---:|---:|---|
   | Output TPS (~467 in) | 13.5 | 15.4 | +14% faster (more EUs) |
   | Output TPS (~1058 in) | 11.0 | 13.9 | +26% faster |
   | Output TPS (~2075 in) | 8.3 | 11.7 | +41% faster |
   | Total peak (~467 in) | 10.4 GB | 10.6 GB | +0.2 GB (comparable) |
   | Prefill (~467 in) | 258.4 | 1366.0 | +5.3× faster |
   | Memory bandwidth | 6800 MT/s | 8533 MT/s | +25% |

   Our system is consistently faster due to higher memory bandwidth (8533 vs 6800 MT/s) and
   more iGPU EUs (96 vs unknown). Peak memory is comparable (~10.4-10.6 GB).

6. **Recommendation for 16 GB deployment:**
   - **Best TPS:** No cache, mmap ON (default) — 15.4 t/s, 10.6 GB peak
   - **Lowest memory:** Cache + mmap OFF — 9.3 GB peak, but 27% TPS penalty
   - **Best balance:** No cache, mmap OFF — 15.2 t/s, 10.9 GB peak (negligible diff from default)

### Status

- [x] Environment verified on this system (OV 2026.2 + GenAI PR#3644)
- [x] 32 GB baseline benchmark Run 1 (manual np.fromfile workaround) — done
- [x] 32 GB baseline benchmark Run 2 (ENABLE_MMAP='NO' kwarg) — done
- [x] Memory reconfigured to 16 GB
- [x] 16 GB baseline benchmark (4 configs: mmap ON/OFF × cache ON/OFF) — done
- [x] Per-process memory profile (subprocess isolation) — done
- [x] ASUS comparison analysis — done
- [ ] 4K I/O extended test
- [ ] Internal ticket filed (if gaps confirmed)
