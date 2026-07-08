# Gemma4 E4B & 26B-A4B KPI Benchmark Results
**Date:** 2026-07-08  
**Runtime:** OpenVINO GenAI 0706 build (`2026.3.0.0-3240-b99a4dd60f5`)  
**Feature:** PagedAttention enabled (`SchedulerConfig`)  
**Platform:** Intel Arc B390 iGPU · 84.84 GB GPU mem (shared) · 95.5 GB RAM · Windows 11

---

## Environment & Conversion Info

### Inference Runtime
| Item | Value |
|---|---|
| OV GenAI build | `2026.3.0.0-3240-b99a4dd60f5` (0706 nightly) |
| OV Runtime | `2026.3.0-22323-ca498d09be7` |
| PagedAttention | ON — `SchedulerConfig(enable_prefix_caching=False, max_num_batched_tokens=maxsize)` |
| Device | GPU (Intel Arc B390 iGPU) |
| GPU EU count | 96 |
| GPU uarch | 30.0.4 |

### Conversion Environment ("Johnson env")
| Package | Version |
|---|---|
| `optimum-intel` | `2.1.0.dev0+f335745` |
| `optimum` | `2.2.0` |
| `transformers` | `5.5.0` |
| `torch` | `2.12.1+cpu` |
| `safetensors` | `0.7.0` |
| `openvino` | `2026.3.0.dev20260630` (in venv) |

> **Note:** `optimum-onnx` must be **uninstalled** before conversion (causes namespace conflict).  
> **Conversion command:** `optimum-cli export openvino` (not `-m optimum.exporters.openvino`).  
> **Patch applied to `convert.py`:** `_save_model` forces `compress_to_fp16=True` for INT4/INT8 exports to avoid `ios_base::badbit` OOM during serialization of large (26B) models.

---

## Model Inventory

### Gemma-4-E4B (4B active params)
| Directory | Source | Conv env | Notes |
|---|---|---|---|
| `gemma-4-E4B-it-int4-ov` | `OpenVINO/gemma-4-E4B-it-int4-ov` (HF) | OV team | Pre-converted; **fails PA** (missing `v13.SDPA` op) |
| `gemma-4-E4B-it-qat-ov-int4-gs64-johnson` | `google/gemma-4-E4B-it-qat-q4_0-unquantized` | t5.5.0 + f335745 | QAT INT4 gs64 |
| `gemma-4-E4B-it-ov-int4-gs64-johnson` | `google/gemma-4-E4B-it` | t5.5.0 + f335745 | nonQAT INT4 gs64 |

### Gemma-4-26B-A4B (4B active params, 26B total)
| Directory | Source | Conv env | Notes |
|---|---|---|---|
| `gemma-4-26b-a4b-it-int4-ov` | `OpenVINO/gemma-4-26b-a4b-it-int4-ov` (HF) | OV team | Pre-converted; **PA works** |
| `gemma-4-26B-A4B-it-qat-ov-int4-gs64-johnson` | `google/gemma-4-26B-A4B-it-qat-q4_0-unquantized` | t5.5.0 + f335745 | QAT INT4 gs64 |
| `gemma-4-26B-A4B-it-ov-int4-gs64-johnson` | `google/gemma-4-26B-A4B-it` | t5.5.0 + f335745 | nonQAT INT4 gs64 |

**NNCF quantization distribution (26B LM):**
- QAT: 97% `int4_asym gs64` + 3% `int8_asym per-channel`
- nonQAT: 97% `int4_asym gs64` + 3% `int8_asym/float`

---

## Gemma-4-E4B KPI Results

**Config:** warmup=1, runs=3 (avg of runs 2–3), output=128 tok, PA=ON

| Model | InTok (raw) | TTFT_ms | Prefill_tps | TPOT_ms | Decode_tps | E2E_tps | RSS_GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| **QAT-johnson** | 512 (477) | 260 | 1835 | 31.4 | 31.9 | 31.9 | 7.88 |
| | 1024 (989) | 487 | 2032 | 34.0 | 29.4 | 29.4 | 8.07 |
| | 2048 (2013) | 867 | 2324 | 34.7 | 28.8 | 28.8 | 8.61 |
| | 4096 (4061) | 1798 | 2259 | 34.8 | 28.6 | 28.8 | 9.56 |
| **nonQAT-johnson** | 512 (477) | 310 | 1539 | 32.4 | 30.9 | 30.9 | 8.09 |
| | 1024 (989) | 461 | 2146 | 33.1 | 30.2 | 30.2 | 8.30 |
| | 2048 (2013) | 838 | 2402 | 34.6 | 28.9 | 28.9 | 8.80 |
| | 4096 (4061) | 1795 | 2262 | 34.9 | 28.6 | 28.7 | 9.73 |

> Source: `results_kpi_0706_pa.csv`

**E4B Key observations:**
- QAT has ~15% lower TTFT at 512 tok (260ms vs 310ms) vs nonQAT
- Decode throughput converges at longer inputs (~28–29 tps at 4096)
- RSS peaks at ~9.7 GB at 4096 tok (borderline for 8 GB discrete GPU configs)
- OV pre-conv model incompatible with PA (missing `v13.ScaledDotProductAttention` op — needs re-conversion)

---

## Gemma-4-26B-A4B KPI Results

**Config:** warmup=1, runs=3 (avg of runs 2–3), output=128 tok, PA=ON

| Model | InTok (raw) | TTFT_ms | Prefill_tps | TPOT_ms | Decode_tps | E2E_tps | RSS_GB |
|---|---:|---:|---:|---:|---:|---:|---:|
| **pre-conv** (OV team) | 512 (477) | 590 | 809 | 31.1 | 32.2 | 32.2 | 16.66 |
| | 1024 (989) | 882 | 1122 | 32.7 | 30.6 | 30.6 | 16.54 |
| | 2048 (2013) | 1695 | 1187 | 35.9 | 27.9 | 27.9 | 17.07 |
| | 4096 (4061) | 4089 | 993 | 35.4 | 28.2 | 28.2 | 17.58 |
| **QAT-johnson** | 512 (477) | 551 | 866 | 31.0 | 32.3 | 32.3 | 16.91 |
| | 1024 (989) | 784 | 1261 | 31.8 | 31.5 | 31.5 | 16.78 |
| | 2048 (2013) | 1540 | 1307 | 33.9 | 29.4 | 29.5 | 17.31 |
| | 4096 (4061) | 3768 | 1078 | 34.7 | 28.8 | 28.8 | 17.85 |
| **nonQAT-johnson** | 512 (477) | 583 | 818 | 30.7 | 32.5 | 32.5 | 16.99 |
| | 1024 (989) | 831 | 1191 | 32.2 | 31.0 | 31.0 | 16.80 |
| | 2048 (2013) | 1640 | 1228 | 34.1 | 29.2 | 29.3 | 17.33 |
| | 4096 (4061) | 3797 | 1070 | 34.8 | 28.7 | 28.7 | 17.88 |

> Source: `results_kpi_26B_0706_pa.csv`

**26B Key observations:**
- **QAT is fastest**: ~7% lower TTFT vs pre-conv at 512 tok (551ms vs 590ms), ~8% at 4096 tok (3768ms vs 4089ms)
- **Decode speed nearly identical to E4B** (~28–32 tps) — MoE architecture activates only ~4B params per token
- **RSS ~17 GB** — 2× vs E4B (26B total weights loaded to GPU even though only 4B active per token)
- Pre-conv (OV team) works with PA unlike E4B pre-conv — OV team re-exported with PA-compatible ops
- nonQAT slightly slower prefill than QAT but decode is equivalent

---

## E4B vs 26B Comparison (at 512 tok input)

| Model | Variant | TTFT_ms | Prefill_tps | Decode_tps | RSS_GB |
|---|---|---:|---:|---:|---:|
| E4B | QAT-johnson | 260 | 1835 | 31.9 | 7.88 |
| E4B | nonQAT-johnson | 310 | 1539 | 30.9 | 8.09 |
| 26B | pre-conv | 590 | 809 | 32.2 | 16.66 |
| 26B | QAT-johnson | 551 | 866 | 32.3 | 16.91 |
| 26B | nonQAT-johnson | 583 | 818 | 32.5 | 16.99 |

> E4B prefill is ~2× faster than 26B (more dense attention layers hit higher throughput).  
> Decode throughput is essentially the same (MoE: both activate ~4B params per decode step).  
> 26B requires ~2× RSS vs E4B.

---

## Issues & Fixes

| Issue | Root cause | Fix |
|---|---|---|
| `ios_base::badbit` during 26B conversion | OV `save_model` OOM — 80+ GB working set during serialization | Patched `_save_model` in `convert.py` to use `compress_to_fp16=True` for INT4/INT8 exports |
| E4B pre-conv fails PagedAttention | Missing `v13.ScaledDotProductAttention` op | Needs re-conversion with current OV runtime |
| `kv_shared_layer_index` AttributeError | transformers 5.12.1 moved attribute | Use transformers 5.5.0 (no patch needed) |
| `python -m optimum.exporters.openvino` silent no-op | f335745 `__main__.py` has no `main()` | Use `optimum-cli export openvino` instead |
| `optimum-onnx` namespace conflict | Multiple distributions for `optimum` | Uninstall `optimum-onnx` |
| Windows pagefile consuming disk during 26B conversion | ~100 GB pagefile growth with 80 GB WS | Freed old model dirs before conversion |

---

## Benchmark Scripts

| Script | Purpose |
|---|---|
| `benchmark_kpi2.py` | Extended KPI benchmark: TTFT/prefill/TPOT/decode/E2E × multiple input lengths |
| `parse_results_kpi.py` | Parse `results_kpi_0706_pa.csv` (E4B) |
| `parse_results_26b_kpi.py` | Parse `results_kpi_26B_0706_pa.csv` (26B) |
| `benchmark.py` | Original benchmark with `--paged-attention` and `--text-only` flags |

### Run command template (0706 + PA)
```powershell
& "C:\working\gemma4\openvino_genai_windows_2026.3.0.0.dev20260706_x86_64\setupvars.ps1" -python_version "3.12"
cd "C:\working\gemma4\gemma4-openvino-genai"
& "C:\working\gemma4\.venv\Scripts\python.exe" benchmark_kpi2.py `
    --model-dir models/<model1> models/<model2> `
    --device GPU --warmup 1 --runs 3 `
    --input-lengths 512 1024 2048 4096 `
    --output-csv results_<name>.csv
```

### Conversion command template (johnson env)
```powershell
$env:HF_XET_ENABLED = "0"
& "C:\working\gemma4\.venv\Scripts\optimum-cli.exe" export openvino `
    --model "google/<model>" --weight-format int4 --group-size 64 --ratio 1.0 `
    --task image-text-to-text "<output-dir>"
```
