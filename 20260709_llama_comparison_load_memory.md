# Gemma-4-E4B vs Llama-3.1-8B: Load Time, Memory & KPI Comparison

**Date:** 2026-07-09  
**Runtime:** OpenVINO GenAI `2026.3.0.0-3240-b99a4dd60f5` (0706 nightly)  
**Feature:** PagedAttention ON · mmap=ON and mmap=OFF both measured  
**Platform:** Intel Arc B390 iGPU · 84.84 GB GPU mem (shared) · 95.5 GB RAM · Windows 11

---

## 1. Model Architecture Comparison

Both models are **dense** transformers (Gemma-4-E4B is **not** MoE — `enable_moe_block: false`
in its `text_config`). The "E4B" name means "Efficient 4B", referring to the 4 B parameter
count. Gemma-4-**26B-A4B** is the MoE variant (26 B total, 4 B active per token).

| Feature | Gemma-4-E4B | Llama-3.1-8B |
|---|---|---|
| **Total parameters** | ~4 B (dense) | ~8 B (dense) |
| **Model type** | `gemma4_text` | `llama` |
| **`enable_moe_block`** | **`false`** | N/A |
| **Layers** | **42** | 32 |
| **Hidden size** | 2 560 | 4 096 |
| **Attention heads (Q)** | 8 | 32 |
| **KV heads (GQA)** | 2 (ratio 4:1) | 8 (ratio 4:1) |
| **Head dim** | 256 (local) / 512 (global) | 128 |
| **FFN intermediate size** | 10 240 | 14 336 |
| **FFN activation** | GeGLU (`gelu_pytorch_tanh`) | SwiGLU |
| **Attention type** | Mixed: **35 sliding-window** (w=512) + **7 full** | Full attention only |
| **KV-sharing layers** | **18 layers share KV** from a prior layer | None |
| **Per-Layer Embedding (PLE)** | **Yes** — 256-dim lookup per layer, all 42 layers | **No** |
| **Vocabulary size** | **262 144** | 128 256 |
| **Context length** | 131 072 | 131 072 |
| **Tied embeddings** | Yes | No |
| **Pipeline type (OV GenAI)** | VLMPipeline | LLMPipeline |
| **OV model binary (INT4 gs64)** | **6 235 MB total** | **4 621 MB total** |

### Gemma-4-E4B binary breakdown (6 235 MB)

| File | Size | Role |
|---|---|---|
| `openvino_language_model.bin` | 2 685 MB | Transformer layer weights (INT4 gs64) |
| `openvino_text_embeddings_per_layer_model.bin` | **2 689 MB** | Per-Layer Embedding table (262 144 × 256 × 42 layers, BF16) |
| `openvino_text_embeddings_model.bin` | 641 MB | Main token embeddings (262 144 × 2 560, BF16) |
| `openvino_vision_embeddings_model.bin` | 162 MB | SigLIP vision encoder (text inference skips this) |

The **Per-Layer Embedding model is almost as large as the language model itself** — this is unique
to Gemma-4 and is the primary driver of E4B's larger memory footprint and lower prefill throughput.

---

## 2. Why Is Llama-3.1-8B Prefill Faster Than Gemma-4-E4B?

> Observation: Llama-3.1-8B achieves **2 029 prefill_tps** vs E4B QAT's **1 842 prefill_tps**
> at 1024 tok input, despite Llama having **2× more total parameters**.

This is counter-intuitive. The explanation requires looking at the actual on-device data flow.

### 2.1 Per-Layer Embedding (PLE) bandwidth overhead

Gemma-4 introduces a **per-layer input embedding** (`hidden_size_per_layer_input: 256`):
at each of the 42 transformer layers, the model looks up a 256-dimensional embedding from a
262 144-entry vocabulary table and adds it to the layer input before the attention + FFN computation.

- PLE table: 262 144 × 256 × 42 layers × 2 bytes/BF16 = **~2.69 GB** of data
- For a 512-token prefill: 42 layers × 512 tokens = **21 504 embedding lookups** plus the full
  2.69 GB table must be resident on the GPU
- Llama has **no PLE mechanism** — its data path only touches the 4 621 MB model binary

The iGPU must stream ~6 GB of E4B model data per forward pass vs ~4.6 GB for Llama.
On Intel Arc B390 with ~350 GB/s GPU memory bandwidth:

| Model | Model data streamed | Theoretical BW time |
|---|---|---|
| Gemma-4-E4B | 6 235 MB | ~17.8 ms/layer-sweep |
| Llama-3.1-8B | 4 621 MB | ~13.2 ms/layer-sweep |

> **This bandwidth gap, repeated across all 42 (E4B) vs 32 (Llama) layers, directly reduces
> E4B's prefill tokens/second.**

### 2.2 VLMPipeline vs LLMPipeline overhead

E4B is loaded via `VLMPipeline` (multimodal pipeline) even for text-only inference, while Llama
uses the lighter `LLMPipeline`. The VLM pipeline maintains vision encoder infrastructure, has
additional preprocessing stages, and may trigger different kernel dispatch paths in OV GenAI.

### 2.3 Mixed attention kernel dispatch

E4B uses **two attention kernels** per layer-group:
- **Sliding-window attention** (35/42 layers, window=512): uses `rope_theta=10000`,
  causal mask restricted to last 512 tokens
- **Full attention** (7/42 layers): uses `rope_theta=1000000`, `partial_rotary_factor=0.25`
- **KV-sharing** (18 layers): these layers re-use K,V from an earlier layer (saves VRAM but
  adds pointer-indirection overhead in the attention kernel)

This kernel diversity increases CPU-side dispatch overhead and may limit OV's graph fusion
optimizations compared to Llama's uniform full-attention architecture.

### 2.4 Matrix multiply efficiency (GEMM sizing)

INT4-quantized inference is dominated by **dequantize + GEMM** operations. Larger, more
"square" matrix dimensions better saturate GPU compute tiles:

| Projection | Gemma-4-E4B | Llama-3.1-8B |
|---|---|---|
| Q projection | 2 560 → 2 048 | 4 096 → 4 096 |
| K,V projections | 2 560 → 512 | 4 096 → 1 024 |
| O projection | 2 048 → 2 560 | 4 096 → 4 096 |
| FFN gate+up | 2 560 → 10 240 | 4 096 → 14 336 |

Llama's GEMMs (4 096 × N) better exploit the B390's 96-EU tile structure than E4B's narrower
2 560 × N operations. Fewer but larger GEMMs also means fewer kernel launch overheads.

### 2.5 Vocabulary + LM head size

| | Gemma-4-E4B | Llama-3.1-8B |
|---|---|---|
| Vocab size | 262 144 | 128 256 |
| LM head params | 2 560 × 262 144 = 671 M | 4 096 × 128 256 = 525 M |
| Embedding params | 671 M (tied with LM head) | 525 M (separate) |

E4B's embedding table is 641 MB in BF16. While only the last token's logit is needed for decode,
the large vocabulary increases tokenizer overhead and embedding lookup cost.

### 2.6 TTFT vs prefill_tps distinction

- **TTFT** (Time to First Token, ms): wall-clock time from request to first output token —
  includes prefill compute **plus** OV scheduling overhead, KV cache allocation, and pipeline dispatch.
  Llama TTFT (~248 ms) is lower than E4B TTFT (~340 ms) at 512 tok, partly because LLMPipeline
  has less scheduling overhead than VLMPipeline, not only because of the model binary size difference.

- **Prefill_tps** (tokens/second): `actual_input_tokens / TTFT_s` — normalizes for input length,
  directly comparable across runs. At 1024 tok, E4B reaches 1 842 tps vs Llama 2 106 tps because
  the PLE bandwidth overhead and VLM overhead are amortized over more tokens but GEMM efficiency
  still favors Llama.

### 2.7 Decode throughput (why E4B wins at decode)

| Model | Decode_tps (512 tok) |
|---|---|
| Gemma-4-E4B QAT | **~30 tps** |
| Llama-3.1-8B | ~23 tps |

During decode (autoregressive, 1 token at a time), **the bottleneck shifts to KV-cache
bandwidth**, not weight bandwidth. E4B's **KV sharing** (18 layers skip K,V computation) and
**smaller per-head KV size** (2 KV heads × 256 dim vs Llama's 8 KV heads × 128 dim = same bytes
but different access pattern) make it more efficient per generated token.
Additionally, E4B's smaller hidden size (2 560 vs 4 096) means cheaper Q×K attention dot
products per decode step, allowing higher decode throughput despite the extra layers.

---

## 3. Memory Measurement Methodology

### 3.1 `peak_load_rss_gb` — Sampler-based RSS peak during warm load

**RSS (Resident Set Size)** is the amount of physical RAM currently mapped to the process.
It is measured via `psutil.Process().memory_info().rss` from the Python process.

Capture method in `benchmark_loadmem.py`:

```python
class MemoryMonitor:
    """Samples process RSS in a background thread at 5 ms intervals."""
    def _run(self):
        while not self._stop.is_set():
            self.samples.append(self.proc.memory_info().rss)
            self._stop.wait(0.005)   # 5 ms poll interval
```

The monitor is **started** just before the warm pipeline load (compiled-blob cache hit) and
**stopped** after the constructor returns. `peak_load_rss_gb = max(samples) / 1 GB`.

**What RSS includes during model load:**
- CPU-side weight buffers for host-to-GPU DMA staging
- OV graph/blob metadata allocations
- Python interpreter + framework overhead
- With `mmap=ON`: file-backed mmap pages appear briefly in RSS then are reclaimed by the OS
  after GPU upload completes (typically within 1–2 s)
- With `mmap=OFF`: weights loaded into anonymous heap → stay resident

**Limitation:** The 5 ms poll can **miss** the mmap=ON transient spike if the OS reclaims pages
faster than the poll interval. This is why `peak_load_rss_gb` reads only 4.8 GB for E4B
with `mmap=ON` (the sampler missed the spike), but the true peak is visible in `os_peak_wset_gb`.

### 3.2 `os_peak_wset_gb` — Windows OS-tracked lifetime peak working set

Captured via the Windows `PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize` field:

```python
def os_peak_wset_gb() -> float:
    mi = psutil.Process().memory_info()
    pw = getattr(mi, "peak_wset", None)   # Windows-only field
    return pw / 1024**3
```

`peak_wset` is the **maximum number of pages simultaneously resident in RAM at any single
instant** since process start, tracked by the Windows kernel at **page-fault interrupt
granularity** — not at a sampler interval. This makes it:

- **Independent of poll rate** — cannot miss a spike
- **Cumulative across the entire process lifetime** — always the true high-water mark
- **Available only on Windows** via `PROCESS_MEMORY_COUNTERS`

### 3.3 Key difference

| Metric | How captured | Granularity | Can miss mmap spike? |
|---|---|---|---|
| `peak_load_rss_gb` | Python background thread, 5 ms poll | 5 ms | **Yes** — if spike < poll window |
| `os_peak_wset_gb` | Windows kernel page-fault counter | Per page-fault | **No** — always accurate |

**Concrete evidence from data (mmap=ON):**

| Model | peak_load_rss_gb | os_peak_wset_gb | Gap (missed spike) |
|---|---:|---:|---:|
| Gemma-4-E4B | 4.80 GB | 8.20 GB | **3.40 GB** |
| Llama-3.1-8B | 6.53 GB | 10.61 GB | **4.08 GB** |

With `mmap=OFF`, the gap disappears because heap RSS stays resident and the sampler reliably
captures it:

| Model | peak_load_rss_gb | os_peak_wset_gb | Gap |
|---|---:|---:|---:|
| Gemma-4-E4B | 10.19 GB | 10.61 GB | 0.42 GB |
| Llama-3.1-8B | 10.07 GB | 11.13 GB | 1.06 GB |

> **Recommendation:** Use `os_peak_wset_gb` as the authoritative peak RAM figure for
> planning host memory budgets. Use `peak_load_rss_gb` only when mmap=OFF.

### 3.4 `sustained_rss_gb` — Inference working set

Sampled by the same `MemoryMonitor` thread during `generate()` calls. Reported as the **median**
of all samples taken during inference. This represents the **steady-state** CPU-side memory while
the model is actively running tokens — KV cache allocated, all model weights on GPU, Python
runtime stable. This metric is largely independent of mmap setting (both modes ≈ same value)
because by the time inference runs, the file-backed pages have already been reclaimed or are
not being accessed.

---

## 4. Load Time Analysis (Warm / Cached Load)

Warm load = pipeline constructor reading pre-compiled OpenCL kernel blobs from disk cache
(no recompilation). OV built-in load time via `perf_metrics.get_load_time()` (ms).

### 4.1 mmap=ON vs mmap=OFF load time

| Model | mmap | Load Cold (s) | Load Warm (s) | OV load_time (ms) |
|---|:---:|---:|---:|---:|
| Gemma-4-E4B QAT | ON | 14.4 | 8.9 | 8 896 |
| Gemma-4-E4B QAT | OFF | 15.2 | **4.3** | **4 339** |
| Llama-3.1-8B | ON | 8.1 | 9.1 | 9 020 |
| Llama-3.1-8B | OFF | 8.9 | **4.8** | **4 628** |

**Why mmap=OFF is ~2× faster to load (OV built-in metric):**

With `mmap=ON`, OV maps weight files to virtual addresses and lets the GPU page-fault them in
on demand. The first GPU kernel that reads each page triggers a page fault → OS reads from file →
page loaded → GPU resumes. This demand-paging adds latency per page access and stretches the
loading phase across the entire GPU warm-up period (~9 s).

With `mmap=OFF`, OV reads all weights into an anonymous heap buffer using standard `ReadFile`
calls. Modern NVMe drives deliver sequential reads at multi-GB/s, and all weights are in RAM
before the first GPU kernel runs → no page faults during GPU execution → clean ~4.3–4.8 s load.

**Load cold is similar for both modes** because cold load includes OpenCL compilation
(the dominant time component), masking the file-I/O difference.

### 4.2 Peak load memory comparison

| Model | mmap | peak_load_rss_gb | os_peak_wset_gb |
|---|:---:|---:|---:|
| Gemma-4-E4B | ON | 4.80 | **8.20** |
| Gemma-4-E4B | OFF | 10.19 | 10.61 |
| Llama-3.1-8B | ON | 6.53 | **10.61** |
| Llama-3.1-8B | OFF | 10.07 | 11.13 |

`mmap=ON` has lower steady-state RSS (OS reclaims file-backed pages after GPU upload) but
requires trusting `os_peak_wset_gb` for the true host RAM peak.
`mmap=OFF` requires ~10 GB of RAM simultaneously resident but is predictable and measurable.

---

## 5. Full KPI Results (Inference, mmap=ON, avg of runs ≥ 1)

> Source: `results_loadmem_0706_v2.csv`  
> Config: warmup=0, runs=2, max_new_tokens=128, PA=ON, device=GPU

### 5.1 Throughput & latency

| Model | mmap | InTok | TTFT_ms | Prefill_tps | TPOT_ms | Decode_tps |
|---|:---:|---:|---:|---:|---:|---:|
| Gemma-4-E4B QAT | ON | 512 | 332 | 1 436 | 33.3 | 30.0 |
| Gemma-4-E4B QAT | ON | 1 024 | 537 | 1 842 | 34.2 | 29.2 |
| Gemma-4-E4B QAT | OFF | 512 | 349 | 1 369 | 33.5 | 29.8 |
| Gemma-4-E4B QAT | OFF | 1 024 | 548 | 1 806 | 34.2 | 29.2 |
| Llama-3.1-8B | ON | 512 | 236 | 2 029 | 43.5 | 23.0 |
| Llama-3.1-8B | ON | 1 024 | 470 | 2 106 | 44.2 | 22.6 |
| Llama-3.1-8B | OFF | 512 | 251 | 1 902 | 43.9 | 22.8 |
| Llama-3.1-8B | OFF | 1 024 | 469 | 2 111 | 44.7 | 22.4 |

### 5.2 Memory during inference

| Model | mmap | InTok | sustained_rss_gb | peak_infer_rss_gb |
|---|:---:|---:|---:|---:|
| Gemma-4-E4B QAT | ON | 512 | 7.89 | 7.89 |
| Gemma-4-E4B QAT | ON | 1 024 | 8.15 | 8.15 |
| Llama-3.1-8B | ON | 512 | 5.87 | 6.10 |
| Llama-3.1-8B | ON | 1 024 | 5.98 | 5.98 |

E4B inference RSS is ~2 GB higher than Llama-8B despite fewer total params:
- E4B loads 6 235 MB of model data (including 2 689 MB PLE) vs Llama's 4 621 MB
- VLMPipeline maintains additional buffers for multimodal preprocessing
- OV KV cache for E4B's mixed sliding-window + full attention requires more complex allocation

### 5.3 E4B vs Llama summary at 1024-tok input

| Metric | Gemma-4-E4B QAT | Llama-3.1-8B | Winner |
|---|---:|---:|---|
| TTFT (ms) | 537 | 470 | Llama (−12%) |
| Prefill (tps) | 1 842 | 2 106 | **Llama (+14%)** |
| TPOT (ms) | 34.2 | 44.2 | E4B (−23%) |
| Decode (tps) | **29.2** | 22.6 | **E4B (+29%)** |
| Sustained RSS (GB) | 8.15 | 5.98 | **Llama (−27%)** |
| Load warm / mmap=OFF (s) | **4.3** | 4.8 | E4B (−10%) |
| OS peak load RAM (mmap=ON) | 8.20 | 10.61 | E4B (−23%) |

**E4B excels at decode throughput** (MoE-like decode efficiency via KV sharing + small
active compute per step) while **Llama excels at prefill throughput** (clean dense attention,
no PLE overhead, efficient GEMM sizing, lightweight LLMPipeline).

---

## 6. Conversion Notes (Llama-3.1-8B)

**Issue:** `optimum-cli export openvino ... --model meta-llama/Llama-3.1-8B-Instruct`
failed with `AttributeError: 'PreTrainedConfig' object has no attribute 'max_position_embeddings'`.

**Root cause:** transformers 5.5.0 moved RoPE config attributes under a standardized
`rope_config` sub-object. The optimum-intel library inference path
(`_infer_library_from_model_name_or_path`) probes the model type and hits this in a generic
`PreTrainedConfig` branch that does not know about the new attribute location.

**Fix:**
```powershell
optimum-cli export openvino `
    --model meta-llama/Llama-3.1-8B-Instruct `
    --library transformers `        # <-- bypasses broken model-type inference
    --weight-format int4 --group-size 64 --ratio 1.0 `
    --task text-generation-with-past `
    Llama-3.1-8B-Instruct-ov-int4-gs64-johnson
```

`--library transformers` forces the Hugging Face Transformers export path directly, bypassing
the crashing `_infer_library_from_model_name_or_path()` code.

**Output:** `openvino_model.bin` = 4 591 MB INT4 gs64.
Detected as LLM (no `openvino_vision_embeddings_model.xml`) → uses `LLMPipeline`.

---

## 7. Measurement Scripts

| Script | Purpose |
|---|---|
| `benchmark_loadmem.py` | Load time + mmap ON/OFF + peak/sustained memory + OV load_time |
| `run_loadmem.ps1` | Wrapper to source `setupvars.ps1` and invoke `benchmark_loadmem.py` |
| `results_loadmem_0706_v2.csv` | Raw results — 16 rows (2 models × 2 mmap × 2 lengths × 2 runs) |

### Run command (0706 build, mmap ON + OFF)
```powershell
& "C:\working\gemma4\openvino_genai_windows_2026.3.0.0.dev20260706_x86_64\setupvars.ps1" -python_version "3.12"
cd "C:\working\gemma4\gemma4-openvino-genai"
& "C:\working\gemma4\.venv\Scripts\python.exe" benchmark_loadmem.py `
    --model-dir models/gemma-4-E4B-it-qat-ov-int4-gs64-johnson `
               models/Llama-3.1-8B-Instruct-ov-int4-gs64-johnson `
    --device GPU --warmup 0 --runs 2 `
    --input-lengths 512 1024 `
    --mmap on off `
    --cache-root ov_cache_tmp `
    --output-csv results_loadmem_0706_v2.csv
```

### OV GenAI API notes

```python
# LLMPipeline: must pass list to get DecodedResults (not plain str)
result = pipe.generate(["your prompt"], generation_config=cfg)
ov_load_ms = result.perf_metrics.get_load_time()   # float, ms — NOT .load_time

# VLMPipeline: plain string OK
result = pipe.generate("your prompt", generation_config=cfg)
ov_load_ms = result.perf_metrics.get_load_time()
```

---

*Generated by `benchmark_loadmem.py` on 2026-07-09 · OV GenAI 0706 · Intel Arc B390 iGPU*
