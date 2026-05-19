# Gemma-4-E4B-it Dense Weight Streaming — Release Package

## Overview

This release package enables running **Gemma-4-E4B-it (INT4)** on Intel iGPU with
**Dense Weight Streaming** — weights are loaded from NVMe on-the-fly during inference,
enabling the model to run on systems with limited GPU memory.

**Key feature:** Dual-NVMe parallel IO for ~2x NVMe bandwidth.

---

## System Requirements

- **OS:** Windows 11 (22H2+)
- **CPU:** Intel with integrated GPU (12+ Xe EUs recommended)
- **RAM:** 16 GB (target: 8 GB with streaming)
- **Storage:** 1-2 NVMe SSDs (Gen4+ recommended, Gen5 for best results)
- **Python:** 3.12 (must match the included DLLs)
- **OpenVINO:** 2026.2.0 (included in this package)

---

## Package Contents

```
release/
├── README.md                     ← This file
├── run_gemma4.py                 ← Main inference script
├── pack_dense_weights_dual.py    ← Tool to regenerate stripe files
├── benchmark.py                  ← Performance benchmark script
├── requirements.txt              ← Python dependencies
│
├── dlls/                         ← Modified OpenVINO DLLs
│   ├── openvino_intel_gpu_plugin.dll  ← GPU plugin with streaming support
│   ├── openvino.dll
│   ├── openvino_genai.dll
│   ├── openvino_tokenizers.dll
│   ├── tbb12.dll
│   └── ... (other runtime DLLs)
│
├── model/                        ← Model files (place on fastest NVMe)
│   ├── openvino_language_model.xml
│   ├── openvino_language_model.bin           (~2.69 GB)
│   ├── openvino_text_embeddings_per_layer_model.xml
│   ├── openvino_text_embeddings_per_layer_model_revised.bin (~3.07 GB)
│   ├── openvino_vision_embeddings_model.xml
│   ├── openvino_vision_embeddings_model.bin  (~162 MB)
│   ├── openvino_tokenizer.xml / .bin
│   ├── openvino_detokenizer.xml / .bin
│   ├── tokenizer.json
│   ├── config.json / generation_config.json / etc.
│   └── model_cache/              ← Compiled GPU kernels (saves ~12s on first run)
│
├── streaming_nvme0/              ← Stripe file for NVMe 0
│   ├── dense_weights_streaming_0.bin   (~776 MB)
│   └── dense_weights_streaming_0.json  (metadata)
│
└── streaming_nvme1/              ← Stripe file for NVMe 1
    └── dense_weights_streaming_1.bin   (~776 MB)
```

---

## File Size Explanation

You might notice different large binary files — they serve completely different purposes:

| File | Size | Purpose |
|------|------|---------|
| `openvino_language_model.bin` | 2.69 GB | Full decoder model (all 42 layers, all weights+scale+zp) |
| `openvino_text_embeddings_per_layer_model_revised.bin` | 3.07 GB | Per-layer embedding lookup table (vocab=262144). Used via mmap — only ~10 KB read per token. Saves ~2.82 GB GPU memory. |
| `dense_weights_streaming_0.bin` | 776 MB | **First half** of 32 streamed decoder layers' FC weights only |
| `dense_weights_streaming_1.bin` | 776 MB | **Second half** of same 32 layers' FC weights |

**Why streaming files are smaller than the language model:**
- Streaming only includes FC weight tensors (not scale/zp/small constants)
- Only 32 layers are streamed (layers 5-36); head/tail layers stay pinned in GPU memory
- Scale and zero-point tensors (~5 MB/layer) remain in GPU memory (they get reordered by GPU compiler)
- Total: 776 MB × 2 = 1.55 GB out of 2.69 GB

**Per-layer embedding offload (automatic):**
- When `openvino_text_embeddings_per_layer_model_revised.bin` exists in the model directory, VLMPipeline automatically uses memory-mapped (mmap) access
- Each token only reads ~10 KB from the 3 GB file (embedding lookup)
- This saves ~2.82 GB of GPU/system memory — critical for 8 GB systems
- No configuration needed — just keep the file in the model directory

---

## Installation Steps

### 1. Install Python 3.12 and dependencies

```powershell
pip install openvino==2026.2.0 openvino-genai openvino-tokenizers
pip install -r requirements.txt
```

### 2. Replace the GPU plugin DLL

Copy the modified `openvino_intel_gpu_plugin.dll` over the default one:

```powershell
# Find your OpenVINO installation
$ovLibs = python -c "import openvino; import os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))"

# Backup original
Copy-Item "$ovLibs\openvino_intel_gpu_plugin.dll" "$ovLibs\openvino_intel_gpu_plugin.dll.bak"

# Install modified version
Copy-Item "dlls\openvino_intel_gpu_plugin.dll" "$ovLibs\openvino_intel_gpu_plugin.dll"
```

### 3. Place model files

Copy the `model/` folder to your preferred location (fastest NVMe recommended).

### 4. Place streaming stripe files

**Single NVMe setup:**
```powershell
# Place both stripe files on the same drive (still works, ~10% benefit from extra parallelism)
Copy-Item "streaming_nvme0\dense_weights_streaming_0.bin" "C:\model_dir\"
Copy-Item "streaming_nvme1\dense_weights_streaming_1.bin" "C:\model_dir\"
```

**Dual NVMe setup (recommended):**
```powershell
# Place stripe 0 on NVMe 0 (e.g., C:\)
Copy-Item "streaming_nvme0\dense_weights_streaming_0.bin" "C:\model_dir\"
Copy-Item "streaming_nvme0\dense_weights_streaming_0.json" "C:\model_dir\"

# Place stripe 1 on NVMe 1 (e.g., D:\)
mkdir "D:\gemma4_streaming"
Copy-Item "streaming_nvme1\dense_weights_streaming_1.bin" "D:\gemma4_streaming\"
```

---

## Running Inference

> **Note:** `run_gemma4.py` automatically handles:
> - `CACHE_DIR` — auto-detects `model/model_cache/` for compiled kernel caching
> - Per-layer embedding offload — auto-detects `_revised.bin` and uses mmap
> - No extra flags needed for these features.

### Single NVMe Mode

```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\model_dir\dense_weights_streaming_0.bin"
python run_gemma4.py --model-dir "C:\model_dir" --prompt "What is quantum computing?"
```

### Dual NVMe Mode (2x bandwidth)

```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\model_dir\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2 = "D:\gemma4_streaming\dense_weights_streaming_1.bin"
python run_gemma4.py --model-dir "C:\model_dir" --prompt "What is quantum computing?"
```

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OV_DENSE_STREAM_WEIGHTS` | (none) | Path to stripe file 0 (enables streaming) |
| `OV_DENSE_STREAM_WEIGHTS_2` | (none) | Path to stripe file 1 (enables dual-NVMe) |
| `OV_DENSE_STREAM_NO_PREFETCH` | `0` | Set to `1` to disable prefetch (debug only) |
| `OV_DENSE_STREAM_DEBUG` | `0` | Set to `1` for verbose debug output |

---

## Expected Performance

Tested on Intel Panther Lake (12 Xe EUs, 16 GB LPDDR5):

| Mode | TPOT | tok/s | Notes |
|------|------|-------|-------|
| No streaming (all in memory) | 42 ms | 24.0 | Requires all weights in GPU memory |
| v2 Pipeline (single NVMe) | 143 ms | 7.0 | ~12 GB/s NVMe bandwidth |
| **v2 Dual-path (same disk)** | **129 ms** | **7.8** | Extra parallelism on single NVMe |
| **v2 Dual NVMe (2 drives)** | **~77 ms** | **~13** | ~24 GB/s combined bandwidth |

---

## Correctness Verification

Tested on 2026-05-19 with dual-path (same disk) — all answers correct:

```
Test1: "What is the capital of Japan?" → Tokyo ✅
Test2: "What is 7 * 8?"               → 56 ✅
Test3: "Translate to French: Hello"    → Bonjour, comment allez-vous? ✅
```

To run your own verification:
```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\model_dir\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2 = "D:\streaming\dense_weights_streaming_1.bin"
python run_gemma4.py --model-dir "C:\model_dir" --prompt "What is the capital of Japan?" --max-new-tokens 10
# Expected: "Tokyo"
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `[DenseStreaming]` messages not appearing | Ensure `OV_DENSE_STREAM_WEIGHTS` is set before running |
| "Cannot open file" error | Check file path, ensure no other process locks the file |
| Same TPOT as without streaming (~42ms) | Weights already fit in memory — streaming not needed |
| Very high TPOT (>200ms) | Check NVMe health, close other disk-heavy applications |
| First run takes extra 12s | Normal — compiling GPU kernels (cached for subsequent runs) |
| model_cache not used | Set `CACHE_DIR` in script or copy model_cache/ to model dir |

---

## Regenerating Stripe Files

If you modify the model or want different group sizes:

```powershell
python pack_dense_weights_dual.py `
    --model_dir "C:\model_dir" `
    --output_dir "C:\output" `
    --first_streamed 5 --last_streamed 36 `
    --group_size 4
```

This produces:
- `dense_weights_streaming_0.bin` + `.json` (NVMe 0)
- `dense_weights_streaming_1.bin` (NVMe 1)
- `dense_weights_streaming_dual.json` (combined metadata)

---

## Architecture Notes

### Dense Weight Streaming Pipeline (per token)

```
Token N generation:
  ┌─ GPU executes pinned HEAD layers (0-4) ─────────────────────┐
  │                                                              │
  │  Meanwhile: NVMe loads Group 0 into Buffer A                 │
  │  (Dual mode: NVMe0 loads first half, NVMe1 loads second half)│
  └──────────────────────────────────────────────────────────────┘
  
  For each Group G (0..7):
    1. Wait IO complete (Group G already prefetched)
    2. Swap weight pointers → Buffer with Group G data
    3. Re-bind kernel arguments
    4. Prefetch Group G+1 into other buffer (async)
    5. GPU executes Group G layers
  
  ┌─ GPU executes pinned TAIL layers (37-41) ───────────────────┐
  └──────────────────────────────────────────────────────────────┘
```

### File Format (V2)

```
Header (48 + 8*num_groups bytes):
  [0-3]   magic: "DNSW"
  [4-5]   version: 2
  [6-7]   num_layers: 32
  [8-9]   num_groups: 8
  [10-11]  layers_per_group: 4
  [12-15]  first_streamed_layer: 5
  [16-19]  last_streamed_layer: 36
  [20-23]  sector_size: 512
  [24-31]  total_weight_bytes
  [32-39]  per_layer_size
  [40-43]  reserved
  [44-45]  num_stripes: 2
  [46-47]  stripe_index: 0 or 1
  [48+]    full_group_aligned_bytes[num_groups] (uint64 each)

Group Table (16 bytes per group):
  [0-3]   first_layer
  [4-7]   num_layers
  [8-15]  file_offset (to this stripe's portion)
  [16-23] group_bytes (this stripe's portion size)

Layer Table (per group, 16 bytes per layer):
  [0-7]   offset_in_group
  [8-15]  layer_bytes
```

---

## Version History

- **v2.0 (2026-05-19):** Dual-NVMe parallel IO, Group-Half Striping
- **v1.1 (2026-05-15):** Pipeline overlap (async prefetch), no_prefetch flag
- **v1.0 (2026-05-11):** Initial FC weight streaming with group execution
