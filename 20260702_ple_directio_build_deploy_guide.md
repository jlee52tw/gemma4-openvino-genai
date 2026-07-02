# 2026-07-02: Per-Layer Embedding DirectIO Offload — Build, Deploy & Test Guide

> **Model:** `gemma-4-E4B-it-ov-qat-w4a16` (w4a16-ct QAT, 6.02 GB OV IR)  
> **System:** Intel Panther Lake, Arc B390 iGPU (96 EUs), 32 GB LPDDR5 8533 MT/s, Windows 11  
> **OV GenAI source:** `C:\working\gemma4\openvino_genai_src` (branch `as/vlm_enable_1`, commit `4f296476`)  
> **OV 2026.2 package (cmake reference):** `C:\working\gemma4\openvino_genai_windows_2026.2.0.0_x86_64`  
> **Python venv:** `C:\working\gemma4\.venv` (Python 3.12, transformers 5.12.1)

---

## What Is PLE Offload?

Gemma4 E4B uses a dedicated per-layer embedding model
(`openvino_text_embeddings_per_layer_model.xml/.bin`) to look up
token-by-token embedding vectors before each forward pass. At inference this
model is normally compiled onto the GPU alongside the main language model,
consuming ~2.7 GB of shared RSS.

**PLE DirectIO offload** replaces the GPU-compiled path with a Win32
`FILE_FLAG_NO_BUFFERING` reader that reads the packed binary row-by-row at
token lookup time, bypassing both the GPU compilation step and the OS page cache.
The per-layer model weight memory never enters RSS.

---

## Result: RSS vs Perf (2026-07-02, 32 GB system)

### PLE OFF (GPU compiled per-layer model — baseline)

| Prompt | Tok/s (3 runs) | TTFT | Peak RSS |
|--------|---------------|------|---------|
| short-text (7 tok) | 22.33 / 20.21 / 21.17 | ~0.35 s | **6.93 GB** |
| long-text (1024 tok) | 20.37 / 20.35 | ~0.86 s | **7.59 GB** |
| short-image | 23.48 / 23.08 / 22.46 | ~0.49 s | **7.78 GB** |

### PLE ON (DirectIO binary reader — this patch)

| Prompt | Tok/s (3 runs) | TTFT | Peak RSS | RSS Saved |
|--------|---------------|------|---------|-----------|
| short-text (7 tok) | 25.14 / 25.04 / 24.15 | ~0.28 s | **4.29 GB** | **−2.64 GB** |
| long-text (1024 tok) | 19.02 / 19.06 / 17.93 | ~0.96 s | **4.87 GB** | **−2.72 GB** |
| short-image | 23.35 / 21.97 / 22.07 | ~0.53 s | **5.12 GB** | **−2.66 GB** |

### Summary

| Metric | PLE OFF | PLE ON | Delta |
|--------|---------|--------|-------|
| Peak RSS (worst case) | 7.78 GB | 5.12 GB | **−2.66 GB (−34%)** |
| short-text tok/s | ~21.2 | ~24.4 | **+15% ↑** |
| long-text tok/s | ~20.4 | ~18.7 | −8% ↓ (sequential NVMe I/O for 1024 tokens) |
| short-image tok/s | ~23.0 | ~22.5 | −2% ≈ parity |

> **Key insight:** Short-text improves because GPU is no longer time-sharing with the
> per-layer compiled model. Long-text regresses slightly because 1024 sequential
> DirectIO reads (~12.5 MB) add to TTFT. On an 8 GB system the 2.7 GB RSS savings
> are critical for fitting in memory.

---

## File Layout After Patch

```
openvino_genai_src/src/cpp/src/visual_language/gemma4/
├── classes.hpp                        ← MODIFIED: add include + m_per_layer_reader member
├── classes.cpp                        ← MODIFIED: constructor dispatch + get_per_layer_embeddings
└── per_layer_embedding_reader.hpp     ← NEW: Win32 DirectIO reader class

models/gemma-4-E4B-it-ov-qat-w4a16/
└── per_layer_embedding_directio.bin   ← GENERATED: 3.22 GB packed binary (PLEB v2 format)

.venv/Lib/site-packages/openvino_genai/
├── openvino_genai.dll                 ← REPLACED: patched build (4.85 MB)
└── openvino_genai.dll.bak_official    ← BACKUP: original official DLL (5.60 MB)
```

---

## Step 1 — Source Patches

Three files need to be patched/created in `openvino_genai_src`.

### 1a. Create `per_layer_embedding_reader.hpp` (new file)

Path: `src/cpp/src/visual_language/gemma4/per_layer_embedding_reader.hpp`

This is a self-contained header-only class.  The file lives in the repository at
`gemma4-openvino-genai` under the same relative path for reference.

Key design points:
- Win32 `FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN` (bypasses OS page cache)
- `VirtualAlloc`-aligned 4096-byte I/O buffer (satisfies NVMe 512-byte sector alignment)
- Per-token read via `ReadFile` with `OVERLAPPED` struct (positional read, no `SetFilePointerEx`)
- **Header-aware:** reads PLEB magic + version from file header at open
  - v1 (INT8 weight): `float(int8[i]) × fp16_scale × 16.0f`  — q4_0 model
  - v2 (UINT8 weight + UINT8 ZP): `(float(uint8[i]) − zp) × fp16_scale × 16.0f` — w4a16-ct model
- Special token remapping: IDs `{258880, 258884, 258881}` → row 0; OOV → zeros

### 1b. Patch `classes.hpp`

Add include and new member:

```diff
 #include <filesystem>

 #include "visual_language/inputs_embedder.hpp"
+#include "visual_language/gemma4/per_layer_embedding_reader.hpp"
 #include "visual_language/vision_encoder.hpp"

...

-    // Per-layer text embeddings model (Gemma4-specific)
-    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_per_layer_embeddings_requests;
+    // Per-layer text embeddings: either DirectIO binary reader OR compiled GPU model
+    std::unique_ptr<PerLayerEmbeddingReader>               m_per_layer_reader;
+    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_per_layer_embeddings_requests;
```

### 1c. Patch `classes.cpp`

In `InputsEmbedderGemma4::InputsEmbedderGemma4(model_dir ...)` constructor, add DirectIO branch **before** the existing compile_model call:

```diff
     : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
+    // Prefer DirectIO binary if present alongside the model
+    auto directio_bin = model_dir / "per_layer_embedding_directio.bin";
+    if (std::filesystem::exists(directio_bin)) {
+        m_per_layer_reader = std::make_unique<PerLayerEmbeddingReader>(directio_bin);
+        return;
+    }
+
     auto per_layer_model_path = model_dir / "openvino_text_embeddings_per_layer_model.xml";
```

In `get_per_layer_embeddings()`, add DirectIO dispatch **before** the GPU path:

```diff
 ov::Tensor InputsEmbedderGemma4::get_per_layer_embeddings(const ov::Tensor& input_ids) {
+    // DirectIO path: bypass GPU compiled model
+    if (m_per_layer_reader) {
+        return m_per_layer_reader->get_embeddings(input_ids);
+    }
+
     OPENVINO_ASSERT(m_per_layer_embeddings_requests, "Per-layer embeddings model is not loaded");
```

---

## Step 2 — Build

Open **x64 Native Tools Command Prompt for VS 2022** or use PowerShell from VS dev env.

### 2a. CMake Configure

```powershell
cd C:\working\gemma4\openvino_genai_src

cmake -B build -G "Visual Studio 17 2022" `
    "-DOpenVINO_DIR=C:\working\gemma4\openvino_genai_windows_2026.2.0.0_x86_64\runtime\cmake" `
    -DENABLE_PYTHON=OFF `
    -DENABLE_JS=OFF `
    -DENABLE_TESTS=OFF `
    -DENABLE_TOOLS=OFF `
    -DENABLE_SAMPLES=OFF `
    -DENABLE_GGUF=ON `
    -DENABLE_XGRAMMAR=OFF
```

> **Note:** `ENABLE_GGUF=ON` is required even though we don't need GGUF. There is a
> pre-existing source tree bug where `gguf_tokenizer.cpp` always compiles and includes
> `gguflib.h` (which is only fetched when GGUF is enabled).

Expected output:
```
-- Configuring done
-- Generating done
-- Build files have been written to: C:/working/gemma4/openvino_genai_src/build
```

### 2b. Build

```powershell
cd C:\working\gemma4\openvino_genai_src
cmake --build build --target openvino_genai --config Release -- /p:CL_MPCount=6
```

Build takes ~5–10 minutes (first time; incremental is ~30 s for single file change).

Output DLL: `build\openvino_genai\openvino_genai.dll` (≈4.85 MB)

### Known Build Gotchas

| Issue | Fix |
|-------|-----|
| `error C2589: '(' illegal token on right side of '::'` in `classes.cpp` | `<windows.h>` min/max macros — ensure `#define NOMINMAX` is before `#include <windows.h>` in the reader header |
| `Cannot open include file: 'gguflib.h'` | Reconfigure with `-DENABLE_GGUF=ON` |

---

## Step 3 — Deploy

Back up the original venv DLL first (only needed once):

```powershell
$venv_genai = "C:\working\gemma4\.venv\Lib\site-packages\openvino_genai"

# One-time backup of official DLL
if (-not (Test-Path "$venv_genai\openvino_genai.dll.bak_official")) {
    Copy-Item "$venv_genai\openvino_genai.dll" "$venv_genai\openvino_genai.dll.bak_official"
}

# Deploy patched DLL
Copy-Item "C:\working\gemma4\openvino_genai_src\build\openvino_genai\openvino_genai.dll" `
    "$venv_genai\openvino_genai.dll" -Force

Write-Host "Deployed: $((Get-Item "$venv_genai\openvino_genai.dll").Length) bytes"
```

### Restore original DLL (if needed)

```powershell
$venv_genai = "C:\working\gemma4\.venv\Lib\site-packages\openvino_genai"
Copy-Item "$venv_genai\openvino_genai.dll.bak_official" "$venv_genai\openvino_genai.dll" -Force
```

### Verify DLL loads

```powershell
& C:\working\gemma4\.venv\Scripts\python.exe -c `
    "import openvino_genai; print('OK version:', openvino_genai.__version__)"
```

Expected: `OK version: 2026.2.0.0-3030-4f296476d4f-as/vlm_enable_1`

---

## Step 4 — Generate PLE Binary

The packed binary must be generated once per model directory. It is **not** part of the
model conversion step.

```powershell
cd C:\working\gemma4

.\.venv\Scripts\python.exe gemma4-openvino-genai\pack_per_layer_embedding.py `
    --model-dir "gemma4-openvino-genai\models\gemma-4-E4B-it-ov-qat-w4a16" `
    --output    "gemma4-openvino-genai\models\gemma-4-E4B-it-ov-qat-w4a16\per_layer_embedding_directio.bin" `
    --verify
```

Expected output:
```
Format: v2 (UINT8 weight, UINT8 ZP)
Scale range: [0.001444, 0.007244]  NaN count: 0
Done! Output: 3.221 GB in ~3 s
Output size: ✅ correct (3221229568 bytes)
...
Max absolute error: 0.000000   ✅ All 20 samples PASS
```

### Binary Format Details (w4a16-ct = v2)

```
Source binary layout (openvino_text_embeddings_per_layer_model.bin):
  offset           0 : UINT8 weight [262144, 10752] = 2,818,572,288 bytes
  offset 2818572288 : UINT8 ZP      [262144,   1]   =       262,144 bytes
  offset 2818834432 : FP16  scale   [262144,   1]   =       524,288 bytes

Packed binary layout (per_layer_embedding_directio.bin, PLEB v2):
  [0,     4096) : 4096-byte header  (magic "PLEB", version=2, metadata)
  [4096,  ...)  : 262144 rows × 12288 bytes
    Row = [UINT8 weight[10752] | UINT8 ZP[1] | FP16 scale[2] | pad[1533]]

Dequant: (float(uint8_weight[i]) - float(uint8_zp)) × float(fp16_scale) × 16.0
```

### Toggle PLE on/off without rebuilding

```powershell
# Disable PLE (GPU compiled model path)
Rename-Item "...\per_layer_embedding_directio.bin" "per_layer_embedding_directio.bin.off"

# Re-enable PLE
Rename-Item "...\per_layer_embedding_directio.bin.off" "per_layer_embedding_directio.bin"
```

---

## Step 5 — Test

### Quick smoke test (single run)

```powershell
cd C:\working\gemma4\gemma4-openvino-genai
& C:\working\gemma4\.venv\Scripts\python.exe benchmark.py `
    --model-dir models\gemma-4-E4B-it-ov-qat-w4a16 `
    --device GPU `
    --warmup 1 `
    --runs 1
```

Watch for `Model loaded in ~9s, RSS: ~4.x GB` (vs ~6.9 GB without PLE).

### Full benchmark (3 runs)

```powershell
cd C:\working\gemma4\gemma4-openvino-genai
& C:\working\gemma4\.venv\Scripts\python.exe benchmark.py `
    --model-dir models\gemma-4-E4B-it-ov-qat-w4a16 `
    --device GPU `
    --warmup 1 `
    --runs 3 `
    --output-csv results_ple_on.csv
```

### Verify binary dequant correctness (no GPU needed)

```powershell
cd C:\working\gemma4
.\.venv\Scripts\python.exe gemma4-openvino-genai\pack_per_layer_embedding.py `
    --model-dir "gemma4-openvino-genai\models\gemma-4-E4B-it-ov-qat-w4a16" `
    --output    "gemma4-openvino-genai\models\gemma-4-E4B-it-ov-qat-w4a16\per_layer_embedding_directio.bin" `
    --verify-only
```

Expected: `✅ All 20 samples PASS (abs_err < 0.01)`

---

## Disk Footprint

Location: `C:\working\gemma4\gemma4-openvino-genai\models\gemma-4-E4B-it-ov-qat-w4a16\`

| File | Size | Loaded at runtime? |
|------|------|--------------------|
| `per_layer_embedding_directio.bin` | 3.22 GB | Read row-by-row via DirectIO, **never in RSS** |
| `openvino_text_embeddings_per_layer_model.bin` | 2.69 GB | **No** — skipped when directio.bin present |
| `openvino_language_model.bin` | 2.62 GB | Yes → GPU |
| `openvino_text_embeddings_model.bin` | 0.63 GB | Yes → GPU |
| `openvino_vision_embeddings_model.bin` | 0.10 GB | Yes → GPU |
| Other (xml, json, tokenizer) | ~0.06 GB | Small |
| **Total directory** | **9.32 GB** | |
| **OV IR only (without directio.bin)** | **6.10 GB** | |

> The 2.69 GB source `.bin` can be deleted after generating `per_layer_embedding_directio.bin`
> if disk space is tight — the directio.bin is the only file needed for PLE at runtime.
> **Do NOT delete it** if you may need to re-run `pack_per_layer_embedding.py --verify`.

---

## Source Tree Quick Reference

```
openvino_genai_src/
└── src/cpp/src/visual_language/gemma4/
    ├── per_layer_embedding_reader.hpp   ← NEW (Win32 DirectIO, POSIX O_DIRECT fallback)
    ├── classes.hpp                      ← PATCHED (+include, +m_per_layer_reader member)
    └── classes.cpp                      ← PATCHED (constructor + dispatch in get_per_layer_embeddings)

gemma4-openvino-genai/
├── pack_per_layer_embedding.py          ← UPDATED (v1/v2 auto-detect, ZP support, --verify)
└── models/gemma-4-E4B-it-ov-qat-w4a16/
    └── per_layer_embedding_directio.bin ← GENERATED (3.22 GB, PLEB v2)

.venv/Lib/site-packages/openvino_genai/
├── openvino_genai.dll                   ← PATCHED (deployed from build/)
└── openvino_genai.dll.bak_official      ← BACKUP of original official DLL
```

---

## Reproducing from Scratch on a New Session

```powershell
# 1. Activate venv
& C:\working\gemma4\.venv\Scripts\Activate.ps1

# 2. Apply patches (already in repo — skip if openvino_genai_src already patched)
#    - Create per_layer_embedding_reader.hpp
#    - Edit classes.hpp  (add include + m_per_layer_reader)
#    - Edit classes.cpp  (constructor dispatch + get_per_layer_embeddings dispatch)

# 3. Configure (one-time)
cd C:\working\gemma4\openvino_genai_src
cmake -B build -G "Visual Studio 17 2022" `
    "-DOpenVINO_DIR=C:\working\gemma4\openvino_genai_windows_2026.2.0.0_x86_64\runtime\cmake" `
    -DENABLE_PYTHON=OFF -DENABLE_JS=OFF -DENABLE_TESTS=OFF -DENABLE_TOOLS=OFF `
    -DENABLE_SAMPLES=OFF -DENABLE_GGUF=ON -DENABLE_XGRAMMAR=OFF

# 4. Build
cmake --build build --target openvino_genai --config Release -- /p:CL_MPCount=6

# 5. Deploy DLL
Copy-Item "build\openvino_genai\openvino_genai.dll" `
    "C:\working\gemma4\.venv\Lib\site-packages\openvino_genai\openvino_genai.dll" -Force

# 6. Generate packed binary (skip if per_layer_embedding_directio.bin already exists)
cd C:\working\gemma4
.\.venv\Scripts\python.exe gemma4-openvino-genai\pack_per_layer_embedding.py `
    --model-dir "gemma4-openvino-genai\models\gemma-4-E4B-it-ov-qat-w4a16" `
    --output    "gemma4-openvino-genai\models\gemma-4-E4B-it-ov-qat-w4a16\per_layer_embedding_directio.bin"

# 7. Benchmark
cd C:\working\gemma4\gemma4-openvino-genai
.\.venv\Scripts\python.exe benchmark.py `
    --model-dir models\gemma-4-E4B-it-ov-qat-w4a16 --device GPU --warmup 1 --runs 3
```
