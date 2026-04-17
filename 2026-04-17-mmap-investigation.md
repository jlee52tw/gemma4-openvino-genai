# 2026-04-17 — ENABLE_MMAP Investigation & Pre-16GB Checkpoint

## Summary

Before swapping to 16 GB memory, we investigated why `ENABLE_MMAP='NO'` shows
**identical peak memory** to mmap ON (13.07 GB vs 13.06 GB) on VLMPipeline with
GPU device. On the native OpenVINO API path (CPU), mmap OFF is expected to
reduce peak RSS because file-backed pages are replaced by heap-allocated buffers
that are freed after model compilation. On GPU (integrated), the behavior differs.

---

## 為什麼 VLMPipeline 在 GPU 上 mmap ON/OFF 的 Peak Memory 相同？（詳細機制）

### 背景

在 CPU 推論路徑中，`ENABLE_MMAP=NO` 確實能降低尖峰記憶體（peak RSS），因為
CPU plugin 的 `compile_model()` 可以**直接引用 mmap 頁面**作為權重來源，不需要
複製。關閉 mmap 會多一份 heap 拷貝，反而增加記憶體。

但在 **integrated GPU (iGPU)** 上，我們觀察到 mmap ON 和 OFF 的 peak 完全相同
（~13 GB）。以下是完整的技術分析。

### VLMPipeline 載入流程（以 Gemma4 E4B 為例）

```
時間軸 →

[Step 1]  read_model("openvino_language_model.xml")
          ├─ mmap ON:  OS 做 memory-map，.bin 檔案映射到虛擬位址空間（file-backed pages）
          └─ mmap OFF: malloc + fread，配置 heap 記憶體讀入 .bin 全部內容
          ⟹ CPU 記憶體 +4.5 GB（language model 權重）

[Step 2]  compile_model(language_model, "GPU", properties)
          ├─ GPU plugin 巡訪所有 Constant 節點
          ├─ 為每個權重張量配置 OpenCL buffer（GPU 可存取的共享系統記憶體）
          └─ 將 CPU 端的權重資料 **完整複製** 到 GPU buffer
          ⟹ GPU 記憶體 +6.9 GB（含對齊/padding）
          ⟹ 此時 CPU 端原始權重 **仍然存活**（shared_ptr 還在 scope 內）
          ⟹ ★ 尖峰 = CPU 權重 + GPU 權重 同時存在 ★

[Step 3]  InputsEmbedder 建構
          ├─ VisionEncoder:  compile_model(path, "GPU") — path-based，CPU 端暫存自動釋放 ✓
          ├─ EmbeddingsModel: read_model() + compile_model() — local var，constructor 結束釋放 ✓
          └─ PerLayerEmbeddings: compile_model(path, "GPU") — path-based ✓
          ⟹ ★ 但 language_model 的 CPU 權重持續佔用記憶體，因為外層 scope 還持有 shared_ptr ★

[Step 4]  finalize_initialization(language_model, ...)
          ├─ 從 language_model 讀取 cache_types
          └─ language_model shared_ptr 仍存活

[Step 5]  VLMPipeline constructor return
          └─ language_model 離開 scope → shared_ptr destroy → CPU 權重終於釋放
          ⟹ RSS 降至 ~7.4 GB（只剩 GPU compiled model + runtime overhead）
```

### 記憶體時間線圖

```
記憶體
(GB)
 13 ─ ─ ─ ─ ─┬─────────────────────────────────┐ ← PEAK（CPU weights + GPU weights 共存）
              │  language_model CPU 權重          │
              │  +                                │
              │  compiled GPU language model      │
              │  +                                │
              │  vision/embedding 短暫配置        │
  7 ─ ─ ─ ─ ─┤─────────────────────────────────┘ ← language_model scope 結束，CPU 權重釋放
              │  只剩 GPU compiled models
              │  + runtime buffers
              │  + KV cache（隨推論增長）
  0 ──────────┴──────────────────────────────→ 時間
         load        compile      inference
```

### 為什麼 mmap ON 和 OFF 的 peak 相同？

| | mmap ON | mmap OFF |
|---|---|---|
| `read_model` 時 | OS 建立 file-backed mapping，pages 按需載入 | `malloc` + `fread`，全部讀入 heap |
| `compile_model` 時 | GPU plugin 逐頁存取 mmap pages → **OS 將它們全部載入 RSS** | GPU plugin 存取 heap buffer（已在 RSS 中） |
| compile 完成後 | **mmap pages 仍在 RSS 中**（shared_ptr 持有 mapping） | **heap buffer 仍在 RSS**（shared_ptr 持有 Model） |
| peak 時刻 | file-backed pages (~4.5 GB) + GPU buffers (~6.9 GB) ≈ 13 GB | heap pages (~4.5 GB) + GPU buffers (~6.9 GB) ≈ 13 GB |

**關鍵點：** 在 `compile_model()` 執行期間，GPU plugin 必須存取**所有**權重頁面
（不論是 mmap 還是 heap），所以它們全部都會出現在 working set / RSS 中。
同時 GPU compiled model 的 buffers 也在系統記憶體中（iGPU 共享記憶體）。
兩者重疊的瞬間就是 peak。

**mmap ON 和 OFF 的唯一差別**是頁面的*類型*（file-backed vs anonymous），
而不是*數量*。Windows 的 Peak Working Set 兩種都計算在內。

### 為什麼 CPU 路徑上 mmap 有幫助？

在 CPU 上，`compile_model()` 可以選擇**直接引用 mmap 頁面**作為推論時的權重來源
（zero-copy）。不需要另外配置記憶體複製權重。所以：

- mmap ON + CPU: peak ≈ 1× model size（mmap pages 既是來源也是推論用）
- mmap OFF + CPU: peak ≈ 2× model size（heap 拷貝 + compiled model 拷貝）

GPU 無法做 zero-copy（需要特殊格式的 OpenCL buffer），所以永遠是 2×。

---

## 可行的修復方案

### 方案 1：Sequential Read + Compile + Release（推薦）

**目前的問題：** `language_model` 的 `shared_ptr<ov::Model>` 從 pipeline.cpp:797
一路存活到 constructor return（line 822），橫跨了所有子模型的載入。

**修改方向：** 在 `compile_model()` 之後立即釋放 `language_model`：

```cpp
// ── 修改前（pipeline.cpp:797-822）──
auto language_model = read_model(language_model_path, {}, properties);
// ... transformations ...
// language_model 持續存活到 line 822

// ── 修改後 ──
auto language_model = read_model(language_model_path, {}, properties);
// 提前擷取 finalize 需要的資訊
auto cache_types = utils::get_cache_types(*language_model);
auto kv_pos = utils::get_kv_axes_pos(language_model);
// ... transformations ...
auto compiled_lm = compile_model(language_model, device, lm_properties);
language_model.reset();  // ★ 立即釋放 ~4.5 GB CPU 權重 ★

// 接下來載入 vision/embedding 子模型時，peak 會低 ~4.5 GB
auto inputs_embedder = std::make_shared<InputsEmbedder>(...);
```

**預估節省：** ~4.5 GB（language model 的 CPU 端權重在載入其他子模型前就釋放）

**可行性：** ★★★★★ — image_generation pipeline 已有相同的 `m_model.reset()` 模式
（見 `autoencoder_kl.cpp:257`, `clip_text_model.cpp:124`）。這是 codebase 中已驗證的做法。

### 方案 2：使用 path-based compile_model

```cpp
// 直接用路徑，不經過 read_model
auto compiled_lm = compile_model(language_model_path, device, properties);
```

**問題：** 無法在 compile 前做 graph transformations（`apply_slice_before_matmul`）。
除非把 transformation 移到 compile 內部或用其他機制。**不建議**，改動太大。

### 方案 3：結合 mmap + madvise/VirtualUnlock

在 `compile_model()` 之後呼叫 OS API 釋放 mmap pages：
- Linux: `madvise(MADV_DONTNEED)` 
- Windows: `EmptyWorkingSet(GetCurrentProcess())`

**問題：** VLMPipeline 層級無法存取底層 mmap 指標。需要在 OpenVINO Core 層面實作。**不實際**。

### 建議

**方案 1 是最佳解**。改動量小（~10-15 行），有既有 codebase 先例，預估在 16 GB 系統上可將 peak 從 ~13 GB 降至 ~8-9 GB，大幅改善記憶體壓力。

值得向 openvino.genai 提交 PR 或 issue。

---

## Memory Profile (32 GB, Gemma-4-E4B-it INT4, GPU)

| Stage | mmap ON | mmap OFF | Delta |
|---|---:|---:|---:|
| Before load | 0.06 GB | 0.06 GB | — |
| **After load (peak)** | **13.07 GB** | **13.06 GB** | **~0 GB** |
| Post-load stabilized | 7.43 GB | 7.43 GB | ~0 GB |
| After warmup gen | 7.47 GB | 7.47 GB | ~0 GB |
| After short inference | 7.59 GB | 7.59 GB | ~0 GB |
| After long inference | 7.93 GB | 7.92 GB | ~0 GB |
| Final settle | 7.87 GB | 7.81 GB | ~0 GB |
| After del+gc | 1.33 GB | 1.32 GB | ~0 GB |

---

## Source Code Analysis

Inspected the openvino.genai source at
`C:\working\gemma4-openvino\openvino_genai_src` (PR#3644, branch `as/vlm_enable_1`).

### ENABLE_MMAP IS correctly forwarded in VLMPipeline

| Model component | Load call | ENABLE_MMAP passed? |
|---|---|---|
| `openvino_language_model.xml/.bin` | `singleton_core().read_model(path, {}, properties)` in `pipeline.cpp:797` | **YES** |
| `openvino_vision_embeddings_model.xml/.bin` | `singleton_core().compile_model(path, device, properties)` in `vision_encoder.cpp:24` | **YES** (compile_model from path reads internally) |
| `openvino_text_embeddings_model.xml/.bin` | `core.read_model(path, {}, properties)` in `embedding_model.cpp:48` | **YES** |
| `openvino_text_embeddings_per_layer_model.xml/.bin` | same EmbeddingsModel path | **YES** |
| `openvino_tokenizer.xml/.bin` | `core.read_model(path, {}, filtered_properties)` in `tokenizer_impl.cpp:338` | **YES** |

### No code bug — ENABLE_MMAP property flows correctly through all sub-models

The VLMPipeline constructor:
1. Extracts `ATTENTION_BACKEND` (only) and passes remaining properties (including `ENABLE_MMAP`) onward
2. Calls `read_model(language_model_path, {}, properties)` — ENABLE_MMAP controls mmap here
3. Passes properties into `InputsEmbedder` which forwards to `VisionEncoder`, `EmbeddingsModel`, `Tokenizer`
4. All call `ov::Core::read_model()` or `ov::Core::compile_model()` with `ENABLE_MMAP` in the properties map

This matches `LLMPipeline`'s flow — both correctly forward `ENABLE_MMAP` to `ov::Core::read_model()`.

---

## Root Cause: Why Peak Memory is Identical on iGPU

### The GPU plugin uses shared system memory

```
DEVICE_TYPE:              Type.INTEGRATED
FULL_DEVICE_NAME:         Intel(R) Arc(TM) B390 GPU (iGPU)
GPU_DEVICE_TOTAL_MEM_SIZE: 27,296,473,088 (~25.4 GB shared)
```

The Arc B390 is an **integrated GPU** — it has no dedicated VRAM. It shares
system DRAM with the CPU. When `compile_model()` runs for GPU:

1. **With mmap ON:** `read_model()` memory-maps the `.bin` files → file-backed
   pages in process address space. Then `compile_model()` copies/transforms
   weights into GPU-accessible shared memory buffers. Peak = mmap pages + GPU
   buffers coexisting. After compile, OS can evict mmap pages → RSS drops.

2. **With mmap OFF:** `read_model()` reads `.bin` into heap (anonymous pages).
   Then `compile_model()` copies/transforms weights into GPU-accessible shared
   memory buffers. Peak = heap pages + GPU buffers coexisting. After compile,
   heap is freed → RSS drops.

**In both cases, the peak is dominated by the model weights existing in two
places simultaneously** during the `read_model` → `compile_model` transition:

- ~6.05 GB (model weights on disk, read into memory via mmap or heap)
- ~6.9 GB (GPU-compiled model in shared system memory, with padding/alignment)
- Total peak ≈ 13 GB regardless of mmap mode

### Why it matters on CPU but not GPU

On **CPU** (native OV path), `compile_model` can potentially **use the mmap'd
pages directly** as the weight source — no copy needed. Disabling mmap forces a
heap copy, doubling memory. Enabling mmap means the weights are file-backed and
OS can reclaim them → lower peak.

On **integrated GPU**, `compile_model` always copies weights into GPU-accessible
buffers (different memory format/layout). The source pages (mmap or heap) are
**always discarded** after compile. So mmap ON vs OFF makes no difference to
peak — peak is always ~2× model size during the transition.

### The previous np.fromfile() workaround showed higher peak because...

The old manual loading approach created **Python numpy arrays** (heap) + then
**ov.Tensor** wrappers, then passed them to VLMPipeline's second constructor.
The Python-side numpy arrays persisted (Python GC doesn't free immediately),
creating a **third copy** of weights in process memory during pipeline
construction. This inflated peak to 13.7–14.6 GB.

---

## Conclusion

| Aspect | Finding |
|---|---|
| Code correctness | `ENABLE_MMAP` is correctly passed through all VLMPipeline sub-model loading paths |
| Peak memory behavior | Identical on iGPU because GPU compile always copies weights — source (mmap/heap) doesn't matter |
| CPU vs GPU difference | On CPU, mmap pages can be used in-place; on iGPU, weights are always copied to shared GPU memory |
| Practical impact on 16 GB | ENABLE_MMAP alone won't help reduce peak. But **CACHE_DIR + ENABLE_MMAP='NO'** reduces peak from 13 GB to ~10 GB. |
| Old workaround impact | np.fromfile() inflated peak by ~1.5 GB due to extra Python-side copies |

---

## CACHE_DIR Discovery — Peak Memory Reduction

After investigating the 2× peak issue, we discovered that `CACHE_DIR` (GPU compiled
model cache) combined with `ENABLE_MMAP='NO'` dramatically reduces peak memory.

### How It Works

1. **First load** (cache miss): `read_model()` + `compile_model()` → 2× peak (same as before) + writes `.blob` cache files
2. **Subsequent loads** (cache hit): `compile_model()` loads from `.blob` cache, skipping most of the heavyweight compilation. With `ENABLE_MMAP='NO'`, the original `.bin` weights don't need to be fully materialized → peak drops

### Clean Per-Process Measurements (32 GB system)

| Configuration | Load Time | **Peak** | Stable RSS | Savings |
|---|---|---|---|---|
| No cache, mmap ON | 11.7s | **13.07 GB** | 7.44 GB | baseline |
| No cache, mmap OFF | 11.6s | **13.08 GB** | 7.44 GB | — |
| Cache, mmap ON | 7.8s / 6.0s | **12.77 / 11.90 GB** | 7.12 GB | −1.2 GB peak |
| **Cache, mmap OFF** | **6.8s** | **9.99 / 9.31 GB** | **7.12 GB** | **−3.1 GB peak** |

### Full Benchmark Results — Cache + mmap OFF (best config)

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1720.5 | 22.2 | 276.7 | 45.0 | 9.31 |
| 1067 | 300 | 1108.2 | 20.1 | 962.8 | 49.7 | 9.31 |
| 2084 | 300 | 1804.8 | 16.9 | 1154.7 | 59.3 | 9.31 |

### Full Benchmark Results — Cache + mmap ON

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1749.4 | 23.0 | 272.1 | 43.4 | 11.90 |
| 1067 | 300 | 1148.7 | 20.6 | 928.9 | 48.5 | 11.90 |
| 2084 | 300 | 1809.4 | 17.3 | 1151.8 | 57.9 | 11.90 |

### Usage

```bash
# First run — creates cache (same peak as before, ~13 GB)
python benchmark_asus_kpi.py --model-dir ... --device GPU \
    --cache-dir MODEL_DIR/model_cache --no-mmap

# Subsequent runs — uses cache (peak reduced to ~9-10 GB)
python benchmark_asus_kpi.py --model-dir ... --device GPU \
    --cache-dir MODEL_DIR/model_cache --no-mmap --skip-cache-measurement
```

### Cache Disk Usage

The `.blob` cache files total ~6.5 GB (comparable to model `.bin` files).
This is a **disk-for-memory tradeoff**: +6.5 GB disk space → −3 GB peak RAM.

### Implications for 16 GB Config

| Config | Peak | Headroom on 16 GB |
|---|---|---|
| No cache (default) | ~13.1 GB | ~2.9 GB (tight) |
| Cache + mmap OFF | ~9.3 GB | **~6.7 GB (comfortable)** |

**Cache + mmap OFF is strongly recommended for 16 GB deployment.**

---

## Code Change — CACHE_DIR + ENABLE_MMAP Support

### Python — VLMPipeline constructor kwargs

The fix is simple. Both `CACHE_DIR` and `ENABLE_MMAP` are passed as **keyword
arguments** to `VLMPipeline`. GenAI forwards them through `ov::AnyMap` to
`ov::Core::compile_model()`, which recognizes `CACHE_DIR` as a GPU plugin
property.

```python
import openvino_genai as ov_genai

model_dir = r"C:\working\gemma4-openvino\gemma-4-E4B-it-ov"
cache_dir = model_dir + r"\model_cache"

# ── Best config: cache + no mmap ──
# First load creates .blob cache (~6.5 GB on disk), peak ~13 GB
# Subsequent loads hit cache, peak drops to ~9-10 GB
pipe = ov_genai.VLMPipeline(
    str(model_dir),
    "GPU",
    CACHE_DIR=cache_dir,        # GPU compiled model cache directory
    ENABLE_MMAP='NO',           # Don't memory-map .bin weights
)
```

### C++ — ov::AnyMap properties

```cpp
#include "openvino/genai/visual_language/pipeline.hpp"

std::filesystem::path model_dir = "gemma-4-E4B-it-ov";
std::string cache_dir = (model_dir / "model_cache").string();

ov::AnyMap props;
props["CACHE_DIR"]    = cache_dir;   // GPU compiled model cache
props["ENABLE_MMAP"]  = false;       // Don't memory-map .bin weights

ov::genai::VLMPipeline pipe(model_dir, "GPU", props);
```

### What changed in `benchmark_asus_kpi.py`

1. **New `--cache-dir` argument** — specifies the GPU compiled model cache directory
2. **Pipeline construction** — builds a `kwargs` dict and unpacks it:

```python
# Before (only mmap control):
if args.no_mmap:
    pipe = ov_genai.VLMPipeline(str(model_dir), args.device, ENABLE_MMAP='NO')
else:
    pipe = ov_genai.VLMPipeline(str(model_dir), args.device)

# After (mmap + cache control):
load_kwargs = {}
if args.no_mmap:
    load_kwargs['ENABLE_MMAP'] = 'NO'
if args.cache_dir:
    load_kwargs['CACHE_DIR'] = args.cache_dir

if load_kwargs:
    pipe = ov_genai.VLMPipeline(str(model_dir), args.device, **load_kwargs)
else:
    pipe = ov_genai.VLMPipeline(str(model_dir), args.device)
```

Same pattern applied to both `measure_cache_creation()` and the main scenario
loading section.

### Why this works — internal flow

```
VLMPipeline(model_dir, "GPU", CACHE_DIR=cache_dir, ENABLE_MMAP='NO')
  │
  ├─ ENABLE_MMAP='NO' → passed to ov::Core::read_model()
  │   └─ Reads .bin into heap (no mmap), but with cache hit
  │      the weights from .bin are NOT fully needed
  │
  └─ CACHE_DIR=cache_dir → passed to ov::Core::compile_model()
      └─ GPU plugin checks cache_dir for matching .blob files
         ├─ Cache MISS: compile normally, write .blob → peak = 2× model size
         └─ Cache HIT:  load pre-compiled .blob directly → skip heavyweight
                        GPU compilation → peak reduced by ~3 GB
```

The GPU plugin's cache stores the **compiled graph + transformed weights** in
`.blob` format. On cache hit, `compile_model()` loads the blob directly into
GPU-accessible shared memory, bypassing the `read_model()` → transform →
compile pipeline that causes the 2× peak.

---

## Pre-16GB Swap Checklist

- [x] 32 GB benchmarks collected (Run 1 + Run 2 + Cache runs)
- [x] ENABLE_MMAP code path verified correct in VLMPipeline source
- [x] Memory profile documented at all stages
- [x] CACHE_DIR + ENABLE_MMAP='NO' validated as peak reduction strategy
- [x] Scripts updated with `--cache-dir` support (benchmark_asus_kpi.py)
- [x] SESSION_NOTES.md updated

## 32 GB Benchmark Results (Run 2 — ENABLE_MMAP kwarg, no cache)

### mmap ON

| Cache time | Cache peak | Stabilized RSS |
|---:|---:|---:|
| 10.1s | 12.18 GB | 7.43 GB |

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1747.1 | 23.5 | 272.4 | 42.5 |
| 1067 | 300 | 1200.6 | 21.0 | 888.7 | 47.6 |
| 2084 | 300 | 1897.0 | 17.5 | 1098.6 | 57.2 |

### mmap OFF

| Cache time | Cache peak | Stabilized RSS |
|---:|---:|---:|
| 11.5s | 12.18 GB | 7.43 GB |

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) |
|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1720.0 | 23.4 | 276.7 | 42.8 |
| 1067 | 300 | 1168.0 | 21.0 | 913.5 | 47.6 |
| 2084 | 300 | 1829.8 | 17.5 | 1138.9 | 57.3 |

---

*Ready for 16 GB memory swap.*

---

## 16 GB Benchmark Results

**System:** Intel Panther Lake, Arc B390 iGPU (96 EUs), 15.7 GB LPDDR5 8533 MT/s, iGPU override ~13 GB  
**GPU_DEVICE_TOTAL_MEM_SIZE:** 8.5 GB (reported by OV)

### Per-Process Memory Profile (16 GB)

| Configuration | Load Time | **Peak** | Stable RSS | After Infer |
|---|---|---|---|---|
| No cache, mmap ON | 21.3s | **10.22 GB** | 6.45 GB | 6.52 GB |
| No cache, mmap OFF | 17.3s | **11.35 GB** | 2.89 GB | 3.08 GB |
| Cache, mmap ON | 11.3s | **12.60 GB** | 7.04 GB | 7.10 GB |
| **Cache, mmap OFF** | **7.8s** | **9.99 GB** | **7.11 GB** | **7.19 GB** |

### Full Benchmark — No cache, mmap ON

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1366.0 | 15.4 | 348.5 | 65.0 | 10.62 |
| 1067 | 300 | 860.5 | 13.9 | 1240.0 | 72.2 | 10.62 |
| 2084 | 300 | 1106.4 | 11.7 | 1883.6 | 85.3 | 10.62 |

### Full Benchmark — No cache, mmap OFF

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1384.0 | 15.2 | 343.9 | 65.9 | 10.87 |
| 1067 | 300 | 964.1 | 13.8 | 1106.7 | 72.4 | 10.87 |
| 2084 | 300 | 1263.8 | 11.8 | 1649.0 | 85.0 | 10.87 |

### Full Benchmark — Cache + mmap ON

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1337.1 | 14.6 | 356.0 | 68.3 | 11.19 |
| 1067 | 300 | 884.6 | 13.2 | 1206.2 | 75.6 | 11.19 |
| 2084 | 300 | 1091.7 | 11.5 | 1909.0 | 87.0 | 11.19 |

### Full Benchmark — Cache + mmap OFF (best peak memory)

| Input tok | Output tok | Prefill (t/s) | Output TPS | TTFT (ms) | TPOT (ms) | Peak (GB) |
|---:|---:|---:|---:|---:|---:|---:|
| 476 | 300 | 1385.5 | 11.2 | 343.6 | 89.0 | 9.31 |
| 1067 | 300 | 845.3 | 10.2 | 1262.3 | 97.7 | 9.31 |
| 2084 | 300 | 1201.1 | 9.2 | 1735.1 | 108.1 | 9.31 |

### Observations (16 GB)

**Memory pressure is real:**
- On 32 GB, Output TPS at ~467 was 23.5 t/s; on 16 GB it drops to 15.4 t/s (no cache) or 11.2 t/s (cache + no-mmap) — a **35-52% regression**
- Cache + mmap OFF has lowest peak (9.31 GB) but **worst TPS** (11.2 t/s) — the 6.5 GB cache blobs consume disk I/O and the combined cache + model files (~12.7 GB) may exceed disk cache capacity, causing page thrashing
- No cache configs (mmap ON/OFF) actually deliver **better TPS** (15.2-15.4 t/s) despite higher peak, because less disk I/O contention during inference
- TPOT increased from 42-57 ms (32 GB) to 65-108 ms (16 GB) across all configs

**Comparison: 32 GB vs 16 GB (no cache, mmap ON)**

| Metric | 32 GB | 16 GB | Regression |
|---|---|---|---|
| Output TPS (~467) | 23.5 | 15.4 | −34% |
| Output TPS (~1058) | 21.0 | 13.9 | −34% |
| Output TPS (~2075) | 17.5 | 11.7 | −33% |
| TTFT (~467) | 272 ms | 349 ms | +28% |
| TPOT (~467) | 42.5 ms | 65.0 ms | +53% |
| Load time | 10.1s | 14.9s | +48% |
| Peak | 12.18 GB | 10.62 GB | −13% |

**Key finding:** The ~34% TPS regression is consistent across all input lengths, suggesting it's caused by **memory bandwidth contention** — the iGPU and CPU are competing for the same LPDDR5 bandwidth, and with only 16 GB, the OS has less room for file cache and background services, increasing memory pressure.

**Best config for 16 GB depends on use case:**
- **Lowest peak memory:** Cache + mmap OFF (9.31 GB peak) — but TPS is worst
- **Best TPS on 16 GB:** No cache, mmap ON/OFF (15.2-15.4 t/s) — peak ~10.6-10.9 GB, fits fine
