# 20260514 — Pipeline Overlap Fix: clFinish + Early Prefetch

## 根因分析

### 為什麼 IO 與 GPU 完全序列化？

透過原始碼確認，`execute_impl_streamed()` 中的 "GPU fence" 實際上是 `get_stream().flush()` = `clFlush()` = **非阻塞**：

```
ocl_stream::flush()  → cl_queue.flush()  = clFlush()  ← 不等 GPU 完成！
ocl_stream::finish() → cl_queue.finish() = clFinish() ← 阻塞等 GPU 完成
```

**結果：** CPU 不等 GPU，瞬間跑完 group 的所有 primitive enqueue（~1ms），
然後立刻進入下一個 group transition 的 `wait_for_load()`。
Overlap window = 1ms，但 IO 每 group 需要 ~2.8ms（1-layer）或 ~11ms（4-layer）。

### 修正前 Pipeline 時序

```
Group N transition:
  flush()=clFlush     → 0ms (non-blocking!)
  wait_IO(N)          → blocks ~2.8ms (IO was only running 1ms from prev prefetch)
  swap                → 0.04ms
  set_args            → 0.75ms
  prefetch(N+1)       → 0ms (start async IO)
  enqueue GPU(N)      → 1ms (non-blocking clEnqueue × ~70 prims)
  ← Total: IO(~2.8) + overhead(~2ms) + enqueue(~1ms) ≈ 5ms per group

Overlap window: ~1ms (enqueue time)
IO utilization: 1/2.8 = 36% — 大部分時間 NVMe 閒置等 CPU
```

### 修正後 Pipeline 時序

```
Group N transition:
  finish()=clFinish   → blocks 1-10ms (等 GPU(N-1) 完成)
                        ← 同時 prefetch(N) 的 NVMe DMA 持續運行！
  wait_IO(N)          → 0ms! (IO 已在 GPU wait 期間完成)
  swap                → 0.04ms
  set_args            → 0.75ms
  prefetch(N+1)       → 0ms (start async IO)
  enqueue GPU(N)      → 1ms
  ← Total: max(GPU, IO) + overhead(~2ms) ≈ max(IO, GPU) + 2ms

Overlap: IO 完全被 GPU wait + enqueue 時間 overlap！
```

---

## 方案設計

### 改動 1: `flush()` → `finish()` (network.cpp)

將 group transition 的 GPU fence 從 `get_stream().flush()` (clFlush, 非阻塞)
改為 `get_stream().finish()` (clFinish, 阻塞等 GPU)。

**效果：** clFinish 阻塞 CPU 直到 GPU 完成。在此期間 NVMe DMA 持續運行。
當 clFinish 返回時，IO 很可能已完成 → `wait_for_load()` 瞬間返回。

### 改動 2: Early Prefetch for Group 0

原本 Group 0 是 cold start（synchronous load），浪費了 pre-decoder 的 ~30ms。
新流程：在 `set_arguments()` 之後立即 `prefetch_next_group(0)`，
讓 IO 在 pre-decoder + HEAD 期間就開始。到達 Group 0 transition 時 IO 已完成。

### 改動 3: Debug Logging

在 critical path 的每個步驟加入 per-group timing log（debug mode only）。
Log 輸出到 `OV_DENSE_STREAM_LOG_FILE` 指定的檔案。

### 使用 4-layer groups

與之前 Phase 2 實測數據直接可比較：
- 8 groups, 32 streamed layers (5-36), H5+T5 pinning
- 每 group IO ~170 MB, ~11ms at 10 GB/s

---

## 預期效能

### 4-layer groups (8 groups)

| 指標 | 修正前 (序列) | 修正後 (overlap) | 改善 |
|---|---:|---:|---:|
| Per-group time | IO+GPU ≈ 22ms | max(IO,GPU) ≈ 11ms | ~50% |
| 8 groups total | ~176ms | ~88ms | ~50% |
| + pre/post overhead | ~30ms | ~30ms (overlapped) | — |
| **Total TPOT** | **~155ms** | **~90-110ms** | **30-40%** |
| **tok/s** | **~6.45** | **~9-11** | **+40-70%** |

### 理論下限 (IO-bound, 1×NVMe)

- Streamed data: ~1.55 GB
- NVMe throughput: ~10 GB/s
- Floor: 1.55/10 = 155ms per token (pure IO)
- 但 pre-decoder + pinned layers ≈ 30ms 可 overlap → 有效 floor ≈ 125ms
- 加上 overhead → 實際目標 ~100-120ms (8.3-10 tok/s)

---

## 實作步驟

### Step 1: 修改 network.cpp

1. Group transition 的 `get_stream().flush()` → `get_stream().finish()`
2. 新增 early prefetch: `set_arguments()` 後立即 `m_dense_streaming->prefetch_next_group(0)`
3. Group 0 transition 改為 `wait_for_load(0)` 而非 `load_group(0) + wait_for_load(0)`
4. Per-group debug logging（finish/IO/swap/args 各步驟的 wall-clock time）
5. 最後的 `get_stream().flush()` → `get_stream().finish()`（確保最後一組 GPU 完成）

### Step 2: 驗證 dense_weight_streaming_manager.cpp

確認 `prefetch_next_group(0)` 在 `set_arguments()` 之後可安全呼叫。
確認 `wait_for_load()` 正確處理 early prefetch 的 group 0。

---

## Build & Deploy & Test

### Build GPU Plugin DLL

```powershell
# 1. Copy source to openvino tree
Copy-Item "C:\working\gemma4-openvino\gemma4-openvino-genai\cpp\dense_weight_streaming_manager.cpp" `
    "C:\working\gemma4-openvino\openvino\src\plugins\intel_gpu\src\graph\dense_weight_streaming_manager.cpp" -Force
Copy-Item "C:\working\gemma4-openvino\gemma4-openvino-genai\cpp\dense_weight_streaming_manager.hpp" `
    "C:\working\gemma4-openvino\openvino\src\plugins\intel_gpu\src\graph\include\dense_weight_streaming_manager.hpp" -Force

# 2. Build (network.cpp is already in openvino tree — it was modified in-place)
cd C:\working\gemma4-openvino\openvino
$env:CI_BUILD_NUMBER = "2026.2.0-21571-9c4a2eb9ad3"
cmake --build build --target openvino_intel_gpu_plugin --config Release -- /v:m /p:CL_MPCount=4

# 3. Deploy
Copy-Item "C:\working\gemma4-openvino\openvino\bin\intel64\Release\openvino_intel_gpu_plugin.dll" `
    "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" -Force
```

### Test — Streaming with Pipeline Fix

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai

# Enable streaming with debug logging
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\dense_weights_streaming.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
$env:OV_DENSE_STREAM_LOG_FILE = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\streaming_pipeline_fix.log"

# Quick correctness test
python -c "
import openvino_genai as ov_genai
pipe = ov_genai.VLMPipeline(r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov', 'GPU',
    CACHE_DIR=r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache')
config = ov_genai.GenerationConfig()
config.max_new_tokens = 5
out = pipe.generate('What is the capital of Japan? One word answer.', generation_config=config)
print(f'Output: [{out}]')
" 2>&1 | Tee-Object "temp\test_pipeline_fix_quick.log"

# Performance test — 256 tokens
python -c "
import openvino_genai as ov_genai
pipe = ov_genai.VLMPipeline(r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov', 'GPU',
    CACHE_DIR=r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache')
config = ov_genai.GenerationConfig()
config.max_new_tokens = 256
out = pipe.generate('Write a short essay about artificial intelligence.', generation_config=config)
print(f'Output: {str(out)[:200]}')
" 2>&1 | Tee-Object "temp\test_pipeline_fix_perf.log"
```

### Verification Criteria

| 指標 | 修正前 | 修正後目標 | 說明 |
|---|---:|---:|---|
| `flush_ms` (avg) | ~0 ms | **~5-15 ms** | clFinish 阻塞等 GPU — 正常 |
| `load_ms` (avg) | ~91 ms | **~0-5 ms** | IO 在 GPU wait 期間完成 |
| `total_ms` (avg) | ~155 ms | **~90-120 ms** | 30-40% 改善 |
| `tok/s` | ~6.45 | **~8-11** | 目標 |
| Output | Tokyo | Tokyo | 正確性不變 |

### Log 分析

```powershell
# 查看 per-group timing
Select-String -Path temp\streaming_pipeline_fix.log -Pattern "group_transition|finish_ms|io_wait_ms"
```

---

## 與之前數據的比較基線

### Phase 2 原始（4-layer groups, sync thread IO）

| Token | Total (ms) | Load (ms) | GPU (ms) | tok/s |
|---|---:|---:|---:|---:|
| AVG | 147.0 | 106.7 | 38.7 | 6.78 |

### Phase 2 + Async ReadFile（今天之前的版本）

| Token | Total (ms) | Load (ms) | GPU (ms) | tok/s |
|---|---:|---:|---:|---:|
| AVG | 154.97 | 90.80 | 62.23 | 6.45 |

### Phase 2 + Pipeline Fix（本次修改目標）

| Token | Total (ms) | Load (ms) | GPU (ms) | Flush (ms) | tok/s |
|---|---:|---:|---:|---:|---:|
| AVG (target) | ~100-120 | ~0-5 | ~60 | ~5-15 | ~8-10 |
