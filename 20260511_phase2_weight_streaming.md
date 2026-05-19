# Dense Weight Streaming — Phase 2 Implementation Progress
**Date:** 2026-05-19 (updated)  
**Author:** jlee52tw  
**Status:** Phase 2 FC weight streaming — ✅ 完成，4-layer/3-buffer + Dual-NVMe 確認

---

## 0. 最新進度摘要（2026-05-19）

### Dual-NVMe Parallel IO 已實作並驗證

| 模式 | Avg TPOT | tok/s | Load% | GPU% |
|------|----------|-------|-------|------|
| v1 Sequential (baseline) | 177.48 ms | 5.63 | — | — |
| v2 Pipeline (single NVMe) | 142.81 ms | 7.00 | 56.0% | 42.7% |
| **v2 Dual-path (same disk)** | **128.56 ms** | **7.78** | **48.2%** | **50.6%** |
| v2 Dual NVMe (estimated) | ~77 ms | ~13.0 | — | — |

### 架構：Group-Half Striping
- 每個 group (~200 MB) 在 sector-aligned 中點切半
- 兩半分別存入 `streaming_0.bin` (NVMe 0) 和 `streaming_1.bin` (NVMe 1)
- Runtime 平行讀取兩個檔案，寫入同一個 USM buffer（連續區域）
- 4 個 async handles，每個 NVMe 分配 2 個

### 使用方式
```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\path\to\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2 = "D:\path\to\dense_weights_streaming_1.bin"
python run_gemma4.py --model-dir <model_dir> --prompt "..."
```

---

## 1. 目標

Phase 1 已驗證 streamed execution path（group transition 偵測 + 排序）對效能無影響 (23.4 tps = baseline)。

Phase 2 目標：**實際從 NVMe 載入權重並替換 GPU kernel 的 USM 指標**，完成完整的 weight streaming pipeline。

---

## 2. Phase 2 Pipeline 設計

每個 token 生成時，`execute_impl_streamed()` 按 group 順序執行：

```
對每個 group transition (group >= 0 && group != last_streamed_group):
  1. GPU fence: get_stream().flush()     // 等前一個 group 的 GPU 計算完成
  2. IO fence:  wait_for_load(group)     // 等 NVMe 載入完成
  3. Swap:      swap_weight_pointers()   // 替換 data primitive 的 output memory
  4. Re-bind:   _reset_arguments = true  // 繞過 set_arguments() 的 guard
               set_arguments()          // 重新綁定所有 kernel arguments
  5. Prefetch:  prefetch_next_group()    // 非同步預載下一個 group
```

### 2.1 首次 token（cold start）
- 同步載入 group 0：`load_group(0)` + `wait_for_load(0)`
- 之後走正常 pipeline

### 2.2 雙緩衝策略
- Buffer A / Buffer B 交替使用
- GPU 執行 group N 時，NVMe 同步預載 group N+1
- 理想情況：IO 完全被 GPU 計算遮蔽

---

## 3. 已解決的技術問題

### Bug 1: oneDNN 記憶體不相容

**症狀：** `[CLDNN] Can't convert memory object to onednn`

**原因：** `engine.attach_memory(layout, ptr)` 建立的是 `simple_attached_memory`，缺少 USM 中繼資料（allocation type、OpenCL context）。oneDNN kernel 需要完整的 `gpu_usm` 物件。

**修復：** 改用 `engine.create_subbuffer(memory, layout, byte_offset)`
- 從現有的 USM allocation 建立子區域（zero-copy）
- 保留原始 USM allocation type 和 memory tracker
- 位於 `ocl_engine.cpp:134`，使用 `UsmMemory(get_usm_helper(), ptr, byte_offset)`

**結果：** 不再 crash，程式正常完成。

### Bug 2: `set_arguments()` 被 guard 阻擋

**症狀：** 權重指標已替換，但 GPU kernel 仍使用舊的 arguments → 計算結果錯誤。

**原因：** `set_arguments()` 內部有 `_reset_arguments` flag。首次呼叫後設為 `false`，後續呼叫都直接 return → no-op。

**修復：** 在 `execute_impl_streamed()` 中，每次 swap 後：
```cpp
_reset_arguments = true;  // 強制重置 guard
set_arguments();           // 重新綁定所有 kernel arguments
```
移除了 `swap_weight_pointers()` 內的 `net->set_arguments()` 呼叫（本來就是 no-op）。

### Bug 3: JSON 層索引錯位 ✅ 已解決（被 FC-Scan 取代）

**症狀：** `build_weight_mapping_from_json()` cross-reference 為 0 matched, 全部 unmatched。

**原因：** JSON metadata 中的 `"layer_idx"` 使用 **model layer index**（0~41），但 C++ 使用 packed index（0~31）來搜尋 `"layer_idx": 0`。streamed layers 實際上是 model layer 5~36。

**最終解法：** 放棄 JSON tensor name matching，改用 **FC-scan** 方式直接從 compiled network 的 FullyConnected primitives 反推 weight mapping（見 Section 10）。

### Bug 4: Tensor 名稱不匹配 ✅ 已解決（被 FC-Scan 取代）

**症狀：** 即使修復層索引，JSON cross-reference 仍為 0 matched。

**原因：** compiled network 中的 primitive ID 經過 GPU compiler 重新命名，format 與 JSON 不一致。加上 GPU compiler 會將 weight/scale/zp 融合為 FC internal dependencies，external names 已不可靠。

**最終解法：** FC-scan 方法完全繞過名稱匹配問題（見 Section 10）。

### 附加問題: Offset 計算 ✅ 已解決

**問題：** OV IR 模型每層有 **33 個 constants**（weight + scale + zero_point + 小型常數），但 compiled GPU network 只有 **14 個 data primitives**（GPU compiler 融合了 weight/scale/zp）。

**影響：** Binary layout 基於 33 tensors 打包，與 compiled network 的 14 primitives 不匹配。

**解法：** FC-scan 方法直接讀取 JSON metadata 中每個 tensor 的 `offset`/`size`，根據 FC primitive 的 dependency name 做 key lookup，完全不依賴 tensor index 或排序。

---

## 4. ✅ FC-Scan Weight Mapping — 突破性方法

### 4.1 核心思路

放棄從 JSON tensor name 比對 compiled network primitive ID，改為：
1. 遍歷 compiled network 的所有 `fully_connected` primitives
2. 取得每個 FC 的 dependencies（`dep[1]` = weight, `dep[2]` = scale, `dep[3]` = zero_point）
3. 用 dependency name 作為 key，在 JSON offset table 中查找 binary offset

### 4.2 FC Primitive 結構

每個 compiled FullyConnected primitive 有 2-4 個 dependencies：
```
dep[0] = input activation（不需要 swap）
dep[1] = weight（INT4 量化權重）← 這是我們要 swap 的目標
dep[2] = scale（GPU compiler 已 reorder，不能 swap）
dep[3] = zero_point（同上，已 reorder）
```

### 4.3 Scale/ZP Reorder 重要發現

**GPU compiler 會將 scale 和 zero_point tensors 重新排列（reorder）為 GPU-optimal layout。**
- 編譯後的 primitive name 包含 `_reorder_` 標記
- Binary 中存的是原始 IR 格式（row-major），但 compiled kernel 預期的是 reorder 後的格式
- 如果 swap scale/zp → 讀到 raw IR data → 計算出亂碼

**解法：** 只 swap FC weights（`dep[1]`），scale/zp 留在 GPU 記憶體（~5 MB/layer，可以承受）。

### 4.4 FC-Scan 結果

```
FC primitives found: 224 (全部 42 layers)
  Layers 0-4 (pinned head): 19 FC — 跳過
  Layers 5-36 (streamed): 205 FC — mapped ✅
  Layers 37-41 (pinned tail): 0 FC — 無 FC
  
Weights mapped: 205 tensors, 1352.5 MB
  Layers 5-23: 6 FC weights/layer (q/k/v/o_proj + gate/up_proj)
  Layers 24-36: 7 FC weights/layer (+down_proj exposed)
```

### 4.5 驗證結果

```
Prompt: "What is the capital of Japan?"
Output: "The capital of Japan is **Tokyo**."
→ ✅ 正確！
```

---

## 5. 效能分析（實測數據）

### 5.1 Baseline vs Streaming 對比

| 指標 | Baseline (no streaming) | Phase 2 Streaming | 降幅 |
|---|---:|---:|---:|
| Output TPS | 24.0 | 4.54 | -81% |
| TPOT | 41.7 ms | 220.3 ms | +5.3× |
| TTFT | 300 ms | 750.9 ms | +2.5× |

### 5.2 Per-Token Timing Breakdown（實測，9 tokens）

```
 Token  Total_ms  Load_ms  Swap_ms  Args_ms Flush_ms   GPU_ms Groups
     0    525.85    77.27    16.64     2.36     0.01   429.58     32
     1    317.02    67.37    15.62     1.91     0.01   232.11     32
     2    186.61   112.87    16.61     1.95     0.01    55.17     32
     3    188.83   115.57    18.41     1.88     0.00    52.97     32
     4    185.51   115.94    17.46     1.92     0.00    50.19     32
     5    193.15   113.58    17.77     2.31     0.01    59.48     32
     6    190.21   115.04    17.21     2.25     0.01    55.71     32
     7    190.62   113.00    16.45     2.40     0.00    58.76     32
     8    244.19   114.23    24.61     2.76     0.01   102.58     32
------------------------------------------------------------
   AVG    246.89   104.99    17.86     2.19     0.01   121.84
 TOTAL   2221.98   944.87   160.76    19.74     0.05  1096.56
```

### 5.3 Overhead Breakdown（穩態 tokens 2-7 平均）

| 項目 | 時間 (ms) | 佔比 | 說明 |
|---|---:|---:|---|
| **NVMe load** | 114.3 | 60.3% | 🔴 最大瓶頸 — IO 無法被完全遮蔽 |
| **GPU compute** | 55.2 | 29.1% | 含 kernel launch overhead |
| **Swap pointers** | 17.3 | 9.1% | create_subbuffer × 205 tensors × 32 groups |
| **set_arguments** | 2.1 | 1.1% | 重綁所有 2549 primitives |
| **GPU fence** | 0.01 | 0.0% | flush() 幾乎為零 |

### 5.4 為什麼 IO 無法被完全遮蔽？

雙緩衝 pipeline 設計：GPU 計算 group N 時，NVMe 預載 group N+1。
但每個 group 只有 1 層（~48 MB），GPU 計算只需 ~1.7 ms/group，
而 NVMe 載入需 ~3.6 ms/group → **IO 是 GPU 的 2 倍慢 → pipeline stall**。

```
Per-group timing (估算):
  GPU compute: ~55 ms / 32 groups = ~1.7 ms/group
  NVMe load:   ~115 ms / 32 groups = ~3.6 ms/group
  → IO:GPU ratio = 2.1:1 → pipeline 只能遮蔽 ~47% 的 IO 時間
```

### 5.5 Token 0-1 異常

- Token 0 GPU=429 ms — 首次 GPU kernel JIT + cache warmup
- Token 1 GPU=232 ms — 部分 cache 已暖，仍在 JIT
- Token 2+ GPU=55 ms — 穩態，GPU cache 已完全暖

---

## 6. 關鍵發現

### `create_subbuffer()` 是正確的 memory 替換方式

```cpp
// ✅ 正確：建立 gpu_usm sub-buffer (zero-copy, preserves USM metadata)
auto new_mem = engine.create_subbuffer(*usm_buffer, layout, byte_offset);

// ❌ 錯誤：建立 simple_attached_memory (oneDNN 無法使用)
auto new_mem = engine.attach_memory(layout, ptr);
```

### `set_arguments()` 有 re-entry guard

```cpp
void network::set_arguments() {
    if (!_reset_arguments) return;  // ← 第二次呼叫後永遠直接 return
    _reset_arguments = false;
    // ... 實際綁定 kernel arguments ...
}
```
必須在每次 swap 後設定 `_reset_arguments = true` 再呼叫。

### FC-scan 比 JSON name matching 更可靠

| 方法 | 優點 | 缺點 |
|---|---|---|
| JSON name matching | 理論上精確 | GPU compiler 重命名 → 名稱不匹配 |
| **FC-scan（目前使用）** | 直接從 compiled graph 取得 | 只能 map FC weights（夠用） |

### Scale/ZP 必須留在 GPU 記憶體

- GPU compiler 將 scale/zp reorder 為 GPU-optimal layout
- Binary 中存的是 raw IR 格式 → swap 後格式不對 → 亂碼
- Scale/ZP 總共只佔 ~5 MB/layer → pinned in memory 完全可行

### OV IR constants ≠ Compiled network primitives

| 來源 | 每層 constants | 說明 |
|---|---|---|
| OV IR (openvino_language_model.xml) | 33 | weight + scale + zp + tiny constants |
| Compiled GPU network | ~14 data primitives | GPU compiler 融合 weight/scale/zp |
| **FC-scan mapped** | 6-7 FC weights | 只有 FC 的 dep[1] 需要 swap |

---

## 7. Debug Logging 與 Timing 工具

### 7.1 環境變數

| 變數 | 用途 | 預設值 |
|---|---|---|
| `OV_DENSE_STREAM_WEIGHTS` | Streaming binary 檔案路徑 | _(必要)_ |
| `OV_DENSE_STREAM_DEBUG` | 開啟 verbose stderr 輸出 | `"0"` |
| `OV_DENSE_STREAM_LOG_FILE` | Per-token timing 輸出到檔案 | _(不輸出)_ |

### 7.2 Per-Token Timing 機制

啟用 `OV_DENSE_STREAM_LOG_FILE` 後：
1. 每次 `execute_impl_streamed()` 結束時記錄一筆 `TokenTimingRecord`
2. 包含 6 個時間維度：total, load, swap, set_args, flush, gpu_compute
3. Pipeline 銷毀時（destructor）自動 flush 所有記錄到檔案
4. 輸出格式：per-token 表格 + 平均值/總計/百分比 breakdown

### 7.3 使用方式

```powershell
$env:OV_DENSE_STREAM_LOG_FILE = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\streaming_debug.log"
$env:OV_DENSE_STREAM_DEBUG = "1"
python run_gemma4.py --model-dir ... --prompt "..." --max-new-tokens 20

# 查看結果
Get-Content temp\streaming_debug.log
```

---

## 8. 下一步

### 8.1 優化 NVMe IO 遮蔽率（P0 — 效能瓶頸）

目前 IO:GPU = 2.1:1，pipeline 只能遮蔽 ~47% IO。可能方向：
1. **增大 group size** — 每 group 2-4 layers，增加 GPU 計算時間以遮蔽 IO ✅ 已實作
2. **減少需要 stream 的數據量** — 目前 1352 MB 交換 FC weights，但 scale/zp 已 pinned
3. **NVMe 預讀策略** — 嘗試多 group 預載而非只預載 1 個

### Phase 2 最終效能結果（2026-05-11）

| 配置 | TPOT (ms) | tok/s | Load% | Swap% | GPU% | 備註 |
|---|---:|---:|---:|---:|---:|---|
| Baseline (no streaming) | 41.7 | 24.0 | — | — | 100% | 所有權重常駐 |
| 1-layer/group, 2-buffer | 166×32 steps | 6.02 | 77% | 3% | 20% | Phase 2 初始 |
| **4-layer/group, 3-buffer** | **147** | **6.78** | **72.6%** | **0.7%** | **26.3%** | ✅ **最佳配置** |

**記憶體實測：**
- Baseline RSS: 6.664 GB
- 4-layer/3-buffer RSS: 7.241 GB (+0.577 GB for 3×198 MB buffers)
- 尚未實作 Phase 3 記憶體釋放

**Phase 2 結論：** decoder layer weight streaming 本身受限於 `compile_model`/USM memcpy 瓶頸，
最佳可達 ~6.78 tok/s（基線的 28%）。直接優化此路徑的 ROI 有限。

### 8.6 新方向：Per-layer Embedding 2.82 GB Offload（P0 — 轉入 Phase 3）

**2026-05-12 發現：** Gemma4 的 `text_embeddings_per_layer_model` (2.82 GB) 本質上只是
embedding lookup，每個 token 只需讀 1 行 10.5 KB。
可用 **DirectIO/DirectStorage** 從 NVMe 讀取，**完全跳過 compile_model**，省下 2.82 GB 記憶體。

→ 詳見 `20260512_per_layer_embedding_offload.md`

### 8.2 優化 swap pointer 開銷（P1）

穩態每 token ~17 ms 用在 `swap_weight_pointers()`（32 groups × 6-7 tensors × `create_subbuffer`）。
可能方向：
1. **快取 subbuffer** — 同一 offset+layout 不需要每次重建
2. **減少 group 數量** — 增大 group size → 減少 transition 次數

### 8.3 優化 `set_arguments()`（P2）

目前每次 group transition 重綁所有 2549 primitives（~2 ms/次 × 32 次 = ~65 ms/token），
但只有 6-7 個 FC primitives 的 weight 有變化。
可能方向：
1. **只重綁受影響的 primitives** — 需修改 `set_arguments()` 接受 partial rebind
2. **標記 dirty primitives** — 只有 weight dependency 變化的 FC 需要 rebind

### 8.4 記憶體釋放（Phase 3）

釋放 streamed layers 的原始 USM 記憶體，為 KV cache 留出空間。
目前 streaming 沒有減少 peak memory — 只是多了雙緩衝的 ~110 MB。

### 8.5 Scale/ZP Streaming（長期）

如果未來需要 stream scale/zp（進一步減少 pinned memory），
需要在 binary 中以 GPU-reorder 格式打包，而非 IR 格式。
需要理解 GPU compiler 的 reorder 規則（per-layout variable）。

---

## 9. 修改的檔案清單

| 檔案 | 位置 | 變更 |
|---|---|---|
| `dense_weight_streaming_manager.cpp` | workspace + OpenVINO | FC-scan `build_weight_mapping_from_json()`: 從 FC deps 反推 weight mapping |
| `dense_weight_streaming_manager.cpp` | workspace + OpenVINO | `swap_weight_pointers()`: 只 swap FC weights (dep[1])，跳過 scale/zp |
| `dense_weight_streaming_manager.cpp` | workspace + OpenVINO | `record_token_timing()` + `flush_token_timings()`: per-token timing log |
| `dense_weight_streaming_manager.hpp` | workspace + OpenVINO | `TokenTimingRecord` struct, `debug_log_path`, `token_timings_count()` |
| `network.cpp` | OpenVINO only | `execute_impl_streamed()`: 分離 swap/set_args timing，記錄所有 tokens |
| `network.cpp` | OpenVINO only | `try_init_dense_streaming()`: JSON path derivation + fallback |
| `primitive_inst.h` | OpenVINO only | `update_weights_cache()`: 加入 public 方法供 FC weight swap |
| `pack_dense_weights.py` | workspace | sorted tensors, offset metadata for FC-scan

---

## 10. 完整 Build → Deploy → Test 流程

### 10.1 前置環境設定

本專案使用 **pip 安裝的 OpenVINO** (`openvino==2026.2.0`)，透過替換 pip site-packages 中的 GPU plugin DLL 來測試修改。

#### 路徑總覽

| 項目 | 路徑 |
|---|---|
| OpenVINO 原始碼 | `C:\working\gemma4-openvino\openvino\` |
| Build 輸出目錄 | `C:\working\gemma4-openvino\openvino\bin\intel64\Release\` |
| Pip OpenVINO libs | `C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\` |
| 模型目錄 | `C:\working\gemma4-openvino\gemma-4-E4B-it-ov\` |
| Blob cache | `C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache\` |
| Workspace | `C:\working\gemma4-openvino\gemma4-openvino-genai\` |
| Streaming binary | `C:\working\gemma4-openvino\gemma4-openvino-genai\temp\dense_weights_streaming.bin` |
| Streaming JSON | `C:\working\gemma4-openvino\gemma4-openvino-genai\temp\dense_weights_streaming.json` |

#### 修改的 OpenVINO 檔案

| 檔案 | OpenVINO 路徑 | Workspace 備份 |
|---|---|---|
| `network.cpp` | `src/plugins/intel_gpu/src/graph/network.cpp` | _(僅在 OpenVINO tree)_ |
| `network.hpp` | `src/plugins/intel_gpu/include/intel_gpu/graph/network.hpp` | _(僅在 OpenVINO tree)_ |
| `primitive_inst.h` | `src/plugins/intel_gpu/src/graph/include/primitive_inst.h` | _(僅在 OpenVINO tree)_ |
| `CMakeLists.txt` | `src/plugins/intel_gpu/src/graph/CMakeLists.txt` | _(僅在 OpenVINO tree)_ |
| `dense_weight_streaming_manager.cpp` | `src/plugins/intel_gpu/src/graph/dense_weight_streaming_manager.cpp` | `cpp/dense_weight_streaming_manager.cpp` |
| `dense_weight_streaming_manager.hpp` | `src/plugins/intel_gpu/src/graph/include/dense_weight_streaming_manager.hpp` | `cpp/dense_weight_streaming_manager.hpp` |

> **注意：** `dense_weight_streaming_manager.cpp/.hpp` 同時存在於 workspace `cpp/` 和 OpenVINO tree。
> 修改時應編輯 workspace 版本，再複製到 OpenVINO tree。

---

### 10.2 Step 1: 複製原始碼到 OpenVINO tree

如果有修改 `dense_weight_streaming_manager.cpp` 或 `.hpp`：

```powershell
# 從 workspace 複製到 OpenVINO source tree
Copy-Item "C:\working\gemma4-openvino\gemma4-openvino-genai\cpp\dense_weight_streaming_manager.cpp" `
    "C:\working\gemma4-openvino\openvino\src\plugins\intel_gpu\src\graph\dense_weight_streaming_manager.cpp" -Force

Copy-Item "C:\working\gemma4-openvino\gemma4-openvino-genai\cpp\dense_weight_streaming_manager.hpp" `
    "C:\working\gemma4-openvino\openvino\src\plugins\intel_gpu\src\graph\include\dense_weight_streaming_manager.hpp" -Force
```

---

### 10.3 Step 2: 建置 GPU Plugin DLL

```powershell
cd C:\working\gemma4-openvino\openvino
$env:CI_BUILD_NUMBER = "2026.2.0-21571-9c4a2eb9ad3"
cmake --build build --target openvino_intel_gpu_plugin --config Release -- /v:m /p:CL_MPCount=4
```

- **Target:** `openvino_intel_gpu_plugin`（只建置 GPU plugin，不需要整個 OpenVINO）
- **產出 DLL:** `C:\working\gemma4-openvino\openvino\bin\intel64\Release\openvino_intel_gpu_plugin.dll`（約 34-36 MB）
- **建置時間:** 約 1-3 分鐘（增量編譯，4 parallel）
- **`CI_BUILD_NUMBER`:** 必須設定，否則版本號不匹配會導致 plugin 載入失敗

> **Tip:** 如果建置失敗，嘗試降低 `CL_MPCount` 到 4（記憶體不足時 8 會失敗）。

---

### 10.4 Step 3: 部署 DLL 到 pip site-packages

```powershell
# 複製建置產出的 DLL 到 pip openvino libs
Copy-Item "C:\working\gemma4-openvino\openvino\bin\intel64\Release\openvino_intel_gpu_plugin.dll" `
    "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" -Force
```

#### 驗證部署

```powershell
# 確認 DLL 已更新（檢查時間戳和大小）
Get-ChildItem "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" | Select-Object Name, Length, LastWriteTime

# 確認只有一份 DLL（不應有多個版本）
Get-ChildItem "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin*" | Select-Object Name, LastWriteTime
```

> **備份：** pip 目錄中保留了原始 DLL 的備份：
> - `openvino_intel_gpu_plugin.dll.backup` — 首次備份
> - `openvino_intel_gpu_plugin.dll.backup_apr16` — Apr 16 備份

---

### 10.5 Step 4: 設定環境變數

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai

# [必要] 指定 streaming binary 檔案路徑（使用絕對路徑）
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\dense_weights_streaming.bin"

# [可選] 開啟 debug logging
$env:OV_DENSE_STREAM_DEBUG = "1"

# [可選] 輸出 per-token timing 到檔案
$env:OV_DENSE_STREAM_LOG_FILE = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\streaming_debug.log"
```

> **不需要 `setupvars.ps1`：** 因為使用 pip 安裝的 OpenVINO，Python 環境已自帶完整設定。
> `setupvars.ps1`（位於 `C:\working\gemma4-openvino\openvino\scripts\setupvars\setupvars.ps1`）
> 僅在使用 OpenVINO 原始碼建置的 C++ 應用程式時才需要。

---

### 10.6 Step 5: 執行測試

#### 快速正確性測試（max_new_tokens=5，預期含 "Tokyo"）

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\dense_weights_streaming.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
python -c "
import openvino_genai as ov_genai
pipe = ov_genai.VLMPipeline(r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov', 'GPU',
    CACHE_DIR=r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache')
config = ov_genai.GenerationConfig()
config.max_new_tokens = 5
out = pipe.generate('What is the capital of Japan? One word answer.', generation_config=config)
print(f'Output: [{out}]')
" 2>&1 | Tee-Object "temp\test_streaming_latest.log"
```

#### 基線效能測試（無 streaming，對照用）

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai
Remove-Item Env:\OV_DENSE_STREAM_WEIGHTS -ErrorAction SilentlyContinue
Remove-Item Env:\OV_DENSE_STREAM_DEBUG -ErrorAction SilentlyContinue
python benchmark.py 2>&1 | Tee-Object "temp\test_baseline_latest.log"
```

#### 完整 Pipeline 測試（with log capture + filter）

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\dense_weights_streaming.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
python -c "
import openvino_genai as ov_genai
import sys
sys.stdout.reconfigure(line_buffering=True)
print('Starting...')
pipe = ov_genai.VLMPipeline(r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov', 'GPU',
    CACHE_DIR=r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache')
print('Pipeline created, generating...')
config = ov_genai.GenerationConfig()
config.max_new_tokens = 5
out = pipe.generate('What is the capital of Japan? One word answer.', generation_config=config)
print(f'Output: [{out}]')
print('DONE')
" 2>&1 | Tee-Object temp\full_output_latest.log | Select-String -Pattern 'Dense|Starting|Pipeline|Output|OUT|DONE|Error|Exception'
```

---

### 10.7 一鍵快速流程（複製貼上用）

完整的 build → deploy → test 流程：

```powershell
# === 1. Copy source ===
Copy-Item "C:\working\gemma4-openvino\gemma4-openvino-genai\cpp\dense_weight_streaming_manager.cpp" "C:\working\gemma4-openvino\openvino\src\plugins\intel_gpu\src\graph\dense_weight_streaming_manager.cpp" -Force
Copy-Item "C:\working\gemma4-openvino\gemma4-openvino-genai\cpp\dense_weight_streaming_manager.hpp" "C:\working\gemma4-openvino\openvino\src\plugins\intel_gpu\src\graph\include\dense_weight_streaming_manager.hpp" -Force

# === 2. Build ===
cd C:\working\gemma4-openvino\openvino
$env:CI_BUILD_NUMBER = "2026.2.0-21571-9c4a2eb9ad3"
cmake --build build --target openvino_intel_gpu_plugin --config Release -- /v:m /p:CL_MPCount=4

# === 3. Deploy DLL ===
Copy-Item "C:\working\gemma4-openvino\openvino\bin\intel64\Release\openvino_intel_gpu_plugin.dll" "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" -Force

# === 4. Test ===
cd C:\working\gemma4-openvino\gemma4-openvino-genai
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp\dense_weights_streaming.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
python -c "
import openvino_genai as ov_genai
pipe = ov_genai.VLMPipeline(r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov', 'GPU', CACHE_DIR=r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache')
config = ov_genai.GenerationConfig()
config.max_new_tokens = 5
out = pipe.generate('What is the capital of Japan? One word answer.', generation_config=config)
print(f'Output: [{out}]')
" 2>&1 | Tee-Object "temp\test_streaming_latest.log"
```

---

### 10.8 重新打包 Streaming Binary

如果模型或打包邏輯有變更，需重新產生 `.bin` 和 `.json`：

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai
python pack_dense_weights.py `
    --model_dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --output_dir temp `
    --first_streamed 5 --last_streamed 36 `
    --group_size 1 2>&1 | Tee-Object "temp\pack_output.log"
```

- 產出：`temp/dense_weights_streaming.bin`（~1.55 GB）+ `temp/dense_weights_streaming.json`（~330 KB）
- `--first_streamed 5 --last_streamed 36`：跳過 head 5 層 + tail 5 層（pinned in memory）
- `--group_size 1`：每 group 1 層（32 groups for 32 streamed layers）

---

### 10.9 疑難排解

| 問題 | 解決方法 |
|---|---|
| Build 失敗 `out of memory` | 降低 `CL_MPCount` 到 4 或更低 |
| Plugin 載入失敗 version mismatch | 確認 `$env:CI_BUILD_NUMBER` 設定正確 |
| `[DenseStreaming]` 訊息完全不出現 | 確認環境變數已設定：`echo $env:OV_DENSE_STREAM_WEIGHTS` |
| DenseStreaming 訊息不出現（pipeline 已建立） | `try_init_dense_streaming()` 在 **第一次** `execute_impl()` 時才觸發（lazy init），需呼叫 `generate()` 才會出現 |
| `0 matched, N unmatched` | JSON tensor 名稱與 compiled network primitive ID 不匹配（目前的已知問題） |
| Output 為空 `[]` | Weight mapping offset 不正確，GPU 讀到錯誤數據 |
| DLL 時間戳不更新 | 確認沒有其他 Python process 鎖住 DLL；用 Task Manager 檢查 |
| Blob cache 不一致 | 刪除 `model_cache` 資料夾強制重新編譯（需約 12 秒）|
