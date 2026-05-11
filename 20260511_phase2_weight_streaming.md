# Dense Weight Streaming — Phase 2 Implementation Progress
**Date:** 2026-05-11  
**Author:** jlee52tw  
**Status:** Phase 2 runtime weight swap — 正在除錯中

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

### Bug 3: JSON 層索引錯位（進行中）

**症狀：** `build_weight_mapping_from_json()` cross-reference 為 0 matched, 全部 unmatched。

**原因：** JSON metadata 中的 `"layer_idx"` 使用 **model layer index**（0~41），但 C++ 使用 packed index（0~31）來搜尋 `"layer_idx": 0`。streamed layers 實際上是 model layer 5~36。
- C++ 搜尋 `"layer_idx": 0` → 找到 model layer 0（pinned head，無 tensors）
- 應該搜尋 `"layer_idx": 5` → 找到 model layer 5（第一個 streamed layer）

**修復：** 更新搜尋邏輯：
```cpp
uint32_t model_layer = first_streamed + packed_idx;
std::string marker = "\"layer_idx\": " + std::to_string(model_layer) + ",";
```

### Bug 4: Tensor 名稱不匹配（進行中）

**症狀：** 即使修復層索引，JSON cross-reference 仍為 0 matched。

**原因：** JSON 中的 tensor 命名有兩種格式：
| JSON 名稱格式 | 類型 | 範例 |
|---|---|---|
| `self.model.language_model.layers.5.mlp.down_proj.weight` | 大型權重 | 13 MB |
| `__module.model.language_model.layers.5.self_attn/aten::neg/Constant` | 小型常數 | 4~16 bytes |

compiled network 中的 primitive 名稱格式**尚未確認**。需要比對：
- JSON tensor name vs compiled network primitive ID
- 差異可能在前綴（`self.` vs `__module.`）或其它 GPU compiler 重命名

**目前狀態：** 正在加入 debug 輸出以確認 compiled network 的 primitive 名稱格式。

### 附加問題: Offset 計算

**問題：** OV IR 模型每層有 **33 個 constants**（weight + scale + zero_point + 小型常數），但 compiled GPU network 只有 **14 個 data primitives**（GPU compiler 融合了 weight/scale/zp）。

**影響：** 
- Python packer 依 33 tensors 順序打包，offset 按 33 tensors 累加
- C++ `build_weight_mapping()` 只找到 14 個，按 14 tensors 算 offset → offset 不匹配
- 必須使用 JSON metadata 作為 binary layout 的權威來源

**修復方向：** `build_weight_mapping_from_json()` 讀取全部 33 tensors 計算 offset，只保留在 compiled network 中存在的 14 個（移除 unmatched entries）。

---

## 4. Fallback 路徑: `build_weight_mapping()` 分析

目前 JSON mapping 失敗後，fallback 到基於 compiled network 掃描的 `build_weight_mapping()`：
- 找到 448 tensors（14 per layer × 32 layers）
- 按 primitive name 排序計算 offset
- 每層 swap 14 tensors
- **但 offset 計算基於 14 tensors，不匹配 binary 中的 33-tensor layout → 讀取到錯誤的數據**

這是 output 為空字串的根本原因。

---

## 5. 效能觀測（不正確數據，僅供參考）

以下是 offset 錯誤時的 Phase 2 timing（數據本身反映 pipeline 效率，但推理結果不正確）：

| 指標 | 值 | 說明 |
|---|---|---|
| First token total | 655 ms | 包含 32 group transitions |
| NVMe load total | 55 ms | 32 groups, 平均 1.7 ms/group |
| Swap total | 3.2 ms | 32 groups, 平均 0.1 ms/group |
| Flush (GPU fence) | 0.007 ms | 幾乎為零 |
| NVMe throughput | 8.4-10.4 GB/s | 接近 NVMe 理論值 |
| Per-group load | 5-6 ms | ~48-55 MB/group |

觀察：
- IO 效能非常好（8-10 GB/s Direct I/O）
- Swap 很快（0.1 ms/group = 14 個 `create_subbuffer` + `force_set_output_memory`）
- 主要時間花在 GPU 計算（655 - 55 - 3 ≈ 597 ms）
- 預期正確後 TPS 約為 1000/655 ≈ **1.5 tps**（未優化的非 pipeline 模式，因為 `set_arguments()` 重綁所有 2549 primitives）

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

### OV IR constants ≠ Compiled network primitives

| 來源 | 每層 constants | 說明 |
|---|---|---|
| OV IR (openvino_language_model.xml) | 33 | weight + scale + zp + tiny constants |
| Compiled GPU network | 14 | GPU compiler 融合 weight/scale/zp 為一個 FullyConnected |

Binary layout 基於 33 tensors 打包，但只有 14 個需要在 runtime swap。

---

## 7. 下一步

1. **解決 tensor 名稱匹配** — 確認 compiled network primitive ID 格式，修復 JSON cross-reference
2. **驗證正確性** — 預期 output 包含 "Tokyo"
3. **效能量測** — 正確數據後量測 steady-state TPS
4. **優化 `set_arguments()`** — 只重綁受影響的 primitives，而非全部 2549 個
5. **記憶體釋放** — 釋放 pinned layers 的原始 USM 記憶體（未來，Phase 3）

---

## 8. 修改的檔案清單

| 檔案 | 位置 | 變更 |
|---|---|---|
| `dense_weight_streaming_manager.cpp` | workspace + OpenVINO | `swap_weight_pointers()`: attach_memory → create_subbuffer; 移除 net->set_arguments() |
| `dense_weight_streaming_manager.cpp` | workspace + OpenVINO | `build_weight_mapping_from_json()`: 修復 layer index (packed→model); 移除 unmatched tensors |
| `network.cpp` | OpenVINO only | `execute_impl_streamed()`: 完整 Phase 2 pipeline (load/swap/reset_args/set_args/prefetch) |
| `network.cpp` | OpenVINO only | `try_init_dense_streaming()`: JSON path derivation + fallback |
| `pack_dense_weights.py` | workspace | sorted tensors by name for both data packing and JSON metadata |

---

## 9. Build 指令

```powershell
cd C:\working\gemma4-openvino\openvino
$env:CI_BUILD_NUMBER = "2026.2.0-21571-9c4a2eb9ad3"
cmake --build build --target openvino_intel_gpu_plugin --config Release -- /v:m /p:CL_MPCount=4
```

## 10. 測試指令

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai
$env:OV_DENSE_STREAM_WEIGHTS = "temp\dense_weights_streaming.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
python -c "
import openvino_genai as ov_genai
pipe = ov_genai.VLMPipeline(r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov', 'GPU',
    CACHE_DIR=r'C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache')
config = ov_genai.GenerationConfig()
config.max_new_tokens = 5
out = pipe.generate('What is the capital of Japan? One word answer.', generation_config=config)
print(f'Output: [{out}]')
"
```
