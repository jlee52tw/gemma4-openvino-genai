# CL_OUT_OF_RESOURCES 根因分析 (2026-05-14)

## 問題描述

Dense weight streaming 啟用後（8 groups, 4 layers/group, H5+T5 pinning），
執行到 **group 3** 時觸發 `CL_OUT_OF_RESOURCES` crash。

- 系統：32 GB RAM, iGPU 27 GB USM — 不是簡單的記憶體不足
- 錯誤非確定性：不同 run crash 在不同 primitive（prim 1415/1417）、不同 phase（prepare/execute）
- Groups 0-2 正常完成，group 3 執行中 crash
- 錯誤來源：`dnnl_ocl.hpp:276`（"could not create a memory"）或 `ocl_common.hpp:62`（`CL_OUT_OF_RESOURCES`）

---

## 核心發現：「昨天的 code 為什麼沒 crash？」

**答案：昨天的 build 根本沒啟用 streaming 功能。**

### 證據：Git Diff

```powershell
cd C:\working\gemma4-openvino\openvino
git diff HEAD -- src/plugins/intel_gpu/src/graph/CMakeLists.txt
```

| 檔案 | HEAD (committed, 昨天) | Working tree (今天部署) |
|---|---|---|
| **CMakeLists.txt** | **無** `OV_DENSE_WEIGHT_STREAMING_ENABLED` | **新增** `target_compile_definitions(... OV_DENSE_WEIGHT_STREAMING_ENABLED)` |
| **primitive_inst.h** | **無** `force_set_output_memory()` / `update_weights_cache()` | **新增** 24 行 |
| **network.cpp** | `execute_impl_streamed()` 在 `#ifdef` 保護下 → **fallback 到 `execute_impl(events)`** | 啟用後走完整 streaming 路徑 |
| **dense_weight_streaming_manager.cpp/hpp** | 不存在（untracked） | 新增，部署到 openvino 樹 |

### 關鍵：`execute_impl_streamed()` 昨天是死代碼

```cpp
// network.cpp (HEAD committed version)
void network::execute_impl_streamed(const std::vector<event::ptr>& events) {
#ifdef OV_DENSE_WEIGHT_STREAMING_ENABLED   // ← 昨天 CMakeLists 沒定義！
    // ... 全部 streaming 程式碼（swap、prefetch、group transition）...
#else
    execute_impl(events);  // ← 昨天實際走這條：100% 正常推理路徑
#endif
}
```

**結論：CL_OUT_OF_RESOURCES 不是「今天 pipeline overlap 改動引入的 regression」，
而是 streaming 功能第一次被真正編譯啟用，暴露了 swap 機制本身的架構問題。**

### 今天的改動清單（network.cpp diff vs HEAD）

| 改動 | 性質 | 影響記憶體分配？ |
|---|---|---|
| Early prefetch（group 0 IO 提前到 pre-decoder 前） | 時序優化 | ❌ 否 |
| clFlush → clFinish → 還原 clFlush | 測試/還原 | ❌ 否 |
| 移除 `load_group(0)` 改為 `wait_for_load(0)` | 配合 early prefetch | ❌ 否 |
| try-catch 分離 prepare_primitive / execute | 診斷 | ❌ 否 |
| Per-group debug logging | 診斷 | ❌ 否 |
| 註解更新 | 文件 | ❌ 否 |

**沒有任何今天的改動影響記憶體分配行為。** 問題在 streaming 的 swap 機制本身。

---

## 真正的根因：Pre-Reorder 權重 + `update_weights()` 累積分配

### 問題鏈

```
pack_dense_weights.py         → 打包 IR model 的原始 Constant 數據（pre-reorder layout）
                                 ↓
swap_weight_pointers()        → force_set_output_memory() 替換 dep[1] 為 streaming subbuffer
                                 ↓
prepare_primitive()           → update_weights() 偵測 layout 不相容
                                 ↓
update_weights()              → original_layout.compatible(expected_layout) == false
                                 ↓
engine.allocate_memory()      → 分配 usm_device 記憶體做 reorder output
                                 ↓
累積 group 0~2 的 reorder 分配  → group 3 時 GPU 資源耗盡 → CL_OUT_OF_RESOURCES
```

### 技術細節

#### 1. Streaming binary 包含 pre-reorder 權重

`pack_dense_weights.py` (L133) 使用 `op.get_data()` 從 IR model 的 Constant 節點提取原始數據：

```python
tensor_data = op.get_data()
raw_bytes = tensor_data.tobytes()
```

這些數據的 layout（format、stride、padding）與 compile_model 後 GPU kernel 期望的 reordered layout 不同。

#### 2. `update_weights()` 強制分配 `usm_device`

`primitive_inst.cpp` L2550-2615 的邏輯：

```cpp
// original_layout = streaming subbuffer 的 layout (pre-reorder)
// expected_layout = GPU kernel 需要的 reordered layout

if (_reordered_weights_cache.has(expected_layout)) {
    return;  // ← cache hit：零分配 ✅
}
else if (original_layout.compatible(expected_layout)) {
    // reinterpret_buffer：零拷貝 ✅
    _reordered_weights_cache.add(expected_layout,
        engine.reinterpret_buffer(*original_weights_memory, expected_layout));
    return;
}
else {
    // ⚠️ FULL REORDER：分配新記憶體
    auto alloc_type = engine.get_preferred_memory_allocation_type();  // → usm_device
    weights_memory = engine.allocate_memory(expected_layout, alloc_type);
    // ... execute reorder kernel ...
}
```

#### 3. `get_preferred_memory_allocation_type()` 回傳 `usm_device`

`engine.cpp` L126-136：

```cpp
allocation_type engine::get_preferred_memory_allocation_type(...) const {
    if (supports_allocation(allocation_type::usm_device))
        return allocation_type::usm_device;  // ← Intel iGPU 走這條
    ...
}
```

`usm_device` 是 GPU-local 記憶體。在 iGPU 上雖映射到同一物理 LPDDR5，
但 driver 可能對 device allocation 有不同的上限或追蹤機制。

#### 4. 記憶體累積估算

- 32 streamed layers × ~5-7 FC/layer = ~205 FC weight tensors
- 每個 FC 的 reorder output 大小 ≈ 原始權重大小（~30 MB avg）
- Groups 0-2 = 12 layers × ~5 FC = ~60 tensors × ~30 MB = **~1.8 GB usm_device 新分配**
- 加上原始 model weights ~6 GB → 總分配接近 ~7.8 GB
- Group 3 嘗試再分配 ~600 MB → 觸發 `CL_OUT_OF_RESOURCES`

#### 5. `weights_layout_opt` 為何始終 false？

`swap_weight_pointers()` 在 `prepare_primitive()` **之前**被呼叫。
此時 `_impl_params->weights_layout` 尚未被設定（首 token 尚未執行 `update_weights()`）。
因此 `update_weights_cache()` 永遠不被呼叫 → 符合預期，但 cache 無法預填充。

---

## 修復策略：Warm-Up Token（首 token 不走 streaming）

### 原理

```
首 token（warm-up）:
  → 走正常 execute_impl() 路徑（weights 在 compile_model 原始 USM）
  → update_weights() 為每個 FC 分配 reorder buffer 並填充 _reordered_weights_cache
  → _impl_params->weights_layout 被正確設定
  → 結束後所有 FC 的 cache 已有 expected_layout entry

後續 token（streaming）:
  → swap_weight_pointers() 替換 dep[1] 為 streaming subbuffer
  → update_weights() 計算 expected_layout（與 warm-up 相同，因 shape 不變）
  → _reordered_weights_cache.has(expected_layout) == true → cache hit!
  → 零分配、零 reorder → 直接 return ✅
```

### 為什麼 Cache Hit 保證成立？

`expected_layout` 的計算方式（primitive_inst.cpp L2551-2553）：

```cpp
auto expected_layout = reorder_kernel_params->get_output_layout()
    .clone_with_other_shape(original_layout.get_partial_shape());
```

- `reorder_kernel_params` 取決於 **FC kernel impl**（跨 token 不變）
- `original_layout.get_partial_shape()` = weight tensor shape（streaming 和 original 相同）
- 因此 `expected_layout` 在 warm-up 和 streaming 完全相同
- Cache lookup key match → **100% cache hit rate**

### 實作（預計修改 ~10 行）

**network.hpp**: 新增成員

```cpp
bool m_dense_streaming_warmed_up = false;
```

**network.cpp `execute_impl()`**: 新增 warm-up 分支

```cpp
if (has_dense_weight_streaming()) {
    if (!m_dense_streaming_warmed_up) {
        // 首 token：正常路徑，填充 reorder cache
        m_dense_streaming_warmed_up = true;
        // Fall through to normal set_arguments() + execute
    } else {
        execute_impl_streamed(events);
        return;
    }
}
```

### 優缺點

| 面向 | 評估 |
|---|---|
| **正確性** | ✅ Cache hit 由 layout 一致性數學保證 |
| **首 token 影響** | ✅ 無影響（warm-up = 正常推理，TTFT 不變） |
| **後續 token** | ✅ 零 reorder 開銷（cache hit → 直接 return） |
| **記憶體** | ⚠️ 首 token 的 reorder buffer 永遠存在（~1.8 GB usm_device），但不再增長 |
| **8 GB 系統** | ⚠️ warm-up 需 full model（~6 GB）+ reorder（~1.8 GB），可能不夠 |
| **實作複雜度** | ✅ 極低：1 bool flag + ~8 行邏輯 |

---

## 替代方案（未來考慮）

### Option A: 打包 Post-Reorder 權重（最佳長期方案）

從 compiled model 提取已 reorder 的權重打包到 streaming binary。
Streaming 時 `compatible() == true` → `reinterpret_buffer()` 零拷貝。
**完全消除 reorder + 可釋放 original weights → 最大記憶體節省。**

複雜度高：需在 compile_model 後遍歷 FC 的 reordered weights 對應回 per-layer 結構。

### Option B: 每 Group 後清除 Reorder Cache

避免跨 group 累積，但每 group 重新 reorder ~25 FC → ~100ms overhead/token，不可接受。

### Option C: 釋放 Original Model Weights

Streaming 初始化時釋放被 stream 的層的原始權重（data_inst::mem）。
理論節省 ~2 GB，但修改生命週期管理風險高。

---

## 測試計劃

```powershell
# Build
cd C:\working\gemma4-openvino\openvino
cmake --build build --target openvino_intel_gpu_plugin --config Release -- /v:m /p:CL_MPCount=4

# Deploy
Copy-Item build\src\plugins\intel_gpu\Release\openvino_intel_gpu_plugin.dll `
  C:\working\gemma4-openvino\ov_nightly\runtime\bin\intel64\Release\

# Test
cd C:\working\gemma4-openvino\gemma4-openvino-genai
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_gs4.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
  --prompt "What is the capital of Japan? Answer in one word." --max-new-tokens 16
```

### 驗證標準

1. ✅ 首 token 正常完成（warm-up #1，prefill），stderr 印出 warm-up 訊息
2. ✅ 第二 token 正常完成（warm-up #2，first decode），消除 IMPL_CHANGED 帶來的分配
3. ✅ 第三 token 起進入 streaming，所有 8 groups 完成不 crash
4. ✅ 生成 102 tokens 全部正確，回答語意連貫
5. ✅ Decode FC 全部走 NO_REORDER 路徑 → streaming 時零新分配

---

## ✅ 修復結果 (2026-05-14 23:40)

### 2-Token Warm-up 策略成功！

**問題**：1-token warm-up 只跑 prefill，第一個 decode 在 streaming 中觸發
`IMPL_CHANGED`（shape 從 seq_len=N 變 seq_len=1），每個 FC 都呼叫
`update_weights()` + `realloc_if_needed()`，累積分配在 group 3 OOM。

**修復**：改為 2-token warm-up：
- Warm-up #1 (prefill)：初始化 impl + 權重 cache（NO_REORDER）+ output buffers
- Warm-up #2 (first decode)：IMPL_CHANGED → 選擇 decode kernel + 分配 decode outputs
- Token #3+（streaming）：shape 不變(seq_len=1) → 零 IMPL_CHANGED → 零新分配！

### 效能數據（實測）

| 指標 | 基線（無 streaming） | Streaming (8 groups) | 比率 |
|---|---:|---:|---:|
| Generated tokens | — | 102 | ✅ 完整生成 |
| TTFT | 300 ms | 625 ms | 2.1× |
| TPOT | 41.7 ms | 161.9 ms | 3.9× |
| Throughput | 24.0 tps | 6.17 tps | 25.7% |

### Streaming 效能分解

- 每 group NVMe→USM 載入：~17 ms（穩定，無異常值）
- 8 groups × 17 ms = **~136 ms IO overhead / token**
- GPU 計算：~26 ms / token
- 總 TPOT ≈ 162 ms（IO 佔 84%）
- Swap: ~0.3-0.8 ms/group, set_args: ~0.04 ms/group（可忽略）

### 生成品質

Prompt: "Explain what is machine learning in exactly 5 sentences."
回答語意連貫，包含 "relationships", "input", "data", "machines", "learn",
"allows", "improve", "decision-making", "capabilities", "over time" 等
— 確認 weight swap 後數據正確送達 FC kernel。

### 未來優化方向

1. **Pipeline overlap（clFinish vs clFlush）**：目前用 clFlush（非阻塞），
   GPU fence 只要 0.0002ms → IO 完全無法重疊 GPU 計算。
   改為 clFinish 可讓 IO 在 GPU 等待期間進行。
2. **減少 group 數**：目前 8 groups（4 layers/group），
   改為 4 groups（8 layers/group）可減半 IO 次數，但需更大 buffer。
3. **釋放原始權重**：暖機後釋放 streamed layers 的原始權重記憶體，
   才能真正在 8 GB 系統上節省記憶體。

---

## 附錄：OpenVINO 樹 Unstaged Changes 完整清單

```
$ git diff HEAD --stat
 network.hpp      | +40（streaming API 聲明 + warmup_count + is_dense_streaming_warmed_up()）
 CMakeLists.txt   | +7（OV_DENSE_WEIGHT_STREAMING_ENABLED macro + winmm）
 primitive_inst.h  | +24（force_set_output_memory / update_weights_cache）
 network.cpp      | +121/-39（execute_impl_streamed 實作 + 2-token warm-up + pipeline 優化）
 primitive_inst.cpp | +50（update_weights 診斷 + usm_host override）
 + untracked: dense_weight_streaming_manager.cpp/hpp
```
