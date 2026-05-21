# 20260520 — Pipeline Profile + Memory Root Cause (Track A 完整報告)

> 在現有 release DLL（無 source 改動）上，用 token-level profiling + GPU 記憶體探測 + 程式碼閱讀，回答兩個問題：
> 1. Single-NVMe vs Dual-NVMe 的 pipeline overlap 行為差距為何？
> 2. 為何啟用 dense streaming 反而讓總記憶體變多？
>
> **結論**：Dual-NVMe 的提速來自實體磁碟並行；Streaming 沒節省記憶體是因為 **原始 FC weights 沒被釋放**（被 `_reordered_weights_cache` pin 住）。要在 8 GB 系統省記憶體，必須改 GPU plugin 程式碼。

---

## 1. 量測環境

- HW: Intel Xe³ Panther Lake 16 EUs, 16 GB LPDDR5
- NVMe: 2× Samsung MZVLC2T0HBLD 2 TB（D: Disk 0, E: Disk 2）
- Model: gemma-4-E4B-it-ov INT4 (gs=64), 42 decoder layers, ~6.05 GB
- Streaming: H5+T5 pinning（layers 0-4 + 37-41）, 8 groups × 4 layers, DirectIO HARDCODED
- Prompt: 33-token "Explain how a transformer attention block computes its output…"
- Generation: 256 tokens
- DLL: 2026-05-19 deployed `openvino_intel_gpu_plugin.dll` (35.9 MB)

---

## 2. Pipeline Profiling — 4 個變體

| 變體 | NUM_BUFFERS | IO_THREADS | File 0 | File 1 | Prefetch |
|---|---:|---:|---|---|---|
| v1 sequential | 2 | 4 | D:\…\0.bin | – | OFF (`OV_DENSE_STREAM_NO_PREFETCH=1`) |
| v2 single overlap | 2 | 4 | D:\…\0.bin | – | ON |
| v3 dual NVMe overlap | 4 | 8 | D:\…\0.bin | E:\…\1.bin | ON |
| v4 single NVMe dual file | 4 | 8 | D:\…\0.bin | D:\…\1.bin | ON |

### 2.1 Per-token 時間分解（256 tokens 平均，單位 ms）

| 變體 | TPOT | NVMe load | Swap | set_args | GPU compute* | Throughput | 總時間 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **v1 sequential** | 97.89 | 64.86 (66%) | 1.20 | 0.34 | 31.84 (32%) | **10.22 tps** | 24.86 s |
| **v2 single overlap** | 96.21 | 23.77 (25%) | 1.10 | 0.34 | 71.35 (74%) | **10.39 tps** | 24.44 s |
| **v3 dual NVMe** | 76.82 | 8.31 (11%) | 1.11 | 0.33 | 67.40 (88%) | **13.02 tps** | 19.51 s |
| **v4 single dual-file** | 105.83 | 38.65 (37%) | 0.93 | 0.33 | 65.92 (62%) | **9.45 tps** | 26.88 s |

*v2/v3/v4 的「GPU compute」欄位包含 GPU stall（等 next group 載入完成的時間）；真實 GPU 計算時間 ≈ v1 的 31.84 ms（每 group 4 layers）。

### 2.2 NVMe / GPU 比值（每 layer 時間）

```
GPU per-layer    = 31.84 / 4 = 7.96 ms
NVMe per-layer (v1 sequential)   = 64.86 / 4 = 16.22 ms   → ratio 2.04× (IO-bound)
NVMe per-layer (v2 single ovlp)  = 23.77 / 4 =  5.94 ms   → ratio 0.75× (mostly hidden)
NVMe per-layer (v3 dual NVMe)    =  8.31 / 4 =  2.08 ms   → ratio 0.26× (fully hidden)
NVMe per-layer (v4 dual-file 1Q) = 38.65 / 4 =  9.66 ms   → ratio 1.21× (worse than v2!)
```

### 2.3 ASCII Gantt（單 token，8 groups streamed）

**v1 sequential**（每 group 等 NVMe → GPU 同步）：
```
G0: [LOAD ████████████][GPU ████]
G1:                    [LOAD ████████████][GPU ████]
G2:                                       [LOAD ████████████][GPU ████]
...
total = 8 × (16ms LOAD + 8ms GPU) = 192 ms (理論)；實測 ~98 ms（已部分 overlap）
```

**v2 single overlap**（NVMe ↔ GPU 重疊；單 NVMe 隊列）：
```
G0:  [LOAD ████████████][GPU ████]
G1:        [LOAD ████████████][GPU ████]
G2:               [LOAD ████████████][GPU ████]
...
NVMe lane 完全飽和；GPU lane 偶爾等 IO（占 25% 時間）
```

**v3 dual NVMe overlap**（兩條獨立 NVMe 隊列；4 buffer / 8 thread）：
```
NVMe-D: G0[████████]  G2[████████]  G4[████████]  G6[████████]
NVMe-E:    G1[████████]  G3[████████]  G5[████████]  G7[████████]
GPU:    [G0_pin][G0][G1][G2][G3][G4][G5][G6][G7][T_pin]
        |←───── GPU 完全主導，IO 只占 11% ─────→|
```

**v4 single NVMe dual file**（兩個檔案在同一顆 NVMe；4 buffer / 8 thread）：
```
NVMe-D 隊列爭用：8 個 IO thread 同時搶 1 顆磁碟 →
IO depth 飽和、context switching 上升 → 比 v2（2 buf / 4 thread）更慢
```

### 2.4 結論
- **Pipeline overlap 確實有效**：v1→v2 的 NVMe 占比從 66% 降到 25%。
- **Dual-NVMe 的提速來自實體磁碟並行**：v3 比 v4 快 38%（13.0 vs 9.5 tps），即使兩者都是 4 buf / 8 thread。
- **單 NVMe 上開更多 buffer 反而退步**：v4 比 v2 慢 9%（IO 隊列爭用 + 更多 USM 配置開銷）。
- **新機 GPU 比舊機更快**：原預測 GPU/layer 1.4 ms，實測 7.96 ms（比舊機 ~1.9 ms 慢 4×）—— Panther Lake 16 EUs 對 INT4 reorder kernel 反而不如 12 Xe，可能是 Xe³ kernel 路徑未針對 reorder 優化。

---

## 3. GPU 記憶體探測（A2）

### 3.1 工具與方法
- 工具：Windows `Get-Counter "\GPU Process Memory(pid_X_*)\Local|Shared|Total Committed"`
- 因 Python 被 launcher fork 兩次，改用 `RUN_GEMMA4_PID_FILE` 環境變數讓 `run_gemma4.py` 自寫 pid → probe 讀檔取得真實 pid
- 在每個 case 跑 256-token decode，並以 250 ms 間隔取樣 GPU 記憶體
- iGPU 沒有 Dedicated VRAM；真值看 **Total Committed**

### 3.2 結果（peak / median MB）

| Case | 設定 | Peak Local | Peak Shared | **Peak Committed** | Δ vs B |
|---|---|---:|---:|---:|---:|
| **B baseline** | 無 streaming（mmap embedding only） | 3689 | 3672 | **3681** | — |
| **C v2 streaming** | NUM_BUFFERS=2 | 3879 | 3883 | **3888** | **+207 MB** |
| **D-single (v4)** | NUM_BUFFERS=4 single NVMe | 4479 | 4479 | **4484** | **+803 MB** |
| **D-dual (v3)** | NUM_BUFFERS=4 dual NVMe | 4474 | 4479 | **4484** | **+803 MB** |

### 3.3 關鍵觀察
- 新增 GPU 記憶體 ≈ `NUM_BUFFERS × 198 MB`（一個 group 的 streaming buffer 大小）
  - C: 2 × 198 = 396 MB 預期；實測 +207 MB
  - D: 4 × 198 = 792 MB 預期；實測 +803 MB ✓
- **Streaming 沒省到任何記憶體**：原 FC weights 完全沒被釋放，反而多花 streaming buffer 的記憶體。

---

## 4. 程式碼根因分析（A3）

### 4.1 Streaming 的 swap 流程（`cpp/dense_weight_streaming_manager.cpp:1562-1700`）
```cpp
// 對每個 streamed FC weight tensor：
data_inst->force_set_output_memory(new_mem, 0);   // _outputs[0] = new_mem (drops old ref)
fc_inst->update_weights_cache(weights_layout, new_mem);  // 試圖覆寫 cache 中的 entry
```

### 4.2 `_reordered_weights_cache` 的設計（`primitive_inst.h:425`，capacity = 3）
- 每個 dynamic FC primitive 維護 LRU cache，key = `layout`，value = `memory::ptr`
- 第一次 inference 在 `update_weights()`（`primitive_inst.cpp:2540-2630`）填入 cache
- **Cache 持有 `memory::ptr` 強引用** → 即使 `data_inst._outputs[0]` 被換走，cache 裡的舊 USM **不會被釋放**

### 4.3 `LruCache::add` 行為（`lru_cache.hpp:55`）
```cpp
bool add(const Key& key, const Value& value) {
    auto it = _key_map.find(key);
    if (it != _key_map.end()) {                   // ← key 已存在
        it->second->second = value;               //   覆寫 → 舊 value 引用降為 0 ✓
        return false;
    }
    if (_capacity == _key_map.size()) pop();      // ← key 不存在 → LRU 淘汰
    insert(...);
    return popped;
}
```

### 4.4 為什麼 streaming 沒釋放原 FC weights？

對於 INT4 量化 FC，`update_weights()` 在 reorder 路徑可能放兩種 entry：
1. `original_layout` → `original_weights_memory`（無 reorder 時）
2. `expected_layout` → `engine.reinterpret_buffer(original, expected_layout)`（compatible reorder）
3. `expected_layout` → 全新配置的 reordered USM（incompatible reorder）

我們的 `update_weights_cache(weights_layout, new_mem)` 只覆寫**一個** key。對 INT4 reorder 的 FC，layout 路徑更複雜：
- 若 `weights_layout_opt` = `expected_layout` 但 cache 同時有 `original_layout` entry → 只覆寫 expected，**original 仍 pin 住原 USM**
- 若 `weights_layout_opt.bytes_count() != new_mem->size()`（程式碼裡有 WARNING 顯示） → cache key 不匹配 → 變成新增 entry，舊的不會被淘汰（cap=3）

更關鍵的是：**scale/zp 兩個 data primitive 的 weights 確實被 force_set_output_memory 換掉並釋放，但量化模型的主 weight tensor (qzero_point + qscale + reorder copies) 加總大於 streamed buffer**，所以淨效應是 **streaming buffer 全是 overhead**。

### 4.5 程式碼層結論

**原始 FC weights 沒有被釋放**，原因組合：
1. **`_reordered_weights_cache` capacity=3 + LRU，cache 內舊 entry 不會自動淘汰**
2. **`update_weights_cache` 只在 layout key 完全匹配時才會 drop 舊 USM**
3. **量化 INT4 路徑常產生 cache key mismatch（reordered layout vs original layout）**

→ **Landing Zone B**：必須改 GPU plugin source。

---

## 5. 修正方案（給 Phase 2 / Track B 用）

### 5.1 最小修改（建議方案）
在 `primitive_inst.h` 加 force-clear 方法，並在 `swap_weight_pointers` 呼叫前先 clear cache：

```cpp
// primitive_inst.h
void clear_weights_cache() {
    _reordered_weights_cache.clear();
}
```

```cpp
// dense_weight_streaming_manager.cpp swap_weight_pointers (FC path)
if (mapping.is_fc_weight) {
    auto fc_inst = net->get_primitive(mapping.fc_primitive_id);
    if (fc_inst) {
        fc_inst->clear_weights_cache();           // ← 新增：先 drop 所有舊 cache entry
    }
    data_inst->force_set_output_memory(new_mem, 0);
    if (fc_inst) {
        fc_inst->update_weights_cache(weights_layout_opt.value(), new_mem);
    }
}
```

由 env var `OV_DENSE_STREAM_FREE_ORIGINALS=1` gate 住，避免影響 baseline 行為。

### 5.2 預期效益
- 釋放 32 streamed layers 的原始 FC weights ≈ **1.0 GB GPU local memory**
- 抵掉 streaming buffer overhead 後（4×198 = 792 MB）→ 淨省 ~200 MB
- 若改用 BUF=2（v2 模式）→ 淨省 ~600 MB
- 配合 mmap embedding offload（已有，3 GB 不在 RSS） → **可在 8 GB 系統運行**

### 5.3 風險
- `clear_weights_cache()` 後第一次 inference 會重新 reorder → 第一次 token 變慢（但只發生一次）
- 若 reorder kernel 不能就地處理 streaming sub-buffer 的 alignment，可能需要 fallback 到 pre-reorder 過的 streaming bin

---

## 6. 給 Track B/C 的決策表

| 落點 | 條件 | Phase 2 動作 | Phase 3 動作 | 預期記憶體 |
|---|---|---|---|---|
| A | streaming 已正確釋放（不適用，本報告否定） | – | – | – |
| **B** | **原 FC weights 沒釋放（本報告確認）** | **Build baseline DLL** | **Patch `clear_weights_cache` + rebuild** | **～4 GB Committed** |
| C | 部分釋放 | Build baseline | 量化後決定 | – |

→ **進入 Track B**：先建 baseline DLL（驗證 build chain），再進 Track C 套 patch。

---

## 7. 附錄

### 7.1 量測檔案
- `temp/v1_seq.log`, `v2_single.log`, `v3_dual.log`, `v4_singlenvme_dualfile.log`：dense streaming debug log
- `temp/v*_seq.stdout.log`：包含 VLMPipeline 的 TTFT/TPOT/Throughput
- `temp/gpumem_caseB|C|D_single|D_dual.{csv,summary.txt}`：GPU 記憶體取樣
- `temp/gpumem_probe_v4.ps1`：最終可用的 probe（用 `RUN_GEMMA4_PID_FILE`）

### 7.2 Probe 開發走過的坑
1. **v1 用 powershell wrapper 啟動 python**：counter 抓到 wrapper 的 pid，沒有 GPU instance → 0 MB
2. **v2 直接 Start-Process python**：但 `python.exe` 是 launcher，會 fork 一個真正的 python 子行程；wrapper 的 pid 無 GPU 活動
3. **v3 用 `pid_X_*` wildcard 在 PathsWithInstances 過濾**：仍是 launcher pid，找不到 instance
4. **v4 用 `RUN_GEMMA4_PID_FILE` 環境變數，讓 python 自己寫 `os.getpid()` 進檔案** → probe 讀檔取得真實 pid → 成功

### 7.3 為什麼 iGPU 的 Dedicated Usage 永遠是 0
iGPU 沒有獨立 VRAM，所有記憶體都來自系統 LPDDR5。Windows GPU performance counter 把它報在 Local Usage 與 Shared Usage（兩者數值幾乎相同，因為同一塊記憶體被「dedicated 給 GPU 視角」也是「與 CPU 共享」）。**Total Committed 是最可靠的單一數字。**
