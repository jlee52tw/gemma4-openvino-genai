# Copilot Instructions — Gemma4 OpenVINO GenAI 專案

please response in Traditional Chinese. All instructions and Copilot responses should be in Traditional Chinese.



## 專案概述（Traditional Chinese）

本專案是 **Gemma-4-E4B-it** (INT4 量化) 在 Intel 內顯 (iGPU) 上使用 OpenVINO 推理的研究與實作專案。
主要目標是在 **8 GB 記憶體系統**上實現可接受的推理效能。

---

## 系統環境

- **硬體:** Intel Panther Lake, 12 Xe EUs iGPU, 16 GB LPDDR5（目標：8 GB 系統）
- **OS:** Windows 11
- **Runtime:** OpenVINO 2026.2.0 nightly + openvino.genai PR#3644
- **模型:** gemma-4-E4B-it-ov (INT4, group_size=64, 42 decoder layers, 6.05 GB 總大小)
- **NVMe:** Gen5×4, 持續循序讀取 ~12 GB/s

---

## 基線效能

| 情境 | Output TPS | TPOT (ms) | TTFT (s) | Peak RSS (GB) |
|---|---:|---:|---:|---:|
| short-text | **24.0** | 41.7 | 0.30 | 7.1 |
| long-text (1024 in) | **17.4** | 57.5 | 0.88 | 7.9 |
| short-image | 19.7 | 50.8 | 0.54 | 8.3 |

---

## Dense Weight Streaming 研究（重點方向）

### 問題定義

模型推理時，所有 42 層 decoder weights (~2.09 GB) 必須全部載入 GPU USM 記憶體。
在 8 GB 系統上，記憶體不足以同時容納所有權重 + KV cache + runtime overhead。
目標：讓部分權重留在 NVMe，用「串流」方式逐批載入。

### Option B 實驗結果（Phase 1, 2026-05-08）

**重要說明：以下數據是 blob cache 載入時間（第 2 次以後），不包含首次 IR→blob 編譯時間。**

首次編譯（IR → .blob）是一次性成本，只在模型第一次使用時發生，之後都從 .blob 快取載入。
我們的 Option B 構想是：每個 token 生成時，載入/卸載 sub-model 的 .blob —— 
因此只有 **blob cache 載入時間**是每個 token 都會付出的代價。

#### 完整模型載入計時

| 階段 | 時間 | 說明 |
|---|---:|---|
| 首次編譯 (IR → .blob) | 12.2 s | ❌ 一次性成本，之後不再發生 |
| **Blob cache 載入（第2次以後）** | **2.27 s** | ✅ 每次 compile_model 的實際成本 |
| 卸載 (unload) | 53 ms | ✅ 釋放記憶體很快 |

#### 分割模型 (2-way split) Blob Cache 載入時間

| Sub-model | 層數 | BIN 大小 | Blob 載入時間 | 卸載 | 完整循環 |
|---|---|---:|---:|---:|---:|
| Part 0 | 0-20 (21 layers) | 1.02 GB | **1.03 s** | 29 ms | 1.06 s |
| Part 1 | 21-41 (21 layers) | 1.62 GB | **1.22 s** | 20 ms | 1.24 s |
| **合計** | | | | | **2.30 s** |

#### 為什麼 Blob 載入要 1-2 秒？

`compile_model()` 從 .blob 載入時，瓶頸**不是 NVMe 讀取速度**，而是：

1. **USM 記憶體分配** — 為每個 weight tensor 分配 GPU-accessible 記憶體
2. **memcpy 權重資料** — 從 blob buffer 複製到 USM buffer（有效吞吐量 ~1.2 GB/s）
3. **Graph state 初始化** — 設定 KV cache variables、推理狀態

即使 NVMe 能提供 12 GB/s 的原始讀取頻寬，`compile_model` API 的記憶體管理開銷
將有效吞吐量限制在 ~1.2 GB/s。

#### Option B 結論

```
每 token 需要 2 次 sub-model 載入：
  2 × ~1.1 s = 2.2 s（僅載入開銷）
  + 42 ms （GPU 計算）
  = 2.24 s per token
  = 0.43 tps

結論：Option B（每 token 重新 compile_model）完全不可行。
```

### 下一步：Option A (DirectStorage USM Buffer Swap)

既然 `compile_model` API 的瓶頸在 USM 分配 + memcpy，
正確做法是：

1. **編譯模型一次**（首次或從 .blob cache）— 獲得已編譯的 GPU graph
2. **運行時直接替換 weight USM buffer** — 跳過 compile_model
3. **使用 DirectStorage** — NVMe→USM 零拷貝 DMA，達到真正的 12 GB/s
4. **重用 MoE OTD 的 `get_arguments()` 模式** — 改寫 weight 指標

參考：`jlee52tw/openvino` branch `moe-otd-pr-squash` 中的
`moe_expert_weight_manager.hpp/.cpp`

---

## 程式碼結構

| 檔案 | 用途 |
|---|---|
| `run_gemma4.py` | VLMPipeline 推理（text/image） |
| `benchmark.py` | 效能基準測試 |
| `measure_load_time.py` | 測量 blob cache 載入/卸載時間 |
| `split_language_model.py` | 分割 decoder model 為 sub-models |
| `pack_dense_weights.py` | 打包 decoder 權重為 DirectStorage 串流格式 |
| `20260508_dense_weight_streaming_plan.md` | Weight streaming 完整可行性計畫 |
| `cpp/run_gemma4.cpp` | C++ 版本推理 |
| `cpp/dense_weight_streaming_manager.hpp` | DirectStorage 雙緩衝串流管理器（標頭） |
| `cpp/dense_weight_streaming_manager.cpp` | DirectStorage 雙緩衝串流管理器（實作） |
| `reference/moe_expert_weight_manager.hpp` | MoE OTD 參考原始碼 |

---

## 模型路徑

- 模型目錄: `C:\working\gemma4-openvino\gemma-4-E4B-it-ov\`
- Blob cache: `C:\working\gemma4-openvino\gemma-4-E4B-it-ov\model_cache\`
- 分割模型: `C:\working\gemma4-openvino\gemma-4-E4B-it-ov\split_models\`
- OpenVINO 原始碼: `C:\working\gemma4-openvino\openvino\`
- GenAI 原始碼: `C:\working\gemma4-openvino\openvino_genai_src\`

---

## 開發慣例

- **所有對話回覆與說明文件使用繁體中文**（包含 Copilot 的回應）
- 效能數據必須標註來源（實測 vs 估算）
- 載入時間測量需區分：首次編譯（一次性）vs blob cache 載入（每次付出的成本）
- Commit message 使用英文
- 程式碼註解使用英文

---

## 關鍵技術概念

### Blob Cache 機制
OpenVINO 首次 `compile_model()` 會將 IR 編譯為 GPU kernel 圖（.blob），
存入 `CACHE_DIR`。之後重新載入時跳過編譯步驟，但仍然需要：
- 從 .blob 讀取序列化資料
- 分配 USM 記憶體
- 複製權重到 USM

### USM (Unified Shared Memory)
Intel GPU 使用 USM 而非傳統 GPU VRAM。USM buffer 可被 CPU 和 GPU 共同存取。
分配和複製都走系統記憶體總線，因此受限於記憶體頻寬而非 NVMe 頻寬。

### DirectStorage + BypassIO
Windows DirectStorage API 允許 NVMe→記憶體的零拷貝 DMA 傳輸，
搭配 BypassIO 繞過 OS 檔案快取層。MoE OTD 專案已證明這方法可用於
OpenVINO runtime 內的權重載入。

### KV Cache 結構
- 42 層 decoder 中只有 24 層（0-23）有 KV cache
- 每層 KV cache: 2 heads × 256 dim × seq_len × 2 (K+V) in FP32
- 層 5 是全注意力層 (512 dim)，其餘為 sliding window (256 dim)
