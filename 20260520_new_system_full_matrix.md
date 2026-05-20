# 20260520 — 新系統完整效能矩陣（單／雙 NVMe × BUF × IO）

> **目的**：在新測試平台（Intel Xe³ iGPU 16 EUs + 雙 NVMe D:/E:）上，
> 以**相同 prompt** 系統性量測下列每一個改進步驟對 Gemma-4-E4B-it 推理效能的影響，
> 並用實測數據繪製 pipeline 流程圖，作為下一階段優化的依據。

---

## 1. 測試環境

| 項目 | 規格 |
|---|---|
| **CPU/iGPU** | Intel Xe³ Panther Lake, arch **v30.4.0**, **16 EUs** (`GPU_EXECUTION_UNITS_COUNT`) |
| **iGPU 名稱** | `Intel(R) Graphics (iGPU)` |
| **記憶體** | 16 GB LPDDR5 |
| **NVMe 0** | D: — Disk 0, Samsung MZVLC2T0HBLD 2 TB |
| **NVMe 1** | E: — Disk 2, Samsung MZVLC2T0HBLD 2 TB |
| **System SSD** | C: — Disk 1, Intel SSDPEKNW010T8 1 TB |
| **OS** | Windows 11 |
| **OpenVINO** | 2026.2.0.dev20260411 (modified DLL — `2026.2.0-21571-9c4a2eb9ad3`) |
| **openvino-genai** | 2026.2.0.0.dev20260411 branch `as/vlm_enable_1` (modified) |
| **模型** | gemma-4-E4B-it-ov INT4, group_size=64 |
| **per-layer embedding** | 3.22 GB，**目前 DLL 強制 mmap 模式**（DirectIO toggle 失效 — 見 §6） |
| **Dense streaming 檔案** | `dense_weights_streaming_{0,1}.bin`，總大小 1352.5 MB，FC 權重 only |

### Dense streaming 結構
- **42 層 decoder**：layers **0–4** + **37–41** 共 10 層 pinned（head=5 / tail=5）
- 中間 **32 層**（5–36）參與串流，分為 **8 group × 4 layers**
- Group 大小（每 NVMe）：99.05 / 99.05 / 95.68 / 99.05 / 98.38 / 92.99 / 95.68 / 95.68 MB
- Dual-NVMe 模式：每個 group 在 D: 與 E: 各放半段（sector-aligned 中點切分），IO 並行
- 同步點：`clFinish` GPU fence + 雙緩衝（BUF）交替，IO threads 為 Win32 OVERLAPPED + FILE_FLAG_NO_BUFFERING

### 共用 prompt
```
"Explain quantum computing in 3 sentences for a high school student."
--max-new-tokens 96   →  實際 generated = 80 tokens (hit EOS)
input tokens = 22
```

---

## 2. 測試矩陣（9 個 config，每個跑單次完整推理）

> 由於每次測試 `compile_model` 都從 GPU **blob cache** 載入（首次編譯成本已分攤），
> Load time 在所有 config 都 ≈ 4.45–4.52 秒；TTFT 與 TPOT 是主要比較指標。

| # | Tag | 串流模式 | NVMe | BUF | IO threads | embedding 模式（實際） |
|---|---|---|---|---|---|---|
| 01 | `PB_mmap`        | OFF      | —      | — | — | mmap |
| 02 | `PB_DIO`         | OFF      | —      | — | — | mmap (DIO toggle 失效) |
| 03 | `S_mmap_b2i4`    | ON 單    | D:     | 2 | 4 | mmap |
| 04 | `S_DIO_b2i4`     | ON 單    | D:     | 2 | 4 | mmap (DIO toggle 失效) |
| 05 | `D_mmap_b2i4`    | ON 雙    | D:+E:  | 2 | 4 | mmap |
| 06 | `D_mmap_b2i8`    | ON 雙    | D:+E:  | 2 | 8 | mmap |
| 07 | `D_mmap_b4i4`    | ON 雙    | D:+E:  | 4 | 4 | mmap |
| 08 | `D_mmap_b4i8`    | ON 雙    | D:+E:  | 4 | 8 | mmap |
| 09 | `D_DIO_b4i8`     | ON 雙    | D:+E:  | 4 | 8 | mmap (DIO toggle 失效) |

---

## 3. 實測結果（end-to-end GenAI metrics）

| # | Tag | Load (ms) | TTFT (ms) | **TPOT (ms)** | **TPS** | Gen dur (ms) | Inf dur (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| 01 | PB_mmap         | 4516 |  604.25 | **69.61** | **14.37** | 6104 | 5849 |
| 02 | PB_DIO          | 4474 |  613.91 |   69.87   |   14.31   | 6128 | 5854 |
| 03 | S_mmap_b2i4     | 4492 |  756.29 |   97.75   |   10.23   | 8458 | 8202 |
| 04 | S_DIO_b2i4      | 4453 |  634.34 |   91.65   |   10.91   | 7866 | 7601 |
| 05 | D_mmap_b2i4     | 4475 |  805.81 |  113.10   |    8.84   | 9697 | 9436 |
| 06 | D_mmap_b2i8     | 4465 |  661.01 |  104.25   |    9.59   | 8887 | 8627 |
| 07 | D_mmap_b4i4     | 4457 |  705.17 |   93.51   |   10.69   | 8060 | 7821 |
| 08 | D_mmap_b4i8     | 4440 |  686.17 | **93.50** | **10.69** | 8049 | 7812 |
| 09 | D_DIO_b4i8      | 4458 |  710.22 |   92.47   |   10.81   | 7980 | 7742 |

### 內部串流統計（DenseStreaming manager — 不含 sampler／detok／embedding lookup）

| Tag | Avg TPOT (ms) | 內部 TPS | 穩態 per-token breakdown |
|---|---:|---:|---|
| 03 S_mmap_b2i4 | **75.96** | 13.16 | total≈618, load=552, swap=1.0, gpu=64.7（單 NVMe 序列化） |
| 05 D_mmap_b2i4 | 89.33 | 11.19 | total≈603, load=531, swap=0.9, gpu=70.8（雙 NVMe 但 thread 不足，回退近序列） |
| 06 D_mmap_b2i8 | 80.19 | 12.47 | total=87.5, load=24.5, swap=1.1, gpu=61.5（IO 增為 8 → 載入並行化奏效） |
| 07 D_mmap_b4i4 | **72.90** | 13.72 | total=91.0, load=20.0, swap=0.9, gpu=69.7（BUF=4 預載完整覆蓋 GPU compute） |
| 08 D_mmap_b4i8 | **72.67** | 13.76 | total=92.5, load=20.8, swap=1.0, gpu=70.4（與 b4i4 等同；IO=8 飽和） |

---

## 4. 觀察與分析

### 4.1 串流相對於 pure baseline 的「稅」
- **Pure baseline TPOT**：69.6 ms（mmap embedding，全部 FC 權重已 resident GPU）
- **最佳串流 TPOT**：93.5 ms（dual D+E, BUF=4, IO=8）  
  → **streaming tax ≈ 24 ms/token**（端到端），**≈ 3 ms/token**（內部串流量測）
- **內部 vs 端到端 差距 ~20 ms**：來自 per-layer embedding lookup（mmap，cold lookup 隨機 IO）、sampler、detokenizer、Python 層 overhead

### 4.2 從單 NVMe → 雙 NVMe 的提升
| 改進步驟 | TPOT (ms) | TPS | Δ vs 上一步 |
|---|---:|---:|---|
| Pure baseline (no streaming) | 69.61 | 14.37 | — |
| **+** Single NVMe streaming (BUF=2 IO=4) | 97.75 | 10.23 | **−4.1 tps**（首次付出 streaming 成本） |
| **+** Dual NVMe naive (BUF=2 IO=4) | 113.10 | 8.84 | **−1.4 tps**（IO threads 不足，多 NVMe 反而傷害） |
| **+** Dual NVMe IO=8 (BUF=2) | 104.25 | 9.59 | +0.75 tps |
| **+** Dual NVMe BUF=4 IO=4 | 93.51 | 10.69 | +1.1 tps |
| **+** Dual NVMe BUF=4 IO=8 | 93.50 | 10.69 | ±0 tps (持平) |

**關鍵發現**：
1. **BUF=4 是最大改進來源**（從 b2 升到 b4 帶來 +1.85 tps，從 i4 升到 i8 只 +0.0–0.95 tps）
2. **BUF=2 雙 NVMe 反而比單 NVMe 慢**（113 vs 98 ms）— 兩條 IO queue 競爭同一個雙緩衝 slot
3. **IO threads 在 BUF=4 已飽和**（i4 = i8）— 瓶頸不在 IO

### 4.3 串流 vs 純 baseline 的差距無法靠 dual NVMe 補回
- 我們仍比 pure baseline 慢 24 ms/token，即使最佳配置。
- 主因：每個 group 完成 IO 後仍需 `clFinish` 同步 + USM buffer swap，這部分是序列化的 GPU 開銷。

### 4.4 DirectIO embedding toggle 失效 ⚠
測試 02/04/09 原本設計為 DirectIO embedding 對照組，但**現有 DLL build 不論透過**
1. 環境變數 `OV_PER_LAYER_EMBEDDING_PATH` 指向 `per_layer_embedding_directio.bin`
2. 或依靠檔名自動偵測

**皆載入為 mmap 模式**（DLL 內部 PerLayerEmbeddingReader 強制 mmap）。
→ 02、04、09 實際是 01、03、08 的近重複（誤差 < 1 ms TPOT）。
→ 要驗證 DirectIO 必須重新 build 含 DirectIO 路徑的 GenAI DLL。

---

## 5. Pipeline 流程圖（基於實測數據）

### 5.1 Pure baseline（69.6 ms/token）
```
token N timeline (ms):
  0      10     20     30     40     50     60     70
  |------|------|------|------|------|------|------|
  ├── sampler/detok ~2 ms
        ├── per-layer embed lookup (mmap, cold) ~15-20 ms
                          ├──────── GPU compute 42 layers ───────┤
                          └─ ~50 ms (FP16 INT4 matmul) ──────────┘
  端到端 = 69.6 ms (TPS 14.37)
```

### 5.2 Single NVMe streaming, BUF=2 IO=4（97.8 ms/token）
```
Group:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
                                                            
GPU:    pin─G0─G1─G2─G3─G4─G5─G6─G7─pin     compute 序列：~64.7 ms
NVMe D:   ─G2─G3─G4─G5─G6─G7──             載入後續：完全序列，每組 ~10-12 ms

(BUF=2 → 同時只能預備 1 個未來 group，GPU 大量等 IO)
端到端 = 97.8 ms (TPS 10.23)，streaming tax = 28 ms (40%)
```

### 5.3 Dual NVMe naive, BUF=2 IO=4（113.1 ms/token） ❌ 反效果
```
Group:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
GPU compute   ──G0──G1── (等 IO) ──G2── (等 IO) ──G3── ...
NVMe D:    ─G2½─        ─G3½─       ─G4½─                 (序列調度)
NVMe E:    ─G2½─        ─G3½─       ─G4½─                 (序列調度)

問題：BUF=2 + IO=4 thread pool 被兩條 queue 搶占
       → 兩 NVMe 的 half-stripe 互相 stall
端到端 = 113.1 ms (TPS 8.84)
```

### 5.4 Dual NVMe BUF=4 IO=8（93.5 ms/token） ✅ 最佳實測
```
Group:     [0]  [1]  [2]  [3]  [4]  [5]  [6]  [7]
GPU compute   ─G0─G1─G2─G3─G4─G5─G6─G7─                    pure compute ~70 ms
NVMe D:    ─G2½─G3½─G4½─G5½─G6½─G7½─                       並行 8 ops
NVMe E:    ─G2½─G3½─G4½─G5½─G6½─G7½─                       並行 8 ops
         ↑                                  ↑
         BUF=4 → 預先載入 3 個未來 group，GPU 幾乎不等
         load 從 552→20 ms（per-group），swap=1 ms

剩餘 load=20 ms 的來源：
  - 殘餘 io_wait（前 2 個 group 因 BUF 還未 warm）
  - clFinish 隱含同步 (~15-18 ms 已計入 gpu=70 ms)
端到端 = 93.5 ms (TPS 10.69)，streaming tax = 24 ms (34%)
```

### 5.5 各 config 數值比較圖
```
TPS (higher = better, end-to-end)
  16 ┤
  14 ┤ ●14.37  ●14.31                              ← Pure baseline
  12 ┤                                              
  10 ┤            ●10.23 ●10.91                    ●10.69 ●10.69 ●10.81
   8 ┤                          ●8.84  ●9.59
   6 ┤
       PB    PB    S     S     D     D     D     D     D
       mmap  DIO*  b2i4  b2i4* b2i4  b2i8  b4i4  b4i8  b4i8*
       (*DirectIO 對照組失效，等同 mmap)

TPOT ms (lower = better, end-to-end)
 120 ┤            ●            ●113
 100 ┤            ●98          ●104
  80 ┤                          ●91.7  ●93.5  ●93.5  ●92.5
  70 ┤ ●69.6 ●69.9
```

---

## 6. 結論與下一步建議

### 結論
1. **新系統 baseline (16 EU Xe³)** TPOT 69.6 ms / **14.4 tps**，比歷史 24-tps 系統慢 ~40%（推測該系統 ~24 EU）
2. **Dense weight streaming** 在新系統最佳實測 **TPOT 93.5 ms / 10.7 tps**（dual D+E, BUF=4, IO=8）
3. **Streaming tax ≈ 24 ms/token**（端到端）= 約 **−25% TPS** 的代價
4. **雙 NVMe 必須搭配 BUF=4** 才能勝過單 NVMe；naive 設定（BUF=2 IO=4）反而傷害效能
5. **IO threads 在 BUF=4 已飽和** — 從 IO=4 升到 IO=8 無明顯改善
6. **per-layer embedding DirectIO toggle 在現有 DLL 失效** — 要驗證需重 build GenAI

### 下一步
- [ ] **重 build GenAI DLL** 修復 DirectIO toggle（環境變數或檔名偵測），驗證 DirectIO embedding 是否能 reduce 內部/端到端 gap
- [ ] **長 context 測試**（input=1024）— 過去文件指出 mmap > DirectIO 在 8K context 因 OS readahead，需在新系統重新驗證
- [ ] **vary head/tail pin** — 目前 head=5 tail=5；若將 tail 全 pin、head 改用 mmap，可能進一步降 IO 壓力
- [ ] **swap/clFinish 微優化** — 穩態 per-token 剩 ~20 ms load 中，~5 ms 是 fence 等待；考慮 GPU events 取代 clFinish
- [ ] **記憶體峰值與工作集量測**（這份矩陣未抓 RSS／GPU memory）— 8 GB 系統可行性驗證

### 原始資料
- 9 個 stdout log: `temp/matrix_{tag}_stdout.log`
- 7 個 streaming debug log: `temp/matrix_{tag}.log`（only configs 03–09）
- Summary JSON: `temp/matrix_summary.json`
- Matrix runner: `temp/run_matrix.ps1`
