# Dense Weight Streaming — 雙 NVMe Pipeline 效能分析
**Date:** 2026-05-13  
**Author:** jlee52tw  
**Status:** 分析文件 — 多 NVMe 配置效能推估

---

## 0. Release Package 是否支援 Dense Weight Streaming？

**是的**，但需要額外設定：

| 元件 | Release 是否包含 | 說明 |
|---|:---:|---|
| `openvino_intel_gpu_plugin.dll` (modified) | ✅ | runtime/ 中包含已修改的 GPU plugin |
| `openvino_genai.dll` (modified) | ✅ | bin/ 中包含 per-layer embedding 支援 |
| `dense_weights_streaming.bin` (~1.55 GB) | ❌ | 需自行用 `pack_dense_weights.py` 產生 |
| `dense_weights_streaming.json` (~330 KB) | ❌ | 同上 |
| Per-layer embedding (`*_revised.bin`) | ✅ | model/ 已包含，自動偵測 |

**啟用 dense weight streaming 的額外步驟：**
```cmd
:: 1. 產生 streaming binary（在 scripts/ 中執行）
python pack_dense_weights.py --model_dir model --output_dir . --first_streamed 5 --last_streamed 36

:: 2. 設定環境變數
set OV_DENSE_STREAM_WEIGHTS=C:\path\to\dense_weights_streaming.bin

:: 3. 執行（--no-mmap 推薦）
run_gemma4.exe --model-dir model --no-mmap
```

> **注意：** Dense weight streaming 目前最佳效能為 ~6.78 tok/s（baseline 的 28%），
> 主要用於 8 GB 記憶體系統無法載入完整模型時的 fallback。
> 日常使用推薦不開啟 streaming（24 tok/s baseline）。

---

## 1. Phase 2 實測數據回顧

### 1.1 已知系統參數

| 參數 | 值 | 來源 |
|---|---|---|
| 模型 | Gemma4 E4B INT4, 42 decoder layers | 實測 |
| Streamed layers | 5–36 (32 layers) | 配置 |
| Pinned HEAD | layers 0–4 (5 layers) | 配置 |
| Pinned TAIL | layers 37–41 (5 layers) | 配置 |
| FC weights 總量 | **1352.5 MB** (205 tensors) | 實測 |
| FC weights per layer | ~42.3 MB avg | 1352.5÷32 |
| Scale/ZP (pinned) | ~5 MB/layer, 160 MB total | 設計決策 |
| NVMe 型號 | Gen5×4 | 硬體 |
| NVMe 順序讀取 | **~12 GB/s** 持續 | 實測 |
| iGPU | Intel Panther Lake, 12 Xe EUs | 硬體 |
| 系統記憶體 | 16 GB LPDDR5 | 硬體 |

### 1.2 Phase 2 最終實測結果

| 配置 | TPOT (ms) | tok/s | IO (ms) | GPU (ms) | Swap+Args (ms) | 備註 |
|---|---:|---:|---:|---:|---:|---|
| **Baseline** (no streaming) | **41.7** | **24.0** | — | 41.7 | — | 所有權重常駐 GPU |
| 1-layer/32-group, 2-buf | 188.9 | 5.3 | 114.3 | 55.2 | 19.4 | Phase 2 初始 |
| **4-layer/8-group, 3-buf** | **147.0** | **6.78** | **106.7** | **38.7** | **1.6** | **Phase 2 最佳** |

### 1.3 實測數據推導的關鍵常數

```
從 4-layer/8-group 與 1-layer/32-group 的數據反推：

■ NVMe 有效吞吐量
  1352.5 MB ÷ 106.7 ms = 12.67 GB/s （4-layer 大塊讀取）
  1352.5 MB ÷ 114.3 ms = 11.83 GB/s （1-layer 小塊讀取）

■ GPU 純計算時間 per layer
  HEAD+TAIL ≈ 10 ms (5+5 layers)
  Streamed: (38.7 - 10) ÷ 32 = 0.897 ms/layer
  
■ GPU transition overhead
  55.2 - 38.7 = 16.5 ms for 24 extra transitions
  → 0.69 ms per group transition (kernel re-dispatch cost)

■ IO per group (4-layer, 1 NVMe)
  106.7 ÷ 8 = 13.34 ms (169 MB @ 12.67 GB/s)

■ GPU per group (4-layer, excluding HEAD/TAIL)
  (38.7 - 10) ÷ 8 = 3.59 ms
```

### 1.4 重要發現：目前實作為**純序列**執行

Phase 2 per-token timing 分析顯示 IO + GPU + Swap + Args 加總 = TPOT。
這代表 **pipeline overlap 並未實際生效** — IO 與 GPU 序列執行。

```
實測 per-group 序列：
  [IO load 13.34ms] → [Swap 0.13ms] → [Args 0.08ms] → [GPU 3.59ms]
                                                                    ↓
  [IO load 13.34ms] → [Swap 0.13ms] → [Args 0.08ms] → [GPU 3.59ms]

理論 pipeline 設計（尚未達成）：
  [IO load 13.34ms] → [Swap] → [GPU 3.59ms]──────┐
                                  ↕ OVERLAP        IO prefetch
  ────────────────── [IO load G+1 ........] → [Swap] → [GPU]
```

原因可能是：
1. iGPU 與 NVMe 共用系統記憶體頻寬，GPU 計算時佔用匯流排
2. 目前的 async prefetch 實作不夠真正非同步
3. USM 分配 + memcpy 不支援真正的 DMA overlap

---

## 2. 多 NVMe 配置分析

### 2.1 場景定義

| 場景 | NVMe 配置 | 有效讀取頻寬 | 說明 |
|---|---|---:|---|
| **1×NVMe** (current) | 單 Gen5×4 | 12 GB/s | 現有配置 |
| **2×NVMe** | RAID-0 或 striped 讀取 | 24 GB/s | 第二個 NVMe 存放 streaming binary |
| **3×NVMe** | 三路 striped | 36 GB/s | 理論上限探索 |

> **假設：** NVMe 頻寬可線性疊加。實際上 RAID-0 或 DirectStorage 多佇列
> 在 Gen5 NVMe 上可接近線性擴展（PCIe 5.0 ×4 = 15.75 GB/s per drive，
> 2 drives ≈ 28-30 GB/s，保守用 24 GB/s）。

### 2.2 IO 時間縮放

```
Total FC data = 1352.5 MB

1×NVMe (12 GB/s):  1352.5 ÷ 12 = 112.7 ms  (實測 106.7-114.3 ms ✓)
2×NVMe (24 GB/s):  1352.5 ÷ 24 =  56.4 ms
3×NVMe (36 GB/s):  1352.5 ÷ 36 =  37.6 ms
```

---

## 3. 效能推估矩陣

### 3.1 Sequential Mode（當前行為 — IO 與 GPU 不重疊）

計算公式：
$$\text{TPOT}_{\text{seq}} = T_{\text{IO}} + T_{\text{GPU}} + T_{\text{overhead}}$$

其中：
- $T_{\text{IO}} = \frac{1352.5}{K \times 12} \text{ ms}$ （K = NVMe 數量）
- $T_{\text{GPU}}$ 取決於 group 數（含 transition overhead）
- $T_{\text{overhead}}$ = swap + set_arguments + misc

| # Groups (G layers) | NVMe | IO (ms) | GPU (ms) | Overhead (ms) | **TPOT (ms)** | **tok/s** | vs Baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8 groups (4L) | **1** | 106.7 | 38.7 | 1.6 | **147.0** | **6.78** | 28% ← **實測** |
| 8 groups (4L) | **2** | 53.4 | 38.7 | 1.6 | **93.7** | **10.7** | 44% |
| 8 groups (4L) | **3** | 35.6 | 38.7 | 1.6 | **75.9** | **13.2** | 55% |
| 4 groups (8L) | 1 | 103.0 | 36.2 | 0.6 | 139.8 | 7.2 | 30% |
| 4 groups (8L) | **2** | 51.5 | 36.2 | 0.6 | **88.3** | **11.3** | 47% |
| 4 groups (8L) | 3 | 34.3 | 36.2 | 0.6 | 71.1 | 14.1 | 59% |
| 2 groups (16L) | 1 | 100.0 | 35.2 | 0.3 | 135.5 | 7.4 | 31% |
| 2 groups (16L) | **2** | 50.0 | 35.2 | 0.3 | **85.5** | **11.7** | 49% |
| 2 groups (16L) | 3 | 33.3 | 35.2 | 0.3 | 68.8 | 14.5 | 60% |
| 1 group (32L) | 2 | 56.4 | 35.0 | 0.2 | 91.6 | 10.9 | 45% |

> **GPU 時間估算：**
> - 8 groups: 實測 38.7 ms（含 HEAD/TAIL 10ms + 32×0.897 + 8×0.69 transition overhead ≈ 44.2ms... 
>   實測 38.7 較低，取實測值）
> - 4 groups: 38.7 - 4×0.69 = 36.2 ms（4 fewer transitions）
> - 2 groups: 36.2 - 2×0.69 = 35.2 ms
> - 1 group: 35.2 - 0.69 = 35.0 ms

### 3.2 Pipeline Mode（理論 — IO 與 GPU 完全重疊）

計算公式：
$$\text{TPOT}_{\text{pipe}} = T_{\text{HEAD}} + T_{\text{IO}_0} + (N-1) \times \max(T_{\text{IO}_g}, T_{\text{GPU}_g}) + T_{\text{GPU}_\text{last}} + T_{\text{TAIL}}$$

其中：
- $T_{\text{HEAD}} = 5$ ms, $T_{\text{TAIL}} = 5$ ms
- $T_{\text{IO}_g} = \frac{1352.5}{N \times K \times 12}$ ms
- $T_{\text{GPU}_g} = \frac{28.7}{N} + 0.69$ ms (streamed compute + transition)

| # Groups (G layers) | NVMe | IO/grp (ms) | GPU/grp (ms) | Bottleneck | **TPOT (ms)** | **tok/s** | vs Baseline |
|---|---:|---:|---:|---|---:|---:|---:|
| 8 groups (4L) | 1 | 13.34 | 4.28 | IO ⬤ | 120.9 | 8.3 | 35% |
| 8 groups (4L) | **2** | **6.67** | **4.28** | **IO ⬤** | **67.5** | **14.8** | **62%** |
| 8 groups (4L) | **3** | **4.45** | **4.28** | **≈ 平衡 ⚖** | **49.7** | **20.1** | **84%** |
| 4 groups (8L) | 1 | 28.13 | 8.26 | IO ⬤ | 102.0 | 9.8 | 41% |
| 4 groups (8L) | **2** | **14.06** | **8.26** | **IO ⬤** | **65.5** | **15.3** | **64%** |
| 4 groups (8L) | 3 | 9.38 | 8.26 | IO ⬤ | 51.2 | 19.5 | 81% |
| 16 groups (2L) | 2 | 3.33 | 2.49 | IO ⬤ | 61.6 | 16.2 | 68% |
| 16 groups (2L) | **3** | **2.22** | **2.49** | **GPU ⬤** | **52.4** | **19.1** | **80%** |

> **⬤ IO-bound：** max = IO/grp → 增加 NVMe 直接降低 TPOT
> **⬤ GPU-bound：** max = GPU/grp → 增加 NVMe 無法降低 TPOT
> **⚖ 平衡：** IO ≈ GPU → 最佳效率，pipeline 幾乎 100% 遮蔽

### 3.3 重點摘要

```
                    Sequential (實測行為)      Pipeline (理論最佳)
                    ─────────────────────      ───────────────────
1×NVMe (current):   6.78 tok/s (28%)          8.3 tok/s (35%)
2×NVMe (proposed):  10.7 tok/s (44%)          14.8 tok/s (62%)  ← 🎯 值得投資
3×NVMe (ideal):     13.2 tok/s (55%)          20.1 tok/s (84%)  ← 接近 baseline！
Baseline:           24.0 tok/s (100%)
```

---

## 4. Pipeline 資料流圖

### 4.1 當前配置：1×NVMe, 4-layer/8-group, Sequential（實測行為）

```
Per-Token Decode: 1×NVMe, 4L/8-Group, Sequential
═══════════════════════════════════════════════════
TPOT ≈ 147 ms | 6.78 tok/s

Time (ms) 0    10   20   30   40   50   60   70   80   90  100  110  120  130  140  150
          ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤

HEAD (pinned, ~5ms):
  GPU: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Group 0 (layers 5-8, 169 MB):
  IO:  ░░░░░████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  13.3ms
  GPU: ░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  3.6ms

Group 1 (layers 9-12):
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  13.3ms
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  3.6ms

Group 2 (layers 13-16):
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████░░░░░░░░░░░░░░  13.3ms
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░░  3.6ms

  ... ×8 groups, each ~17ms (IO 13.3 + GPU 3.6 + swap 0.2) ...

Group 7 (layers 33-36):
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████░░░░
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

TAIL (pinned, ~5ms):
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

IO:  ████████████████████████████████████████████████████████████░░░░░░░░░░  106.7ms (73%)
GPU: ░░░░██░░░░░░░░░░░░██░░░░░░░░░░░░██░░░░░░░░░░░░██░░░░░░██░░░░░░░░██░  38.7ms (26%)
      ↑ 完全序列，IO 與 GPU 無重疊                                  overlap = 0%
```

### 4.2 提案配置：2×NVMe, 4-layer/8-group, Sequential

```
Per-Token Decode: 2×NVMe, 4L/8-Group, Sequential
═══════════════════════════════════════════════════
TPOT ≈ 93.7 ms | 10.7 tok/s

Time (ms) 0    10   20   30   40   50   60   70   80   90  100
          ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤

HEAD (pinned, ~5ms):
  GPU: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Group 0 (169 MB from 2×NVMe = 24 GB/s):
  IO:  ░░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  6.7ms (半數！)
  GPU: ░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  3.6ms

Group 1:
  IO:  ░░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░░  6.7ms
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░  3.6ms

Group 2:
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░░  6.7ms
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░  3.6ms

  ... ×8 groups, each ~10.5ms (IO 6.7 + GPU 3.6 + swap 0.2) ...

Group 7:
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████░░░░
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

TAIL (pinned, ~5ms):
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

IO:  ████████████████████████████████████████░░░░░░░░░░░░░░  53.4ms (57%)
GPU: ░░░░██░░░░░░██░░░░░░██░░░░░░██░░░░██░░░░██░░░░██░░██  38.7ms (41%)
      ↑ 仍然序列，但 IO 時間減半                           overlap = 0%
      ↑ TPOT 從 147→93.7ms, 速度提升 1.57×
```

### 4.3 提案配置：2×NVMe, 4-layer/8-group, Pipeline（理論最佳）

```
Per-Token Decode: 2×NVMe, 4L/8-Group, Double-Buffer Pipeline
═══════════════════════════════════════════════════════════════
TPOT ≈ 67.5 ms | 14.8 tok/s

Time (ms) 0    10   20   30   40   50   60   70
          ├────┼────┼────┼────┼────┼────┼────┤

HEAD (pinned, ~5ms):
  GPU: ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Group 0 (cold start — no overlap):
  IO:  ░░░░░████████░░░░░░░░░░░░░░░░░░░░░░░░  Load G0→Buf[0] (6.7ms)
  GPU: ░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░  Execute G0 (3.6ms)

Group 1 (pipelined — IO overlaps with GPU):
  IO:  ░░░░░░░░░░░░░[████]███░░░░░░░░░░░░░░░  Prefetch G1→Buf[1] (6.7ms total)
                      ↑3.6↑ ↑2.1↑              3.6ms hidden, 2.1ms stall
  GPU: ░░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░░░  Execute G1 (3.6ms)

Group 2:
  IO:  ░░░░░░░░░░░░░░░░░░░░░░[████]███░░░░░░  Prefetch G2→Buf[0]
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████░░  Execute G2

  ... pipeline step = max(6.7, 3.6+0.2) = 6.7ms per group ...

Group 7 (last):
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░[████]███
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

TAIL (pinned, ~5ms):
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

IO:  ████████[████████████████████████████████████]░░░░  53.4ms total
GPU: ░░░░██░░[░██░░░░██░░░░██░░░░██░░░░██░░░░██]░██░██  38.7ms total
              ↑──── IO & GPU 交織，pipeline 生效 ────↑
              Overlap = 7 × min(6.67, 3.59) = 25.1ms
              Overlap ratio = 25.1 / 53.4 = 47%

Buffer ping-pong: Buf[0] ↔ Buf[1] alternates each group
Each buffer: ~169 MB (4 layers × 42.3 MB)
```

### 4.4 理想配置：3×NVMe, 4-layer/8-group, Pipeline（接近平衡！）

```
Per-Token Decode: 3×NVMe, 4L/8-Group, Pipeline (Near-Balanced)
═══════════════════════════════════════════════════════════════
TPOT ≈ 49.7 ms | 20.1 tok/s (84% of baseline!)

Time (ms) 0    10   20   30   40   50
          ├────┼────┼────┼────┼────┤

HEAD (pinned):
  GPU: ████░░░░░░░░░░░░░░░░░░░░░░░░░

Group 0 (cold):
  IO:  ░░░░░█████░░░░░░░░░░░░░░░░░░░  4.5ms (IO@36GB/s)
  GPU: ░░░░░░░░░░████░░░░░░░░░░░░░░░  3.6ms

Group 1 (IO ≈ GPU → nearly perfect overlap!):
  IO:  ░░░░░░░░░░[████]░░░░░░░░░░░░░  4.5ms (3.6ms hidden, 0.9ms stall)
  GPU: ░░░░░░░░░░░░░░░████░░░░░░░░░░  3.6ms

Group 2:
  IO:  ░░░░░░░░░░░░░░░[████]░░░░░░░░  Almost fully hidden!
  GPU: ░░░░░░░░░░░░░░░░░░░░████░░░░░  3.6ms

  ... step = max(4.5, 3.8) = 4.5ms ≈ GPU-bound crossover point! ...

Group 7:                                              ┌────── IO 幾乎完全
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░[████]  │   被 GPU 遮蔽！
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████ ◄┘

TAIL:
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████

IO:GPU ratio = 4.45 : 4.28 = 1.04:1  ← ⚖ 幾乎完美平衡！
Overlap ratio ≈ 96%
```

### 4.5 Sequential vs Pipeline 對比（2×NVMe, 8-layer/4-group）

```
Per-Token Decode: 2×NVMe, 8L/4-Group, SEQUENTIAL vs PIPELINE
═════════════════════════════════════════════════════════════

■ SEQUENTIAL (current behavior):  TPOT ≈ 88.3 ms | 11.3 tok/s

  Time:  0────10───20───30───40───50───60───70───80───90
  IO:    ████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  G0: 14.1ms
  GPU:   ░░░░░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░░░░░░░░░  G0: 7.2ms
  IO:    ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████████  G1: 14.1ms
  GPU:   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████  G1: 7.2ms
         ... ×4 groups, each ~21.3ms ...


■ PIPELINE (theoretical):  TPOT ≈ 65.5 ms | 15.3 tok/s

  Time:  0────10───20───30───40───50───60───70
  IO G0: ████████████████████░░░░░░░░░░░░░░░░  14.1ms (cold)
  GPU 0: ░░░░░░░░░░░░░░░░░░░░████████░░░░░░░  7.2ms
  IO G1: ░░░░░░░░░░░░░░░░░░░░[███████]███████  14.1ms (7.2ms hidden)
  GPU 1: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████  7.2ms
  IO G2: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░[███████]███████
  GPU 2: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████
         ... step = max(14.1, 7.2) = 14.1ms ...

  Pipeline saves: 3 × 7.2ms overlap = 21.6ms
```

---

## 5. 完整配置矩陣

### 5.1 所有組合：Sequential Mode

| Layers/Group | Groups | Buffer (MB) | 1×NVMe tok/s | 2×NVMe tok/s | 3×NVMe tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 42 × 2 = 85 | 5.3 | 7.4 | 8.8 |
| 2 | 16 | 85 × 2 = 169 | 6.2 | 9.6 | 12.0 |
| **4** | **8** | **169 × 3 = 507** | **6.8** | **10.7** | **13.2** |
| 8 | 4 | 338 × 3 = 1014 | 7.2 | 11.3 | 14.1 |
| 16 | 2 | 676 × 2 = 1352 | 7.4 | 11.7 | 14.5 |
| 32 | 1 | 1353 × 1 = 1353 | 7.3 | 10.9 | 13.7 |

> **Buffer 說明：** 2-buffer 用於 ≤2 groups，3-buffer 用於 ≥3 groups（允許 prefetch lookahead）

### 5.2 所有組合：Pipeline Mode（理論）

| Layers/Group | Groups | 1×NVMe tok/s | 2×NVMe tok/s | 3×NVMe tok/s | 最佳 #NVMe |
|---:|---:|---:|---:|---:|---|
| 1 | 32 | 5.9 | 9.6 | 14.0 | IO-bound 全部 |
| 2 | 16 | 7.3 | 12.8 | 17.6 | IO→GPU transition @3 |
| **4** | **8** | **8.3** | **14.8** | **20.1** | **IO@2, ⚖ 平衡@3** |
| 8 | 4 | 9.8 | 15.3 | 19.5 | IO@2, near⚖@3 |
| 16 | 2 | 9.0 | 14.7 | 18.5 | IO all |
| 32 | 1 | 7.3 | 10.9 | 13.7 | 無 pipeline |

### 5.3 最佳配置推薦

```
                          Sequential          Pipeline
                          ──────────          ────────
🥇 Best 2×NVMe config:   8L/4-group          4L/8-group
                          11.3 tok/s          14.8 tok/s
                          buffer: 1014 MB     buffer: 507 MB

🥇 Best 3×NVMe config:   16L/2-group         4L/8-group
                          14.5 tok/s          20.1 tok/s (84%!)
                          buffer: 1352 MB     buffer: 507 MB

💡 最佳 cost/perf:        2×NVMe + 4L/8-grp pipeline
                          14.8 tok/s (62% baseline)
                          需修復 async IO overlap
```

---

## 6. IO:GPU 比率分析 — Pipeline 效率關鍵

Pipeline overlap 效率完全取決於 IO:GPU 比率：

```
IO:GPU Ratio    Overlap%    Description
─────────────   ─────────   ──────────────────────────────
   > 3:1          < 33%     IO 嚴重瓶頸，GPU 大量閒置
   2:1            50%       IO 是 2 倍慢，一半 IO 被遮蔽
   1.5:1          67%       
   1:1            100%      ⚖ 完美平衡！IO 完全被 GPU 遮蔽
   0.7:1          100%      GPU-bound，IO 完全隱藏（GPU 成為瓶頸）
```

### Per-Group IO:GPU 各配置比率

| Config | 1×NVMe | 2×NVMe | 3×NVMe |
|---|---:|---:|---:|
| 4L/8-group | 3.12:1 🔴 | **1.56:1** 🟡 | **1.04:1** 🟢 |
| 8L/4-group | 1.71:1 🟡 | **0.86:1** 🟢 | 0.57:1 🟢 |
| 16L/2-group | 1.39:1 🟡 | **0.70:1** 🟢 | 0.46:1 🟢 |
| 2L/16-group | 2.68:1 🔴 | 1.34:1 🟡 | 0.89:1 🟢 |

> 🔴 IO-bound (overlap < 50%) | 🟡 IO-leaning (50-80%) | 🟢 Balanced/GPU-bound (>80%)

**Key Insight：**
- **2×NVMe + 8L/4-group** → IO:GPU = 0.86:1 → **GPU-bound！IO 完全被遮蔽！**
- 但只有 4 個 groups → pipeline 機會少（只 3 步 overlap）
- **2×NVMe + 4L/8-group** → IO:GPU = 1.56:1 → IO 仍主導，但大幅改善
- **3×NVMe + 4L/8-group** → IO:GPU = 1.04:1 → **幾乎完美平衡**

---

## 7. 記憶體影響分析

Dense weight streaming 的目標是省下 GPU 記憶體，讓 8 GB 系統能跑。

| 配置 | Decoder weights in GPU | Buffer overhead | Net savings | Peak RSS est. |
|---|---:|---:|---:|---:|
| Baseline (no streaming) | 1352 MB | 0 | — | 6.66 GB |
| 4L/8-grp, 3-buf | 0 MB (offloaded) | 507 MB | **845 MB** | ~5.82 GB |
| 8L/4-grp, 3-buf | 0 MB | 1014 MB | **339 MB** | ~6.32 GB |
| 16L/2-grp, 2-buf | 0 MB | 1352 MB | **0 MB** | ~6.66 GB |

> **16L/2-group 的 buffer 大小 = decoder weights 大小 → 無記憶體節省！**
> 
> 對 8 GB 系統來說，**4L/8-group** 是唯一能同時節省記憶體和維持合理效能的配置。

---

## 8. Pipeline Overlap 未生效的根因分析

### 8.1 目前 IO 實作方式（不是 std::ifstream！）

> ⚠️ 之前的分析文件有誤 — 實際上 **不是** 使用 `std::ifstream.read()`。
> 目前主要路徑使用的是 **Win32 Direct I/O (`ReadFile` + `FILE_FLAG_NO_BUFFERING`)**。

#### 實際程式碼路徑（`dense_weight_streaming_manager.cpp`）

```
初始化時：
  CreateFileW(path,
      GENERIC_READ,
      FILE_SHARE_READ,
      FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)   ← 4 個 file handles
                                                              （每個 IO thread 一個）

讀取時（load_direct_io）：
  4 個 std::thread workers，每個 worker：
    ReadFile(hFile[thread_id],         ← Win32 同步阻塞式 ReadFile
             dest_ptr + offset,         ← 目標：USM host buffer（已 page-aligned）
             chunk_size,                ← 最大 256 MB/call
             &bytes_read,
             &overlapped)               ← OVERLAPPED 結構只用來指定 file offset
                                          （不是真正的 async I/O！）
```

#### 重點釐清

| 特性 | 目前實作 | 說明 |
|---|---|---|
| API | Win32 `ReadFile` | ✅ 繞過 C++ iostream 層 |
| `FILE_FLAG_NO_BUFFERING` | ✅ 有 | Direct I/O，繞過 OS page cache → 不佔額外記憶體 |
| `FILE_FLAG_SEQUENTIAL_SCAN` | ✅ 有 | 提示 OS 做 read-ahead 優化 |
| `FILE_FLAG_OVERLAPPED` | ❌ **沒有** | 沒有用 async I/O！ |
| 多線程 | ✅ 4 threads | 用 `std::thread` workers 做 parallelism |
| 頻寬 | 11.2 GB/s | 接近 NVMe Gen5 理論值 |

**`std::ifstream` 只出現在 Linux fallback 路徑（目前不會走到）：**
```cpp
#else  // Linux fallback
    std::ifstream file(path, std::ios::binary);
    file.seekg(file_offset);
    file.read(dest_ptr, size);  ← 純 fallback，Windows 不走這條
#endif
```

### 8.2 Pipeline 為什麼沒有 Overlap？

看 `execute_impl_streamed()` 中的實際執行流程（[network.cpp](network.cpp#L1273)）：

```
每個 group transition：

[主線程]
  ┌─────────────────────────────────────────────────────────┐
  │ (1) GPU fence: get_stream().flush()     ← 阻塞！等 GPU  │
  │ (2) IO fence:  wait_for_load(g)         ← 阻塞！等 IO   │
  │     └→ join_prefetch_thread()           ← 等背景線程 join│
  │ (3) swap_weight_pointers(g)             ← 阻塞          │
  │ (4) set_arguments()                     ← 阻塞          │
  │ (5) prefetch_next_group(g+1)            ← 啟動背景線程   │
  │ (6) GPU execute primitives              ← 提交到 GPU     │
  └─────────────────────────────────────────────────────────┘
```

**問題在步驟 (1) + (6) 的互動：**

`inst->execute()` 在 OpenVINO GPU plugin 中是 **非阻塞的** — 它只是
把 OpenCL kernel 提交（enqueue）到 GPU command queue，不等完成。
但由於 iGPU 與 CPU 共用系統記憶體頻寬（LPDDR5），一旦 GPU kernels
開始跑，`ReadFile` 的 DMA 傳輸速度會被 GPU 記憶體存取搶頻寬。

**根本原因拆解：**

```
理想 pipeline（如果 IO 和 GPU 真的能平行）：

  Time: ────────────────────────────────────────►
  IO:   ████████████████░░░░░░████████████████░░
  GPU:  ░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░████
       prefetch G1 ─────┘    prefetch G2 ─────┘

實際發生的事（序列化）：

  Time: ────────────────────────────────────────►
  IO:   ████████████████░░░░████████████████░░░░
  GPU:  ░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░████
                         ↑               ↑
                  flush() 會等 GPU 完成  │
                  然後 wait_for_load()   │
                  也要等 IO 完成         │
                  → 兩者加起來就是序列    │
```

**三個 serialization point：**

1. **GPU `flush()` 是 blocking sync** — `get_stream().flush()` 呼叫
   `clFinish(queue)`，等所有已提交的 GPU kernels 完成。在此期間
   主線程閒置，沒有發起新的 IO。

2. **Prefetch thread 與 GPU 競爭記憶體頻寬** — iGPU 使用 USM（Unified
   Shared Memory），GPU kernel 讀 weight 和 `ReadFile` DMA 到 USM buffer
   走同一條 LPDDR5 匯流排。即使 prefetch 啟動了，throughput 也會被
   GPU 搶走。在 8 GB 系統上，LPDDR5 頻寬 ~50 GB/s，GPU 計算 + IO
   同時跑最多只有 12.7 + GPU_BW，理論上可以 overlap，但...

3. **`ReadFile` 調用本身是同步阻塞的** — 雖然有 `OVERLAPPED` 結構，
   但沒有開 `FILE_FLAG_OVERLAPPED`，所以 `ReadFile` 仍然是同步的。
   每個 worker thread 在 `ReadFile` 返回前 blocked。join 線程也是 blocking。

### 8.3 為什麼用 Direct I/O 而不是 DirectStorage？

Phase 2 的 benchmark 測試結果（見 `20260508_dense_weight_streaming_plan.md §8`）：

| IO 方法 | 吞吐量 | 說明 |
|---|---:|---|
| Direct I/O (ReadFile, 4 threads) | **11.19 GB/s** | ✅ 更快 |
| DirectStorage (batched) | 8.50 GB/s | 較慢 |
| DirectStorage (single) | 5.20 GB/s | 最慢 |

對於 **順序大塊讀取**（每次 150-350 MB），`ReadFile` + `FILE_FLAG_NO_BUFFERING`
比 DirectStorage 更快，因為：
- `FILE_FLAG_SEQUENTIAL_SCAN` 讓 OS kernel 做 read-ahead（預讀優化）
- DirectStorage 的 BypassIO 優勢只在 **隨機小 IO**（MoE experts ~12 MB）時才明顯
- DirectStorage 有 queue management 開銷（`EnqueueRead` → GPU fence → dequeue）

**但 DirectStorage 有一個 Direct I/O 沒有的關鍵能力：真正的非同步 DMA。**

---

## 9. 方案 A：真正的 Async IO（推薦 — ✅ 已實作）

> **Implementation Status (2026-05-14):** 已完成實作。
> - `dense_weight_streaming_manager.hpp/cpp`: `FILE_FLAG_OVERLAPPED` + Event-based async `ReadFile`
> - 移除 `std::thread` prefetch，改用 `start_async_load()` / `wait_async_load()` / `is_async_load_complete()`
> - `network.cpp`: 更新 pipeline 註解，`prefetch_next_group()` 現在立即返回
> - 系統：8533MHz 32GB LPDDR5，iGPU 27GB USM，>100 GB/s 記憶體頻寬
> - NVMe DMA (~12 GB/s) + iGPU DMA (~50 GB/s) << LPDDR5 頻寬 → 無競爭

### 9.1 核心問題

目前 IO 的 11.2 GB/s 吞吐量已經很好，問題不是頻寬而是 **latency hiding**
（IO 延遲無法被 GPU 計算遮蔽）。

### 9.2 修復方案：Win32 Async ReadFile

**最小改動方案** — 不換 DirectStorage，只改 `ReadFile` 為真正的 async：

```cpp
// 現在：同步 ReadFile（FILE_FLAG_NO_BUFFERING 但沒有 FILE_FLAG_OVERLAPPED）
HANDLE hFile = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
    OPEN_EXISTING,
    FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,  // ← 同步
    nullptr);

// 改為：非同步 ReadFile（加上 FILE_FLAG_OVERLAPPED）
HANDLE hFile = CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
    OPEN_EXISTING,
    FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,        // ← 真正 async！
    nullptr);

// 發起非同步讀取
OVERLAPPED ov = {};
ov.hEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
ov.Offset = file_offset_low;
ov.OffsetHigh = file_offset_high;
ReadFile(hFile, dest_buffer, size, nullptr, &ov);
// ↑ 立即返回！不等完成

// ... 此時 GPU 繼續 execute() ...

// 等 IO 完成（只在需要時）
WaitForSingleObject(ov.hEvent, INFINITE);
CloseHandle(ov.hEvent);
```

### 9.3 實際修改內容（已完成）

| 檔案 | 修改 | 狀態 |
|---|---|:---:|
| `dense_weight_streaming_manager.hpp` | 移除 `m_prefetch_thread`，加 `m_io_events[]`、`m_overlapped_storage`、`start_async_load()`、`wait_async_load()`、`is_async_load_complete()` | ✅ |
| `dense_weight_streaming_manager.cpp` | `initialize_direct_io()`: `FILE_FLAG_OVERLAPPED` + 建立 Events | ✅ |
| `dense_weight_streaming_manager.cpp` | 新增 `start_async_load()`: 多 handle async ReadFile | ✅ |
| `dense_weight_streaming_manager.cpp` | 新增 `wait_async_load()`: `WaitForMultipleObjects` | ✅ |
| `dense_weight_streaming_manager.cpp` | `prefetch_next_group()`: 不再 spawn `std::thread`，直接 `start_async_load()` | ✅ |
| `dense_weight_streaming_manager.cpp` | 移除 `join_prefetch_thread()`，`shutdown_io()` 清理 Events | ✅ |
| `network.cpp` | 更新 pipeline 註解（async ReadFile 模式） | ✅ |

### 9.4 改良後的 Pipeline 流程

```
改良前（序列）：
  [flush GPU] → [wait IO] → [swap] → [set_args] → [prefetch] → [GPU exec] → [flush GPU] ...
  │←── stall ───→│←── stall ──→│                  │← async →│←── GPU ──→│

改良後（async overlap）：
  [flush GPU + wait IO]  → [swap] → [GPU exec ──────────────]
                                     ↕ overlap！
                            [async ReadFile(next) ──────────] → ...
  │←── max(flush,IO) ──→│         │← 真正的 overlap ────────→│
```

### 9.5 方案 B：DirectStorage Async Read（更徹底但更複雜）

```cpp
IDStorageFactory* factory;
DStorageGetFactory(IID_PPV_ARGS(&factory));

IDStorageFile* file;
factory->OpenFile(path, IID_PPV_ARGS(&file));

IDStorageQueue* queue;
DSTORAGE_QUEUE_DESC desc = {};
desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
desc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
factory->CreateQueue(&desc, IID_PPV_ARGS(&queue));

// 非同步讀取：kernel DMA，完全繞過 OS
DSTORAGE_REQUEST req = {};
req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
req.Source.File.Source = file;
req.Source.File.Offset = file_offset;
req.Source.File.Size = size;
req.Destination.Memory.Buffer = usm_buffer_ptr;
req.Destination.Memory.Size = size;
queue->EnqueueRequest(&req);

ID3D12Fence* fence;
queue->EnqueueSignal(fence, fence_value);
queue->Submit();
// ↑ 全部非同步！NVMe DMA 硬體直接搬

// GPU 繼續計算...

// 等 fence（只在需要時）
fence->SetEventOnCompletion(fence_value, event);
WaitForSingleObject(event, INFINITE);
```

- 優點：真正的硬體 DMA，零 CPU 介入
- 缺點：需要 D3D12 + DirectStorage SDK，與 OpenCL-based GPU plugin 混用有風險
- 已有 MoE OTD 參考實作（`moe_expert_weight_manager.hpp`）

---

## 10. 方案 C：OpenVINO AsyncInferRequest 模式（深入分析）

### 10.1 AsyncInferRequest 架構

從 OpenVINO 原始碼分析，AsyncInferRequest 的類別階層如下：

```
ov::IAsyncInferRequest (base class)
  └── ov::intel_gpu::AsyncInferRequest (GPU plugin implementation)
        └── wraps: ov::intel_gpu::SyncInferRequest

API 呼叫流程：
  user calls: infer_request.start_async()
    → AsyncInferRequest::start_async()
      → m_infer_request->setup_stream_graph()    // 設定 GPU context
      → m_infer_request->enqueue_notify()         // enqueue 到 GPU queue
      → Parent::start_async()                     // 啟動 pipeline 線程

  user calls: infer_request.wait()
    → m_infer_request->wait_notify()              // 等 GPU 完成
      → SyncInferRequest::wait()
        → network.get_stream().finish()           // clFinish(queue) ← blocking!
        → copy output tensors to host
```

### 10.2 GPU Plugin 中 Sync vs Async 的差異

```cpp
// SyncInferRequest::infer() — 同步模式
void SyncInferRequest::infer() {
    setup_stream_graph();
    std::lock_guard<std::mutex> lk(m_graph->get_mutex());
    enqueue();    // enqueue + execute_impl() → ALL primitives submitted to GPU
    wait();       // clFinish() + copy outputs
}

// AsyncInferRequest — 非同步模式
void AsyncInferRequest::start_async() {
    m_infer_request->setup_stream_graph();
    m_infer_request->enqueue_notify();    // enqueue ALL primitives to GPU
    Parent::start_async();                // start pipeline thread (callbacks)
}
// wait_notify() runs in a separate wait_executor thread
```

**關鍵發現：不論 sync 或 async，`enqueue()` 都是一次性把整個 network
的所有 primitives 提交到 GPU command queue。**

### 10.3 VLMPipeline 目前的呼叫方式

從 `lm_encoding.cpp` 看到：

```cpp
// Prompt phase（首次推理）：同步
m_llm.infer();  // blocking，等所有 token 處理完

// Generation phase（逐 token 生成）：半非同步
m_llm.start_async();   // 提交 GPU 推理
stream_generated_tokens();  // 在 GPU 跑的同時做 token streaming
free_non_running_requests();
m_llm.wait();           // 等 GPU 完成
```

Generation phase 用 `start_async()` + `wait()`，但這個 async 的目的
不是 IO/GPU overlap，而是讓 CPU 在 GPU 推理期間做 token streaming。

### 10.4 AsyncInferRequest 能否用於 Weight Streaming IO/GPU Overlap？

**答案：不適合。原因如下：**

#### 問題 1：AsyncInferRequest 是整個 network 一次提交

AsyncInferRequest 把整個 compiled network（所有 2549 primitives）
一次 enqueue 到 GPU。Weight streaming 需要在 **每個 group transition**
停下來做 IO + swap + re-bind。這兩個模式根本衝突。

```
AsyncInferRequest 執行模式：
  enqueue: [prim_0] [prim_1] ... [prim_2548]  ← 全部一次提交到 GPU queue
  wait:    等全部完成

Weight streaming 需要的模式：
  [HEAD prims] → STOP → [load G0] → [swap] → [G0 prims] → STOP → [load G1] → ...
                 ↑ 不可能！AsyncInferRequest 不支援中斷 execution ↑
```

#### 問題 2：不能對同一模型建立多個 InferRequest 做 pipeline

即使拆成多個 sub-models（每個 group 一個 compiled model），
每個 compiled model 需要**獨立的** `compile_model()` 呼叫。
這就回到了 Option B 的問題 — 每個 sub-model 的 `compile_model()`
需要 1-2 秒（USM 分配 + memcpy），完全不可行。

#### 問題 3：`set_arguments()` 不支援 partial rebind

即使能控制 execution granularity，
`set_arguments()` 仍然重綁所有 2549 primitives。
AsyncInferRequest 沒有 API 可以只 rebind 某幾個 primitive 的 arguments。

#### 問題 4：KV cache 狀態耦合

Gemma4 的 42 層 decoder 共用同一個 KV cache（透過 `VariableState`）。
如果拆成多個 compiled model，KV cache 的 cross-model sharing 需要
極複雜的 memory management。目前的 single-network 架構天然解決這個問題。

### 10.5 AsyncInferRequest 何時有用？

AsyncInferRequest **適合**的場景是 **inter-request pipelining**：

```
                        Request N            Request N+1
                        ─────────            ───────────
GPU:                    ████████████████     ████████████████
CPU (post-process):                    ██                      ██
                                        ↑ overlap ↑
                                 CPU 處理 token N 的同時，GPU 跑 token N+1
```

這就是 VLMPipeline **目前已經在做的事** — `start_async()` + `stream_generated_tokens()` + `wait()`。
它讓 CPU 的 token sampling/streaming 與下一步 GPU 推理 overlap。

但它 **不能解決** intra-request level 的 IO/GPU overlap（= weight streaming 的需求）。

### 10.6 結論

| 方案 | IO/GPU Overlap | 實作難度 | 推薦 |
|---|---|---|---|
| **A. Win32 Async ReadFile** | ✅ 可行 | 低（改 CreateFile flags + Event） | ✅ **推薦** |
| **B. DirectStorage** | ✅ 可行 | 中（DS SDK + D3D12 fence） | 🤔 備選 |
| **C. AsyncInferRequest** | ❌ 不可行 | — | ❌ 不適用 |

---

## 11. 2×NVMe 方案總結與建議

### 11.1 投資效益分析

| 方案 | 硬體成本 | 軟體工作量 | 效能提升 | 建議 |
|---|---|---|---|---|
| +1 NVMe (Sequential) | ~$100-200 | 無（只改 binary 路徑） | 6.78→10.7 (+58%) | ✅ 容易實施 |
| +1 NVMe (Pipeline) | ~$100-200 | 中等（修復 async IO） | 6.78→14.8 (+118%) | ✅ 推薦目標 |
| +2 NVMe (Pipeline) | ~$200-400 | 中等 | 6.78→20.1 (+196%) | 🤔 ROI 遞減 |

### 11.2 推薦行動

1. **短期（零成本）：** 修復 pipeline async IO overlap → 從 6.78→8.3 tok/s (+22%)
2. **中期（+1 NVMe）：** 雙 NVMe + pipeline → 14.8 tok/s (62% baseline)
3. **長期：** 如果 pipeline 可用，3×NVMe 可達 20.1 tok/s (84% baseline)

### 11.3 與 Per-Layer Embedding 的比較

| Feature | Per-Layer Embedding Offload | Dense Weight Streaming |
|---|---|---|
| 目標 | Offload embedding lookup (3.22 GB) | Offload decoder FC weights (1.35 GB) |
| 效能影響 | **0%** (24 tok/s = baseline) | **-72%** (6.78 tok/s) to **-16%** (20.1@3NVMe pipe) |
| 記憶體節省 | **2.8 GB** | 0.85 GB (4L/8-grp) |
| 實作複雜度 | 已完成，穩定 | 需持續優化 |
| **適用場景** | **所有系統** | **8 GB 系統 fallback** |

> **結論：** Per-layer embedding offload 是「免費」的最佳優化。
> Dense weight streaming 在極端記憶體限制下有用，但需要
> pipeline overlap 修復 + 額外 NVMe 才能達到可接受的效能。

---

## Appendix A: 計算驗證

### A.1 Sequential Mode 驗證

```python
# 4-layer/8-group, 1 NVMe (calibration point)
IO    = 1352.5 / 12.67       # = 106.7 ms ✓ (measured)
GPU   = 38.7                  # measured
Over  = 1.6                   # measured
TPOT  = IO + GPU + Over       # = 147.0 ms ✓ (measured)

# 4-layer/8-group, 2 NVMe
IO_2  = 1352.5 / (2 * 12.67)  # = 53.4 ms
TPOT_2 = IO_2 + 38.7 + 1.6    # = 93.7 ms → 10.67 tok/s

# 4-layer/8-group, 3 NVMe
IO_3  = 1352.5 / (3 * 12.67)  # = 35.6 ms
TPOT_3 = IO_3 + 38.7 + 1.6    # = 75.9 ms → 13.18 tok/s
```

### A.2 Pipeline Mode 驗證

```python
# 4-layer/8-group, 2 NVMe, Pipeline
HEAD  = 5.0   # ms
TAIL  = 5.0   # ms
IO_g  = 53.4 / 8    # = 6.68 ms per group
GPU_g = (38.7 - 10) / 8  # = 3.59 ms per group (excluding HEAD/TAIL)
Sa_g  = 1.6 / 8     # = 0.20 ms swap+args per group

cold  = IO_g + GPU_g + Sa_g        # = 10.47 ms (Group 0, no overlap)
step  = max(IO_g, GPU_g + Sa_g)    # = max(6.68, 3.79) = 6.68 ms
pipe  = HEAD + cold + 7 * step + TAIL
      # = 5 + 10.47 + 46.76 + 5
      # = 67.2 ms → 14.88 tok/s ✓
```

### A.3 IO:GPU Balance Point

```python
# 求 N_nvme 使 IO_per_group = GPU_per_group
# IO_g = (1352.5 / 8) / (N * 12.67) = 169 / (N * 12.67)
# GPU_g + Sa = 3.79 ms

# N = 169 / (3.79 * 12.67) = 169 / 48.02 = 3.52
# → 需要 ~3.5 個 NVMe 才能完美平衡！
# → 3 NVMe 時 IO:GPU = 1.04:1 → 接近平衡 ✓
```
