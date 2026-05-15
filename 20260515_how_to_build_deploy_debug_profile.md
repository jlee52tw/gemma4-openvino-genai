# How to Build, Deploy, Debug & Profile — Dense Weight Streaming

> **Date:** 2026-05-15  
> **Project:** Gemma-4-E4B-it OpenVINO Dense Weight Streaming  
> **Author:** Auto-generated from profiling session

---

## 目錄

1. [Build（編譯）](#1-build編譯)
2. [Deploy（部署）](#2-deploy部署)
3. [Debug（除錯）](#3-debug除錯)
4. [Profile（效能分析）](#4-profile效能分析)
5. [v1 vs v2 效能比較](#5-v1-vs-v2-效能比較)
6. [環境變數參考](#6-環境變數參考)
7. [二進位備份](#7-二進位備份)
8. [已知問題與注意事項](#8-已知問題與注意事項)

---

## 1. Build（編譯）

### 1.1 原始碼位置

| 檔案 | 路徑 | 說明 |
|------|------|------|
| 管理器標頭 | `cpp/dense_weight_streaming_manager.hpp` | Config struct、API 定義 |
| 管理器實作 | `cpp/dense_weight_streaming_manager.cpp` | 初始化、IO、雙緩衝、計時器 |
| 管理器標頭（build tree） | `openvino\build\src\plugins\intel_gpu\include\intel_gpu\plugin\dense_weight_streaming_manager.hpp` | ← 同步自 workspace |
| 管理器實作（build tree） | `openvino\build\src\plugins\intel_gpu\src\plugin\dense_weight_streaming_manager.cpp` | ← 同步自 workspace |
| Network 執行路徑 | `openvino\src\plugins\intel_gpu\src\graph\network.cpp` | `execute_impl_streamed()` — 真正的 streaming execution loop |

> **重要：** 實際的 streaming decode 迴圈在 `network.cpp` 的 `execute_impl_streamed()` 中，而非 manager 的 `execute_streamed_decode_token()`。修改管理器 API 後，**必須確認 network.cpp 也被重新編譯**。

### 1.2 同步原始碼到 Build Tree

```powershell
# 同步 .hpp
Copy-Item "cpp\dense_weight_streaming_manager.hpp" `
  "C:\working\gemma4-openvino\openvino\build\src\plugins\intel_gpu\include\intel_gpu\plugin\dense_weight_streaming_manager.hpp"

# 同步 .cpp
Copy-Item "cpp\dense_weight_streaming_manager.cpp" `
  "C:\working\gemma4-openvino\openvino\build\src\plugins\intel_gpu\src\plugin\dense_weight_streaming_manager.cpp"
```

### 1.3 編譯 GPU Plugin DLL

```powershell
cmake --build C:\working\gemma4-openvino\openvino\build `
  --target openvino_intel_gpu_plugin `
  --config Release `
  -- /p:CL_MPCount=4
```

- **輸出路徑：** `C:\working\gemma4-openvino\openvino\bin\intel64\Release\openvino_intel_gpu_plugin.dll`
- **大小：** ~34.25 MB
- **編譯時間：** ~1-3 分鐘（增量編譯）

### 1.4 編譯 Benchmark 工具

```powershell
cd cpp
cmake --build build --config Release
```

輸出：
- `cpp\build\Release\benchmark_ds.exe` — Dense streaming benchmark
- `cpp\build\Release\benchmark_pipeline.exe` — Pipeline benchmark

### 1.5 強制重新編譯 network.cpp

當修改 `DenseStreamingConfig` struct 或 manager API 時，cmake 可能無法偵測到 header dependency（因 build tree copy）。以下方法可強制重新編譯：

```powershell
# 方法 1：touch network.cpp
(Get-Item "C:\working\gemma4-openvino\openvino\src\plugins\intel_gpu\src\graph\network.cpp").LastWriteTime = Get-Date

# 方法 2：在 network.cpp 加一行無害的 comment，再移除
# 然後重新 cmake --build

# 方法 3：清除特定 obj
Remove-Item "C:\working\gemma4-openvino\openvino\build\src\plugins\intel_gpu\CMakeFiles\openvino_intel_gpu_plugin.dir\src\graph\network.cpp.obj" -ErrorAction SilentlyContinue
```

> ⚠️ **ABI 陷阱：** 如果在 struct 中間新增欄位（而非末尾），network.cpp 編譯時看到的 struct layout 會與 manager.cpp 不一致，導致 stack corruption 和錯誤輸出（例如 `"H5+T42 (stream 4112372310 middle layers)"`）。**新欄位永遠加在 struct 末尾。**

---

## 2. Deploy（部署）

### 2.1 部署 DLL 到 Python 環境

```powershell
Copy-Item "C:\working\gemma4-openvino\openvino\bin\intel64\Release\openvino_intel_gpu_plugin.dll" `
  "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" -Force
```

### 2.2 驗證部署

```powershell
# 確認 DLL 時間戳
Get-Item "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" |
  Select-Object Name, Length, LastWriteTime

# 確認 OpenVINO 版本
python -c "import openvino; print(openvino.__version__)"
# 預期: 2026.2.0.dev20260411
```

### 2.3 部署 Streaming 權重檔

權重檔需預先產生：

```powershell
python pack_dense_weights.py `
  --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
  --output "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.bin"
```

---

## 3. Debug（除錯）

### 3.1 啟用 Debug 輸出

設定環境變數：

```powershell
$env:OV_DENSE_STREAM_DEBUG = "1"
```

啟用後，每個 token 的計時明細會輸出到 stderr/console：

```
[DenseStreaming] group_transition g=0 finish_ms=0.0001 io_wait_ms=14.94 swap_ms=0.23 args_ms=0.09 ...
```

### 3.2 將 Debug 輸出重導至檔案

**方法 A：設定 Log File 環境變數**（推薦）

```powershell
$env:OV_DENSE_STREAM_LOG_FILE = "C:\path\to\debug.log"
$env:OV_DENSE_STREAM_DEBUG = "1"
```

管理器初始化時會自動開啟 log file，所有 `[DenseStreaming]` 訊息寫入指定檔案。

**方法 B：PowerShell stderr 重導**

```powershell
$env:OV_DENSE_STREAM_DEBUG = "1"
python benchmark.py 2>&1 | Tee-Object -FilePath "debug_output.log"
```

### 3.3 Debug Log 格式說明

Log 開頭包含啟動資訊：

```
=== Dense Weight Streaming Debug Log ===
Timestamp: Fri May 15 14:30:24 2026
```

每次 streamed token 產生：`[TRACE] execute_impl entry #N` 記錄（前 20 次）

每個 group transition 記錄以下計時：

| 欄位 | 說明 |
|------|------|
| `finish_ms` | GPU fence wait 時間 |
| `io_wait_ms` | NVMe async IO 等待時間 |
| `swap_ms` | swap_weight_pointers 時間 |
| `args_ms` | set_arguments 時間 |

### 3.4 Token 級統計表

Log 結尾包含 per-token 統計表和平均值：

```
 token   total_ms   load_ms   swap_ms   args_ms  flush_ms    gpu_ms  groups
     3    142.81    79.98     1.26     0.58     0.00    60.99      8
   ...
------------------------------------------------------------
   AVG    142.81    79.98     1.26     0.58     0.00    60.99
```

---

## 4. Profile（效能分析）

### 4.1 v1 Sequential（無 Prefetch 重疊）

v1 模式停用 IO/GPU 重疊，每個 group 為：load → wait → swap → compute（全串列）。

```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
$env:OV_DENSE_STREAM_LOG_FILE = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_streaming_debug.log"
$env:OV_DENSE_STREAM_NO_PREFETCH = "1"   # ← 關鍵：停用 prefetch

python run_gemma4.py --prompt "Hello"
```

執行後取回 log：

```powershell
Copy-Item "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_streaming_debug.log" `
  "profiles\v1_sequential_debug.log"
```

### 4.2 v2 Pipeline（Prefetch 雙緩衝重疊）

v2 模式啟用 IO/GPU 重疊：prefetch(N+1) 與 GPU compute(N) 並行。

```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.bin"
$env:OV_DENSE_STREAM_DEBUG = "1"
$env:OV_DENSE_STREAM_LOG_FILE = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_streaming_debug.log"
Remove-Item Env:OV_DENSE_STREAM_NO_PREFETCH -ErrorAction SilentlyContinue  # ← 確保 prefetch 啟用

python run_gemma4.py --prompt "Hello"
```

取回 log：

```powershell
Copy-Item "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_streaming_debug.log" `
  "profiles\v2_pipeline_debug.log"
```

### 4.3 快速重新執行 Profile

```powershell
# 一鍵 v1 profile
$env:OV_DENSE_STREAM_NO_PREFETCH="1"; $env:OV_DENSE_STREAM_DEBUG="1"
python run_gemma4.py --prompt "Hello" 2>&1 | Out-Null
Copy-Item "$env:OV_DENSE_STREAM_LOG_FILE" "profiles\v1_sequential_debug.log"

# 一鍵 v2 profile
Remove-Item Env:OV_DENSE_STREAM_NO_PREFETCH -ErrorAction SilentlyContinue
python run_gemma4.py --prompt "Hello" 2>&1 | Out-Null
Copy-Item "$env:OV_DENSE_STREAM_LOG_FILE" "profiles\v2_pipeline_debug.log"
```

---

## 5. v1 vs v2 效能比較

### 5.1 測試條件

| 項目 | 值 |
|------|------|
| 模型 | gemma-4-E4B-it-ov (INT4, 42 decoder layers) |
| 硬體 | Intel Panther Lake, 12 Xe EUs iGPU, 16 GB LPDDR5 |
| NVMe | Gen5×4, ~12 GB/s sequential read |
| Streaming 層數 | 32 layers (pin head=5, pin tail=5) |
| Groups | 8 |
| Prompt | "Hello"（short-text，最小 prefill） |
| Decode tokens | ~253 |

### 5.2 結果對比

| 指標 | v1 Sequential | v2 Pipeline | 改善 |
|------|---:|---:|---:|
| **Avg TPOT (internal)** | **177.48 ms** | **142.81 ms** | **-19.5%** |
| **Throughput** | **5.63 tok/s** | **7.00 tok/s** | **+24.3%** |
| NVMe load 時間 (avg) | 142.99 ms | 79.98 ms | -44.1% |
| NVMe load 占比 | 80.6% | 56.0% | |
| Swap pointers (avg) | 1.10 ms | 1.26 ms | — |
| set_arguments (avg) | 0.51 ms | 0.58 ms | — |
| GPU compute (avg) | 32.88 ms | 60.99 ms | +85.4%* |
| GPU compute 占比 | 18.5% | 42.7% | |
| GPU fence (avg) | ~0 ms | ~0 ms | — |

> \* GPU compute 的絕對時間上升是因為 v2 的 IO 與 GPU 重疊，GPU 在等 IO 時也在 compute 其它 group，計時不完全可比。

### 5.3 Pipeline 效果分析

```
v1 Sequential (NO_PREFETCH=1):
  ┌─────────────────────────────┐     ┌──────┐     ┌─────────────────────────────┐     ┌──────┐
  │     NVMe load (143 ms)      │     │GPU   │     │     NVMe load (143 ms)      │     │GPU   │
  │     group N                 │ ... │(33ms)│ ... │     group N+1               │ ... │(33ms)│
  └─────────────────────────────┘     └──────┘     └─────────────────────────────┘     └──────┘
  Total per token: 143 + 33 ≈ 177 ms (all serial)

v2 Pipeline (prefetch enabled):
  ┌─────────────────────────────┐     ┌──────┐
  │     NVMe load group N       │     │GPU N │
  └─────────────────────────────┘     └──────┘
       ┌─────────────────────────────┐     ┌──────┐
       │     NVMe load group N+1     │     │GPU   │
       └─────────────────────────────┘     │ N+1  │
                                           └──────┘
  IO(N+1) overlaps with GPU(N) → effective TPOT ≈ max(IO, GPU) + overhead ≈ 143 ms
```

### 5.4 關鍵觀察

1. **Pipeline 有效：** v2 的 NVMe load 等待時間從 143ms 降至 80ms（-44%），證明 IO 與 GPU overlap 成功
2. **GPU utilization 提升：** 從 18.5% → 42.7%，GPU 不再空等 NVMe IO
3. **瓶頸仍在 NVMe IO：** 即使 overlap，NVMe load 仍占 56% 的 TPOT
4. **改善空間：** 增加 IO queue depth、減少 group 數量（更大 group 更好 overlap）、或使用更快 NVMe

---

## 6. 環境變數參考

| 環境變數 | 預設值 | 說明 |
|----------|--------|------|
| `OV_DENSE_STREAM_WEIGHTS` | *(無)* | Streaming 權重二進位檔路徑。設定此變數才會啟用 streaming |
| `OV_DENSE_STREAM_DEBUG` | `0` | 設 `1` 啟用詳細計時 log |
| `OV_DENSE_STREAM_LOG_FILE` | *(無)* | Debug log 輸出檔路徑（不設則輸出到 stderr） |
| `OV_DENSE_STREAM_NO_PREFETCH` | `0` | 設 `1` 停用 IO/GPU 重疊（v1 sequential 模式） |
| `OV_DENSE_STREAM_NUM_BUFFERS` | `2` | IO buffer 數量（clamp: 2-4） |
| `OV_DENSE_STREAM_IO_THREADS` | `4` | Async IO handles 數量 |
| `OV_DENSE_STREAM_PIN_HEAD` | `5` | 常駐 GPU 的前 N 層（不做 streaming） |
| `OV_DENSE_STREAM_PIN_TAIL` | `5` | 常駐 GPU 的後 N 層 |
| `OV_DENSE_STREAM_TOTAL_LAYERS` | `42` | Decoder 總層數 |
| `OV_DENSE_STREAM_MAX_BUFFER_GB` | *(auto)* | 單一 buffer 最大 GB |
| `OV_DENSE_STREAM_FILE` | *(auto)* | 權重檔路徑（alias） |
| `OV_DENSE_STREAM_LOCK_MEMORY` | `0` | 鎖定 buffer 記憶體（VirtualLock） |
| `OV_DENSE_STREAM_TIMING` | `0` | 額外計時資訊 |

---

## 7. 二進位備份

### 7.1 備份結構

```
binaries/
├── v1_sequential/                    # Sequential mode (NO_PREFETCH=1)
│   ├── openvino_intel_gpu_plugin.dll  (34.25 MB)
│   ├── benchmark_ds.exe               (0.19 MB)
│   └── benchmark_pipeline.exe         (0.31 MB)
└── v2_pipeline/                      # Pipeline mode (prefetch enabled)
    ├── openvino_intel_gpu_plugin.dll  (34.25 MB)
    ├── benchmark_ds.exe               (0.19 MB)
    └── benchmark_pipeline.exe         (0.31 MB)
```

> **注意：** v1 和 v2 使用**相同的 DLL**。差異在於 `OV_DENSE_STREAM_NO_PREFETCH` 環境變數。備份的目的是保留已知可運作的二進位版本。

### 7.2 回滾指令

```powershell
# 回滾到 v1 (sequential)
Copy-Item "binaries\v1_sequential\openvino_intel_gpu_plugin.dll" `
  "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" -Force

# 回滾到 v2 (pipeline)
Copy-Item "binaries\v2_pipeline\openvino_intel_gpu_plugin.dll" `
  "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs\openvino_intel_gpu_plugin.dll" -Force
```

### 7.3 Profile Logs

```
profiles/
├── v1_sequential_debug.log   (1154.8 KB, 12980 lines)
└── v2_pipeline_debug.log     (1002.8 KB, 11202 lines)
```

---

## 8. 已知問題與注意事項

### 8.1 Struct ABI 陷阱

在 `DenseStreamingConfig` struct 中間新增欄位會導致 network.cpp 和 manager.cpp 之間的 struct layout 不一致。症狀：

- 奇怪的初始化訊息（如 `H5+T42 (stream 4112372310 middle layers)`）
- TPOT 異常低（如 40ms，表示 streaming 根本沒啟用）
- Stack corruption、UB

**解決方案：** 新欄位永遠加在 struct 末尾，且修改 header 後強制重建 network.cpp。

### 8.2 num_buffers Clamp

`read_from_env()` 中 `num_buffers` 被 clamp 到 `[2, 4]` 範圍。設 `NUM_BUFFERS=1` 無效。要測試 sequential 模式，使用 `OV_DENSE_STREAM_NO_PREFETCH=1`。

### 8.3 Warm-up Tokens

前 2 個 decode token 走 warm-up 路徑（不做 streaming），第 3 個 token 開始才進入 `execute_impl_streamed()`。Profile 數據從 token #3 開始。

### 8.4 TRACE 訊息

前 20 次 `execute_impl` 呼叫會印出 `[TRACE] execute_impl entry #N` 到 console/log，這是 network.cpp 的診斷訊息，不在 hot path 上。

### 8.5 第一次編譯成本

首次 `compile_model()` 會產生 GPU kernel blob cache（~12s）。之後從 blob cache 載入（~2.3s）。Streaming 是在 blob cache 載入完成後才啟動的。

---

## 附錄：完整 Profile 指令碼

```powershell
# === 完整 v1 + v2 Profile 流程 ===

$MODEL_DIR = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov"
$WEIGHTS   = "$MODEL_DIR\dense_weights_streaming.bin"
$LOG_FILE  = "$MODEL_DIR\dense_streaming_debug.log"

# 共通設定
$env:OV_DENSE_STREAM_WEIGHTS  = $WEIGHTS
$env:OV_DENSE_STREAM_DEBUG    = "1"
$env:OV_DENSE_STREAM_LOG_FILE = $LOG_FILE

# --- v1 Sequential ---
$env:OV_DENSE_STREAM_NO_PREFETCH = "1"
python run_gemma4.py --prompt "Hello"
Copy-Item $LOG_FILE "profiles\v1_sequential_debug.log"

# --- v2 Pipeline ---
Remove-Item Env:OV_DENSE_STREAM_NO_PREFETCH -ErrorAction SilentlyContinue
python run_gemma4.py --prompt "Hello"
Copy-Item $LOG_FILE "profiles\v2_pipeline_debug.log"

Write-Host "=== Profile 完成 ==="
Write-Host "v1 log: profiles\v1_sequential_debug.log"
Write-Host "v2 log: profiles\v2_pipeline_debug.log"
```
