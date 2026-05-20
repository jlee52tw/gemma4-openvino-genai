# 2026-05-20 — Dual-NVMe 系統交接與下一步計畫

**Date:** 2026-05-20  
**Author:** jlee52tw  
**Status:** 準備移轉到另一台具備 2 顆 NVMe 的系統，進行實體 Dual-NVMe 驗證與後續開發  
**Current Git HEAD:** `9ddd537` (`main`, 已推送到 `origin/main`)  

---

## 0. 本文件目的

此文件是換到下一台系統前的交接摘要，包含：

1. 歷史脈絡與目前完成狀態
2. 目前關鍵架構設計
3. Release v2 套件內容與限制
4. 到新系統後的下載、安裝、執行方式
5. 2 顆 NVMe 實測計畫
6. 下一步 TODO / 風險 / 注意事項
7. 相關 Markdown 文件索引

> **重要：** GitHub repo 只包含程式碼、腳本與文件。  
> `release_v2/` 與 `gemma4_streaming_release_v2.tar` 因為 13.59 GB，已被 `.gitignore` 排除，**不會 push 到 GitHub**。  
> 若要在新系統直接跑完整模型，需要另外搬移 `gemma4_streaming_release_v2.tar` 或模型目錄。

---

## 1. Git / Workspace 狀態

### 1.1 目前最近提交

```text
9ddd537 Add auto CACHE_DIR + per-layer embedding docs to run_gemma4.py and README
c3e1e72 Add release packaging (v2) with dual-NVMe support
eeabcd8 Implement dual-NVMe parallel IO for 2x bandwidth
0cb4bcc Add build/deploy/debug/profile doc and no_prefetch support for v1/v2 comparison
5d297c8 Implement async ReadFile (FILE_FLAG_OVERLAPPED) for true GPU/IO overlap
eb7d5c1 Deep analysis: why pipeline overlap fails, AsyncInferRequest infeasibility, Win32 async ReadFile proposal
c4225b5 Add streaming files to release package, dual NVMe analysis doc
86ebe77 Release package: rename ple bin, build C++ exe, add package script
```

### 1.2 目前已推送狀態

- Branch: `main`
- Remote: `origin/main`
- 最新已推送 commit: `9ddd537`
- 工作區在建立本文件前是 clean；本文件會 commit + push 後再移轉。

---

## 2. 歷史脈絡摘要

### 2.1 Phase 0 / mmap 起點

參考：`2026-04-17-mmap-investigation.md`

早期主要問題是模型太大，必須研究 mmap / lazy loading / cache 對 RSS 的影響。
後續確認單純依賴 OpenVINO blob cache 或 mmap 不足以在 8 GB 系統上穩定容納所有權重 + KV cache + runtime overhead。

### 2.2 Dense Weight Streaming 初始計畫

參考：`20260508_dense_weight_streaming_plan.md`

核心問題：

- Decoder dense weights 約 2.09 GB 必須常駐 GPU USM
- 8 GB 系統上記憶體不足
- 嘗試讓部分 decoder weights 留在 NVMe，token decode 時串流載入

早期 Option B（每 token 重新 `compile_model()`）已被實測否定：

```text
2-way split blob cache load:
  Part 0: ~1.03 s
  Part 1: ~1.22 s
  合計: ~2.30 s/token
  → 約 0.43 tok/s，不可行
```

結論：必須走 Option A：模型編譯一次，runtime 直接替換 GPU weight USM buffer 指標。

### 2.3 Phase 2：FC Weight Streaming 成功

參考：`20260511_phase2_weight_streaming.md`

完成內容：

- `DenseWeightStreamingManager`
- Runtime 以 group 為單位讀取 NVMe weight binary
- 使用 `create_subbuffer()` 替換 FC weight memory
- `set_arguments()` guard 問題已修復
- FC-scan mapping 成功：從 compiled network 的 fully-connected dependencies 反推 weight mapping
- 只 stream FC weights；scale / zero-point 保持 pinned，避免 GPU reorder 格式錯誤

正確性驗證：

```text
Prompt: What is the capital of Japan?
Output: Tokyo ✅
```

### 2.4 Per-layer Embedding Offload

參考：`20260512_per_layer_embedding_offload.md`

Gemma4 的 `openvino_text_embeddings_per_layer_model_revised.bin` 約 3.07 GB。
它不是 decoder dense weights，而是 per-layer embedding lookup table。

目前做法：

- 保留此檔在 model dir
- VLMPipeline 自動使用 mmap reader
- 每 token 只讀約 10 KB
- 可節省約 2.82 GB GPU/system memory

最新 `run_gemma4.py` 已自動偵測並印出：

```text
Per-layer embedding offload: enabled (mmap, 3.00 GB)
```

### 2.5 Pipeline Overlap / Async ReadFile

參考：

- `20260514_pipeline_overlap_fix.md`
- `20260515_how_to_build_deploy_debug_profile.md`

已完成：

- Win32 `ReadFile` + `FILE_FLAG_OVERLAPPED`
- Async I/O handles
- Prefetch next group
- `OV_DENSE_STREAM_NO_PREFETCH` 可切換 v1 sequential vs v2 pipeline

實測：

| 模式 | TPOT | tok/s | 說明 |
|------|------:|------:|------|
| v1 Sequential | 177.48 ms | 5.63 | 無 prefetch |
| v2 Pipeline (single NVMe) | 142.81 ms | 7.00 | async prefetch 成功 |

### 2.6 Dual-NVMe Parallel IO

參考：`20260513_dual_nvme_streaming_analysis.md` 與最新 commit `eeabcd8`

完成內容：

- `pack_dense_weights_dual.py`
- File format V2
- `OV_DENSE_STREAM_WEIGHTS_2`
- `read_tables_file2()`
- `start_async_load_dual()`
- 4 個 async handles，平均分配：2 per NVMe
- Group-Half Striping：每個 group 在 sector-aligned midpoint 切成兩半

同碟 dual-path 實測結果：

| 模式 | Avg TPOT | tok/s | Load% | GPU% |
|------|----------:|------:|------:|-----:|
| Single NVMe v2 pipeline | 142.81 ms | 7.00 | 56.0% | 42.7% |
| Dual-path same disk | 128.56 ms | 7.78 | 48.2% | 50.6% |
| Physical Dual NVMe estimated | ~77 ms | ~13.0 | — | — |

同碟測試正確性：

```text
Test1: Japan capital → Tokyo ✅
Test2: 7 * 8 → 56 ✅
Test3: French translation → 合理輸出 ✅
```

---

## 3. 關鍵架構設計

### 3.1 Dense Weight Streaming Pipeline

每個 decode token：

```text
Pinned HEAD layers 0-4 run on GPU

For each streamed group G (8 groups total, 4 layers/group):
  1. Wait async NVMe load complete
  2. Swap FC weight pointers to current USM buffer
  3. Force `_reset_arguments = true`
  4. Rebind GPU kernel arguments
  5. Prefetch next group into alternate buffer
  6. Execute group G GPU kernels

Pinned TAIL layers 37-41 run on GPU
```

### 3.2 Streamed Layer Range

```text
Pinned head: layers 0-4
Streamed:    layers 5-36  (32 layers)
Pinned tail: layers 37-41
Group size:  4 layers/group
Groups:      8 groups
```

### 3.3 FC-scan Mapping

不要依賴 JSON tensor name 與 compiled primitive name 完全一致。
目前可靠做法：

1. 掃描 compiled graph 的 fully-connected primitives
2. 取 FC dependency：
   - `dep[0]`: activation
   - `dep[1]`: FC weight ← 只 swap 這個
   - `dep[2]`: scale ← pinned
   - `dep[3]`: zero point ← pinned
3. 用 dependency name 對 JSON offset table 查 offset / size

結果：

```text
FC primitives found: 224
Weights mapped: 205
Total mapped FC weights: 1352.5 MB
```

### 3.4 Group-Half Striping

雙 NVMe 模式不是複製整份 weight，而是每個 group 切半：

```text
Group G logical data (~190-198 MB):
  First half  → dense_weights_streaming_0.bin  (NVMe 0)
  Second half → dense_weights_streaming_1.bin  (NVMe 1)

Runtime:
  read stripe0 → USM buffer offset 0
  read stripe1 → USM buffer offset stripe0_size
  GPU sees one contiguous group buffer
```

### 3.5 File Size 說明

| 檔案 | 大小 | 說明 |
|------|------:|------|
| `openvino_language_model.bin` | ~2.69 GB | 完整 decoder IR weights，包含所有 42 層 + scale/zp/small constants |
| `openvino_text_embeddings_per_layer_model_revised.bin` | ~3.07 GB | Per-layer embedding lookup table，透過 mmap offload |
| `dense_weights_streaming_0.bin` | ~776 MB | Streamed FC weights 的第一半 |
| `dense_weights_streaming_1.bin` | ~776 MB | Streamed FC weights 的第二半 |
| 兩個 streaming stripe 合計 | ~1.55 GB | 只含 layers 5-36 的 FC weights，不含 scale/zp/small constants |

---

## 4. Release v2 套件狀態

### 4.1 本機 release 產物

```text
release_v2/                              13.59 GB
  dlls/                                  61 MB
  model/                                 12.02 GB
  streaming_nvme0/                       0.76 GB
  streaming_nvme1/                       0.76 GB

gemma4_streaming_release_v2.tar          13.59 GB
```

### 4.2 重要限制

`release_v2/` 與 `gemma4_streaming_release_v2.tar` 已被 `.gitignore` 排除，不會在 GitHub repo 裡。

若到下一台系統只做：

```powershell
git clone https://github.com/jlee52tw/gemma4-openvino-genai.git
```

只會取得程式碼、腳本與文件；**不會取得 13.59 GB 模型與 DLL release 套件**。

請另外用以下方式搬移：

- USB / external SSD
- NAS / SMB share
- `scp` / `robocopy`
- 手動建立 GitHub Release asset（若網路與 quota 允許）
- 重新在新系統上準備模型與 streaming binaries

---

## 5. 新系統 Quick Start（2 顆 NVMe）

假設：

- Repo clone 到：`C:\working\gemma4-openvino\gemma4-openvino-genai`
- Release tar 已搬到新系統
- NVMe 0: `C:\`
- NVMe 1: `D:\`

### 5.1 Clone repo

```powershell
mkdir C:\working\gemma4-openvino
cd C:\working\gemma4-openvino
git clone https://github.com/jlee52tw/gemma4-openvino-genai.git
cd gemma4-openvino-genai
```

### 5.2 解開 release tar

```powershell
# 假設 tar 位於 C:\working\gemma4-openvino\gemma4_streaming_release_v2.tar
mkdir C:\working\gemma4-openvino\gemma4_release_v2
tar -xf C:\working\gemma4-openvino\gemma4_streaming_release_v2.tar `
    -C C:\working\gemma4-openvino\gemma4_release_v2
```

### 5.3 安裝 Python dependencies

```powershell
pip install openvino==2026.2.0 openvino-genai openvino-tokenizers
pip install -r C:\working\gemma4-openvino\gemma4_release_v2\requirements.txt
```

### 5.4 安裝 modified GPU plugin DLL

```powershell
$ovLibs = python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))"
Copy-Item "$ovLibs\openvino_intel_gpu_plugin.dll" "$ovLibs\openvino_intel_gpu_plugin.dll.bak" -Force
Copy-Item "C:\working\gemma4-openvino\gemma4_release_v2\dlls\openvino_intel_gpu_plugin.dll" `
    "$ovLibs\openvino_intel_gpu_plugin.dll" -Force
```

### 5.5 放置 2 顆 NVMe 的 streaming stripe

建議 layout：

```text
C:\working\gemma4-openvino\gemma4_release_v2\model\
  ├─ openvino_language_model.xml/bin
  ├─ openvino_text_embeddings_per_layer_model_revised.bin
  ├─ model_cache\
  └─ dense_weights_streaming_0.json  ← metadata 放 model dir 或跟 stripe0 同目錄

C:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin
D:\gemma4_streaming_nvme1\dense_weights_streaming_1.bin
```

Copy：

```powershell
mkdir C:\gemma4_streaming_nvme0 -Force
mkdir D:\gemma4_streaming_nvme1 -Force

Copy-Item C:\working\gemma4-openvino\gemma4_release_v2\streaming_nvme0\dense_weights_streaming_0.bin `
    C:\gemma4_streaming_nvme0\ -Force
Copy-Item C:\working\gemma4-openvino\gemma4_release_v2\streaming_nvme0\dense_weights_streaming_0.json `
    C:\gemma4_streaming_nvme0\ -Force
Copy-Item C:\working\gemma4-openvino\gemma4_release_v2\streaming_nvme1\dense_weights_streaming_1.bin `
    D:\gemma4_streaming_nvme1\ -Force
```

### 5.6 執行 dual-NVMe 推理

```powershell
cd C:\working\gemma4-openvino\gemma4_release_v2

$env:OV_DENSE_STREAM_WEIGHTS = "C:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2 = "D:\gemma4_streaming_nvme1\dense_weights_streaming_1.bin"

python run_gemma4.py `
  --model-dir "C:\working\gemma4-openvino\gemma4_release_v2\model" `
  --prompt "What is the capital of Japan? Answer in one word." `
  --max-new-tokens 20
```

預期輸出：

```text
Using model cache: ...\model_cache
Per-layer embedding offload: enabled (mmap, 3.00 GB)
[DenseStreaming] NVMe mode: Dual (2x bandwidth)
Response: Tokyo
```

### 5.7 若要先做同碟 dual-path sanity test

```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2 = "C:\gemma4_streaming_nvme0\dense_weights_streaming_1.bin"
```

注意：這只驗證 dual-path code path，不代表真正 2 顆 NVMe 頻寬。

---

## 6. 新系統測試計畫

### 6.1 正確性測試（必做）

```powershell
python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma4_release_v2\model" `
  --prompt "What is the capital of Japan? Answer in one word." `
  --max-new-tokens 20

python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma4_release_v2\model" `
  --prompt "What is 7 * 8? Answer with just the number." `
  --max-new-tokens 20
```

預期：

```text
Tokyo ✅
56 ✅
```

### 6.2 效能測試：Single vs Dual

#### Single NVMe v2 pipeline

```powershell
Remove-Item Env:\OV_DENSE_STREAM_WEIGHTS_2 -ErrorAction SilentlyContinue
$env:OV_DENSE_STREAM_WEIGHTS = "C:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma4_release_v2\model" `
  --prompt "Explain quantum computing in 3 sentences." `
  --max-new-tokens 64
```

#### Dual NVMe

```powershell
$env:OV_DENSE_STREAM_WEIGHTS = "C:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2 = "D:\gemma4_streaming_nvme1\dense_weights_streaming_1.bin"
python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma4_release_v2\model" `
  --prompt "Explain quantum computing in 3 sentences." `
  --max-new-tokens 64
```

### 6.3 要收集的數據

每次測試請記錄：

| 欄位 | 說明 |
|------|------|
| TPOT | `run_gemma4.py` perf metrics |
| Throughput | tok/s |
| DenseStreaming avg | destructor log: `avg XX ms/tok` |
| Load% | IO 佔比 |
| GPU% | GPU compute 佔比 |
| TTFT | first token latency |
| System RAM / Peak RSS | 可用 `--show-memory` |
| NVMe model / PCIe generation | 方便對照頻寬 |

### 6.4 Debug log 檢查

```powershell
Get-Content "C:\working\gemma4-openvino\gemma4_release_v2\model\dense_streaming_debug.log" |
  Select-String "start_async_load_dual|NVMe0|NVMe1|avg|Breakdown" |
  Select-Object -First 30
```

應看到類似：

```text
start_async_load_dual: group 0, 4 ops (NVMe0: 99.05 MB, NVMe1: 99.05 MB)
...
[DenseStreaming] 62 tokens, avg 128.56 ms/tok (7.78 tok/s)
[DenseStreaming] Breakdown: load=48.2% swap=0.9% args=0.4% flush=0.0% gpu=50.6%
```

在實體 2 NVMe 上，希望 Load% 明顯下降、TPOT 接近 ~77 ms。

---

## 7. 下一步 TODO

### P0 — 實體 2 NVMe 驗證

- [ ] 在新系統確認 C: 與 D: 是不同 physical NVMe，不只是同一顆磁碟分割
- [ ] 將 stripe0 放 C:，stripe1 放 D:
- [ ] 跑 correctness tests：Tokyo / 56 / translation
- [ ] 跑 64-token perf test
- [ ] 收集 `dense_streaming_debug.log`
- [ ] 比較 single NVMe vs physical dual NVMe

### P0 — Release 使用性驗證

- [ ] 解 tar 到乾淨系統
- [ ] 安裝 pip OpenVINO / GenAI / Tokenizers
- [ ] 替換 `openvino_intel_gpu_plugin.dll`
- [ ] 確認 `run_gemma4.py` 自動啟用 `CACHE_DIR`
- [ ] 確認 per-layer embedding offload 顯示 enabled
- [ ] 確認 `model_cache` 有效，第二次啟動不需要重新 compile

### P1 — 若 dual NVMe 未達預期

- [ ] 檢查 NVMe 是否真的不同 controller / physical drive
- [ ] 檢查 Windows Defender / indexing / compression 是否影響 Direct I/O
- [ ] 嘗試 8 handles total（4 per NVMe）
- [ ] 測試不同 group size（4 目前最佳，但 dual NVMe 後可重新掃）
- [ ] 記錄每 group load latency，而不只總 Load%

### P1 — Release package 改善

- [ ] 若需要真正 `.zip`，在系統安裝 7-Zip 後重新打包
- [ ] 考慮產生 split archives：`release_v2.part01.zip` 等
- [ ] README 補上 SHA256 checksum
- [ ] 可考慮 GitHub Release asset，但 13.59 GB 可能超出限制或不方便

### P2 — 程式碼清理

- [ ] `dense_weights_streaming_dual.json` 與 runtime `.json` lookup 規則統一
- [ ] 讓 runtime 在 dual mode 自動 fallback `_dual.json` 或 stripe0 metadata
- [ ] 減少 debug output 預設噪音
- [ ] 將 correctness tests 做成 `test_dual_streaming.py`

---

## 8. 已知風險 / 注意事項

### 8.1 Release tar 不在 Git

`.gitignore` 已排除：

```text
release_v2/
*.tar
*.zip
temp/
binaries/
results_*.json
```

所以換系統後不要期待 `git clone` 會拿到模型或 release tar。

### 8.2 Python 版本必須一致

目前 release 包含 Python 3.12 的 `py_openvino_genai.cp312-win_amd64.pyd`。
建議新系統使用 Python 3.12。

### 8.3 DLL 替換是必要步驟

若未替換 `openvino_intel_gpu_plugin.dll`，會看不到 `DenseStreaming` 相關訊息，或不支援 dual-NVMe code path。

### 8.4 Per-layer embedding offload 是自動的，但檔名要對

必須保留：

```text
openvino_text_embeddings_per_layer_model_revised.bin
```

若檔名錯誤或不存在，可能回到 GPU/USM 常駐 embedding，導致 RSS 大幅上升。

### 8.5 同碟 dual-path 不是真正雙碟頻寬

同碟 dual-path 實測 128.56 ms 代表程式路徑正確，但真正目標是 C: + D: 不同 physical NVMe。

---

## 9. 相關文件索引

| 文件 | 用途 |
|------|------|
| `2026-04-17-mmap-investigation.md` | mmap / memory 起始分析 |
| `20260508_dense_weight_streaming_plan.md` | Dense weight streaming Option A/B 原始計畫 |
| `20260510_group_size_tuning_and_verification.md` | group size tuning 與早期 verification |
| `20260511_architecture_review.md` | 架構 review |
| `20260511_phase2_weight_streaming.md` | Phase 2 FC streaming 主文件 |
| `20260512_per_layer_embedding_offload.md` | 3GB per-layer embedding mmap offload |
| `20260513_dual_nvme_streaming_analysis.md` | Dual NVMe 架構分析 |
| `20260514_pipeline_overlap_fix.md` | async pipeline overlap 修正 |
| `20260514_cl_out_of_resources_analysis.md` | CL_OUT_OF_RESOURCES 分析 |
| `20260515_how_to_build_deploy_debug_profile.md` | build / deploy / debug / profile 指南 |
| `RELEASE_README.md` | Release v2 使用者 README |

---

## 10. 下一台系統第一個建議操作順序

1. `git clone` repo
2. 搬移並解開 `gemma4_streaming_release_v2.tar`
3. 安裝 Python 3.12 dependencies
4. 替換 modified GPU plugin DLL
5. 確認兩顆 NVMe physical drive
6. 將 stripe0 放 NVMe0、stripe1 放 NVMe1
7. 跑 Tokyo / 56 correctness tests
8. 跑 64-token perf test
9. 收集 debug log
10. 回填一份新的 `20260520_dual_nvme_physical_test_results.md` 或更新本文件

---

## 11. Quick Commands（可直接複製）

```powershell
# Clone
mkdir C:\working\gemma4-openvino
cd C:\working\gemma4-openvino
git clone https://github.com/jlee52tw/gemma4-openvino-genai.git

# Extract release tar
mkdir C:\working\gemma4-openvino\gemma4_release_v2
tar -xf C:\path\to\gemma4_streaming_release_v2.tar -C C:\working\gemma4-openvino\gemma4_release_v2

# Install deps
pip install openvino==2026.2.0 openvino-genai openvino-tokenizers
pip install -r C:\working\gemma4-openvino\gemma4_release_v2\requirements.txt

# Replace plugin
$ovLibs = python -c "import openvino, os; print(os.path.join(os.path.dirname(openvino.__file__), 'libs'))"
Copy-Item "$ovLibs\openvino_intel_gpu_plugin.dll" "$ovLibs\openvino_intel_gpu_plugin.dll.bak" -Force
Copy-Item "C:\working\gemma4-openvino\gemma4_release_v2\dlls\openvino_intel_gpu_plugin.dll" "$ovLibs\openvino_intel_gpu_plugin.dll" -Force

# Prepare stripes
mkdir C:\gemma4_streaming_nvme0 -Force
mkdir D:\gemma4_streaming_nvme1 -Force
Copy-Item C:\working\gemma4-openvino\gemma4_release_v2\streaming_nvme0\* C:\gemma4_streaming_nvme0\ -Force
Copy-Item C:\working\gemma4-openvino\gemma4_release_v2\streaming_nvme1\* D:\gemma4_streaming_nvme1\ -Force

# Run dual-NVMe
cd C:\working\gemma4-openvino\gemma4_release_v2
$env:OV_DENSE_STREAM_WEIGHTS = "C:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2 = "D:\gemma4_streaming_nvme1\dense_weights_streaming_1.bin"
python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma4_release_v2\model" --prompt "What is the capital of Japan? Answer in one word." --max-new-tokens 20
```
