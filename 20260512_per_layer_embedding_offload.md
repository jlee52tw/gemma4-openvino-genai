# Per-Layer Embedding 2.82 GB Offload — NVMe mmap + CPU Dequant
**Date:** 2026-05-12  
**Author:** jlee52tw  
**Status:** ✅ 已實作完成，驗證通過

---

## 1. 摘要

將 Gemma4 的 `text_embeddings_per_layer_model`（2.82 GB）**從 GPU compiled model
卸載到 NVMe**，改用 OS mmap + CPU dequant 取代。配合所有 42 decoder layers 保留在
GPU 記憶體中，達到：

- **24.83 tok/s**（vs 基線 24.0 tok/s = +3.5%）
- **穩定 RSS: 4,416 MB**（vs 基線 ~7,100 MB = **省 2,684 MB / -37.8%**）
- **回答品質不受影響**（bit-exact dequant，20/20 samples abs_err = 0.000000）

---

## 2. 效能實測結果

### 2.1 Per-Layer Embedding Offload 效能（2026-05-12 實測）

| 指標 | 基線（原始 GPU） | **Per-layer Offload** | 變化 |
|---|---:|---:|---|
| Output TPS | 24.0 | **24.83** | +3.5% ✅ |
| TPOT | 41.7 ms | **40.27 ms** | -3.4% ✅ |
| TTFT | 300 ms | **563 ms** | +263 ms (首次 mmap page fault) |
| 穩定 RSS | ~7,100 MB | **4,416 MB** | **-2,684 MB (-37.8%)** ✅ |
| Peak RSS (loading) | ~8,020 MB | **7,553 MB** | -467 MB |
| Model Load Time | ~9.0 s | **8.8 s** | -0.2 s (跳過 per-layer compile) |

**測試條件：** short-text, prompt 25 tokens, generate 256 tokens, GPU device

### 2.2 為什麼 TTFT 增加？

首次推理時 OS 需要 **page fault 載入 mmap pages** 到實體記憶體。
3.22 GB 檔案被 mmap，但首次存取 prefill 的 N 個 token 時，
每個 token 觸發 3 頁 page fault = ~12 KB。第二次推理以後 OS 已 cache
這些 pages，TTFT 會恢復正常。

### 2.3 為什麼 TPOT 反而變快？

卸載 2.82 GB 後 GPU USM 記憶體壓力降低，GPU scheduler 更高效地
排程 decoder layers 的計算。此外 per-layer embedding lookup 的 CPU dequant
只需 ~10-30 µs（vs GPU 端 ~42 ms compute），overhead 完全可忽略。

---

## 3. 使用的模型檔案

### 3.1 原始模型檔案（輸入）

| 檔案 | 路徑 | 大小 | 說明 |
|---|---|---:|---|
| Per-layer model IR | `openvino_text_embeddings_per_layer_model.xml` | ~5 KB | IR 圖定義 |
| Per-layer model weights | `openvino_text_embeddings_per_layer_model.bin` | 2,819 MB | 包含 INT8 weight + FP16 scale |
| Language model | `openvino_language_model.xml` / `.bin` | 2,095 MB | 42 decoder layers (使用 per_layer_inputs) |
| Text embeddings model | `openvino_text_embeddings_model.xml` / `.bin` | 672 MB | 主 token embedding |
| Vision encoder | `openvino_vision_embeddings_model.xml` / `.bin` | 171 MB | 圖片 encoder |

模型目錄：`C:\working\gemma4-openvino\gemma-4-E4B-it-ov\`

### 3.2 轉換後檔案（offload 專用）

| 檔案 | 路徑 | 大小 | 說明 |
|---|---|---:|---|
| **Repacked binary** | `per_layer_embedding_directio.bin` | **3,221 MB** | 4K-aligned 格式，mmap 讀取 |

放置位置：與模型目錄相同（`model_dir/per_layer_embedding_directio.bin`）

### 3.3 二進制格式

```
File: per_layer_embedding_directio.bin (3,221,229,568 bytes)

Header (4096 bytes = 1 page):
  [0..3]:     Magic "PLEB" (0x42454C50 little-endian)
  [4..7]:     Version = 1
  [8..11]:    vocab_size = 262144
  [12..15]:   per_layer_dim = 10752
  [16..19]:   num_layers = 42
  [20..23]:   layer_dim = 256
  [24..27]:   weight_dtype = 1 (INT8)
  [28..31]:   scale_dtype = 2 (FP16)
  [32..35]:   row_stride = 12288
  [36..4095]: reserved zeros

Data (262144 rows × 12288 bytes each):
  Row N: [10752 INT8 weight | 2 FP16 scale | 1534 pad] = 12,288 bytes (3 pages)
  
  File offset for token N = 4096 + N × 12288
```

---

## 4. 修改的檔案與各自用途

### 4.1 新建檔案

| # | 檔案 | 位置 | 用途 |
|---|---|---|---|
| 1 | `per_layer_embedding_reader.hpp` | `openvino_genai_src/src/cpp/src/visual_language/gemma4/` | **C++ mmap reader + CPU dequant**。header-only，使用 Win32 `CreateFileMapping`/`MapViewOfFile`（Windows）或 POSIX `mmap`（Linux）將 3.22 GB 檔案映射到虛擬記憶體。每次 lookup 從 mmap 記憶體直接讀取 12 KB row 並做 INT8→FP32 dequant。 |
| 2 | `pack_per_layer_embedding.py` | `gemma4-openvino-genai/` (workspace) | **Python repack 工具**。讀取原始 per-layer model `.bin`，將 weight 和 scale 合併到每行 12,288 bytes 的 4K-aligned 格式，輸出 `per_layer_embedding_directio.bin`。內建 `--verify` 模式驗證 bit-exact 正確性。 |

### 4.2 修改的檔案

| # | 檔案 | 變更 | 用途 |
|---|---|---|---|
| 3 | `classes.hpp` | + `#include per_layer_embedding_reader.hpp`<br>+ `std::unique_ptr<PerLayerEmbeddingReader> m_per_layer_reader` member | **新增 offload reader 成員**。與原本的 `m_per_layer_embeddings_requests`（compiled model）互斥——只有一個會被初始化。 |
| 4 | `classes.cpp` | 修改兩個建構函式 + `get_per_layer_embeddings()` | **核心攔截邏輯**。建構函式中：(1) 檢查環境變數 `OV_PER_LAYER_EMBEDDING_PATH`，(2) 自動偵測 `model_dir/per_layer_embedding_directio.bin`。若找到則建立 `PerLayerEmbeddingReader`（mmap），跳過 `compile_model()` 載入 2.82 GB 到 GPU。`get_per_layer_embeddings()` 新增 Path A（reader.lookup）/ Path B（原始 compiled model）分流。 |

### 4.3 原始碼位置

```
openvino_genai_src/
└── src/cpp/src/visual_language/gemma4/
    ├── classes.hpp                          ← 修改：新增 m_per_layer_reader 成員
    ├── classes.cpp                          ← 修改：建構函式 + get_per_layer_embeddings()
    └── per_layer_embedding_reader.hpp       ← 新建：mmap reader + dequant

gemma4-openvino-genai/  (workspace)
    └── pack_per_layer_embedding.py          ← 新建：repack 工具
```

### 4.4 各檔案修改細節

#### `per_layer_embedding_reader.hpp` — mmap Reader（新建）

```
class PerLayerEmbeddingReader:
  構建方式: open_and_map(path) → mmap 整個 3.22 GB 檔案
  核心方法: lookup(input_ids: Tensor[batch, seq_len]) → Tensor[batch, seq_len, 42, 256]
  
  每個 token 的處理:
    1. resolve_token_id(): 特殊 token {258880, 258884, 258881} → row 0; OOV → zeros
    2. dequant_token(): row = m_data[HEADER + id * 12288]
       int8 weights[10752] × fp16_scale × 16.0 → float32[10752]
    3. 輸出 reshape 為 [42, 256]
  
  記憶體映射方式:
    Windows: CreateFileW(FILE_FLAG_RANDOM_ACCESS) → CreateFileMappingW → MapViewOfFile
    Linux:   open(O_RDONLY) → mmap(PROT_READ, MAP_PRIVATE) + madvise(MADV_RANDOM)
  
  清理: ~PerLayerEmbeddingReader() 自動 UnmapViewOfFile / munmap
```

#### `classes.hpp` — 新增成員（修改）

```diff
+ #include <memory>
+ #include "visual_language/gemma4/per_layer_embedding_reader.hpp"

  private:
    std::unique_ptr<CircularBufferQueue<ov::InferRequest>> m_per_layer_embeddings_requests;
+   std::unique_ptr<PerLayerEmbeddingReader> m_per_layer_reader;
```

#### `classes.cpp` — 建構函式邏輯（修改）

```
建構函式邏輯（兩個 constructor 都做相同修改）：

1. 檢查路徑:
   a. 環境變數 OV_PER_LAYER_EMBEDDING_PATH → 如果設定且檔案存在，使用它
   b. 自動偵測 model_dir / "per_layer_embedding_directio.bin" → 如果存在，使用它

2. 如果找到 repacked binary:
   → m_per_layer_reader = make_unique<PerLayerEmbeddingReader>(path)
   → 印出 "[Gemma4] Per-layer embeddings: using DirectIO reader"
   → 跳過 compile_model()，不分配 GPU USM 記憶體

3. 如果沒找到:
   → 走原本路徑：compile_model() → CircularBufferQueue<InferRequest>
   → 印出 "[Gemma4] Per-layer embeddings: using compiled GPU model"
```

#### `classes.cpp` — `get_per_layer_embeddings()` 方法（修改）

```
ov::Tensor get_per_layer_embeddings(input_ids):
  Path A (m_per_layer_reader != null):
    return m_per_layer_reader->lookup(input_ids)
    // mmap 讀取 + CPU dequant，~10-30 µs per token

  Path B (原始路徑):
    req.set_tensor("input_ids", input_ids)
    req.infer()  // GPU compiled model 推理
    memcpy output → result
```

---

## 5. 背景：Gemma4 Per-Layer Embedding 架構

### 5.1 什麼是 Per-Layer Embedding？

Gemma4 的獨特設計：**每個 decoder layer 接收一個額外的 256-dim token-dependent signal**。

```
config.json:
  "hidden_size_per_layer_input": 256
  "vocab_size_per_layer_input": 262144
  "num_hidden_layers": 42
```

權重矩陣：`embed_tokens_per_layer.weight` shape `[262144, 10752]` (INT8)
- 262144 = vocab_size
- 10752 = 42 layers × 256 dim

### 5.2 資料流

```
Decode phase (每個 token):

  token_id (int64)
       │
       ├─→ text_embeddings_model → inputs_embeds [1,1,2560]     (主 embedding)
       │
       └─→ per_layer lookup → per_layer_inputs [1,1,42,256]
             │
             │  Dequant: int8 × fp16_scale × 16.0 → fp32
             │  Reshape → [1, 1, 42, 256]
             │
             ↓
  language_model.infer(inputs_embeds, per_layer_inputs, ...)
       │
       └─→ 每個 decoder layer i:
             Gather(idx=i) from per_layer_inputs → [1, 1, 256]
             → Gelu gate × slice
             → MatMul(per_layer_projection) → [1, 1, 2560]
             → RMSNorm → Add to hidden_states (residual injection)
```

**關鍵洞察：** per_layer_model 就是一個 **embedding lookup + dequant**。
每個 decode token 只讀 1 行 = 10,752 bytes。
不需要 GPU compiled model — mmap + CPU dequant 就夠了。

### 5.3 Dequant 公式

```
raw_int8[10752] = row from repacked binary
scale_fp16[1]   = row + offset 10752

output[i] = float(raw_int8[i]) × float(scale_fp16[0]) × 16.0f

POST_GATHER_SCALE = 16.0 — 從 IR 圖中的 Constant_209981 發現
```

### 5.4 特殊 Token 處理

```
Token IDs {258880, 258884, 258881} → remap to row 0
Token ID < 0 or >= 262144          → return zeros([1,1,42,256])
```

---

## 6. 如何轉換（Repack）模型

### 6.1 前提條件

- 已匯出的 Gemma4 OpenVINO 模型目錄（包含 `openvino_text_embeddings_per_layer_model.bin`）
- Python 3.10+，numpy

### 6.2 Repack 指令

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai

# Repack: 產生 4K-aligned binary（約 60 秒，輸出 3.22 GB）
python pack_per_layer_embedding.py `
    --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --output "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin"
```

### 6.3 驗證正確性

```powershell
# Verify: 與 compiled model 比對 bit-exact（約 30 秒）
python pack_per_layer_embedding.py `
    --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --output "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin" `
    --verify
```

預期輸出：
```
=== Verification: 20 random samples ===
Sample  0: token= 39457, abs_err=0.000000, rel_err=0.000000 ✓ PASS
...
Sample 19: token=158630, abs_err=0.000000, rel_err=0.000000 ✓ PASS
All 20 samples PASSED (bit-exact)
```

### 6.4 Repack 流程說明

```
Input:  openvino_text_embeddings_per_layer_model.bin (2,819 MB)
  ├── offset 0:                INT8 weights [262144, 10752]
  └── offset 2,818,572,288:    FP16 scales  [262144, 1]

Processing:
  For each row (0..262143):
    read weight[10752] from offset row × 10752
    read scale[1]      from offset 2,818,572,288 + row × 2
    write [weight | scale | 1534-byte padding] = 12,288 bytes

Output: per_layer_embedding_directio.bin (3,221 MB)
  ├── Header: 4096 bytes (magic "PLEB", metadata)
  └── Data:   262144 rows × 12,288 bytes
```

---

## 7. 如何建置 (Build)

### 7.1 前提條件

- Visual Studio 2022 (MSVC v143+)
- CMake 3.23+
- Python 3.12 + pip 安裝的 `openvino==2026.2.0`
- OpenVINO GenAI 原始碼：`C:\working\gemma4-openvino\openvino_genai_src\`

### 7.2 Build 指令

```powershell
cd C:\working\gemma4-openvino\openvino_genai_src

# Configure（只需第一次）
cmake -B build -G "Visual Studio 17 2022" `
    -DCMAKE_BUILD_TYPE=Release `
    -DENABLE_PYTHON=OFF `
    -DENABLE_JS=OFF `
    -DENABLE_TESTS=OFF `
    -DENABLE_TOOLS=OFF `
    -DENABLE_XGRAMMAR=OFF `
    -DENABLE_GGUF=ON

# Build（約 3-5 分鐘）
cmake --build build --target openvino_genai --config Release -- /v:m /p:CL_MPCount=4
```

### 7.3 部署 (Deploy)

```powershell
# 備份原始 DLL
$dst = "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino_genai\openvino_genai.dll"
if (-not (Test-Path "$dst.bak")) {
    Copy-Item $dst "$dst.bak"
}

# 部署修改後的 DLL
Copy-Item "C:\working\gemma4-openvino\openvino_genai_src\build\openvino_genai\openvino_genai.dll" $dst -Force
```

### 7.4 還原（如果需要）

```powershell
$dst = "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino_genai\openvino_genai.dll"
Copy-Item "$dst.bak" $dst -Force
```

---

## 8. 如何執行 (Run)

### 8.1 自動偵測模式（推薦）

只需將 `per_layer_embedding_directio.bin` 放在模型目錄中，GenAI 會自動偵測：

```powershell
cd C:\working\gemma4-openvino\gemma4-openvino-genai

# 確認 repacked binary 存在於模型目錄
ls "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin"

# 正常執行（GenAI 自動偵測，print log: "using DirectIO reader"）
python run_gemma4.py `
    --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --device GPU `
    --prompt "Explain quantum computing in simple terms." `
    --max-new-tokens 256 `
    --show-memory
```

預期輸出中應看到：
```
[PerLayerEmbeddingReader] Loaded from: ...\per_layer_embedding_directio.bin (vocab_size=262144, file_size=3221229568)
[Gemma4] Per-layer embeddings: using DirectIO reader (saving ~2.82 GB GPU memory)
```

### 8.2 環境變數模式

如果 repacked binary 不在模型目錄中，可用環境變數指定路徑：

```powershell
$env:OV_PER_LAYER_EMBEDDING_PATH = "D:\models\per_layer_embedding_directio.bin"
python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" --device GPU --prompt "Hello" --max-new-tokens 64
```

### 8.3 停用 offload（回到原始行為）

移除或重命名 repacked binary 即可回到 compiled GPU model 路徑：

```powershell
# 暫時停用
Rename-Item "...\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin" "per_layer_embedding_directio.bin.disabled"

# 重新啟用
Rename-Item "...\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin.disabled" "per_layer_embedding_directio.bin"
```

### 8.4 搭配 decoder weight streaming

Per-layer offload 和 decoder streaming 是**相互獨立**的功能，可以同時啟用：

| 配置 | Per-layer GPU 記憶體 | Decoder GPU 記憶體 | 備註 |
|---|---:|---:|---|
| 基線（都不用） | 2.82 GB | 2.09 GB | RSS ~7.1 GB, 24 tok/s |
| **Per-layer offload only** | **0 GB** | 2.09 GB | **RSS ~4.4 GB, 24.8 tok/s** ✅ |
| Decoder streaming only | 2.82 GB | ~0.5 GB | 6.78 tok/s（Phase 2 最佳） |
| 兩者兼用 | 0 GB | ~0.5 GB | RSS 最低，tok/s ~5-7 |

---

## 9. 技術深入：mmap vs DirectIO 讀取機制

### 9.1 實際實作使用 mmap

原始設計提到 "DirectIO"，但最終實作使用 **OS mmap（記憶體映射）**，
而非 Win32 `FILE_FLAG_NO_BUFFERING` 的 DirectIO。

#### mmap 方式（目前實作）

```
CreateFileW(FILE_FLAG_RANDOM_ACCESS)    // 告訴 OS 是隨機存取模式
CreateFileMappingW(PAGE_READONLY)       // 建立檔案映射
MapViewOfFile(FILE_MAP_READ)            // 映射到虛擬記憶體空間

讀取時: 直接解引用 m_data[offset]，OS 透過 page fault 自動從 NVMe 載入
```

**工作流程：**
1. mmap 建立虛擬位址映射，但不實際載入資料
2. 首次存取某 row 時觸發 page fault → OS 從 NVMe 讀取 4K page → 載入到 page cache
3. 後續存取同一 row → page cache hit → 直接從 RAM 讀取（~100 ns，零 IO）
4. OS 根據實體記憶體壓力自動 evict 不常用的 pages

#### 真正的 DirectIO 方式（未使用）

```
CreateFileW(FILE_FLAG_NO_BUFFERING)     // 繞過 OS page cache
VirtualAlloc(aligned buffer)            // 分配 4K-aligned buffer
ReadFile(hFile, buf, 12288, ...)        // 每次手動觸發 NVMe 讀取
```

### 9.2 mmap vs DirectIO 優缺點比較

| 面向 | mmap（目前實作） | DirectIO（未使用） |
|---|---|---|
| **Decode 延遲** | 首次 ~4 µs (page fault)，之後 ~0.1 µs (cache hit) | 每次 ~20-30 µs (NVMe IO) |
| **Page Cache** | ✅ OS page cache，常用 token 零 IO | ❌ 繞過 cache，每次都做 IO |
| **記憶體影響** | ⚠️ OS 可能 cache 熱門 pages（~數 MB~數百 MB） | ✅ 完全不佔額外記憶體 |
| **實作複雜度** | ✅ 簡單（mmap + 指標解引用） | ⚠️ 需要 aligned buffer + OVERLAPPED |
| **Prefill 效能** | ✅ OS 自動預讀（readahead） | ⚠️ 需要手動 scatter-gather IO |
| **跨平台** | ✅ Windows + Linux 都有 mmap | ⚠️ Windows/Linux API 差異大 |
| **NVMe 壽命** | ✅ 常用 token cache hit，減少實際 IO 次數 | ❌ 每次都觸發 NVMe read |

### 9.3 Long Context 對兩種方案的影響

#### Prefill 階段（N tokens 一次處理）

| Context 長度 | mmap 首次 IO 量 | DirectIO IO 量 | mmap 重複 token 收益 |
|---:|---:|---:|---|
| 25 tok | 25 × 12 KB = 300 KB | 同左 | 少量重複 |
| 256 tok | 256 × 12 KB = 3 MB | 同左 | ~10% hit |
| 1024 tok | 1024 × 12 KB = 12 MB | 同左 | ~20% hit |
| 4096 tok | 4096 × 12 KB = 48 MB | 同左 | ~25-30% hit |
| 16384 tok | 16384 × 12 KB = 192 MB | 同左 | ~30-40% hit（Zipf 分佈）|

**mmap 的 long context 優勢：**
- Token 重複時（Zipf 分佈），page cache 直接命中（~100 ns vs ~4 µs page fault）
- OS 可能自動 readahead 相鄰 pages，加速連續新 token 載入
- 多輪對話時前一輪的 token pages 可能還在 cache

**DirectIO 的 long context 問題：**
- 每個 token 都強制做 NVMe IO，即使剛讀過同一 token
- Prefill 16K tokens → 16384 × ~20 µs = ~328 ms 純 IO（mmap 可能只需 ~100 ms）

#### Decode 階段（每次 1 token）

| 場景 | mmap 延遲 | DirectIO 延遲 |
|---|---:|---:|
| 首次新 token | ~4 µs (page fault) | ~20 µs (NVMe read) |
| 重複 token (cache hit) | ~0.1 µs | ~20 µs |
| 記憶體壓力下 | ~4 µs (re-fault) | ~20 µs |

#### Long Context 的記憶體影響

```
mmap：OS 會 cache 已存取過的 pages
  25 tokens  →  ~300 KB resident pages
  1024 tokens → ~12 MB resident pages
  16K tokens  → ~192 MB resident pages（最壞情況，全部不重複）
  
  但 OS 在記憶體壓力下會自動回收不活躍的 pages。
  實測穩定 RSS 只增加 ~86 MB（4416 - 4330 = 86 MB），
  因為 token 重複率高 + OS eviction。

DirectIO：永遠精確 0 額外記憶體
  完全不使用 page cache。
  適合極端記憶體受限場景（如 4 GB 系統）。
```

### 9.4 結論：為什麼選 mmap？

1. **Decode 效能最優**：常用 token (Zipf top-100) 幾乎 0 延遲
2. **Prefill long context 最佳**：OS 自動 readahead + cache 重複 token
3. **實作簡單可靠**：不需處理 aligned buffer、OVERLAPPED IO、error recovery
4. **記憶體自動管理**：OS page cache 會在壓力下自動 evict
5. **真正的 bottleneck 在 GPU**：每 token ~40 ms GPU 計算，IO overhead <0.1%

**何時需要切換到 DirectIO？**
- 系統只有 4 GB RAM，不能容忍任何 page cache 記憶體
- 需要精確控制每一 byte 的記憶體分配
- 目前的 row-aligned 格式完全相容 DirectIO（12,288 bytes = 3 × 4K pages）

---

## 10. Embedding 重複問題分析

### 10.1 `embed_tokens.weight` 出現在 3 個地方

| 位置 | 大小 | 量化 | 用途 |
|---|---:|---|---|
| `text_embeddings_model.bin` | 672 MB | INT8 symmetric | Token embedding (前端) |
| `language_model.bin` (offset 2.14 GB) | 672 MB | UINT8 asymmetric + ZP | LM head / logits (後端) |
| `per_layer_model.bin` | 2,819 MB | INT8 symmetric + per-row FP16 scale | Per-layer injection |

### 10.2 這不是 Bug

- `tie_word_embeddings: true` → 同一原始權重被分別量化為不同格式
- Per-layer embedding 是**完全不同的權重矩陣**（[262144, 10752] vs [262144, 2560]）

---

## 11. 風險與注意事項

### 11.1 TTFT 增加

首次推理時 OS 需載入 mmap pages，對短 prompt 影響不大（~200 ms），
但長 prompt (>1K tokens) 可能增加額外延遲。
**緩解：** 第二次推理以後 TTFT 恢復正常（pages 已在 cache）。

### 11.2 記憶體壓力下的行為

如果系統記憶體非常緊張（<4 GB free），OS 可能 evict mmap pages，
導致 decode 時產生額外 page fault（~4 µs vs ~0.1 µs）。
**影響：** 每 token 最多增加 ~4 µs = 0.01% of TPOT，可忽略。

### 11.3 儲存設備需求

mmap 依賴底層儲存設備的 random read 效能。
- NVMe (Gen5): <2 µs per page fault ← **最佳**
- NVMe (Gen3/4): ~4 µs per page fault ← 推薦
- SATA SSD: ~50 µs per page fault ← 可用但稍慢
- HDD: ~5-10 ms per page fault ← ❌ **不適用**

---

## 12. 實作步驟追蹤

| Step | 描述 | 狀態 |
|---|---|---|
| 1 | Repack 工具 (`pack_per_layer_embedding.py`) | ✅ 完成 |
| 2 | Python 數值驗證 (bit-exact, 20/20 pass) | ✅ 完成 |
| 3 | C++ mmap Reader (`per_layer_embedding_reader.hpp`) | ✅ 完成 |
| 4 | GenAI 整合 (`classes.cpp`/`.hpp` 修改) | ✅ 完成 |
| 5 | Build GenAI DLL (`openvino_genai.dll`) | ✅ 完成 |
| 6 | Deploy + 效能驗證 (24.83 tok/s, -2.68 GB) | ✅ 完成 |
| 7 | 選配：LRU Cache（目前不需要） | ⬜ 待評估 |
| 8 | 選配：AVX2/AVX-512 SIMD dequant 加速 | ⬜ 待評估 |

---

## 13. 完整一鍵流程

```powershell
# ===== Step 1: Repack 模型 =====
cd C:\working\gemma4-openvino\gemma4-openvino-genai
python pack_per_layer_embedding.py `
    --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --output "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin" `
    --verify

# ===== Step 2: Build GenAI DLL =====
cd C:\working\gemma4-openvino\openvino_genai_src
cmake -B build -G "Visual Studio 17 2022" `
    -DCMAKE_BUILD_TYPE=Release `
    -DENABLE_PYTHON=OFF -DENABLE_JS=OFF -DENABLE_TESTS=OFF `
    -DENABLE_TOOLS=OFF -DENABLE_XGRAMMAR=OFF -DENABLE_GGUF=ON
cmake --build build --target openvino_genai --config Release -- /v:m /p:CL_MPCount=4

# ===== Step 3: Deploy DLL =====
$dst = "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino_genai\openvino_genai.dll"
Copy-Item "$dst" "$dst.bak" -ErrorAction SilentlyContinue
Copy-Item "build\openvino_genai\openvino_genai.dll" $dst -Force

# ===== Step 4: Run =====
cd C:\working\gemma4-openvino\gemma4-openvino-genai
python run_gemma4.py `
    --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --device GPU `
    --prompt "Explain quantum computing." `
    --max-new-tokens 256 `
    --show-memory
```

---

## 14. 相關文件

| 文件 | 說明 |
|---|---|
| `pack_per_layer_embedding.py` | Repack 工具 + 驗證 |
| `20260511_phase2_weight_streaming.md` | Phase 2 decoder weight streaming |
| `20260508_dense_weight_streaming_plan.md` | 初始 streaming 計畫 |
| `cpp/dense_weight_streaming_manager.hpp/.cpp` | Phase 2 decoder streaming manager |
| `reference/moe_expert_weight_manager.hpp` | MoE OTD DirectStorage 參考 |
