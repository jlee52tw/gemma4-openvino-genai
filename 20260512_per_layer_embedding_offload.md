# Per-Layer Embedding 2.82 GB Offload — DirectIO from NVMe
**Date:** 2026-05-12  
**Author:** jlee52tw  
**Status:** 設計完成，準備實作

---

## 1. 目標

**省下 2.82 GB 記憶體**，透過將 Gemma4 的 `text_embeddings_per_layer_model`
從 GPU compiled model 改為 **DirectIO NVMe 讀取**。

### 記憶體預算

```
Before:  6.664 GB RSS (baseline)
After:   6.664 - 2.819 = 3.845 GB  ← 省下 2.82 GB！
+ LRU:   3.845 + 0.004 = 3.849 GB  (4 MB optional cache)
```

### 效能影響

```
Decode (每個 token):
  NVMe read 12 KB:     ~20-30 µs
  CPU dequant:          ~5-10 µs
  Total overhead:       ~30-40 µs
  vs GPU decode time:   42,000 µs
  Overhead:             0.07-0.10%  ← 可忽略！
```

---

## 2. 背景：Gemma4 Per-Layer Embedding 架構

### 2.1 什麼是 Per-Layer Embedding？

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

### 2.2 資料流

```
Decode phase (每個 token):

  token_id (int64, 前一步生成的 token)
       │
       ├─→ text_embeddings_model → inputs_embeds [1,1,2560]     (主 embedding)
       │
       └─→ text_embeddings_per_layer_model → per_layer_inputs [1,1,42,256]
             │
             │  模型內部操作：
             │  1. Gather(token_id) from [262144, 10752] INT8 weight → [1, 10752]
             │  2. Dequant: int8 × scale(FP16) → FP32
             │  3. Reshape → [1, 1, 42, 256]
             │
             ↓
  language_model.infer(inputs_embeds, per_layer_inputs, ...)
       │
       └─→ 每個 decoder layer i:
             Gather(idx=i) from per_layer_inputs → [1, 1, 256]
             → Gelu gate × slice
             → MatMul(per_layer_projection [2560, 256]^T) → [1, 1, 2560]
             → RMSNorm → Add to hidden_states (residual injection)
```

**關鍵洞察：** per_layer_model 就是一個 **embedding lookup + dequant**。
每個 decode token 只讀 1 行 = 10,752 bytes。
不需要 GPU compiled model，DirectIO 就夠了。

### 2.3 注意：不是 KV Cache

Per-layer embedding **不存入 KV cache**。KV cache 存的是 attention 的 K/V 投影。
Per-layer embedding 每次都要重新讀取（lookup），無法跳過。

---

## 3. Embedding 重複問題分析

### 3.1 `embed_tokens.weight` 出現在 3 個地方

| 位置 | 大小 | 量化 | 用途 |
|---|---:|---|---|
| `text_embeddings_model.bin` | 672 MB | INT8 symmetric | Token embedding (前端) |
| `language_model.bin` (offset 2.14 GB) | 672 MB | UINT8 asymmetric + ZP | LM head / logits (後端) |
| `per_layer_model.bin` | 2,819 MB | INT8 symmetric + per-row FP16 scale | Per-layer injection |

### 3.2 這不是 Bug

- `tie_word_embeddings: true` → 同一原始權重被分別量化為不同格式
- Text embedding 用 i8 symmetric → 適合前端 embedding lookup
- LM head 用 u8 asymmetric → 適合 output projection（不同數值分佈）
- Per-layer embedding 是**完全不同的權重矩陣**（[262144, 10752]）

### 3.3 未來優化機會

如果統一 embedding 和 lm_head 的量化方案，可以共用一份 → 省 672 MB。
但這需要修改 optimum-intel 匯出流程，目前不在 scope 內。

---

## 4. 原始二進制佈局分析

### 4.1 `openvino_text_embeddings_per_layer_model.bin` 結構

```
Offset 0:
  embed_tokens_per_layer.weight: [262144, 10752] INT8
  Size: 2,818,572,288 bytes (2.819 GB)
  Layout: row-major, 連續存放
  Row N 的 offset = N × 10,752

Offset 2,818,572,288:
  embed_tokens_per_layer.weight/scale: [262144, 1] FP16
  Size: 524,288 bytes (0.5 MB)
  Row N 的 scale offset = 2,818,572,288 + N × 2

Offset 2,819,096,576:
  其他小常量 (pad token IDs, shape constants)
  Size: ~80 bytes
```

### 4.2 對齊分析

```
Row size:    10,752 bytes
4K aligned?  NO (10752 / 4096 = 2.625)
512 aligned? YES (10752 / 512 = 21 sectors, 恰好整除！)
GCD(10752, 4096) = 512
LCM(10752, 4096) = 86,016 bytes (= 8 rows = 21 pages)
```

---

## 5. DirectIO Repack 格式設計

### 5.1 為什麼需要 Repack？

原始布局的問題：
1. Weight 和 Scale 不在同一位置 → 需要兩次 IO
2. Row 不是 4K 對齊 → DirectIO 需要讀額外的 bytes
3. 需要計算兩個不同的 offset

### 5.2 Repacked 格式

```
File: per_layer_embedding_directio.bin

Header (4096 bytes, 1 page):
  [0..3]:     Magic "PLEB" (Per-Layer EMBedding)
  [4..7]:     Version = 1
  [8..11]:    vocab_size = 262144 (uint32)
  [12..15]:   per_layer_dim = 10752 (uint32)
  [16..19]:   num_layers = 42 (uint32)
  [20..23]:   layer_dim = 256 (uint32)
  [24..27]:   weight_dtype = 1 (INT8)
  [28..31]:   scale_dtype = 2 (FP16)
  [32..35]:   row_stride = 12288 (uint32, bytes per aligned row)
  [36..39]:   reserved
  [40..4095]: padding zeros

Data (starts at offset 4096):
  Row 0:  [10752 INT8 weight | 2 FP16 scale | 1534 pad] = 12,288 bytes (3 pages)
  Row 1:  [10752 INT8 weight | 2 FP16 scale | 1534 pad] = 12,288 bytes
  ...
  Row 262143: [10752 INT8 weight | 2 FP16 scale | 1534 pad] = 12,288 bytes

Total: 4096 + 262144 × 12288 = 3,221,229,568 bytes (3.221 GB)
```

### 5.3 讀取公式

```
對 token_id N:
  # Special token handling
  if N in {258880, 258884, 258881}: N = 0  (remap to row 0)
  if N < 0 or N >= 262144: return zeros([1, 1, 42, 256])

  file_offset = 4096 + N × 12288
  read_size   = 12288  (3 pages, 4K aligned)
  weight_data = buf[0:10752]     → INT8[10752]
  scale_data  = buf[10752:10754] → FP16[1]
  
  # Dequant: weight × scale × POST_GATHER_SCALE(16.0)
  output[i] = (float)weight_data[i] × (float)scale_data[0] × 16.0   for i in 0..10751
  reshape(output, [1, 1, 42, 256])
```

**POST_GATHER_SCALE = 16.0** — 從 IR 圖中的 `Constant_209981` 發現。
這是 Gemma4 per-layer embedding 架構的固有 scaling。

---

## 6. 實作計畫

### Phase A: Repack 工具 (`pack_per_layer_embedding.py`)

```python
# Input:  openvino_text_embeddings_per_layer_model.bin
# Output: per_layer_embedding_directio.bin

def pack():
    # 1. Read weight [262144, 10752] INT8 from offset 0
    # 2. Read scale [262144, 1] FP16 from offset 2,818,572,288
    # 3. Write header (4096 bytes)
    # 4. For each row: write [weight_row | scale_row | padding] = 12288 bytes
```

### Phase B: GenAI 端攔截 (`per_layer_embedding_reader`)

攔截點在 `openvino_genai_src/src/cpp/src/visual_language/gemma4/classes.cpp`:

```cpp
// 目前的 lambda (每 token 呼叫):
get_per_layer_embeddings_callback() {
    return [this](const ov::Tensor& input_ids) {
        return get_per_layer_embeddings(input_ids);  // ← 呼叫 compiled model
    };
}

// 替換為:
get_per_layer_embeddings_callback() {
    if (m_per_layer_directio_reader) {
        return [this](const ov::Tensor& input_ids) {
            int64_t token_id = input_ids.data<int64_t>()[0];
            return m_per_layer_directio_reader->lookup(token_id);
        };
    }
    // fallback to compiled model
    return [this](const ov::Tensor& input_ids) {
        return get_per_layer_embeddings(input_ids);
    };
}
```

**實作位置選擇：**
- 修改 GenAI 原始碼 → 需要 rebuild genai
- 或：用 Python callback 做 prototype 驗證 → 之後再 C++ 化

### Phase C: DirectIO Reader (`per_layer_embedding_reader.hpp/.cpp`)

```cpp
class PerLayerEmbeddingReader {
    HANDLE m_file;          // FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN
    size_t m_row_stride;    // 12288
    size_t m_header_size;   // 4096
    size_t m_weight_size;   // 10752
    void*  m_aligned_buf;   // VirtualAlloc aligned buffer for DirectIO
    
    // Optional LRU cache
    struct CacheEntry { int64_t token_id; float data[10752]; };
    std::vector<CacheEntry> m_cache;
    
public:
    ov::Tensor lookup(int64_t token_id) {
        // 1. Check LRU cache
        // 2. DirectIO read: 12 KB at (4096 + token_id * 12288)
        // 3. Dequant: int8 × fp16_scale → fp32[10752]
        // 4. Reshape → Tensor [1, 1, 42, 256]
        // 5. Update LRU cache
    }
};
```

### Phase D: Python Prototype（快速驗證）

在實作 C++ 之前，先用 Python 驗證正確性和效能：

```python
# 1. Repack per-layer model to aligned binary
# 2. Open with os.open(O_RDONLY | O_DIRECT) on Linux, or
#    CreateFile(FILE_FLAG_NO_BUFFERING) via ctypes on Windows
# 3. Hook into VLMPipeline's per_layer callback
# 4. Verify output matches compiled model output
```

**問題：** GenAI 的 VLMPipeline 不暴露 per_layer callback 給 Python。
**方案：** 先跑獨立驗證 → 確認 repack + dequant 結果正確 → 再做 C++ 整合。

---

## 7. Win32 DirectIO API 細節

### 7.1 File Open

```cpp
HANDLE hFile = CreateFileW(
    L"per_layer_embedding_directio.bin",
    GENERIC_READ,
    FILE_SHARE_READ,
    NULL,
    OPEN_EXISTING,
    FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,  // DirectIO + async
    NULL
);
```

`FILE_FLAG_NO_BUFFERING` 要求：
- 讀取 offset 必須是 sector size (通常 512 或 4096) 的倍數 ✅ (我們用 4K 對齊)
- 讀取 size 必須是 sector size 的倍數 ✅ (12288 = 3 × 4096)
- Buffer 必須是 sector-aligned ✅ (VirtualAlloc 預設 page-aligned)

### 7.2 Aligned Read

```cpp
// Allocate aligned buffer
void* buf = VirtualAlloc(NULL, 12288, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);

// Read one row
OVERLAPPED ov = {};
ov.Offset     = (DWORD)((4096 + token_id * 12288) & 0xFFFFFFFF);
ov.OffsetHigh = (DWORD)((4096 + token_id * 12288) >> 32);
DWORD bytesRead = 0;
ReadFile(hFile, buf, 12288, &bytesRead, &ov);
```

### 7.3 Dequant

```cpp
// SIMD-friendly dequant: int8 → fp32 × scale
const int8_t* weight = (int8_t*)buf;
uint16_t scale_fp16 = *(uint16_t*)(buf + 10752);
float scale = fp16_to_fp32(scale_fp16);

float* output = tensor.data<float>();  // [10752]
for (int i = 0; i < 10752; i++) {
    output[i] = (float)weight[i] * scale;
}
// Can use AVX2/AVX-512 for 16x-32x speedup
```

---

## 8. Prefill 批次讀取策略

Prefill 時 input_ids 可能有 10-1024 個 token，需要一次讀多行。

### 8.1 Scatter-gather 讀取

```
input_ids = [3689, 563, 236743, ...]  (N tokens)

方案 A: N 次獨立 DirectIO 讀取
  - 簡單，但 N 次系統呼叫開銷
  - N=100 → 100 × 20µs = 2ms (still fast)

方案 B: 排序 token IDs → 合併相鄰請求
  - 如果 token_id 相鄰，可以一次讀多行
  - LCM(10752, 4096) = 86016 = 8 rows → 可一次讀 8 行
  - 但 token IDs 通常不相鄰 → 收益有限

方案 C: ReadFileScatter (Win32)
  - 一次呼叫讀多個不連續的頁
  - 需要所有 segment 相同大小且 page-aligned ✅

推薦: 方案 A (簡單直接，prefill 本身就慢，IO 開銷可忽略)
```

---

## 9. LRU Cache 分析

### 9.1 語言的 Zipf 分佈

Token 頻率遵循 Zipf 定律：少數 token 佔大部分出現次數。

### 9.2 模擬結果

| Cache 大小 | 行數 | Vocab 覆蓋率 | Hit Rate (多輪對話) | Hit Rate (decode) |
|---:|---:|---:|---:|---:|
| 0.3 MB | 32 | 0.01% | ~22% | ~39% |
| 0.7 MB | 64 | 0.02% | ~30% | ~45% |
| 1.4 MB | 128 | 0.05% | ~33% | ~51% |
| 2.8 MB | 256 | 0.10% | ~33% | ~51% |

### 9.3 結論

- LRU cache 即使 0% hit rate，IO 開銷仍只有 ~30 µs ≈ 0.07% of TPOT
- Cache 主要價值在減少 NVMe IO 次數 → 延長 NVMe 壽命
- **建議 Phase 1 不加 LRU，Phase 2 可選加 128-256 行 (~1-3 MB)**

---

## 10. 風險與注意事項

### 10.1 特殊 Token 處理

Per-layer model 內部有：
- Pad token → output = zeros
- 其他特殊 token → 使用 trained embedding 值
- Token ID 範圍檢查

**需要在 reader 中正確處理這些 edge cases。**

### 10.2 GenAI 整合深度

修改 GenAI classes.cpp 需要 rebuild openvino.genai，或者：
- 方案 1: 修改 GenAI 原始碼 → 完整整合
- 方案 2: 用 OpenVINO plugin property → 不修改 GenAI，在 GPU plugin 層攔截
- 方案 3: 先 Python prototype → 驗證數值正確性 → 再決定整合方式

### 10.3 數值精度

Direct dequant (int8 × fp16_scale → fp32) 必須與 compiled model 的結果完全一致。
需驗證：
- FP16 scale → FP32 的轉換精確度
- INT8 → FP32 的型別轉換
- 是否有額外的 post-scaling（compiled model 中有 `Multiply × scaling_constant`）

---

## 11. 實作步驟（Action Items）

### Step 1: 建立 Repack 工具 ✅ → `pack_per_layer_embedding.py`
- 讀取原始 .bin → 輸出 4K-aligned .bin
- 加入 header (magic, version, metadata)
- 驗證每行 weight + scale 正確

### Step 2: Python 數值驗證 ✅
- 從 repacked .bin 讀取 row N → dequant → compare with compiled model output
- **bit-exact (abs_err = 0.000000) — 20/20 samples PASS**
- 發現 IR 圖中有 `Constant_209981 = 16.0` 的 post-Gather scaling factor
- 完整 dequant: `int8 × fp16_scale × 16.0 → fp32`
- 特殊 token: `{258880, 258884, 258881}` → remap to row 0; out-of-range → zeros

### Step 3: C++ DirectIO Reader ⬜
- `per_layer_embedding_reader.hpp/.cpp`
- Win32 CreateFile + ReadFile + VirtualAlloc
- Dequant loop (可選 AVX2 加速)

### Step 4: GenAI 整合 ⬜
- 修改 gemma4/classes.cpp 的 callback
- 環境變數控制啟用/停用
- Prefill 批次讀取

### Step 5: 效能驗證 ⬜
- 比較 tok/s: DirectIO reader vs compiled model
- 比較 RSS: 應省下 ~2.82 GB
- 比較 TTFT: prefill 時的 IO 影響

---

## 12. 相關文件

| 文件 | 說明 |
|---|---|
| `20260511_phase2_weight_streaming.md` | Phase 2 decoder weight streaming (前一階段) |
| `20260508_dense_weight_streaming_plan.md` | 初始計畫 |
| `cpp/dense_weight_streaming_manager.hpp/.cpp` | Phase 2 streaming manager |
| `pack_dense_weights.py` | Phase 2 decoder weight packer |
| `reference/moe_expert_weight_manager.hpp` | MoE OTD DirectStorage 參考 |

---

## 13. 一鍵流程（待實作完成後更新）

```powershell
# === Step 1: Repack ===
cd C:\working\gemma4-openvino\gemma4-openvino-genai
python pack_per_layer_embedding.py `
    --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --output "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin"

# === Step 2: Verify ===
python verify_per_layer_embedding.py `
    --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" `
    --repacked "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin"

# === Step 3: Test with DirectIO reader ===
$env:OV_PER_LAYER_DIRECTIO = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\per_layer_embedding_directio.bin"
python run_gemma4.py --model-dir "C:\working\gemma4-openvino\gemma-4-E4B-it-ov" --prompt "Hello" --max-new-tokens 20
```
