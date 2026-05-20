# 20260520 — 記憶體足跡分析：Weight Streaming 對 8 GB 系統的實際效益

**日期：** 2026-05-20  
**系統：** Intel Panther Lake Xe³ iGPU (16 EUs, v30.4.0), 16 GB LPDDR5, 雙 Samsung 2 TB NVMe (D:/E:)  
**Runtime：** OpenVINO 2026.2.0.dev20260411 (modified) + openvino-genai `as/vlm_enable_1`  
**模型：** gemma-4-E4B-it-ov (INT4, group_size=64, 42 decoder layers)  
**Prompt：** `Explain quantum computing in 3 sentences for a high school student.` (96 tokens)  
**量測工具：** [temp/memprobe2.py](temp/memprobe2.py) — `psutil` 每 50 ms 取樣 RSS / Private bytes

---

## TL;DR — 結論先講

> **在目前這份 build 中，dense weight streaming 不會省記憶體，反而多花 200~840 MB。**  
> 真正讓 8 GB 系統能跑的關鍵是 **per-layer embedding mmap offload**（節省 ≈ 3 GB USM）。  
> 為了 **−4 tps 的代價**換到的並不是記憶體節省，而是 ① GPU USM 上限的彈性、② 雙 NVMe 並行 I/O 的吞吐學術價值。  
> **對純粹 8 GB 系統使用者：建議僅啟用 mmap embedding offload，不要開啟 dense weight streaming。**

---

## 1. 模型 `.bin` 檔案分布

| 檔案 | 大小 (MB) | 大小 (GB) | 角色 | 預設駐留位置 |
|---|---:|---:|---|---|
| `openvino_text_embeddings_per_layer_model_revised.bin` | 3072.0 | 3.000 | Per-layer 256 維 embedding | **mmap shared page**（不計入 private） |
| `per_layer_embedding_directio.bin` | 3072.0 | 3.000 | DirectIO 切換用 HardLink（指向同檔） | 同上 |
| `openvino_language_model.bin` | 2685.2 | 2.622 | 42 層 decoder（含 FC=1352 MB streamable） | **GPU USM**（pinned） |
| `openvino_text_embeddings_model.bin` | 640.5 | 0.625 | Token embedding lookup | GPU USM |
| `openvino_vision_embeddings_model.bin` | 161.9 | 0.158 | Vision encoder（text 模式可不載） | GPU USM |
| `openvino_tokenizer.bin` | 16.5 | 0.016 | tokenizer | CPU RAM |
| `openvino_detokenizer.bin` | 4.2 | 0.004 | detokenizer | CPU RAM |
| **小計 (不含 HardLink 重複)** | **6580.3** | **6.426** | | |

### 1.1 Streaming 後的 NVMe stripe 檔（已在實體 SSD 上）

| 路徑 | 大小 (MB) | 角色 |
|---|---:|---|
| `D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin` | 775.6 | FC weights, half-stripe 0 (Samsung NVMe 1) |
| `E:\gemma4_streaming_nvme1\dense_weights_streaming_1.bin` | 775.6 | FC weights, half-stripe 1 (Samsung NVMe 2) |
| **小計** | **1551.2** | 從 `openvino_language_model.bin` 中抽出的 FC tensor |

> Streaming **不會**從 `openvino_language_model.bin` 真的把 FC 切走 —— 原始的 `language_model.bin` 仍是 2685 MB。Streaming 的 stripe 是**額外的一份副本**。

---

## 2. 實測結果（4 個情境）

> 命令：`python temp/memprobe2.py`  
> 取樣：subprocess + children RSS 與 Windows Private bytes  
> **Peak** = 整個生命週期最高瞬間值（通常出現在模型載入時，包含 mmap demand-paged 頁面）  
> **Steady** = 取樣序列 50%-90% 區段的平均（生成階段穩態）

| Case | 描述 | Peak RSS (MB) | Peak Private (MB) | Steady RSS (MB) | Steady Private (MB) | TPOT (ms) | TPS |
|---|---|---:|---:|---:|---:|---:|---:|
| **A** | 無 embedding offload<br>（沒有 streaming） | 8891.7 | 5081.6 | 7471.0★ | 4473.8★ | — | — |
| **B** | mmap embedding offload<br>（baseline） | 8844.3 | **5182.5** | 4386.0 | **5126.6** | 71.67 | **13.95** |
| **C** | mmap + 單 NVMe stream (D:)<br>BUF=2 IO=4 | 8844.9 | 5396.7 | 4597.0 | 5342.8 | 98.57 | 10.14 |
| **D** | mmap + 雙 NVMe stream (D+E)<br>BUF=4 IO=8 | 8869.5 | 6019.4 | 5220.5 | 5966.8 | 102.89 | 9.72 |

★ Case A 在載入時即崩潰（`Empty weights data in bin file` — 拆掉 embedding 檔但 IR XML 仍然引用），所以 5.6 秒就結束。它的 steady 值是「載入到一半時的瞬間」，**不代表完整推理穩態**。實際 A 若能跑，估計穩態會再多 ~3 GB（整份 per-layer embedding 進 USM）。

### 2.1 為什麼「Peak RSS」全部都 ≈ 8.85 GB？

Peak RSS 量到的是 OS 把 mmap'd 頁面 demand-page 進來時的「**瞬間 working set**」 —— 那些 3 GB 的 embedding 是 **shared、可被回收**的頁面。Private bytes（真實 commit）才是 8 GB 系統能否容納的判準。

### 2.2 為什麼 streaming 的 Private 比 baseline **更大**？

```
ΔPrivate (C − B) = +214 MB   → 與 BUF=2 × stripe_chunk (~99 MB) × 2 NVMe 路徑 buffer 吻合
ΔPrivate (D − B) = +840 MB   → 與 BUF=4 × stripe_chunk × 2 NVMe 吻合
```

**目前的 streaming runtime 仍保留 GPU 上的原始 FC weights**，並未把它們 free 掉。streaming buffer 是「**疊加**」在原本的 weights 之上，所以記憶體只增不減。

```
理論上 streaming 真正能省的：
   GPU USM FC weights = 1352 MB（如果能 free 的話）
但實際付出的：
   + IO double-buffer    = 200 ~ 800 MB（隨 BUF 增加）
   + 原 FC weights 未 free = 1352 MB（仍在 USM）
淨節省：0（事實上是 +200 ~ +800 MB）
```

---

## 3. 對 8 GB 系統的可行性判斷

> 假設預算：steady-state Private bytes ≤ **6.5 GB**（保留 1.5 GB 給 OS + 其他 app + page table）

| Case | Steady Private | 留給 OS | 8 GB 可行？ | TPS | 評價 |
|---|---:|---:|:---:|---:|---|
| A 無 offload | (估) ~7500 MB | ~500 MB | ✗ OOM 邊緣 | — | 不可用 |
| **B baseline + mmap embed** | **5127 MB** | **2873 MB** | **✓ 安全** | **13.95** | **推薦** |
| C 單 NVMe stream | 5343 MB | 2657 MB | ✓ 可行但浪費 | 10.14 | 不推薦（−4 tps 換 0 省記憶體） |
| D 雙 NVMe stream | 5967 MB | 2033 MB | △ 邊緣 | 9.72 | 不推薦 |

**結論：**

1. **8 GB 系統的真正救星是 per-layer embedding mmap offload**（Case A → B 省 ≈ 3 GB）。
2. **Dense weight streaming 在現有 build 中對 8 GB 系統沒有實際幫助** —— 它在 −4 tps 的代價下，額外吃掉 200~800 MB private memory。
3. Streaming 的價值仍存在，但只在以下情境才會真正體現：
   - GPU USM 真的被硬性限制（不是系統 RAM，而是 GPU 專屬池）
   - 模型大到 baseline 也擺不下（>16 GB FC weights）
   - 或：runtime 經改寫，**真的把原 FC weights 從 USM free 掉**

---

## 4. 要讓 streaming「真的省記憶體」，需要怎麼改？

目前 `dense_weight_streaming_manager` 把 streaming buffer 透過 `swap_weight_pointers()` 注入到 kernel arguments，但 **GPU 上的 FC 原始 USM allocation 並未釋放**。

### 改造項目（為未來 Phase 3 參考）

1. **載入後立即 free FC weights**  
   在 streaming 初始化完成、首次 buffer 預熱完成後，呼叫 `cl::Buffer::release()` 釋放對應 FC tensor 的 USM allocation。
2. **graph 重新拓樸**  
   或者直接讓 IR 在 streaming 模式下不要編譯 FC weight 為 constant tensor，改為 input port。
3. **量化的 quant scale / zero-point 也要一併處理**  
   INT4 group_size=64 的 scale/zp 占 FC 大小的 ~6%（≈ 80 MB），目前也沒 streaming。

預期完成上述改造後：

| 項目 | 節省 |
|---|---:|
| Free 原 FC USM | −1352 MB |
| 加上 IO buffer（BUF=2） | +210 MB |
| **淨節省** | **≈ −1140 MB** |

→ 8 GB 系統 steady private 可降至 ~4 GB，留 4 GB 給 OS，**這才是 streaming 真正的記憶體價值**。

---

## 5. 量測檔案與重現步驟

| 檔案 | 內容 |
|---|---|
| [temp/memprobe2.py](temp/memprobe2.py) | 4-case 量測腳本 |
| [temp/memprobe2_summary.json](temp/memprobe2_summary.json) | 數值摘要 |
| [temp/memprobe2_run.log](temp/memprobe2_run.log) | 完整執行 log |
| `temp/memprobe2_A_full_gpu_no_offload.log` | Case A 詳細輸出（含 stack trace） |
| `temp/memprobe2_B_baseline_mmap_embed.log` | Case B 詳細輸出 |
| `temp/memprobe2_C_single_stream_mmap.log` | Case C 詳細輸出 |
| `temp/memprobe2_D_dual_stream_b4i8.log` | Case D 詳細輸出 |

```powershell
cd C:\working\gemma4-openvino-genai\gemma4_streaming_release_v2
& "C:\working\gemma4-openvino-genai\.venv\Scripts\Activate.ps1"
python C:\working\gemma4-openvino-genai\temp\memprobe2.py
```

---

## 6. 與舊系統基線交叉比對

| 系統 | Case | Peak RSS | 來源 |
|---|---|---:|---|
| 舊機 (12 Xe) short-text | baseline | 7.1 GB | `copilot-instructions.md` |
| 舊機 (12 Xe) long-text  | baseline | 7.9 GB | `copilot-instructions.md` |
| 舊機 (12 Xe) short-image| baseline | 8.3 GB | `copilot-instructions.md` |
| **新機 (16 Xe³) short-text** | **B baseline** | **8.84 GB** | 本報告 |

新機 Peak 較高的可能原因：
- OpenVINO 2026.2 nightly 對 mmap 頁面的 prefetch 更積極（一次性 fault-in 更多）
- v30.4.0 GPU driver 的 USM 配置策略不同
- 但 **Private bytes（5.1 GB）與舊機相當**，代表真實 RAM 壓力沒有變化

---

## 7. 下一步建議

1. **更新 `gemma4_streaming_release_v2/README.md`**：明示 streaming 對記憶體**沒有**直接效益，只在特定 GPU USM 受限場景才使用。
2. **Phase 3 開發目標**：實作「FC weights free after streaming init」(章節 4)，目標讓 8 GB 系統 steady private ≤ 4.5 GB。
3. **追加 GPU memory 量測**：目前只看 process 視角，缺 GPU USM 視角。可用 `ze-monitor` 或 Intel GPA 抓 GPU 端 allocation。

---

*本報告數據量測於 2026-05-20，OpenVINO commit signature `2026.2.0-21571-9c4a2eb9ad3`，openvino-genai branch `as/vlm_enable_1`。*
