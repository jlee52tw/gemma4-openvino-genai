# OCR Multimodal Test — Gemma 4 E4B on Intel iGPU

**Model:** gemma-4-E4B-it (INT4, OpenVINO IR)
**Device:** Intel iGPU (Panther Lake, 12 Xe EUs)
**Image:** [gemma4-ocr-multimodal.png](gemma4-ocr-multimodal.png) (5.7 MB)

## Prompt

```
Read all the text in this image. Extract every word exactly as shown.
```

## Command

```powershell
python run_gemma4.py `
    --model-dir ./gemma-4-E4B-it-ov `
    --device GPU `
    --prompt "Read all the text in this image. Extract every word exactly as shown." `
    --image gemma4-ocr-multimodal.png `
    --max-new-tokens 1024
```

## Response

```
PROJECT ALPHA vs. PROJECT BETA: Q2 2026PERFORMANCE

AI EVALUATION IMAGE:
TREND ANALYSIS & OCR.
PROMPT: COMPARE TRENDS
BETWEEN PROJECTS AND
READ ALL TEXT DETAILS.

Units Sold
35,0k
30,0k
25,0k
20,0k
15,0k
10,0k
5,0k
Jan 2026
Feb
Mar
Apr
May
Jun 2026
Jan: 10k
Jan: 20k
18, 00k
Mar: 25, 000
May: 12k
Apr: Dip: 18k
May: 30k
25: 000
23: 8k
25: 000
21, 000
Jun: 35k
Project Alpha
Project Beta

REGIONAL DATA BREAKDOWN (OCR TEST)
REGIONAL PERFORMANCE Q2 2026

| Region | Growth% | Growth% |
|---|---|---|
| APAC | +15% | +15% |
| EMEA | -3% | -3% |
| AMER | +8% | +8% |

*Note: Data sourced from internal
reporting systems. Figures for
APAC are preliminary w/roww.
Figures for APAC are preliminary
and may be subject to
revision upon final review.
AMER performance affected by
supply chain constraints.
Font variations: Arial, Times New
Roman, and monospace. *

PRELIMINARY OUTLOOK
Project Gamma
July
```

## Performance Metrics

```
  Input tokens          : 291
  Generated tokens      : 367

  Load time             : 10043.00 ms
  TTFT                  : 2657.83 ms
  TPOT                  : 41.23 ms

  Throughput            : 24.25 tok/s

  Generate duration     : 17750.46 ms
  Inference duration    : 15523.85 ms
  Tokenization duration : 79.61 ms
  Detokenization dur.   : 0.36 ms
```

---

## Test 2: Chinese Chart Reasoning

### Prompt

```
根據圖表趨勢，如果 Project Alpha 保持目前的增長率，預測 7 月份的 Units Sold 大約是多少？並說明理由。
```

### Command

```powershell
python run_gemma4.py `
    --model-dir ./gemma-4-E4B-it-ov `
    --device GPU `
    --prompt "根據圖表趨勢，如果 Project Alpha 保持目前的增長率，預測 7 月份的 Units Sold 大約是多少？並說明理由。" `
    --image gemma4-ocr-multimodal.png `
    --max-new-tokens 1024
```

### Response

根據圖表趨勢，如果 Project Alpha 保持目前的增長率，預測 7 月份的 Units Sold 大約會在 **35,000 到 40,000 之間**。

以下是推斷的理由：

#### 1. 分析 Project Alpha 的趨勢

* **數據點觀察：**
    * 1 月份：10k
    * 3 月份：23k (約在 18k 到 25k 之間)
    * 5 月份：30k (圖中標註的 May: 30k)
    * 6 月份：35k (圖中標註的 Jun: 35k)
* **增長趨勢：** Project Alpha 的增長是穩健且持續的，尤其是在進入 5 月和 6 月後，增長速度似乎正在放緩，但仍維持在較高的水平。
* **最近的增長率（從 5 月到 6 月）：** 增長了 5,000 個單位（從 30k 到 35k）。

#### 2. 預測 7 月份的銷售量

如果我們假設進入 7 月份時，Project Alpha 的增長率（或至少是月度增長量）能維持在或略高於最近的增長趨勢：

* **保守預測（維持 5 月到 6 月的增長量）：** 35k + 5k = 40k
* **樂觀預測（維持較早期的強勁增長或進入新階段）：** 由於 6 月份已經達到了 35k，如果它能維持類似於 3 月份到 5 月份的強勁勢頭，增長量可能會超過 5k。

**結論：**

基於圖表顯示的強勁勢頭和 6 月份的 35k 這個高點，預測 7 月份的銷售量會是一個**小於或等於 40k 的數值**，並且極有可能會接近或超過 35k 的水平。因此，**35,000 到 40,000** 是最合理的預測範圍。

---
**備註：**

這個預測是基於**線性外推法**（即假設趨勢線會繼續向上延伸）。在實際商業情境中，由於市場飽和、競爭或產品生命週期等因素，增長率可能會放緩（進入平台期），但從圖表上看，它仍處於上升階段。

### Performance Metrics

```
  Input tokens          : 309
  Generated tokens      : 578

  Load time             : 10862.00 ms
  TTFT                  : 2733.00 ms
  TPOT                  : 43.49 ms

  Throughput            : 22.99 tok/s

  Generate duration     : 27828.34 ms
  Inference duration    : 25360.41 ms
  Tokenization duration : 76.67 ms
  Detokenization dur.   : 0.25 ms
```
