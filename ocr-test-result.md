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
