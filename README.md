# Gemma 4 with OpenVINO GenAI — VLMPipeline Example

> **Status — Work in Progress**
> Gemma 4 support in OpenVINO GenAI is currently under active optimization.
> The results below reflect the current state of development; performance is
> expected to improve in future releases.

This repository demonstrates how to run **Google Gemma 4** vision-language
models using the [`openvino.genai`](https://github.com/openvinotoolkit/openvino.genai)
`VLMPipeline` API.  It includes:

| File | Description |
|------|-------------|
| `run_gemma4.py` | Simple inference script — text and image+text |
| `benchmark.py` | Throughput / TTFT / memory benchmark |
| `requirements.txt` | Runtime dependencies (inference only) |
| `requirements-export.txt` | Dependencies for model conversion |

---

## Supported Models

| HuggingFace ID | Type | Total Params | Active Params | Notes |
|---|---|---:|---:|---|
| `google/gemma-4-E2B-it` | Dense | ~2 B | ~2 B | Fastest, smallest |
| `google/gemma-4-E4B-it` | Dense | ~4 B | ~4 B | |
| `google/gemma-4-26B-A4B-it` | MoE | ~25.2 B | ~3.8 B | 128 experts, top-8 routing |
| `google/gemma-4-31B-it` | Dense | ~31 B | ~31 B | Largest dense variant |

All models support **text**, **image+text**, **audio+text**, and
**video+text** inputs (multimodal).

---

## 1. Environment Setup

### 1.1 Create a Python virtual environment

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 1.2 Install runtime dependencies

```bash
pip install -r requirements.txt
```

### 1.3 Build & install openvino-genai from source (PR #3644)

Gemma 4 VLMPipeline support is provided by
[PR #3644](https://github.com/openvinotoolkit/openvino.genai/pull/3644).
Until it is merged, you need to build from source:

```bash
git clone --recursive --branch as/vlm_enable_1 \
    https://github.com/as-suvorov/openvino.genai.git openvino_genai_src

pip install ./openvino_genai_src \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly \
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release
```

> **Build requirements:** C++ compiler (MSVC 2022 / GCC 11+ / Clang 14+)
> and CMake ≥ 3.23.

After installation, verify:

```python
import openvino_genai
print(openvino_genai.__version__)            # should contain "as/vlm_enable_1"
print(hasattr(openvino_genai, "VLMPipeline"))  # True
```

---

## 2. Model Conversion (HuggingFace → OpenVINO IR)

### 2.1 Install export dependencies

```bash
pip install -r requirements-export.txt
```

### 2.2 Export with `optimum-cli`

Use [`optimum-cli`](https://huggingface.co/docs/optimum/intel/inference) to
convert the HuggingFace model to OpenVINO IR format with INT4 weight
compression:

```bash
# Gemma 4 E2B (smallest — good for quick testing)
optimum-cli export openvino \
    --model google/gemma-4-E2B-it \
    --weight-format int4 \
    --group-size 64 \
    --ratio 1.0 \
    gemma-4-E2B-it-ov

# Gemma 4 E4B
optimum-cli export openvino \
    --model google/gemma-4-E4B-it \
    --weight-format int4 \
    --group-size 64 \
    --ratio 1.0 \
    gemma-4-E4B-it-ov

# Gemma 4 26B-A4B (MoE — routes excluded from quantization)
optimum-cli export openvino \
    --model google/gemma-4-26B-A4B-it \
    --weight-format int4 \
    --group-size 64 \
    --ratio 1.0 \
    gemma-4-26B-A4B-it-ov

# Gemma 4 31B (largest dense)
optimum-cli export openvino \
    --model google/gemma-4-31B-it \
    --weight-format int4 \
    --group-size 64 \
    --ratio 1.0 \
    gemma-4-31B-it-ov
```

Each command produces a directory containing:

```
gemma-4-*-it-ov/
├── config.json
├── openvino_language_model.xml / .bin
├── openvino_vision_embeddings_model.xml / .bin
├── openvino_text_embeddings_model.xml / .bin
├── openvino_text_embeddings_per_layer_model.xml / .bin
├── openvino_tokenizer.xml / .bin
├── openvino_detokenizer.xml / .bin
├── openvino_config.json
├── tokenizer.json, tokenizer_config.json
├── chat_template.jinja
├── generation_config.json
└── preprocessor_config.json, processor_config.json
```

> **Note:** Conversion requires significant RAM.  The 31B model may need
> 64 GB+ of system memory during export.

---

## 3. Running Inference

### Text-only

```bash
python run_gemma4.py \
    --model-dir ./gemma-4-E2B-it-ov \
    --device GPU \
    --prompt "Explain quantum computing in simple terms."
```

### Image + Text

```bash
python run_gemma4.py \
    --model-dir ./gemma-4-E2B-it-ov \
    --device GPU \
    --prompt "Describe this image in detail." \
    --image photo.jpg
```

### On CPU

```bash
python run_gemma4.py \
    --model-dir ./gemma-4-E2B-it-ov \
    --device CPU \
    --prompt "Hello, what can you do?"
```

---

## 4. Running the Benchmark

```bash
# Single model — quick test
python benchmark.py \
    --model-dir ./gemma-4-E2B-it-ov \
    --device GPU \
    --max-new-tokens 128 \
    --warmup 1 --runs 3

# Multiple models — full sweep
python benchmark.py \
    --model-dir ./gemma-4-E2B-it-ov \
                ./gemma-4-E4B-it-ov \
                ./gemma-4-26B-A4B-it-ov \
                ./gemma-4-31B-it-ov \
    --device GPU \
    --max-new-tokens 128 \
    --warmup 1 --runs 3 \
    --output-csv results.csv
```

The benchmark tests three prompt types per model:

| Prompt Type | Description |
|---|---|
| `short-text` | ~7-token text prompt |
| `long-text` | ~1 024-token text prompt (prefill stress test) |
| `short-image` | short text + synthetic 336×336 image |

---

## 5. Benchmark Results (Current Progress)

**Hardware:** Intel iGPU (Panther Lake, 12 Xe EUs)
**Software:** OpenVINO 2026.2.0.dev nightly + openvino.genai PR #3644
**Config:** INT4 weights, `max_new_tokens=128`, 1 warmup + 3 measured runs

| Model | Prompt | Avg Tok/s | Avg TTFT (s) | RSS (GB) |
|---|---|---:|---:|---:|
| **gemma-4-E2B-it** | short-text | 34.7 | 0.24 | 5.1 |
| | long-text (1024 in) | 26.5 | 0.58 | 5.7 |
| | short-image | 29.5 | 0.42 | 6.1 |
| **gemma-4-E4B-it** | short-text | 24.0 | 0.30 | 7.1 |
| | long-text (1024 in) | 17.4 | 0.88 | 7.9 |
| | short-image | 19.7 | 0.54 | 8.3 |
| **gemma-4-26B-A4B-it** | short-text | 5.80 | 0.73 | 15.7 |
| | long-text (1024 in) | 4.36 | 4.69 | 16.2 |
| | short-image | 5.32 | 1.97 | 16.6 |
| **gemma-4-31B-it** | short-text | 5.06 | 0.67 | 19.4 |
| | long-text (1024 in) | 3.35 | 3.86 | 20.4 |
| | short-image | 4.55 | 1.73 | 20.3 |

> These numbers represent **current progress** — Gemma 4 support is under
> active optimization and performance will improve in future releases.

---

## Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `openvino` | ≥ 2026.2.0.dev (nightly) | OpenVINO runtime |
| `openvino-genai` | built from PR #3644 | VLMPipeline with Gemma 4 support |
| `openvino-tokenizers` | ≥ 2026.2.0.dev (nightly) | Tokenizer support |
| `numpy` | ≥ 1.24 | Array operations |
| `pillow` | ≥ 10.0 | Image loading |
| `psutil` | ≥ 5.9 | Memory monitoring (benchmark) |
| `optimum-intel` | ≥ 2.1.0.dev | Model conversion only |
| `transformers` | ≥ 4.50 | Model conversion only |

---

## License

The example scripts in this repository are provided under the
[Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0).

Gemma 4 model weights are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).

---

## References

- [Gemma 4 on HuggingFace](https://huggingface.co/collections/google/gemma-4-67ab86bcb29e840132e11f4b)
- [OpenVINO GenAI — VLMPipeline docs](https://openvinotoolkit.github.io/openvino.genai/docs/use-cases/image-processing/)
- [PR #3644 — Gemma 4 VLMPipeline support](https://github.com/openvinotoolkit/openvino.genai/pull/3644)
- [Optimum Intel — Model export](https://huggingface.co/docs/optimum/intel/inference)
