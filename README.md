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
| `benchmark_kpi.py` | KPI benchmark (prefill, output TPS, peak memory) |
| `cpp/run_gemma4.cpp` | C++ inference — same features as `run_gemma4.py` |
| `create_release.ps1` | Builds a self-contained runtime folder with `setupvars` |
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

Gemma 4 VLMPipeline support is provided by
[PR #3644](https://github.com/openvinotoolkit/openvino.genai/pull/3644),
which has **not yet been merged** into the main openvino.genai package.
This means `openvino-genai` from PyPI **does not** include Gemma 4 support —
you must **build from source**.

The setup flow is:

```
1.1  Clone this repo
1.2  Create Python virtual environment
1.3  Install OpenVINO nightly runtime + Python dependencies
1.4  Clone openvino.genai PR #3644 source
1.5  Build & install openvino-genai from source (C++ compilation)
1.6  Verify the complete environment
```

### 1.1 Clone this repository

```powershell
git clone https://github.com/jlee52tw/gemma4-openvino-genai.git
cd gemma4-openvino-genai
```

### 1.2 Create a Python virtual environment

```powershell
python -m venv .venv
```

Activate it (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Or from `cmd`:

```cmd
.venv\Scripts\activate.bat
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip
```

### 1.3 Install OpenVINO nightly runtime & Python dependencies

The `requirements.txt` installs the **OpenVINO 2026.2.0.dev nightly** runtime
(required for Gemma 4), plus `numpy`, `pillow`, and `psutil`.

> **Important:** This step installs the OpenVINO *runtime* — **not**
> `openvino-genai`. The GenAI package is built from source in the next steps.

```powershell
pip install -r requirements.txt `
    --trusted-host storage.openvinotoolkit.org
```

Verify OpenVINO is installed:

```powershell
python -c "import openvino; print('OpenVINO', openvino.__version__)"
# Expected output: OpenVINO 2026.2.0.dev...
```

### 1.4 Clone the openvino.genai source (PR #3644)

Clone the branch that contains Gemma 4 VLMPipeline support:

```powershell
git clone --recursive --branch as/vlm_enable_1 `
    https://github.com/as-suvorov/openvino.genai.git openvino_genai_src
```

This creates an `openvino_genai_src/` directory with the C++ source code
and all submodules (including `openvino_tokenizers`).

### 1.5 Build & install openvino-genai from source

#### Prerequisites

A C++ compiler and CMake are required to build the native code.

| Requirement | Details |
|---|---|
| **Visual Studio 2022** | [Community edition](https://visualstudio.microsoft.com/) with the **"Desktop development with C++"** workload (includes MSVC & CMake) |
| **CMake** | ≥ 3.23 (bundled with VS 2022) |
| **Python** | 3.10 – 3.12 |
| **OpenVINO** | ≥ 2026.2.0.dev nightly (installed in step 1.3) |

Make sure to run the build from a terminal that has the VS environment
loaded — e.g. **"Developer PowerShell for VS 2022"**, or activate
manually with:

```powershell
& "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\Launch-VsDevShell.ps1"
```

#### Build command

`pip install .` runs CMake under the hood to compile the C++ core, Python
bindings, and tokenizer libraries. This typically takes **5–15 minutes**
depending on your machine.

```powershell
pip install ./openvino_genai_src `
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly `
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release `
    --trusted-host storage.openvinotoolkit.org
```

The build installs **two** packages:

| Package | Description |
|---|---|
| `openvino-genai` | GenAI pipeline APIs (LLMPipeline, VLMPipeline, etc.) |
| `openvino-tokenizers` | Tokenizer / detokenizer support |

> **Tip:** Append `-v` and log the output for easier debugging when the
> build fails:
>
> ```powershell
> pip install ./openvino_genai_src `
>     --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly `
>     --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release `
>     --trusted-host storage.openvinotoolkit.org `
>     -v 2>&1 | Tee-Object -FilePath build.log
> ```

#### Rebuilding after a source update

```powershell
cd openvino_genai_src
git pull --recurse-submodules
cd ..

pip install ./openvino_genai_src --no-build-isolation --force-reinstall `
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly `
    --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/pre-release `
    --trusted-host storage.openvinotoolkit.org
```

> If the rebuild fails due to stale CMake cache, delete the build
> directory and retry:
> ```powershell
> Remove-Item -Recurse -Force openvino_genai_src\.py-build-cmake_cache
> ```

### 1.6 Verify the complete environment

Run the following script to confirm that all required packages are
installed and Gemma 4 VLMPipeline support is available:

```powershell
python -c "
import openvino as ov
import openvino_genai as genai
import numpy, PIL, psutil

print('OpenVINO          :', ov.__version__)
print('openvino-genai    :', genai.__version__)
print('numpy             :', numpy.__version__)
print('Pillow            :', PIL.__version__)
print('psutil            :', psutil.__version__)
print()
print('VLMPipeline available :', hasattr(genai, 'VLMPipeline'))
print('Available devices     :', ov.Core().available_devices)
"
```

Expected output (versions may differ):

```
OpenVINO          : 2026.2.0.dev20250411
openvino-genai    : 2026.2.0.0
numpy             : 2.4.4
Pillow            : 12.2.0
psutil            : 7.2.2

VLMPipeline available : True
Available devices     : ['CPU', 'GPU']
```

> If `GPU` does not appear in the device list, ensure the
> [Intel GPU driver](https://www.intel.com/content/www/us/en/download/726609/intel-arc-iris-xe-graphics-whql-windows.html)
> is installed.

Once the verification passes, your environment is ready.
Proceed to **Section 2** to convert models, or **Section 3** to run
inference with pre-converted models.

---

## 2. Model Conversion (HuggingFace → OpenVINO IR)

### 2.1 Install export dependencies

```powershell
pip install -r requirements-export.txt
```

### 2.2 Export with `optimum-cli`

Use [`optimum-cli`](https://huggingface.co/docs/optimum/intel/inference) to
convert the HuggingFace model to OpenVINO IR format with INT4 weight
compression:

```powershell
# Gemma 4 E2B (smallest — good for quick testing)
optimum-cli export openvino `
    --model google/gemma-4-E2B-it `
    --weight-format int4 `
    --group-size 64 `
    --ratio 1.0 `
    gemma-4-E2B-it-ov

# Gemma 4 E4B
optimum-cli export openvino `
    --model google/gemma-4-E4B-it `
    --weight-format int4 `
    --group-size 64 `
    --ratio 1.0 `
    gemma-4-E4B-it-ov

# Gemma 4 26B-A4B (MoE — routes excluded from quantization)
optimum-cli export openvino `
    --model google/gemma-4-26B-A4B-it `
    --weight-format int4 `
    --group-size 64 `
    --ratio 1.0 `
    gemma-4-26B-A4B-it-ov

# Gemma 4 31B (largest dense)
optimum-cli export openvino `
    --model google/gemma-4-31B-it `
    --weight-format int4 `
    --group-size 64 `
    --ratio 1.0 `
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

## 3. Running Inference (Python)

### Text-only

```powershell
python run_gemma4.py `
    --model-dir ./gemma-4-E2B-it-ov `
    --device GPU `
    --prompt "Explain quantum computing in simple terms."
```

### Image + Text

```powershell
python run_gemma4.py `
    --model-dir ./gemma-4-E2B-it-ov `
    --device GPU `
    --prompt "Describe this image in detail." `
    --image photo.jpg
```

> **Note:** `--device GPU` targets the Intel integrated GPU (iGPU).
> CPU is also supported (`--device CPU`) but significantly slower.

---

## 3.1 Running Inference (C++)

A C++ version of `run_gemma4.py` is provided in the `cpp/` subfolder.
It links against the `openvino::genai` shared library (`.dll`)
built from PR #3644 and prints the same performance metrics.

| File | Description |
|---|---|
| `cpp/run_gemma4.cpp` | Main source — arg parsing, VLMPipeline, PerfMetrics |
| `cpp/load_image.cpp` | Image loader using `stb_image.h` (downloaded by CMake) |
| `cpp/load_image.hpp` | Header for image loader |
| `cpp/CMakeLists.txt`  | CMake build script |

### 3.1.1 Create a release folder (one-time setup)

After building openvino-genai from source (step 1.5), run
`create_release.ps1` to assemble a self-contained **release folder**
with all DLLs, headers, CMake configs, and `setupvars` scripts:

```powershell
.\create_release.ps1 `
    -GenaiSrc  .\openvino_genai_src `
    -VenvDir   .\.venv `
    -InstallDir .\openvino_genai_release
```

This produces:

```
openvino_genai_release/
├── setupvars.ps1           # <-- source this before building / running
├── setupvars.bat           #     (cmd.exe version)
├── runtime/
│   ├── bin/intel64/Release/   openvino_genai.dll + openvino_tokenizers.dll
│   ├── lib/intel64/Release/   openvino_genai.lib
│   ├── include/               GenAI C++ headers
│   └── cmake/                 CMake configs (OpenVINO + GenAI)
└── openvino/
    ├── libs/                  OpenVINO core DLLs + .lib files
    └── include/               OpenVINO core C++ headers
```

### 3.1.2 Build the C++ sample

```powershell
# Initialize the environment (adds DLL dirs to PATH, sets CMake vars)
. .\openvino_genai_release\setupvars.ps1

cd cpp
cmake -B build
cmake --build build --config Release
```

The executable is produced at `cpp\build\Release\run_gemma4.exe`.

> **Note:** After sourcing `setupvars.ps1`, the environment variables
> `OpenVINO_DIR` and `OpenVINOGenAI_DIR` are set automatically — you
> do **not** need to pass `-DOpenVINO_DIR=...` manually.

### 3.1.3 Run

Always source `setupvars.ps1` in every new terminal session before
running the executable:

```powershell
# Initialize DLL paths (once per terminal)
. .\openvino_genai_release\setupvars.ps1

# Text-only
.\cpp\build\Release\run_gemma4.exe `
    --model-dir .\gemma-4-E2B-it-ov `
    --device GPU `
    --prompt "Explain quantum computing in simple terms."

# Image + text
.\cpp\build\Release\run_gemma4.exe `
    --model-dir .\gemma-4-E2B-it-ov `
    --device GPU `
    --prompt "Describe this image." `
    --image photo.jpg

# Prompt from file
.\cpp\build\Release\run_gemma4.exe `
    --model-dir .\gemma-4-E2B-it-ov `
    --prompt-file prompt.txt
```

The output includes streaming text generation followed by a performance
metrics summary:

```
------------------------------------------------------------
  OpenVINO GenAI - Performance Metrics
------------------------------------------------------------
  Input tokens          : 13
  Generated tokens      : 64

  Load time             : 7569.00 ms
  TTFT                  : 468.57 +/- 0.00 ms
  TPOT                  : 28.18 +/- 15.21 ms
  iPOT                  : 32.34 +/- 41.91 ms

  Throughput            : 35.48 +/- 19.15 tok/s

  Generate duration     : 2245.30 +/- 0.00 ms
  Inference duration    : 2069.54 +/- 0.00 ms
  Tokenization duration : 72.96 +/- 0.00 ms
  Detokenization dur.   : 0.26 +/- 0.00 ms
  Prepare embeddings    : 74.78 +/- 0.00 ms
------------------------------------------------------------
```

---

## 4. Running the Benchmark

```powershell
# Single model — quick test
python benchmark.py `
    --model-dir ./gemma-4-E2B-it-ov `
    --device GPU `
    --max-new-tokens 128 `
    --warmup 1 --runs 3

# Multiple models — full sweep
python benchmark.py `
    --model-dir ./gemma-4-E2B-it-ov `
                ./gemma-4-E4B-it-ov `
                ./gemma-4-26B-A4B-it-ov `
                ./gemma-4-31B-it-ov `
    --device GPU `
    --max-new-tokens 128 `
    --warmup 1 --runs 3 `
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
