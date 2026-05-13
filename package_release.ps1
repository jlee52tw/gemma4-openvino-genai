# ============================================================
# package_release.ps1 — Create self-contained Gemma4 release
# ============================================================
#
# Creates a release directory with:
#   bin/          run_gemma4.exe + openvino_genai.dll + openvino_tokenizers.dll
#   runtime/      OpenVINO core DLLs (GPU + CPU + IR frontend + TBB)
#   model/        Model IR files + revised per-layer embedding
#   scripts/      Python test scripts
#   setupvars.bat Environment setup (adds bin/ and runtime/ to PATH)
#
# Usage:
#   .\package_release.ps1
#   .\package_release.ps1 -OutputDir "C:\release\gemma4-release"
#   .\package_release.ps1 -Zip   # also create .zip archive
#
# On target system:
#   call setupvars.bat
#   bin\run_gemma4.exe --model-dir model --prompt "Hello" --no-mmap
# ============================================================

param(
    [string]$OutputDir = "C:\working\gemma4-openvino\gemma4-release",
    [switch]$Zip
)

$ErrorActionPreference = "Stop"

# ── Source paths ─────────────────────────────────────────────
$ModelDir    = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov"
$ExeDir      = "C:\working\gemma4-openvino\gemma4-openvino-genai\cpp\build\Release"
$GenAIBuild  = "C:\working\gemma4-openvino\openvino_genai_src\build\openvino_genai"
$OVLibs      = "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs"
$ScriptDir   = "C:\working\gemma4-openvino\gemma4-openvino-genai"

# ── Validate sources ────────────────────────────────────────
$missing = @()
if (-not (Test-Path "$ExeDir\run_gemma4.exe"))        { $missing += "run_gemma4.exe" }
if (-not (Test-Path "$GenAiBuild\openvino_genai.dll")) { $missing += "openvino_genai.dll" }
if (-not (Test-Path "$ModelDir\openvino_language_model.bin")) { $missing += "model files" }
if ($missing.Count -gt 0) {
    Write-Error "Missing: $($missing -join ', '). Build first."
}

# ── Clean and create output structure ────────────────────────
if (Test-Path $OutputDir) { Remove-Item $OutputDir -Recurse -Force }
$dirs = @("bin", "runtime", "model", "scripts", "streaming")
foreach ($d in $dirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $OutputDir $d) | Out-Null
}
Write-Host "[1/7] Output directory: $OutputDir" -ForegroundColor Cyan

# ── bin/ : exe + genai DLLs ──────────────────────────────────
Write-Host "[2/7] Copying executables and GenAI DLLs ..." -ForegroundColor Cyan
Copy-Item "$ExeDir\run_gemma4.exe" (Join-Path $OutputDir "bin") -Force
Copy-Item "$GenAiBuild\openvino_genai.dll" (Join-Path $OutputDir "bin") -Force
Copy-Item "$GenAiBuild\openvino_tokenizers.dll" (Join-Path $OutputDir "bin") -Force

# ── runtime/ : OpenVINO core DLLs (GPU essential only) ───────
Write-Host "[3/7] Copying OpenVINO runtime DLLs ..." -ForegroundColor Cyan
$ovDlls = @(
    "openvino.dll",
    "openvino_c.dll",
    "openvino_intel_gpu_plugin.dll",
    "openvino_intel_cpu_plugin.dll",
    "openvino_ir_frontend.dll",
    "openvino_auto_plugin.dll",
    "openvino_auto_batch_plugin.dll",
    "openvino_hetero_plugin.dll",
    "tbb12.dll",
    "tbbbind_2_5.dll",
    "tbbmalloc.dll",
    "tbbmalloc_proxy.dll"
)
foreach ($dll in $ovDlls) {
    $src = Join-Path $OVLibs $dll
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $OutputDir "runtime") -Force
    } else {
        Write-Warning "  Skipped (not found): $dll"
    }
}

# ── model/ : IR files + revised per-layer embedding ──────────
Write-Host "[4/7] Copying model files ..." -ForegroundColor Cyan
$modelFiles = @(
    # Language model (decoder)
    "openvino_language_model.xml",
    "openvino_language_model.bin",
    # Token embeddings
    "openvino_text_embeddings_model.xml",
    "openvino_text_embeddings_model.bin",
    # Per-layer embedding (our revised/repacked binary)
    "openvino_text_embeddings_per_layer_model_revised.bin",
    # Vision encoder
    "openvino_vision_embeddings_model.xml",
    "openvino_vision_embeddings_model.bin",
    # Tokenizer / detokenizer
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
    # Config files
    "config.json",
    "generation_config.json",
    "openvino_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja"
)
foreach ($f in $modelFiles) {
    $src = Join-Path $ModelDir $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $OutputDir "model") -Force
    } else {
        Write-Warning "  Skipped (not found): $f"
    }
}

# ── streaming/ : Dense weight streaming binary + metadata ────
Write-Host "[5/7] Copying dense weight streaming files ..." -ForegroundColor Cyan
$StreamingDir = "C:\working\gemma4-openvino\gemma4-openvino-genai\temp"
$streamingFiles = @("dense_weights_streaming.bin", "dense_weights_streaming.json")
foreach ($f in $streamingFiles) {
    $src = Join-Path $StreamingDir $f
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $OutputDir "streaming") -Force
    } else {
        Write-Warning "  Skipped (not found): $f — run pack_dense_weights.py first"
    }
}

# ── scripts/ : Python test scripts + packing tool ────────────
Write-Host "[6/7] Copying scripts ..." -ForegroundColor Cyan
$scripts = @("run_gemma4.py", "benchmark.py", "pack_dense_weights.py")
foreach ($s in $scripts) {
    $src = Join-Path $ScriptDir $s
    if (Test-Path $src) {
        Copy-Item $src (Join-Path $OutputDir "scripts") -Force
    }
}

# ── setupvars.bat + README ───────────────────────────────────
Write-Host "[7/7] Creating setupvars.bat and README ..." -ForegroundColor Cyan

$setupvarsBat = @'
@echo off
rem ============================================================
rem  Gemma4 OpenVINO GenAI — Environment Setup
rem  Usage:  call setupvars.bat
rem ============================================================
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "BIN_DIR=%SCRIPT_DIR%\bin"
set "RT_DIR=%SCRIPT_DIR%\runtime"

if not exist "%BIN_DIR%\run_gemma4.exe" (
    echo [ERROR] run_gemma4.exe not found in %BIN_DIR%
    goto :eof
)

echo %PATH% | findstr /C:"%BIN_DIR%" >nul 2>&1 || set "PATH=%BIN_DIR%;%PATH%"
echo %PATH% | findstr /C:"%RT_DIR%" >nul 2>&1 || set "PATH=%RT_DIR%;%PATH%"

echo.
echo [setupvars] Gemma4 OpenVINO GenAI environment initialized
echo   Binaries : %BIN_DIR%
echo   Runtime  : %RT_DIR%
echo.
echo Usage (normal):
echo   run_gemma4.exe --model-dir model --prompt "Hello" --no-mmap --show-memory
echo.
echo Usage (dense weight streaming - for 8GB systems):
echo   set OV_DENSE_STREAM_WEIGHTS=%SCRIPT_DIR%\streaming\dense_weights_streaming.bin
echo   run_gemma4.exe --model-dir model --prompt "Hello" --no-mmap --show-memory
echo.
'@

$readme = @"
# Gemma4 OpenVINO GenAI Release

## Quick Start (C++ exe)

```cmd
call setupvars.bat
run_gemma4.exe --model-dir model --prompt "Explain quantum computing." --no-mmap --show-memory
run_gemma4.exe --model-dir model --prompt "Describe this image." --image photo.jpg --no-mmap
```

## Quick Start (Python)

```cmd
call setupvars.bat
pip install openvino openvino-genai
python scripts\run_gemma4.py --model-dir model --prompt "Hello" --no-mmap --show-memory
```

## Directory Layout

```
gemma4-release/
├── bin/                     Executables and GenAI DLLs
│   ├── run_gemma4.exe       C++ inference tool
│   ├── openvino_genai.dll   Modified GenAI runtime
│   └── openvino_tokenizers.dll
├── runtime/                 OpenVINO core DLLs
│   ├── openvino.dll
│   ├── openvino_intel_gpu_plugin.dll
│   └── ...
├── model/                   Gemma4 E4B INT4 model
│   ├── openvino_language_model.xml/.bin
│   ├── openvino_text_embeddings_per_layer_model_revised.bin
│   └── ...
├── streaming/               Dense weight streaming (for 8GB systems)
│   ├── dense_weights_streaming.bin   (~1.55 GB, decoder FC weights)
│   └── dense_weights_streaming.json  (metadata)
├── scripts/                 Python scripts & tools
│   ├── run_gemma4.py
│   ├── benchmark.py
│   └── pack_dense_weights.py
├── setupvars.bat            Environment setup
└── README.md
```

## Dense Weight Streaming (for memory-constrained systems)

Enable dense weight streaming for systems with limited RAM (e.g. 8 GB):

```cmd
call setupvars.bat
set OV_DENSE_STREAM_WEIGHTS=%CD%\streaming\dense_weights_streaming.bin
run_gemma4.exe --model-dir model --prompt "Hello" --no-mmap --show-memory
```

This streams decoder FC weights from NVMe instead of loading them all into GPU memory.
Performance: ~6.78 tok/s (vs 24 tok/s baseline). Memory savings: ~845 MB.

## Command-Line Options

| Option | Description |
|---|---|
| ``--model-dir <path>`` | Model directory (required) |
| ``--device <CPU\|GPU>`` | Inference device (default: GPU) |
| ``--prompt <text>`` | Text prompt |
| ``--prompt-file <path>`` | Read prompt from file |
| ``--image <path>`` | Image for multimodal inference |
| ``--max-new-tokens <N>`` | Max output tokens (default: 256) |
| ``--no-mmap`` | Disable mmap loading (recommended for iGPU) |
| ``--show-memory`` | Print RSS/peak memory stats |

## Performance (Intel Panther Lake iGPU, 16 GB LPDDR5)

| Scenario | TPS | TPOT (ms) | TTFT (s) | RSS (GB) |
|---|---:|---:|---:|---:|
| short-text (25 in, 256 out) | **24.1** | 41.4 | 0.74 | 4.3 |

Per-layer embedding offload saves ~2.8 GB GPU memory vs baseline.

## System Requirements

- Windows 10/11 (x64)
- Intel GPU with OpenCL support (iGPU or dGPU)
- 16 GB RAM recommended (8 GB minimum with --no-mmap)
- OpenVINO 2026.2 runtime (included in runtime/)
"@

Set-Content -Path (Join-Path $OutputDir "setupvars.bat") -Value $setupvarsBat -Encoding ASCII
Set-Content -Path (Join-Path $OutputDir "README.md") -Value $readme -Encoding UTF8

# ── Summary ──────────────────────────────────────────────────
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Release package created!" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green

# Calculate sizes
$binSize = (Get-ChildItem (Join-Path $OutputDir "bin") -Recurse | Measure-Object Length -Sum).Sum / 1MB
$rtSize  = (Get-ChildItem (Join-Path $OutputDir "runtime") -Recurse | Measure-Object Length -Sum).Sum / 1MB
$mdlSize = (Get-ChildItem (Join-Path $OutputDir "model") -Recurse | Measure-Object Length -Sum).Sum / 1MB
$stmSize = (Get-ChildItem (Join-Path $OutputDir "streaming") -Recurse -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum / 1MB
$totalSize = (Get-ChildItem $OutputDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB

Write-Host ""
Write-Host "  Directory: $OutputDir"
Write-Host "  bin/       : $([math]::Round($binSize, 1)) MB"
Write-Host "  runtime/   : $([math]::Round($rtSize, 1)) MB"
Write-Host "  model/     : $([math]::Round($mdlSize, 1)) MB"
Write-Host "  streaming/ : $([math]::Round($stmSize, 1)) MB"
Write-Host "  TOTAL      : $([math]::Round($totalSize, 1)) MB"
Write-Host ""

# List all files
Get-ChildItem $OutputDir -Recurse -File | ForEach-Object {
    $rel = $_.FullName.Substring($OutputDir.Length + 1)
    $mb = [math]::Round($_.Length / 1MB, 1)
    Write-Host "  $rel  ($mb MB)"
}

# ── Optional ZIP ─────────────────────────────────────────────
if ($Zip) {
    $zipPath = "$OutputDir.zip"
    Write-Host ""
    Write-Host "Creating ZIP archive: $zipPath ..." -ForegroundColor Cyan
    Write-Host "  (This may take several minutes for ~6 GB of data)"
    Compress-Archive -Path "$OutputDir\*" -DestinationPath $zipPath -Force
    $zipMB = [math]::Round((Get-Item $zipPath).Length / 1MB, 1)
    Write-Host "  ZIP created: $zipMB MB" -ForegroundColor Green
}

Write-Host ""
Write-Host "To use on another system:" -ForegroundColor Yellow
Write-Host "  1. Copy the entire folder (or extract .zip)"
Write-Host "  2. call setupvars.bat"
Write-Host "  3. run_gemma4.exe --model-dir model --prompt `"Hello`" --no-mmap"
Write-Host ""
