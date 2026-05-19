# ============================================================
# create_release_v2.ps1 — Build a complete release .zip for
#   Dense Weight Streaming (Dual-NVMe) deployment
# ============================================================
#
# This packages everything needed to run Gemma-4-E4B-it with
# streaming on another system with 1 or 2 NVMe drives.
#
# Usage:
#   .\create_release_v2.ps1
#   .\create_release_v2.ps1 -SkipModelCache   # Skip 6GB model cache
#   .\create_release_v2.ps1 -SkipModelBin     # Skip large model .bin files
#
# Output: gemma4_streaming_release_v2.zip (~8-14 GB depending on options)
# ============================================================

param(
    [string]$OutputDir = ".\release_v2",
    [string]$ZipName = "gemma4_streaming_release_v2.zip",
    [switch]$SkipModelCache,
    [switch]$SkipModelBin,
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"
$ModelDir = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov"
$OvLibs = "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino\libs"
$GenAIDir = "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino_genai"
$TokenizersDir = "C:\Users\Local_Admin\AppData\Local\Programs\Python\Python312\Lib\site-packages\openvino_tokenizers"
$ScriptDir = "C:\working\gemma4-openvino\gemma4-openvino-genai"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Gemma4 Streaming Release Builder v2" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ── Clean and create output directory ─────────────────────────
if (Test-Path $OutputDir) {
    Write-Host "[0/7] Cleaning existing output: $OutputDir" -ForegroundColor Yellow
    Remove-Item $OutputDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir\dlls" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir\model" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir\streaming_nvme0" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutputDir\streaming_nvme1" | Out-Null

# ── 1. Copy scripts ──────────────────────────────────────────
Write-Host "[1/7] Copying scripts..." -ForegroundColor Cyan
$scripts = @(
    "run_gemma4.py",
    "benchmark.py",
    "pack_dense_weights_dual.py",
    "pack_dense_weights.py",
    "requirements.txt"
)
foreach ($s in $scripts) {
    $src = Join-Path $ScriptDir $s
    if (Test-Path $src) {
        Copy-Item $src -Destination $OutputDir
        Write-Host "  + $s"
    } else {
        Write-Host "  ! $s NOT FOUND (skipped)" -ForegroundColor Yellow
    }
}
# Copy release README
Copy-Item (Join-Path $ScriptDir "RELEASE_README.md") -Destination "$OutputDir\README.md"
Write-Host "  + README.md"

# ── 2. Copy DLLs ─────────────────────────────────────────────
Write-Host "[2/7] Copying OpenVINO DLLs..." -ForegroundColor Cyan

# Core OpenVINO DLLs (only what's needed for GPU inference)
$requiredDlls = @(
    "openvino.dll",
    "openvino_intel_gpu_plugin.dll",
    "openvino_ir_frontend.dll",
    "openvino_auto_plugin.dll",
    "openvino_hetero_plugin.dll",
    "openvino_c.dll",
    "tbb12.dll",
    "tbbbind_2_5.dll",
    "tbbmalloc.dll"
)

foreach ($dll in $requiredDlls) {
    $src = Join-Path $OvLibs $dll
    if (Test-Path $src) {
        Copy-Item $src -Destination "$OutputDir\dlls\"
        $sizeMB = [math]::Round((Get-Item $src).Length / 1MB, 1)
        Write-Host "  + $dll ($sizeMB MB)"
    } else {
        Write-Host "  ! $dll not found" -ForegroundColor Yellow
    }
}

# GenAI DLLs
$genaiDlls = Get-ChildItem $GenAIDir -Recurse | Where-Object { $_.Extension -in '.dll','.pyd' }
foreach ($dll in $genaiDlls) {
    Copy-Item $dll.FullName -Destination "$OutputDir\dlls\"
    $sizeMB = [math]::Round($dll.Length / 1MB, 1)
    Write-Host "  + $($dll.Name) ($sizeMB MB)"
}

# Tokenizer DLL
$tokDlls = Get-ChildItem $TokenizersDir -Recurse | Where-Object { $_.Extension -eq '.dll' }
foreach ($dll in $tokDlls) {
    Copy-Item $dll.FullName -Destination "$OutputDir\dlls\"
    $sizeMB = [math]::Round($dll.Length / 1MB, 1)
    Write-Host "  + $($dll.Name) ($sizeMB MB)"
}

# ── 3. Copy streaming stripe files ───────────────────────────
Write-Host "[3/7] Copying streaming stripe files..." -ForegroundColor Cyan

$stripe0 = Join-Path $ModelDir "dense_weights_streaming_0.bin"
$stripe0json = Join-Path $ModelDir "dense_weights_streaming_0.json"
$stripe1 = Join-Path $ModelDir "dense_weights_streaming_1.bin"

if (Test-Path $stripe0) {
    Copy-Item $stripe0 -Destination "$OutputDir\streaming_nvme0\"
    Write-Host "  + streaming_0.bin ($([math]::Round((Get-Item $stripe0).Length/1MB,1)) MB)"
}
if (Test-Path $stripe0json) {
    Copy-Item $stripe0json -Destination "$OutputDir\streaming_nvme0\"
    Write-Host "  + streaming_0.json"
}
if (Test-Path $stripe1) {
    Copy-Item $stripe1 -Destination "$OutputDir\streaming_nvme1\"
    Write-Host "  + streaming_1.bin ($([math]::Round((Get-Item $stripe1).Length/1MB,1)) MB)"
}

# ── 4. Copy model XML/config files ───────────────────────────
Write-Host "[4/7] Copying model configuration files..." -ForegroundColor Cyan

$modelConfigs = @(
    "openvino_language_model.xml",
    "openvino_text_embeddings_per_layer_model.xml",
    "openvino_text_embeddings_model.xml",
    "openvino_vision_embeddings_model.xml",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
    "config.json",
    "generation_config.json",
    "openvino_config.json",
    "tokenizer_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "tokenizer.json"
)

foreach ($f in $modelConfigs) {
    $src = Join-Path $ModelDir $f
    if (Test-Path $src) {
        Copy-Item $src -Destination "$OutputDir\model\"
        $sizeMB = [math]::Round((Get-Item $src).Length / 1MB, 1)
        Write-Host "  + $f ($sizeMB MB)"
    }
}

# ── 5. Copy model binary weights ─────────────────────────────
if (-not $SkipModelBin) {
    Write-Host "[5/7] Copying model binary weights (this may take a while)..." -ForegroundColor Cyan

    $modelBins = @(
        "openvino_language_model.bin",
        "openvino_text_embeddings_per_layer_model_revised.bin",
        "openvino_vision_embeddings_model.bin",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.bin"
    )

    foreach ($f in $modelBins) {
        $src = Join-Path $ModelDir $f
        if (Test-Path $src) {
            $sizeMB = [math]::Round((Get-Item $src).Length / 1MB, 1)
            Write-Host "  Copying $f ($sizeMB MB)..." -NoNewline
            Copy-Item $src -Destination "$OutputDir\model\"
            Write-Host " done" -ForegroundColor Green
        } else {
            Write-Host "  ! $f not found" -ForegroundColor Yellow
        }
    }
} else {
    Write-Host "[5/7] SKIPPED model binaries (--SkipModelBin)" -ForegroundColor Yellow
}

# ── 6. Copy model cache ──────────────────────────────────────
if (-not $SkipModelCache) {
    Write-Host "[6/7] Copying model cache (compiled GPU kernels)..." -ForegroundColor Cyan
    $cacheSrc = Join-Path $ModelDir "model_cache"
    if (Test-Path $cacheSrc) {
        $cacheFiles = Get-ChildItem $cacheSrc -File
        $cacheSizeMB = [math]::Round(($cacheFiles | Measure-Object Length -Sum).Sum / 1MB, 1)
        Write-Host "  Copying $($cacheFiles.Count) files ($cacheSizeMB MB)..." -NoNewline
        Copy-Item $cacheSrc -Destination "$OutputDir\model\model_cache" -Recurse
        Write-Host " done" -ForegroundColor Green
    }
} else {
    Write-Host "[6/7] SKIPPED model cache (--SkipModelCache)" -ForegroundColor Yellow
    Write-Host "  Note: First run will take ~12s extra to compile GPU kernels" -ForegroundColor Yellow
}

# ── 7. Create ZIP ────────────────────────────────────────────
if (-not $NoZip) {
    Write-Host "[7/7] Creating ZIP archive..." -ForegroundColor Cyan
    $zipPath = Join-Path (Get-Location) $ZipName
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    
    # Calculate total size first
    $totalSize = (Get-ChildItem $OutputDir -Recurse -File | Measure-Object Length -Sum).Sum
    $totalGB = [math]::Round($totalSize / 1GB, 2)
    Write-Host "  Total content: $totalGB GB"
    Write-Host "  Compressing (this may take several minutes)..." -NoNewline
    
    Compress-Archive -Path "$OutputDir\*" -DestinationPath $zipPath -CompressionLevel Optimal
    
    $zipSize = [math]::Round((Get-Item $zipPath).Length / 1GB, 2)
    Write-Host " done" -ForegroundColor Green
    Write-Host "  ZIP size: $zipSize GB"
} else {
    Write-Host "[7/7] SKIPPED ZIP creation (--NoZip)" -ForegroundColor Yellow
}

# ── Summary ──────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================" -ForegroundColor Green
Write-Host "  Release package ready!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Green
Write-Host ""

# Size summary
$folders = @{
    "Scripts" = (Get-ChildItem "$OutputDir\*.py","$OutputDir\*.txt","$OutputDir\*.md" -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    "DLLs" = (Get-ChildItem "$OutputDir\dlls" -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
    "Streaming Files" = ((Get-ChildItem "$OutputDir\streaming_nvme0" -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum + 
                         (Get-ChildItem "$OutputDir\streaming_nvme1" -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum)
    "Model" = (Get-ChildItem "$OutputDir\model" -Recurse -File -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
}

Write-Host "Size breakdown:"
foreach ($k in $folders.Keys) {
    $mb = [math]::Round($folders[$k] / 1MB, 1)
    $gb = [math]::Round($folders[$k] / 1GB, 2)
    if ($gb -ge 1) {
        Write-Host "  $k`: $gb GB"
    } else {
        Write-Host "  $k`: $mb MB"
    }
}

$total = ($folders.Values | Measure-Object -Sum).Sum
Write-Host "  ────────────────"
Write-Host "  TOTAL: $([math]::Round($total/1GB,2)) GB"
Write-Host ""
Write-Host "To deploy on another system:" -ForegroundColor Yellow
Write-Host "  1. Extract the ZIP"
Write-Host "  2. pip install openvino==2026.2.0 openvino-genai openvino-tokenizers"
Write-Host "  3. Copy dlls\openvino_intel_gpu_plugin.dll over the installed one"
Write-Host "  4. Set OV_DENSE_STREAM_WEIGHTS=<path_to_streaming_0.bin>"
Write-Host "  5. Set OV_DENSE_STREAM_WEIGHTS_2=<path_to_streaming_1.bin>  (for dual NVMe)"
Write-Host "  6. python run_gemma4.py --model-dir model --prompt '...'"
Write-Host ""
