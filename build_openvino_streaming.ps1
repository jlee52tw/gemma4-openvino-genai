# build_openvino_streaming.ps1
# =============================
# Build OpenVINO with Dense Weight Streaming support.
#
# Prerequisites:
#   1. Run deploy_streaming_to_openvino.ps1 first
#   2. OpenVINO CMake already configured in C:\working\gemma4-openvino\openvino\build
#
# Usage:
#   .\build_openvino_streaming.ps1            # Build GPU plugin only
#   .\build_openvino_streaming.ps1 -Full      # Full OpenVINO rebuild
#   .\build_openvino_streaming.ps1 -Configure # Re-run CMake configure first

param(
    [switch]$Full,
    [switch]$Configure
)

$ErrorActionPreference = "Stop"

$OV_ROOT = "C:\working\gemma4-openvino\openvino"
$BUILD_DIR = "$OV_ROOT\build"
$JOBS = 8

# Check prerequisites
if (-not (Test-Path "$OV_ROOT\src\plugins\intel_gpu\src\graph\dense_weight_streaming_manager.cpp")) {
    Write-Host "ERROR: Streaming manager not deployed. Run deploy_streaming_to_openvino.ps1 first." -ForegroundColor Red
    exit 1
}

# Find Visual Studio
$vcvarsall = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if (-not (Test-Path $vcvarsall)) {
    $vcvarsall = "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
}
if (-not (Test-Path $vcvarsall)) {
    Write-Host "ERROR: Visual Studio 2022 not found" -ForegroundColor Red
    exit 1
}

Write-Host "=== Building OpenVINO with Dense Weight Streaming ===" -ForegroundColor Cyan
Write-Host "  Build dir: $BUILD_DIR"
Write-Host "  Jobs: $JOBS"

if ($Configure) {
    Write-Host ""
    Write-Host "--- CMake Configure ---" -ForegroundColor Yellow
    
    if (-not (Test-Path $BUILD_DIR)) {
        New-Item -ItemType Directory -Path $BUILD_DIR | Out-Null
    }
    
    Push-Location $BUILD_DIR
    try {
        # Minimal GPU-focused build
        cmake $OV_ROOT `
            -G "Visual Studio 17 2022" `
            -A x64 `
            -DCMAKE_BUILD_TYPE=Release `
            -DENABLE_INTEL_GPU=ON `
            -DENABLE_INTEL_CPU=OFF `
            -DENABLE_INTEL_NPU=OFF `
            -DENABLE_TESTS=OFF `
            -DENABLE_SAMPLES=OFF `
            -DENABLE_PYTHON=OFF `
            -DENABLE_WHEEL=OFF `
            -DENABLE_OV_TF_FRONTEND=OFF `
            -DENABLE_OV_TF_LITE_FRONTEND=OFF `
            -DENABLE_OV_PADDLE_FRONTEND=OFF `
            -DENABLE_OV_PYTORCH_FRONTEND=OFF `
            -DENABLE_OV_JAX_FRONTEND=OFF
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CMake configure failed!" -ForegroundColor Red
            exit 1
        }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "--- Building ---" -ForegroundColor Yellow

Push-Location $BUILD_DIR
try {
    if ($Full) {
        Write-Host "  Full build..."
        cmake --build . --config Release -j $JOBS
    } else {
        Write-Host "  GPU plugin incremental build..."
        # Build just the GPU plugin target
        cmake --build . --target openvino_intel_gpu_plugin --config Release -j $JOBS
    }
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Build FAILED!" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "=== Build complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "To test Dense Weight Streaming:" -ForegroundColor Yellow
Write-Host '  $env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.bin"'
Write-Host '  $env:OV_DENSE_STREAM_TIMING = "1"'
Write-Host '  $env:OV_DENSE_STREAM_DEBUG = "1"'
Write-Host "  run_gemma4.exe --model-dir C:\working\gemma4-openvino\gemma-4-E4B-it-ov --prompt `"What is 2+2?`""
