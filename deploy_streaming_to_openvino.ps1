# deploy_streaming_to_openvino.ps1
# ================================
# Deploys Dense Weight Streaming Manager into the OpenVINO GPU plugin source tree.
#
# What it does:
#   1. Copies dense_weight_streaming_manager.hpp/cpp to GPU plugin graph directory
#   2. Patches the .cpp to use real OpenVINO headers instead of cldnn stubs
#   3. Adds OV_DENSE_WEIGHT_STREAMING_ENABLED compile definition to CMakeLists.txt
#   4. Adds winmm.lib link dependency (Windows timer resolution)
#
# Usage:
#   .\deploy_streaming_to_openvino.ps1
#   .\deploy_streaming_to_openvino.ps1 -Undo  # reverse all changes
#
# After deploying, rebuild OpenVINO:
#   cd C:\working\gemma4-openvino\openvino\build
#   cmake --build . --target openvino_intel_gpu_graph --config Release -j8

param(
    [switch]$Undo
)

$ErrorActionPreference = "Stop"

$OV_ROOT = "C:\working\gemma4-openvino\openvino"
$PROJECT_ROOT = "C:\working\gemma4-openvino\gemma4-openvino-genai"
$GRAPH_DIR = "$OV_ROOT\src\plugins\intel_gpu\src\graph"
$GRAPH_INCLUDE_DIR = "$GRAPH_DIR\include"

$HPP_SRC = "$PROJECT_ROOT\cpp\dense_weight_streaming_manager.hpp"
$CPP_SRC = "$PROJECT_ROOT\cpp\dense_weight_streaming_manager.cpp"
$HPP_DST = "$GRAPH_INCLUDE_DIR\dense_weight_streaming_manager.hpp"
$CPP_DST = "$GRAPH_DIR\dense_weight_streaming_manager.cpp"
$CMAKE_FILE = "$GRAPH_DIR\CMakeLists.txt"

if ($Undo) {
    Write-Host "=== Undoing streaming deployment ===" -ForegroundColor Yellow
    
    if (Test-Path $HPP_DST) { Remove-Item $HPP_DST; Write-Host "  Removed $HPP_DST" }
    if (Test-Path $CPP_DST) { Remove-Item $CPP_DST; Write-Host "  Removed $CPP_DST" }
    
    # Remove CMake additions
    $cmake = Get-Content $CMAKE_FILE -Raw
    $cmake = $cmake -replace "(?s)\n# Dense Weight Streaming integration.*?# End Dense Weight Streaming\n", ""
    Set-Content $CMAKE_FILE $cmake -NoNewline
    Write-Host "  Cleaned CMakeLists.txt"
    
    Write-Host "=== Undo complete ===" -ForegroundColor Green
    exit 0
}

Write-Host "=== Deploying Dense Weight Streaming to OpenVINO ===" -ForegroundColor Cyan

# --- Step 1: Copy header (unchanged) ---
Write-Host "  [1/4] Copying header..."
Copy-Item $HPP_SRC $HPP_DST -Force
Write-Host "    -> $HPP_DST"

# --- Step 2: Copy and patch .cpp (replace stubs with real headers) ---
Write-Host "  [2/4] Copying and patching .cpp..."
$cpp = Get-Content $CPP_SRC -Raw

# Replace the cldnn_stubs.hpp include block with real GPU plugin headers
$oldInclude = @"
// OpenVINO GPU plugin internals
// In actual integration, these would be the real headers:
// #include "intel_gpu/runtime/engine.hpp"
// #include "intel_gpu/runtime/memory.hpp"
// #include "intel_gpu/runtime/stream.hpp"
//
// For standalone builds, use shared stubs:
#include "cldnn_stubs.hpp"
"@

$newInclude = @"
// OpenVINO GPU plugin internals — real headers (deployed integration)
#ifdef OV_DENSE_WEIGHT_STREAMING_ENABLED
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/graph/network.hpp"
#include "primitive_inst.h"
#define OPENVINO_GPU_RUNTIME_AVAILABLE
#else
// Standalone stubs for standalone benchmark builds
#include "cldnn_stubs.hpp"
#endif
"@

$cpp = $cpp.Replace($oldInclude, $newInclude)
Set-Content $CPP_DST $cpp -NoNewline
Write-Host "    -> $CPP_DST (patched for real headers)"

# --- Step 3: Patch CMakeLists.txt ---
Write-Host "  [3/4] Patching CMakeLists.txt..."

$cmake = Get-Content $CMAKE_FILE -Raw
$marker = "# Dense Weight Streaming integration"

if ($cmake -match [regex]::Escape($marker)) {
    Write-Host "    CMakeLists.txt already patched, skipping."
} else {
    # Add after the target_compile_options line
    $insertion = @"

# Dense Weight Streaming integration
if(WIN32)
    target_compile_definitions(`${TARGET_NAME} PRIVATE OV_DENSE_WEIGHT_STREAMING_ENABLED)
    target_link_libraries(`${TARGET_NAME} PRIVATE winmm.lib)
endif()
# End Dense Weight Streaming
"@
    
    # Insert before the final ov_install_static_lib line
    $cmake = $cmake -replace "(ov_install_static_lib\(`\$\{TARGET_NAME\})", "$insertion`n`$1"
    Set-Content $CMAKE_FILE $cmake -NoNewline
    Write-Host "    Added OV_DENSE_WEIGHT_STREAMING_ENABLED + winmm.lib"
}

# --- Step 4: Verify ---
Write-Host "  [4/4] Verifying..."
$ok = $true
if (-not (Test-Path $HPP_DST)) { Write-Host "    ERROR: $HPP_DST not found!" -ForegroundColor Red; $ok = $false }
if (-not (Test-Path $CPP_DST)) { Write-Host "    ERROR: $CPP_DST not found!" -ForegroundColor Red; $ok = $false }

$verCmake = Get-Content $CMAKE_FILE -Raw
if ($verCmake -notmatch "OV_DENSE_WEIGHT_STREAMING_ENABLED") {
    Write-Host "    ERROR: CMake define not found!" -ForegroundColor Red; $ok = $false
}

if ($ok) {
    Write-Host ""
    Write-Host "=== Deployment complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "Files deployed:" -ForegroundColor White
    Write-Host "  $HPP_DST"
    Write-Host "  $CPP_DST"
    Write-Host "  $CMAKE_FILE (patched)"
    Write-Host ""
    Write-Host "OpenVINO patches (already applied):" -ForegroundColor White
    Write-Host "  primitive_inst.h  -> force_set_output_memory()"
    Write-Host "  network.hpp       -> invalidate_arguments(), get_exec_order(), execute_primitive_group()"
    Write-Host "  network.cpp       -> execute_primitive_group(), try_init_dense_streaming(), execute_impl_streamed()"
    Write-Host ""
    Write-Host "Next: rebuild OpenVINO" -ForegroundColor Yellow
    Write-Host "  cd $OV_ROOT\build"
    Write-Host "  cmake --build . --target openvino_intel_gpu_plugin --config Release -j8"
    Write-Host ""
    Write-Host "Then test:" -ForegroundColor Yellow
    Write-Host '  $env:OV_DENSE_STREAM_WEIGHTS = "C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.bin"'
    Write-Host "  run_gemma4.exe --model-dir gemma-4-E4B-it-ov --prompt `"What is 2+2?`""
} else {
    Write-Host "=== Deployment FAILED ===" -ForegroundColor Red
    exit 1
}
