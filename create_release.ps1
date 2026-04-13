# ============================================================
# create_release.ps1 — Build a self-contained OpenVINO GenAI
#                       runtime folder with setupvars scripts
# ============================================================
#
# Prerequisites:
#   - openvino-genai built from source via `pip install .`
#     (step 1.5 in the README)
#   - CMake >= 3.23 available on PATH
#
# Usage:
#   .\create_release.ps1 -GenaiSrc .\openvino_genai_src `
#                         -VenvDir .\.venv `
#                         -InstallDir .\openvino_genai_release
#
# The resulting folder can be distributed and used with:
#   . .\openvino_genai_release\setupvars.ps1
#   .\run_gemma4.exe --model-dir ... --device GPU --prompt "..."
# ============================================================

param(
    [Parameter(Mandatory)]
    [string]$GenaiSrc,        # Path to openvino.genai source (cloned PR #3644)

    [Parameter(Mandatory)]
    [string]$VenvDir,         # Path to Python venv (with openvino + openvino-genai installed)

    [string]$InstallDir = ".\openvino_genai_release"
)

$ErrorActionPreference = "Stop"

# ── Resolve paths ────────────────────────────────────────────
$GenaiSrc   = Resolve-Path $GenaiSrc
$VenvDir    = Resolve-Path $VenvDir

# ── Locate build cache ──────────────────────────────────────
$buildCacheBase = Join-Path $GenaiSrc ".py-build-cmake_cache"
if (-not (Test-Path $buildCacheBase)) {
    Write-Error "Build cache not found: $buildCacheBase`nRun 'pip install .' on the genai source first."
}
$buildDir = Get-ChildItem $buildCacheBase -Directory | Select-Object -First 1
if (-not $buildDir) {
    Write-Error "No build configuration found in $buildCacheBase"
}
$buildDir = $buildDir.FullName
Write-Host "[1/5] Using build cache: $buildDir" -ForegroundColor Cyan

# ── cmake --install ─────────────────────────────────────────
$InstallDir = [System.IO.Path]::GetFullPath($InstallDir)
Write-Host "[2/5] Installing GenAI to: $InstallDir" -ForegroundColor Cyan
cmake --install $buildDir --prefix $InstallDir --config Release
if ($LASTEXITCODE -ne 0) { Write-Error "cmake --install failed" }

# ── Copy OpenVINO core DLLs + libs ──────────────────────────
$ovLibs = Join-Path $VenvDir "Lib\site-packages\openvino\libs"
$ovInc  = Join-Path $VenvDir "Lib\site-packages\openvino\include"
$targetLibs = Join-Path $InstallDir "openvino\libs"
$targetInc  = Join-Path $InstallDir "openvino\include"

Write-Host "[3/5] Copying OpenVINO DLLs + headers from venv" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $targetLibs | Out-Null
Copy-Item "$ovLibs\*.dll" -Destination $targetLibs -Force
Copy-Item "$ovLibs\*.lib" -Destination $targetLibs -Force
if (Test-Path $ovInc) {
    Copy-Item $ovInc -Destination (Split-Path $targetInc) -Recurse -Force
}

# ── Copy openvino_tokenizers.dll ────────────────────────────
$tokSrc = Join-Path $VenvDir "Lib\site-packages\openvino_tokenizers\lib\openvino_tokenizers.dll"
$genaiDllDir = Join-Path $InstallDir "runtime\bin\intel64\Release"
Write-Host "[4/5] Copying openvino_tokenizers.dll" -ForegroundColor Cyan
Copy-Item $tokSrc -Destination $genaiDllDir -Force

# ── Create setupvars scripts ────────────────────────────────
Write-Host "[5/5] Creating setupvars.ps1 and setupvars.bat" -ForegroundColor Cyan

$setupvarsPs1 = @'
# OpenVINO GenAI - Environment Setup (PowerShell)
# Usage:  . .\setupvars.ps1
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$BIN_DIR   = Join-Path $SCRIPT_DIR "runtime\bin\intel64\Release"
$OV_LIBS   = Join-Path $SCRIPT_DIR "openvino\libs"
$CMAKE_DIR = Join-Path $SCRIPT_DIR "runtime\cmake"
if (-not (Test-Path "$BIN_DIR\openvino_genai.dll")) {
    Write-Host "[ERROR] openvino_genai.dll not found in $BIN_DIR" -ForegroundColor Red; return
}
if ($env:PATH -notlike "*$BIN_DIR*") { $env:PATH = "$BIN_DIR;$env:PATH" }
if ((Test-Path $OV_LIBS) -and ($env:PATH -notlike "*$OV_LIBS*")) { $env:PATH = "$OV_LIBS;$env:PATH" }
$env:OpenVINO_DIR      = $CMAKE_DIR
$env:OpenVINOGenAI_DIR = $CMAKE_DIR
if (-not $env:CMAKE_PREFIX_PATH) { $env:CMAKE_PREFIX_PATH = $SCRIPT_DIR }
elseif ($env:CMAKE_PREFIX_PATH -notlike "*$SCRIPT_DIR*") { $env:CMAKE_PREFIX_PATH = "$SCRIPT_DIR;$env:CMAKE_PREFIX_PATH" }
Write-Host ""
Write-Host "[setupvars] OpenVINO GenAI environment initialized" -ForegroundColor Green
Write-Host "  INSTALL_DIR : $SCRIPT_DIR"
Write-Host "  DLLs        : $BIN_DIR"
Write-Host "  OV DLLs     : $OV_LIBS"
Write-Host ""
'@

$setupvarsBat = @'
@echo off
rem OpenVINO GenAI - Environment Setup (cmd.exe)
rem Usage:  call setupvars.bat
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "BIN_DIR=%SCRIPT_DIR%\runtime\bin\intel64\Release"
set "OV_LIBS=%SCRIPT_DIR%\openvino\libs"
set "CMAKE_DIR=%SCRIPT_DIR%\runtime\cmake"
if not exist "%BIN_DIR%\openvino_genai.dll" ( echo [ERROR] openvino_genai.dll not found in %BIN_DIR% & goto :eof )
echo %PATH% | findstr /C:"%BIN_DIR%" >nul 2>&1 || set "PATH=%BIN_DIR%;%PATH%"
if exist "%OV_LIBS%" ( echo %PATH% | findstr /C:"%OV_LIBS%" >nul 2>&1 || set "PATH=%OV_LIBS%;%PATH%" )
set "OpenVINO_DIR=%CMAKE_DIR%"
set "OpenVINOGenAI_DIR=%CMAKE_DIR%"
if not defined CMAKE_PREFIX_PATH ( set "CMAKE_PREFIX_PATH=%SCRIPT_DIR%" ) else (
    echo %CMAKE_PREFIX_PATH% | findstr /C:"%SCRIPT_DIR%" >nul 2>&1 || set "CMAKE_PREFIX_PATH=%SCRIPT_DIR%;%CMAKE_PREFIX_PATH%"
)
echo.
echo [setupvars] OpenVINO GenAI environment initialized
echo   INSTALL_DIR : %SCRIPT_DIR%
echo   DLLs        : %BIN_DIR%
echo   OV DLLs     : %OV_LIBS%
echo.
'@

Set-Content -Path (Join-Path $InstallDir "setupvars.ps1") -Value $setupvarsPs1 -Encoding UTF8
Set-Content -Path (Join-Path $InstallDir "setupvars.bat") -Value $setupvarsBat -Encoding ASCII

# ── Summary ──────────────────────────────────────────────────
Write-Host ""
Write-Host "================================================" -ForegroundColor Green
Write-Host "  Release folder created: $InstallDir" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Layout:"
Write-Host "  runtime\bin\intel64\Release\  - openvino_genai.dll + openvino_tokenizers.dll"
Write-Host "  runtime\lib\intel64\Release\  - openvino_genai.lib (link library)"
Write-Host "  runtime\include\              - GenAI C++ headers"
Write-Host "  runtime\cmake\                - CMake configs (OpenVINO + GenAI)"
Write-Host "  openvino\libs\                - OpenVINO core DLLs + .lib files"
Write-Host "  openvino\include\             - OpenVINO core C++ headers"
Write-Host "  setupvars.ps1 / .bat          - Environment setup scripts"
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  # 1. Initialize environment"
Write-Host "  . $InstallDir\setupvars.ps1"
Write-Host ""
Write-Host "  # 2. Build your C++ app"
Write-Host "  cmake -B build -DOpenVINOGenAI_DIR=`"$InstallDir\runtime\cmake`""
Write-Host "  cmake --build build --config Release"
Write-Host ""
Write-Host "  # 3. Run"
Write-Host "  .\build\Release\run_gemma4.exe --model-dir ... --device GPU --prompt `"...`""
Write-Host ""
