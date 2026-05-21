# Track A — three profiling runs (v1 seq, v2 single overlap, v3 dual overlap)
# Each run: max_new_tokens=256, same short prompt; sleeps 5s between to settle USM/page cache.

$ErrorActionPreference = "Continue"
$PSNativeCommandUseErrorActionPreference = $false
$root  = "C:\working\gemma4-openvino-genai"
$rel   = Join-Path $root "gemma4_streaming_release_v2"
$venv  = Join-Path $root ".venv\Scripts\Activate.ps1"
$model = Join-Path $rel "model"
$tmp   = Join-Path $root "temp"
$prompt_text = "Explain how a transformer attention block computes its output in plain English. Cover queries, keys, values, scaled dot-product, masking, and the softmax. Keep it concise."

. $venv

Push-Location $rel

function Clear-StreamEnv {
    Remove-Item Env:OV_DENSE_STREAM_FILE       -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_WEIGHTS    -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_WEIGHTS_2  -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_NUM_BUFFERS -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_IO_THREADS  -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_NO_PREFETCH -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_DEBUG       -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_LOG_FILE    -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_TIMING      -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_PIN_HEAD    -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_PIN_TAIL    -ErrorAction SilentlyContinue
    Remove-Item Env:OV_DENSE_STREAM_TOTAL_LAYERS -ErrorAction SilentlyContinue
}

function Run-Variant($tag, $extraEnv) {
    Write-Host "`n========== Profile $tag ==========`n" -ForegroundColor Cyan
    Clear-StreamEnv
    $env:OV_PER_LAYER_EMBEDDING_PATH = (Join-Path $model "openvino_text_embeddings_per_layer_model_revised.bin")
    $env:OV_DENSE_STREAM_DEBUG       = "1"
    $env:OV_DENSE_STREAM_TIMING      = "1"
    $env:OV_DENSE_STREAM_LOG_FILE    = (Join-Path $tmp "$tag.log")
    foreach ($k in $extraEnv.Keys) {
        Set-Item -Path "Env:$k" -Value $extraEnv[$k]
    }
    $stdout = Join-Path $tmp "$tag.stdout.log"
    & python run_gemma4.py --model-dir model --device GPU `
        --prompt $prompt_text --max-new-tokens 256 --show-memory *>&1 |
        Tee-Object -FilePath $stdout
    Write-Host "`n[$tag] done. stdout=$stdout, debug=$($env:OV_DENSE_STREAM_LOG_FILE)" -ForegroundColor Green
    Start-Sleep -Seconds 5
}

# --- v1 sequential (no prefetch) ---
Run-Variant "v1_seq" @{
    OV_DENSE_STREAM_WEIGHTS     = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
    OV_DENSE_STREAM_FILE        = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
    OV_DENSE_STREAM_NUM_BUFFERS = "2"
    OV_DENSE_STREAM_IO_THREADS  = "4"
    OV_DENSE_STREAM_NO_PREFETCH = "1"
}

# --- v2 single overlap ---
Run-Variant "v2_single" @{
    OV_DENSE_STREAM_WEIGHTS     = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
    OV_DENSE_STREAM_FILE        = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
    OV_DENSE_STREAM_NUM_BUFFERS = "2"
    OV_DENSE_STREAM_IO_THREADS  = "4"
}

# --- v3 dual overlap ---
Run-Variant "v3_dual" @{
    OV_DENSE_STREAM_WEIGHTS     = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
    OV_DENSE_STREAM_FILE        = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
    OV_DENSE_STREAM_WEIGHTS_2   = "E:\gemma4_streaming_nvme1\dense_weights_streaming_1.bin"
    OV_DENSE_STREAM_NUM_BUFFERS = "4"
    OV_DENSE_STREAM_IO_THREADS  = "8"
}

Pop-Location
Write-Host "`nAll variants finished." -ForegroundColor Green
