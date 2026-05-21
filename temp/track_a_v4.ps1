# Track A — additional profile: v4 single-NVMe-dual-file (same as v3 dual but both .bin on D:)
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

foreach ($k in 'OV_DENSE_STREAM_NO_PREFETCH') {
    Remove-Item "Env:$k" -ErrorAction SilentlyContinue
}
$env:OV_PER_LAYER_EMBEDDING_PATH = (Join-Path $model "openvino_text_embeddings_per_layer_model_revised.bin")
$env:OV_DENSE_STREAM_FILE        = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS     = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS_2   = "D:\gemma4_streaming_nvme0\dense_weights_streaming_1.bin"
$env:OV_DENSE_STREAM_NUM_BUFFERS = "4"
$env:OV_DENSE_STREAM_IO_THREADS  = "8"
$env:OV_DENSE_STREAM_DEBUG       = "1"
$env:OV_DENSE_STREAM_TIMING      = "1"
$env:OV_DENSE_STREAM_LOG_FILE    = (Join-Path $tmp "v4_singlenvme_dualfile.log")

$stdout = Join-Path $tmp "v4_singlenvme_dualfile.stdout.log"
& python run_gemma4.py --model-dir model --device GPU `
    --prompt $prompt_text --max-new-tokens 256 --show-memory *>&1 |
    Tee-Object -FilePath $stdout
Pop-Location
