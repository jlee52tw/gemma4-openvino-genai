# Env script: Case C — dense streaming v2 single overlap
foreach ($k in 'OV_DENSE_STREAM_NO_PREFETCH','OV_DENSE_STREAM_WEIGHTS_2','OV_DENSE_STREAM_DEBUG','OV_DENSE_STREAM_LOG_FILE','OV_DENSE_STREAM_TIMING') {
    Remove-Item "Env:$k" -ErrorAction SilentlyContinue
}
$env:OV_PER_LAYER_EMBEDDING_PATH = "C:\working\gemma4-openvino-genai\gemma4_streaming_release_v2\model\openvino_text_embeddings_per_layer_model_revised.bin"
$env:OV_DENSE_STREAM_FILE        = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_WEIGHTS     = "D:\gemma4_streaming_nvme0\dense_weights_streaming_0.bin"
$env:OV_DENSE_STREAM_NUM_BUFFERS = "2"
$env:OV_DENSE_STREAM_IO_THREADS  = "4"
