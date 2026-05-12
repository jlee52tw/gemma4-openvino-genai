"""Analyze model weight distribution for hybrid pinning strategy."""
import json

meta = json.load(open(r"C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.json"))
total_decoder = meta["total_weight_bytes"]

print("=== Per-Group Sizes ===")
for g in meta["groups"]:
    raw_mb = g["raw_bytes"] / 1024**2
    aligned_mb = g["aligned_bytes"] / 1024**2
    print(f"  G{g['group_idx']}: layers {g['first_layer']}-{g['last_layer']} = {raw_mb:.1f} MB (aligned {aligned_mb:.1f} MB)")

print()
print("=== Per-Layer Sizes ===")
for l in meta["layers"]:
    print(f"  Layer {l['layer_idx']:2d}: {l['num_tensors']} tensors, {l['size_bytes']/1024**2:.2f} MB")

print()
decoder_mb = total_decoder / 1024**2
lang_bin = 2685.2
embed_mb = 640.5
vision_mb = 161.9
lm_head_norm_mb = lang_bin - decoder_mb
tokenizer_mb = 20.7

print("=== Weight Budget ===")
print(f"  embed_tokens:    {embed_mb:.1f} MB")
print(f"  vision_embed:    {vision_mb:.1f} MB")
print(f"  lm_head+norm:    {lm_head_norm_mb:.1f} MB")
print(f"  decoder (42 lyr): {decoder_mb:.1f} MB")
print(f"  tokenizer+det:   {tokenizer_mb:.1f} MB")
static_all = embed_mb + vision_mb + lm_head_norm_mb + tokenizer_mb
print(f"  ---")
print(f"  Static total:    {static_all:.1f} MB")
print(f"  Decoder total:   {decoder_mb:.1f} MB")
print(f"  Grand total:     {static_all + decoder_mb:.1f} MB")

print()
print("=== Hybrid Pinning Scenarios ===")
layers = meta["layers"]
layer_sizes = {l["layer_idx"]: l["size_bytes"] / 1024**2 for l in layers}

# IO bandwidth
io_gbps = 11.2
gpu_per_layer_ms = 0.99

scenarios = [
    ("Current: stream all 42", 0, 0),
    ("Pin 3+3, stream 36", 3, 3),
    ("Pin 5+5, stream 32", 5, 5),
    ("Pin 7+7, stream 28", 7, 7),
    ("Pin 10+10, stream 22", 10, 10),
    ("Pin 14+14, stream 14", 14, 14),
]

for name, pin_head, pin_tail in scenarios:
    pinned_layers = list(range(pin_head)) + list(range(42 - pin_tail, 42))
    streamed_layers = [i for i in range(42) if i not in pinned_layers]
    
    pinned_mb = sum(layer_sizes[i] for i in pinned_layers)
    streamed_mb = sum(layer_sizes[i] for i in streamed_layers)
    
    # IO time per token = stream volume / bandwidth
    io_time_ms = (streamed_mb / 1024) / io_gbps * 1000
    
    # GPU compute for streamed layers
    gpu_streamed_ms = len(streamed_layers) * gpu_per_layer_ms
    
    # With double-buffer overlap, effective time = max(io, gpu) per group
    # But IO >> GPU, so roughly: TPOT = io_time + pinned_layers * gpu_ms
    pinned_gpu_ms = len(pinned_layers) * gpu_per_layer_ms
    total_tpot_ms = io_time_ms + pinned_gpu_ms  # GPU for pinned runs free, IO dominates streamed
    tps = 1000.0 / total_tpot_ms if total_tpot_ms > 0 else float("inf")
    
    # Memory: static + pinned decoder layers + 2 * max_group_buffer
    # Group buffer = ceil(streamed / num_groups) for various group counts
    num_streamed = len(streamed_layers)
    
    # Best case: 1-layer groups
    if num_streamed > 0:
        max_layer_mb = max(layer_sizes[i] for i in streamed_layers)
        buf_1layer = 2 * max_layer_mb
    else:
        buf_1layer = 0
    
    total_mem = static_all + pinned_mb + buf_1layer
    
    print(f"  {name}:")
    print(f"    Pinned decoder: {pinned_mb:.1f} MB, Streamed: {streamed_mb:.1f} MB")
    print(f"    IO time/token: {io_time_ms:.1f} ms, Pinned GPU: {pinned_gpu_ms:.1f} ms")
    print(f"    Est TPOT: {total_tpot_ms:.1f} ms -> {tps:.1f} tps")
    print(f"    Memory (static+pinned+2xbuf): {total_mem:.0f} MB = {total_mem/1024:.2f} GB")
    print()
