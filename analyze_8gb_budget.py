"""Realistic 8 GB budget analysis with hybrid pinning strategies."""

print("=== Realistic 8 GB Budget (based on baseline RSS 7.1 GB) ===")
print()
baseline_rss = 7100  # MB (short-text baseline on 16 GB)
decoder_full = 2028  # MB
buf_2x = 109  # MB (2 x 54.6 MB for 1-layer double buffer)

# With streaming: RSS = baseline - decoder_full + pinned_layers + 2*buf
rss_base = baseline_rss - decoder_full + buf_2x  # = 5181 MB
print(f"RSS with zero pinned decoder: {rss_base} MB = {rss_base/1024:.2f} GB")
print()

print("On 8 GB: OS needs ~2.0-2.5 GB")
print("  Conservative (2.5 GB OS): 5692 MB available")
print("  Optimistic   (2.0 GB OS): 6192 MB available")
print()

for name, limit in [("Conservative", 5692), ("Optimistic", 6192)]:
    max_pin = limit - rss_base
    max_layers = int(max_pin / 48.0)
    print(f"  {name}: can pin up to {max_pin:.0f} MB = ~{max_layers} decoder layers")
print()

print("=== Head+Tail Hybrid Performance (1-layer group, double buffer) ===")
print()
hdr = f"{'Strategy':>14} {'Stream':>7} {'PinMB':>7} {'RSS':>7} {'StreamMs':>9} {'PinGPU':>7} {'TPOT':>7} {'TPS':>5} {'8GB':>4}"
print(hdr)
print("-" * len(hdr))

io_per_layer_ms = 48.0 / 1024 / 11.2 * 1000  # ~4.19 ms

scenarios = [(0,0), (3,3), (5,5), (7,7), (10,10), (14,14), (21,21)]
for head_n, tail_n in scenarios:
    pin = head_n + tail_n
    stream = 42 - pin
    pin_mb = pin * 48.0
    rss = rss_base + pin_mb
    
    # Pinned layers: pure GPU, no IO wait
    gpu_pinned_ms = pin * 0.99
    
    # Streamed layers: double-buffer overlap
    # IO for 1 layer ~4.19ms, GPU for 1 layer ~0.99ms
    # Effective per streamed layer = max(IO, GPU) = IO (IO-bound)
    stream_ms = stream * max(io_per_layer_ms, 0.99)
    
    tpot = gpu_pinned_ms + stream_ms
    tps = 1000 / tpot if tpot > 0 else 999
    fits = "Y" if rss < 5692 else ("~" if rss < 6192 else "N")
    
    label = f"H{head_n}+T{tail_n}"
    print(f"{label:>14} {stream:>7} {pin_mb:>6.0f}M {rss:>6.0f}M {stream_ms:>8.1f} {gpu_pinned_ms:>6.1f} {tpot:>6.1f} {tps:>5.1f} {fits:>4}")

print()
print("=== Group Granularity Comparison ===")
print()
print("Buffer memory = 2 x max_group_size")
for layers_per_group in [7, 3, 2, 1]:
    buf_size = layers_per_group * 55  # MB (worst case with expanded layer)
    total_buf = 2 * buf_size
    num_groups = 42 // layers_per_group
    print(f"  {layers_per_group} layer/group:  {num_groups:2d} groups, buffer = 2 x {buf_size:>4d} MB = {total_buf:>4d} MB")

print()
print("=== Why 2 Buffers (A/B) is Optimal ===")
print()
print("  Pipeline: GPU(buf_A) | IO(buf_B) -> swap -> GPU(buf_B) | IO(buf_A)")
print("  GPU only processes ONE group at a time")
print("  IO only loads ONE group at a time")
print("  Buffer C would be idle - no benefit from 3+ buffers")
print()
print("=== Recommended Strategy for 8 GB ===")
print()
print("  1. Use 1-layer groups (42 groups) instead of 7-layer groups (6 groups)")
print("     - Buffer: 2 x 55 MB = 110 MB (vs 2 x 342 MB = 684 MB)")
print("     - Finer overlap, less wasted buffer memory")
print()
print("  2. Pin head 7 + tail 7 = 14 decoder layers permanently")
print("     - Pinned: ~672 MB, RSS: ~5853 MB (fits 8 GB)")
print("     - These 14 layers run at full GPU speed (0.99 ms/layer)")
print()
print("  3. Stream middle 28 layers with double buffer")
print("     - IO per layer: ~4.2 ms (IO-bound)")
print("     - Estimated TPOT: 14*0.99 + 28*4.2 = 131.5 ms -> 7.6 tps")
print()
print("  4. vs baseline 24 tps: streaming adds ~17x overhead")
print("     vs all-stream: hybrid is ~33% faster (7.6 vs 5.7 tps)")
