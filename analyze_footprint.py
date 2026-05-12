"""Analyze overall model memory footprint and fixed/streaming ratios for H5+T5."""
import json

meta_path = r"C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.json"
m = json.load(open(meta_path))

print("=== Current H5+T5 Metadata ===")
print(f"Total decoder layers: {m['total_decoder_layers']}")
print(f"Pin head: {m['pin_head_layers']}, Pin tail: {m['pin_tail_layers']}")
print(f"First streamed: {m['first_streamed_layer']}, Last streamed: {m['last_streamed_layer']}")
print(f"Num groups: {m['num_groups']}, Group size: {m['group_size']}")
print(f"Streamed weight bytes: {m['streamed_weight_bytes']:,} ({m['streamed_weight_bytes']/1024**3:.3f} GB)")
print(f"Total decoder bytes:  {m['total_decoder_weight_bytes']:,} ({m['total_decoder_weight_bytes']/1024**3:.3f} GB)")
print()

print("=== Per-Layer Sizes ===")
for l in m["layers"]:
    status = "PINNED" if l.get("pinned", False) else "stream"
    print(f"  Layer {l['layer_idx']:2d}: {l['num_tensors']:2d} tensors, {l['size_bytes']/1024**2:6.2f} MB  [{status}]")
print()

# Sum by category
head = [l for l in m["layers"] if l["layer_idx"] < m["pin_head_layers"]]
tail = [l for l in m["layers"] if l["layer_idx"] >= m["total_decoder_layers"] - m["pin_tail_layers"]]
streamed = [l for l in m["layers"] if not l.get("pinned", False)]
pin_head_mb = sum(l["size_bytes"] for l in head) / 1024**2
pin_tail_mb = sum(l["size_bytes"] for l in tail) / 1024**2
stream_mb = sum(l["size_bytes"] for l in streamed) / 1024**2
total_dec_mb = sum(l["size_bytes"] for l in m["layers"]) / 1024**2

print(f"Pinned HEAD: {pin_head_mb:.1f} MB ({len(head)} layers)")
print(f"Pinned TAIL: {pin_tail_mb:.1f} MB ({len(tail)} layers)")
print(f"Streamed:    {stream_mb:.1f} MB ({len(streamed)} layers)")
print(f"All decoder: {total_dec_mb:.1f} MB")
print()

# Non-decoder components from model analysis
print("=== Non-Decoder Components (from model analysis) ===")
embed = 640.5
vision = 161.9
tokenizer_det = 20.7
lang_bin_mb = 2685.2
non_dec_in_lang = lang_bin_mb - total_dec_mb

print(f"  embed_tokens.bin:        {embed:.1f} MB")
print(f"  vision_embeddings:       {vision:.1f} MB")
print(f"  lm_head+norms (in lang): {non_dec_in_lang:.1f} MB")
print(f"  tokenizer+detokenizer:   {tokenizer_det:.1f} MB")
static = embed + vision + non_dec_in_lang + tokenizer_det
print(f"  Static (non-decoder):    {static:.1f} MB = {static/1024:.3f} GB")
print()

# Full memory footprint
print("=== Memory Footprint Analysis ===")
print(f"  [A] Non-decoder (always in RAM):    {static:.1f} MB")
print(f"  [B] Pinned decoder HEAD (in RAM):   {pin_head_mb:.1f} MB")
print(f"  [C] Pinned decoder TAIL (in RAM):   {pin_tail_mb:.1f} MB")
pinned_total = pin_head_mb + pin_tail_mb
fixed_total = static + pinned_total
print(f"  [D] Fixed total (A+B+C):            {fixed_total:.1f} MB = {fixed_total/1024:.3f} GB")

max_grp = max(g["aligned_bytes"] for g in m["groups"])
buf2x = 2 * max_grp / 1024**2
print(f"  [E] Streaming buffers (2x max grp): {buf2x:.1f} MB")

kv_runtime = 1200.0
print(f"  [F] KV cache + runtime (est):       {kv_runtime:.0f} MB")
total_rss = fixed_total + buf2x + kv_runtime
print(f"  [G] Total RSS = D+E+F:              {total_rss:.0f} MB = {total_rss/1024:.2f} GB")
print()

# Full model comparison
full_rss = static + total_dec_mb + kv_runtime
print(f"  Full model (no streaming):          {full_rss:.0f} MB = {full_rss/1024:.2f} GB")
print(f"  Savings from streaming:             {full_rss - total_rss:.0f} MB = {(full_rss - total_rss)/1024:.2f} GB")
print()

# Ratios
print("=== Fixed vs Streaming Ratio ===")
print(f"  Fixed+Pinned in RAM:   {fixed_total:.1f} MB ({fixed_total/1024:.3f} GB)")
print(f"  Streaming on NVMe:     {stream_mb:.1f} MB ({stream_mb/1024:.3f} GB)")
ratio = fixed_total / stream_mb
print(f"  Ratio (fixed:stream):  {ratio:.2f} : 1.00")
print(f"  % of model on NVMe:    {stream_mb / (fixed_total + stream_mb) * 100:.1f}%")
print(f"  % of model pinned:     {fixed_total / (fixed_total + stream_mb) * 100:.1f}%")
print()

# Breakdown by component type
print("=== Component Breakdown (pie chart data) ===")
components = [
    ("Embeddings (embed_tokens)", embed),
    ("Vision encoder", vision),
    ("LM head + norms", non_dec_in_lang),
    ("Tokenizer/detok", tokenizer_det),
    ("Pinned HEAD (layers 0-4)", pin_head_mb),
    ("Pinned TAIL (layers 37-41)", pin_tail_mb),
    ("Streamed MIDDLE (layers 5-36)", stream_mb),
    ("Streaming buffers (2x)", buf2x),
    ("KV cache + runtime", kv_runtime),
]
for name, mb in components:
    print(f"  {name:40s} {mb:8.1f} MB  ({mb/total_rss*100:5.1f}%)")
print(f"  {'TOTAL':40s} {total_rss:8.0f} MB")
