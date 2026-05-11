#!/usr/bin/env python3
"""
Dense Weight Streaming — Weight Packing Tool (Hybrid Pinning)
==============================================================
Packs decoder layer weights from OpenVINO model into a contiguous binary
file optimized for Direct I/O sequential reads.

Hybrid Pinning Strategy (H5+T5 default):
  - Head layers (0-4) and tail layers (37-41) stay in GPU USM memory
    permanently ("pinned") and run at full GPU speed.
  - Middle layers (5-36) are packed into this binary file and loaded
    from NVMe per token via double-buffer streaming.

File Format: dense_weights_streaming.bin
  [Header]              — 4096 bytes (sector-aligned)
  [Group Table]         — 4096 bytes (sector-aligned, up to 42 groups)
  [Per-Layer Table]     — 4096 bytes (sector-aligned)
  [Group 0 data]        — sector-aligned, all layers in group packed
  [Group 1 data]        — sector-aligned
  ...

Each group contains weight tensors for its layers, packed contiguously.

Usage:
  # Default: H5+T5 hybrid, 1-layer groups (32 groups for middle layers)
  python pack_dense_weights.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov

  # Custom pinning: pin only 3 head + 3 tail layers
  python pack_dense_weights.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov --pin-head 3 --pin-tail 3

  # No pinning (stream all 42 layers, legacy mode)
  python pack_dense_weights.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov --pin-head 0 --pin-tail 0

  # Dry run: analyze weight structure without writing file
  python pack_dense_weights.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov --dry-run
"""

import argparse
import json
import os
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

# Constants
SECTOR_SIZE = 4096  # DirectStorage sector alignment
MAGIC = b"DNSW"     # Dense Weight Streaming
VERSION = 1
NUM_DECODER_LAYERS = 42
HEADER_SIZE = SECTOR_SIZE       # 4096 bytes
GROUP_TABLE_SIZE = SECTOR_SIZE  # 4096 bytes (room for up to 42 groups)
LAYER_TABLE_SIZE = SECTOR_SIZE * 2  # 8192 bytes (room for 42 layers × ~190 bytes each)

# Weight component patterns per layer (order matters for packing)
# Matches tensor names from Gemma4 model structure
WEIGHT_PATTERNS = [
    # Attention projections
    r"layers\.{layer}.*self_attn.*q_proj.*weight",
    r"layers\.{layer}.*self_attn.*k_proj.*weight",
    r"layers\.{layer}.*self_attn.*v_proj.*weight",
    r"layers\.{layer}.*self_attn.*o_proj.*weight",
    # MLP projections
    r"layers\.{layer}.*mlp.*gate_proj.*weight",
    r"layers\.{layer}.*mlp.*up_proj.*weight",
    r"layers\.{layer}.*mlp.*down_proj.*weight",
    # LayerNorms
    r"layers\.{layer}.*input_layernorm.*weight",
    r"layers\.{layer}.*post_attention_layernorm.*weight",
    r"layers\.{layer}.*pre_feedforward_layernorm.*weight",
    r"layers\.{layer}.*post_feedforward_layernorm.*weight",
]

# Scale/zero-point patterns (associated with INT4 quantized weights)
SCALE_PATTERNS = [
    r"layers\.{layer}.*self_attn.*q_proj.*(scale|zero_point)",
    r"layers\.{layer}.*self_attn.*k_proj.*(scale|zero_point)",
    r"layers\.{layer}.*self_attn.*v_proj.*(scale|zero_point)",
    r"layers\.{layer}.*self_attn.*o_proj.*(scale|zero_point)",
    r"layers\.{layer}.*mlp.*gate_proj.*(scale|zero_point)",
    r"layers\.{layer}.*mlp.*up_proj.*(scale|zero_point)",
    r"layers\.{layer}.*mlp.*down_proj.*(scale|zero_point)",
]


def align_to_sector(offset: int) -> int:
    """Align offset up to next sector boundary."""
    return ((offset + SECTOR_SIZE - 1) // SECTOR_SIZE) * SECTOR_SIZE


def fmt_size(bytes_val: int) -> str:
    if bytes_val < 1024**2:
        return f"{bytes_val / 1024:.1f} KB"
    if bytes_val < 1024**3:
        return f"{bytes_val / (1024**2):.1f} MB"
    return f"{bytes_val / (1024**3):.3f} GB"


def find_layer_constants(model):
    """Find all Constant nodes and categorize by decoder layer.
    
    Returns:
        layer_constants: dict[int, list[(name, constant_node, data_bytes)]]
        non_layer_constants: list[(name, constant_node, data_bytes)]
    """
    layer_constants = {i: [] for i in range(NUM_DECODER_LAYERS)}
    non_layer_constants = []
    
    ops = model.get_ordered_ops()
    constants_found = 0
    
    for op in ops:
        if op.get_type_name() != "Constant":
            continue
        
        name = op.get_friendly_name()
        constants_found += 1
        
        # Try to match to a decoder layer
        # Pattern: __module.model.language_model.layers.{N}/...
        match = re.search(r"layers\.(\d+)", name)
        if match:
            layer_idx = int(match.group(1))
            if 0 <= layer_idx < NUM_DECODER_LAYERS:
                # Get tensor data as bytes
                try:
                    tensor_data = op.get_data()
                    raw_bytes = tensor_data.tobytes()
                    layer_constants[layer_idx].append((name, op, raw_bytes))
                except Exception as e:
                    print(f"  Warning: Cannot extract data from {name}: {e}")
                continue
        
        # Non-layer constant (embedding, vision encoder, etc.)
        non_layer_constants.append((name, op, None))
    
    print(f"  Total Constant nodes found: {constants_found}")
    print(f"  Layer-associated constants: {sum(len(v) for v in layer_constants.values())}")
    print(f"  Non-layer constants: {len(non_layer_constants)}")
    
    return layer_constants, non_layer_constants


def analyze_layer_sizes(layer_constants):
    """Analyze and print per-layer weight sizes."""
    print("\n=== Per-Layer Weight Analysis ===\n")
    print(f"{'Layer':<8} {'Constants':<12} {'Total Size':<12} {'Avg per Const':<14}")
    print("-" * 50)
    
    total_bytes = 0
    layer_sizes = {}
    
    for layer_idx in range(NUM_DECODER_LAYERS):
        consts = layer_constants[layer_idx]
        layer_bytes = sum(len(raw) for _, _, raw in consts)
        layer_sizes[layer_idx] = layer_bytes
        total_bytes += layer_bytes
        
        if consts:
            avg_size = layer_bytes / len(consts)
            print(f"  {layer_idx:<6} {len(consts):<12} {fmt_size(layer_bytes):<12} {fmt_size(int(avg_size)):<14}")
    
    print("-" * 50)
    print(f"  {'TOTAL':<6} {sum(len(layer_constants[i]) for i in range(NUM_DECODER_LAYERS)):<12} {fmt_size(total_bytes):<12}")
    
    return layer_sizes, total_bytes


def pack_weights(model_dir: Path, group_size: int, output_path: Path,
                 pin_head: int = 5, pin_tail: int = 5, dry_run: bool = False):
    """Pack decoder layer weights into streaming binary format.
    
    Only packs the STREAMED (middle) layers. Pinned head/tail layers stay
    in the original compile_model() USM memory and are not included.
    
    Args:
        model_dir: Path to OpenVINO model directory
        group_size: Number of layers per group
        output_path: Output .bin file path
        pin_head: Number of head decoder layers to pin (default: 5)
        pin_tail: Number of tail decoder layers to pin (default: 5)
        dry_run: If True, only analyze without writing
    """
    model_xml = model_dir / "openvino_language_model.xml"
    if not model_xml.exists():
        print(f"ERROR: Model not found: {model_xml}")
        sys.exit(1)
    
    first_streamed = pin_head
    last_streamed = NUM_DECODER_LAYERS - pin_tail - 1
    num_streamed = NUM_DECODER_LAYERS - pin_head - pin_tail
    
    if num_streamed <= 0:
        print(f"ERROR: pin_head({pin_head}) + pin_tail({pin_tail}) >= {NUM_DECODER_LAYERS}")
        print(f"No layers to stream. All layers are pinned.")
        sys.exit(1)
    
    num_groups = (num_streamed + group_size - 1) // group_size
    
    print(f"=== Dense Weight Streaming Packer (Hybrid Pinning) ===")
    print(f"  Model: {model_xml}")
    print(f"  Hybrid pinning: H{pin_head}+T{pin_tail}")
    print(f"  Pinned head: layers 0-{pin_head - 1} (stay in USM)")
    print(f"  Pinned tail: layers {NUM_DECODER_LAYERS - pin_tail}-{NUM_DECODER_LAYERS - 1} (stay in USM)")
    print(f"  Streamed: layers {first_streamed}-{last_streamed} ({num_streamed} layers)")
    print(f"  Group size: {group_size} layer(s) per group")
    print(f"  Num groups: {num_groups}")
    print(f"  Output: {output_path}")
    print(f"  Dry run: {dry_run}")
    print()
    
    # Load model
    print("Loading model (read_model)...")
    t0 = time.perf_counter()
    core = ov.Core()
    model = core.read_model(str(model_xml))
    t1 = time.perf_counter()
    print(f"  Model loaded in {t1-t0:.2f}s")
    
    # Find and categorize constants
    print("\nCategorizing weight constants...")
    layer_constants, non_layer_constants = find_layer_constants(model)
    
    # Analyze sizes
    layer_sizes, total_weight_bytes = analyze_layer_sizes(layer_constants)
    
    # Calculate group layout — only streamed (non-pinned) layers
    streamed_layers = list(range(first_streamed, last_streamed + 1))
    groups = []
    
    print(f"\n=== Group Layout ({num_groups} groups x {group_size} layer(s), streamed only) ===\n")
    
    for g in range(num_groups):
        g_start_offset = g * group_size
        g_end_offset = min((g + 1) * group_size, num_streamed)
        g_layers = streamed_layers[g_start_offset:g_end_offset]
        
        first_layer = g_layers[0]
        last_layer = g_layers[-1]
        num_layers_in_group = len(g_layers)
        
        group_bytes = sum(layer_sizes.get(l, 0) for l in g_layers)
        groups.append({
            "group_idx": g,
            "first_layer": first_layer,
            "last_layer": last_layer,
            "num_layers": num_layers_in_group,
            "raw_bytes": group_bytes,
            "aligned_bytes": align_to_sector(group_bytes),
        })
        
        io_time_ms = (group_bytes / (12 * 1024**3)) * 1000  # at 12 GB/s
        print(f"  Group {g}: layers {first_layer}-{last_layer} "
              f"({num_layers_in_group} layers) = {fmt_size(group_bytes)} "
              f"  IO@12GB/s: {io_time_ms:.1f} ms")
    
    # Show pinned layers summary
    if pin_head > 0:
        head_bytes = sum(layer_sizes.get(l, 0) for l in range(pin_head))
        print(f"\n  [Pinned HEAD] layers 0-{pin_head-1}: {fmt_size(head_bytes)} (stays in USM)")
    if pin_tail > 0:
        tail_start = NUM_DECODER_LAYERS - pin_tail
        tail_bytes = sum(layer_sizes.get(l, 0) for l in range(tail_start, NUM_DECODER_LAYERS))
        print(f"  [Pinned TAIL] layers {tail_start}-{NUM_DECODER_LAYERS-1}: {fmt_size(tail_bytes)} (stays in USM)")
    
    total_aligned = sum(g["aligned_bytes"] for g in groups)
    data_start_offset = HEADER_SIZE + GROUP_TABLE_SIZE + LAYER_TABLE_SIZE
    total_file_size = data_start_offset + total_aligned
    
    # Streamed weight bytes (only middle layers, not pinned head/tail)
    streamed_weight_bytes = sum(layer_sizes.get(l, 0) for l in streamed_layers)
    
    print(f"\n  Streamed weight data: {fmt_size(streamed_weight_bytes)}")
    print(f"  Total decoder weights: {fmt_size(total_weight_bytes)}")
    print(f"  Total file size:   {fmt_size(total_file_size)} (with headers + alignment)")
    print(f"  Stream IO@12GB/s: {(streamed_weight_bytes / (12 * 1024**3)) * 1000:.1f} ms")
    
    # Performance estimates with hybrid pinning
    print(f"\n=== Performance Estimates (H{pin_head}+T{pin_tail} Hybrid Pipeline) ===\n")
    baseline_tpot_ms = 41.7  # short-text baseline
    per_layer_compute_ms = baseline_tpot_ms / NUM_DECODER_LAYERS
    
    # Pinned layers: pure GPU, no IO
    pinned_gpu_ms = (pin_head + pin_tail) * per_layer_compute_ms
    print(f"  Pinned layers ({pin_head}+{pin_tail}): GPU={pinned_gpu_ms:.1f}ms (no IO)")
    
    for g_info in groups:
        gpu_time_ms = g_info["num_layers"] * per_layer_compute_ms
        io_time_ms = (g_info["raw_bytes"] / (12 * 1024**3)) * 1000
        print(f"  Group {g_info['group_idx']}: GPU={gpu_time_ms:.1f}ms  IO={io_time_ms:.1f}ms  "
              f"bottleneck={'IO' if io_time_ms > gpu_time_ms else 'GPU'}")
    
    # Non-overlapped estimate (conservative) — including pinned layers
    non_overlap_tpot_ms = pinned_gpu_ms + sum(
        (g["raw_bytes"] / (12 * 1024**3)) * 1000 + g["num_layers"] * per_layer_compute_ms
        for g in groups
    )
    
    # Overlapped estimate (optimistic, double-buffer) — including pinned layers
    overlap_tpot_ms = pinned_gpu_ms  # pinned head + tail
    overlap_tpot_ms += (groups[0]["raw_bytes"] / (12 * 1024**3)) * 1000  # initial load
    for i, g_info in enumerate(groups):
        gpu_time_ms = g_info["num_layers"] * per_layer_compute_ms
        if i + 1 < num_groups:
            next_io_ms = (groups[i + 1]["raw_bytes"] / (12 * 1024**3)) * 1000
            overlap_tpot_ms += max(gpu_time_ms, next_io_ms)
        else:
            overlap_tpot_ms += gpu_time_ms  # last group, no next IO
    
    print(f"\n  Non-overlapped: TPOT={non_overlap_tpot_ms:.1f}ms -> {1000/non_overlap_tpot_ms:.1f} tps")
    print(f"  Double-buffer:  TPOT={overlap_tpot_ms:.1f}ms -> {1000/overlap_tpot_ms:.1f} tps")
    print(f"  Baseline:       TPOT={baseline_tpot_ms:.1f}ms -> {1000/baseline_tpot_ms:.1f} tps")
    
    if dry_run:
        print("\n[DRY RUN] No file written.")
        return
    
    # === Write the binary file ===
    print(f"\n=== Writing {output_path} ===\n")
    t_write_start = time.perf_counter()
    
    # Calculate actual data offsets
    current_offset = data_start_offset
    for g in groups:
        g["file_offset"] = current_offset
        current_offset += g["aligned_bytes"]
    
    with open(output_path, "wb") as f:
        # --- Header (4096 bytes) ---
        header = bytearray(HEADER_SIZE)
        struct.pack_into("<4s", header, 0, MAGIC)            # magic
        struct.pack_into("<I", header, 4, VERSION)           # version
        struct.pack_into("<I", header, 8, num_streamed)      # num_layers (streamed only)
        struct.pack_into("<I", header, 12, num_groups)       # num_groups
        struct.pack_into("<I", header, 16, group_size)       # group_size
        struct.pack_into("<Q", header, 24, streamed_weight_bytes)  # total_weight_bytes (streamed)
        struct.pack_into("<Q", header, 32, total_file_size)  # total_file_size
        struct.pack_into("<I", header, 40, SECTOR_SIZE)      # sector_size
        f.write(header)
        
        # --- Group Table (4096 bytes) ---
        # Each entry: offset(8) + size(8) + aligned_size(8) + first_layer(4) + num_layers(4) = 32 bytes
        group_table = bytearray(GROUP_TABLE_SIZE)
        for i, g in enumerate(groups):
            base = i * 32
            struct.pack_into("<Q", group_table, base + 0, g["file_offset"])
            struct.pack_into("<Q", group_table, base + 8, g["raw_bytes"])
            struct.pack_into("<Q", group_table, base + 16, g["aligned_bytes"])
            struct.pack_into("<I", group_table, base + 24, g["first_layer"])
            struct.pack_into("<I", group_table, base + 28, g["num_layers"])
        f.write(group_table)
        
        # --- Per-Layer Table (8192 bytes) ---
        # Each entry: offset_in_group(8) + size(8) + num_tensors(4) + layer_idx(4) = 24 bytes
        layer_table = bytearray(LAYER_TABLE_SIZE)
        
        # Calculate per-layer offset within its group
        # Use PACKED index (0-based) so C++ reader can directly index m_layer_table[packed_idx]
        for g in groups:
            offset_in_group = 0
            for layer_idx in range(g["first_layer"], g["first_layer"] + g["num_layers"]):
                layer_bytes = layer_sizes.get(layer_idx, 0)
                num_tensors = len(layer_constants[layer_idx])
                
                packed_idx = layer_idx - first_streamed  # 0-based packed index
                base = packed_idx * 24
                struct.pack_into("<Q", layer_table, base + 0, offset_in_group)
                struct.pack_into("<Q", layer_table, base + 8, layer_bytes)
                struct.pack_into("<I", layer_table, base + 16, num_tensors)
                struct.pack_into("<I", layer_table, base + 20, layer_idx)  # store model layer idx for reference
                
                offset_in_group += layer_bytes
        
        f.write(layer_table)
        
        # --- Weight Data (per group, sector-aligned) ---
        for g in groups:
            group_data = bytearray()
            
            for layer_idx in range(g["first_layer"], g["first_layer"] + g["num_layers"]):
                # IMPORTANT: Sort tensors by name to match C++ build_weight_mapping()
                # which sorts per_layer_tensors by name for deterministic offset_in_layer.
                sorted_layer = sorted(layer_constants[layer_idx], key=lambda x: x[0])
                for name, op, raw_bytes in sorted_layer:
                    group_data.extend(raw_bytes)
            
            # Pad to sector alignment
            padding_needed = g["aligned_bytes"] - len(group_data)
            if padding_needed > 0:
                group_data.extend(b'\x00' * padding_needed)
            
            f.write(group_data)
            
            io_time = (g["raw_bytes"] / (12 * 1024**3)) * 1000
            print(f"  Group {g['group_idx']}: wrote {fmt_size(g['raw_bytes'])} "
                  f"(layers {g['first_layer']}-{g['last_layer']}) "
                  f"@ offset {g['file_offset']} "
                  f"  IO@12GB/s: {io_time:.1f}ms")
    
    t_write_end = time.perf_counter()
    
    actual_size = os.path.getsize(output_path)
    print(f"\n  File written: {fmt_size(actual_size)} in {t_write_end - t_write_start:.2f}s")
    print(f"  Write speed: {actual_size / (t_write_end - t_write_start) / (1024**3):.2f} GB/s")
    
    # Write metadata JSON (for C++ loader to parse)
    meta_path = output_path.with_suffix(".json")
    metadata = {
        "magic": "DNSW",
        "version": VERSION,
        "total_decoder_layers": NUM_DECODER_LAYERS,
        "pin_head_layers": pin_head,
        "pin_tail_layers": pin_tail,
        "num_streamed_layers": num_streamed,
        "first_streamed_layer": first_streamed,
        "last_streamed_layer": last_streamed,
        "num_groups": num_groups,
        "group_size": group_size,
        "sector_size": SECTOR_SIZE,
        "streamed_weight_bytes": streamed_weight_bytes,
        "total_decoder_weight_bytes": total_weight_bytes,
        "total_file_size": actual_size,
        "groups": [
            {
                "group_idx": g["group_idx"],
                "first_layer": g["first_layer"],
                "last_layer": g["last_layer"],
                "num_layers": g["num_layers"],
                "file_offset": g["file_offset"],
                "raw_bytes": g["raw_bytes"],
                "aligned_bytes": g["aligned_bytes"],
            }
            for g in groups
        ],
        "layers": [
            {
                "layer_idx": i,
                "num_tensors": len(layer_constants[i]),
                "size_bytes": layer_sizes.get(i, 0),
                "pinned": i < pin_head or i >= (NUM_DECODER_LAYERS - pin_tail),
                "tensors": [
                    {
                        "name": name,
                        "size_bytes": len(raw),
                        "dtype": str(op.get_output_element_type(0)),
                        "shape": list(op.get_output_shape(0)),
                    }
                    # Sort by name to match C++ build_weight_mapping() order
                    for name, op, raw in sorted(layer_constants[i], key=lambda x: x[0])
                ],
            }
            for i in range(NUM_DECODER_LAYERS)
        ],
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Metadata written: {meta_path}")
    print(f"\nDone! Use this file with DenseWeightStreamingManager in C++.")


def main():
    parser = argparse.ArgumentParser(
        description="Pack decoder weights into streaming binary format (Hybrid Pinning)"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(r"C:\working\gemma4-openvino\gemma-4-E4B-it-ov"),
        help="Path to OpenVINO model directory",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=1,
        help="Number of layers per streaming group (default: 1 for finest granularity)",
    )
    parser.add_argument(
        "--pin-head",
        type=int,
        default=5,
        help="Number of head decoder layers to pin in USM (default: 5)",
    )
    parser.add_argument(
        "--pin-tail",
        type=int,
        default=5,
        help="Number of tail decoder layers to pin in USM (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: model_dir/dense_weights_streaming.bin)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze weight structure without writing file",
    )
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.model_dir / "dense_weights_streaming.bin"
    
    pack_weights(args.model_dir, args.group_size, args.output,
                 pin_head=args.pin_head, pin_tail=args.pin_tail,
                 dry_run=args.dry_run)


if __name__ == "__main__":
    main()
