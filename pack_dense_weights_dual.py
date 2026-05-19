#!/usr/bin/env python3
"""
Dense Weight Streaming — Dual-NVMe Weight Packing Tool
=======================================================
Splits decoder layer weight data across 2 files for parallel NVMe read.

Striping Strategy: Group-Half
  For each group (~200 MB):
    - File 0 (NVMe 0, e.g. C:\): first half of group data (sector-aligned)
    - File 1 (NVMe 1, e.g. D:\): second half of group data (sector-aligned)

At runtime, the C++ manager reads both files simultaneously with
async ReadFile, achieving ~24 GB/s combined throughput instead of ~12 GB/s.

File Format (each stripe file):
  [Header]        — 4096 bytes (includes stripe_index and num_stripes)
  [Group Table]   — 4096 bytes (file_offset and aligned_bytes for THIS stripe)
  [Layer Table]   — 8192 bytes (same in both files)
  [Stripe Data]   — sector-aligned portions of each group

Usage:
  # Split existing single-file weights into 2 stripe files
  python pack_dense_weights_dual.py \\
    --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov \\
    --output-0 C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov\\dense_weights_streaming_0.bin \\
    --output-1 D:\\dense_stream\\dense_weights_streaming_1.bin

  # Dry run to see split info
  python pack_dense_weights_dual.py --model-dir ... --dry-run
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
SECTOR_SIZE = 4096
MAGIC = b"DNSW"
VERSION = 2  # Version 2: dual-NVMe striping support
NUM_DECODER_LAYERS = 42
HEADER_SIZE = SECTOR_SIZE
GROUP_TABLE_SIZE = SECTOR_SIZE
LAYER_TABLE_SIZE = SECTOR_SIZE * 2

# Header V2 fields (extends V1):
#   offset 44: uint16_t num_stripes (1 = single, 2 = dual-NVMe)
#   offset 46: uint16_t stripe_index (0 or 1)
#   offset 48: uint64_t full_group_aligned_bytes[0..num_groups-1] — original full size
#              (stored in header reserved space, max 42 groups × 8 bytes = 336 bytes)
# This allows the C++ reader to know the FULL buffer size needed.

# Weight component patterns per layer
WEIGHT_PATTERNS = [
    r"layers\.{layer}.*self_attn.*q_proj.*weight",
    r"layers\.{layer}.*self_attn.*k_proj.*weight",
    r"layers\.{layer}.*self_attn.*v_proj.*weight",
    r"layers\.{layer}.*self_attn.*o_proj.*weight",
    r"layers\.{layer}.*mlp.*gate_proj.*weight",
    r"layers\.{layer}.*mlp.*up_proj.*weight",
    r"layers\.{layer}.*mlp.*down_proj.*weight",
    r"layers\.{layer}.*input_layernorm.*weight",
    r"layers\.{layer}.*post_attention_layernorm.*weight",
    r"layers\.{layer}.*pre_feedforward_layernorm.*weight",
    r"layers\.{layer}.*post_feedforward_layernorm.*weight",
]

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
    """Find all Constant nodes and categorize by decoder layer."""
    layer_constants = {i: [] for i in range(NUM_DECODER_LAYERS)}
    non_layer_constants = []

    ops = model.get_ordered_ops()
    constants_found = 0

    for op in ops:
        if op.get_type_name() != "Constant":
            continue

        name = op.get_friendly_name()
        constants_found += 1

        match = re.search(r"layers\.(\d+)", name)
        if match:
            layer_idx = int(match.group(1))
            if 0 <= layer_idx < NUM_DECODER_LAYERS:
                try:
                    tensor_data = op.get_data()
                    raw_bytes = tensor_data.tobytes()
                    layer_constants[layer_idx].append((name, op, raw_bytes))
                except Exception as e:
                    print(f"  Warning: Cannot extract data from {name}: {e}")
                continue

        non_layer_constants.append((name, op, None))

    print(f"  Total Constant nodes found: {constants_found}")
    print(f"  Layer-associated constants: {sum(len(v) for v in layer_constants.values())}")
    print(f"  Non-layer constants: {len(non_layer_constants)}")

    return layer_constants, non_layer_constants


def analyze_layer_sizes(layer_constants):
    """Analyze and print per-layer weight sizes."""
    print("\n=== Per-Layer Weight Analysis ===\n")
    print(f"{'Layer':<8} {'Constants':<12} {'Total Size':<12}")
    print("-" * 40)

    total_bytes = 0
    layer_sizes = {}

    for layer_idx in range(NUM_DECODER_LAYERS):
        consts = layer_constants[layer_idx]
        layer_bytes = sum(len(raw) for _, _, raw in consts)
        layer_sizes[layer_idx] = layer_bytes
        total_bytes += layer_bytes

        if consts:
            print(f"  {layer_idx:<6} {len(consts):<12} {fmt_size(layer_bytes):<12}")

    print("-" * 40)
    print(f"  {'TOTAL':<6} {'':<12} {fmt_size(total_bytes):<12}")

    return layer_sizes, total_bytes


def compute_group_split(aligned_bytes: int) -> tuple:
    """Split a group's aligned bytes into two sector-aligned halves.

    Returns (size_0, size_1) where size_0 + size_1 == aligned_bytes
    and both are sector-aligned.
    """
    # Split at midpoint, aligned down to sector
    half = (aligned_bytes // 2 // SECTOR_SIZE) * SECTOR_SIZE
    if half == 0:
        half = SECTOR_SIZE  # Minimum one sector for stripe 0
    size_0 = half
    size_1 = aligned_bytes - half
    assert size_0 % SECTOR_SIZE == 0
    assert size_1 % SECTOR_SIZE == 0
    assert size_0 + size_1 == aligned_bytes
    return size_0, size_1


def pack_weights_dual(model_dir: Path, group_size: int,
                      output_0: Path, output_1: Path,
                      pin_head: int = 5, pin_tail: int = 5,
                      dry_run: bool = False):
    """Pack decoder weights into dual-NVMe striped format.

    Creates 2 files: each contains half of every group's weight data.
    Both files read simultaneously at runtime for ~2x NVMe bandwidth.
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
        sys.exit(1)

    num_groups = (num_streamed + group_size - 1) // group_size

    print(f"=== Dense Weight Streaming Packer (Dual-NVMe Striping) ===")
    print(f"  Model: {model_xml}")
    print(f"  Hybrid pinning: H{pin_head}+T{pin_tail}")
    print(f"  Streamed: layers {first_streamed}-{last_streamed} ({num_streamed} layers)")
    print(f"  Group size: {group_size} layer(s) per group")
    print(f"  Num groups: {num_groups}")
    print(f"  Stripe 0: {output_0}")
    print(f"  Stripe 1: {output_1}")
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
    layer_constants, _ = find_layer_constants(model)

    # Analyze sizes
    layer_sizes, total_weight_bytes = analyze_layer_sizes(layer_constants)

    # Calculate group layout
    streamed_layers = list(range(first_streamed, last_streamed + 1))
    groups = []

    for g in range(num_groups):
        g_start_offset = g * group_size
        g_end_offset = min((g + 1) * group_size, num_streamed)
        g_layers = streamed_layers[g_start_offset:g_end_offset]

        first_layer = g_layers[0]
        last_layer = g_layers[-1]
        num_layers_in_group = len(g_layers)

        group_bytes = sum(layer_sizes.get(l, 0) for l in g_layers)
        full_aligned = align_to_sector(group_bytes)
        split_0, split_1 = compute_group_split(full_aligned)

        groups.append({
            "group_idx": g,
            "first_layer": first_layer,
            "last_layer": last_layer,
            "num_layers": num_layers_in_group,
            "raw_bytes": group_bytes,
            "full_aligned_bytes": full_aligned,
            "stripe_0_bytes": split_0,
            "stripe_1_bytes": split_1,
        })

    # Print group and stripe info
    print(f"\n=== Dual-NVMe Stripe Layout ({num_groups} groups) ===\n")
    print(f"  {'Group':<6} {'Layers':<12} {'Full Size':<12} {'Stripe 0':<12} {'Stripe 1':<12} {'IO@24GB/s'}")
    print(f"  {'-'*70}")

    for g in groups:
        io_time_dual = (g["full_aligned_bytes"] / (24 * 1024**3)) * 1000
        print(f"  {g['group_idx']:<6} {g['first_layer']}-{g['last_layer']:<8} "
              f"{fmt_size(g['full_aligned_bytes']):<12} "
              f"{fmt_size(g['stripe_0_bytes']):<12} "
              f"{fmt_size(g['stripe_1_bytes']):<12} "
              f"{io_time_dual:.1f} ms")

    # Calculate file sizes
    data_start_offset = HEADER_SIZE + GROUP_TABLE_SIZE + LAYER_TABLE_SIZE

    total_stripe_0 = sum(g["stripe_0_bytes"] for g in groups)
    total_stripe_1 = sum(g["stripe_1_bytes"] for g in groups)
    file_0_size = data_start_offset + total_stripe_0
    file_1_size = data_start_offset + total_stripe_1
    total_original = sum(g["full_aligned_bytes"] for g in groups)

    print(f"\n  Original single-file data: {fmt_size(total_original)}")
    print(f"  Stripe 0 file size: {fmt_size(file_0_size)}")
    print(f"  Stripe 1 file size: {fmt_size(file_1_size)}")
    print(f"  Combined: {fmt_size(file_0_size + file_1_size)}")

    # Performance estimates
    print(f"\n=== Performance Estimates (Dual-NVMe Pipeline) ===\n")
    per_layer_compute_ms = 41.7 / NUM_DECODER_LAYERS
    pinned_gpu_ms = (pin_head + pin_tail) * per_layer_compute_ms

    # With 2 NVMe (24 GB/s), IO time halved
    group_io_ms_single = [(g["full_aligned_bytes"] / (12 * 1024**3)) * 1000 for g in groups]
    group_io_ms_dual = [(g["full_aligned_bytes"] / (24 * 1024**3)) * 1000 for g in groups]
    group_gpu_ms = [g["num_layers"] * per_layer_compute_ms for g in groups]

    # Single NVMe overlapped
    single_tpot = pinned_gpu_ms + group_io_ms_single[0]
    for i in range(num_groups):
        if i + 1 < num_groups:
            single_tpot += max(group_gpu_ms[i], group_io_ms_single[i + 1])
        else:
            single_tpot += group_gpu_ms[i]

    # Dual NVMe overlapped
    dual_tpot = pinned_gpu_ms + group_io_ms_dual[0]
    for i in range(num_groups):
        if i + 1 < num_groups:
            dual_tpot += max(group_gpu_ms[i], group_io_ms_dual[i + 1])
        else:
            dual_tpot += group_gpu_ms[i]

    print(f"  Single NVMe (12 GB/s): TPOT ≈ {single_tpot:.1f} ms → {1000/single_tpot:.1f} tps")
    print(f"  Dual NVMe (24 GB/s):   TPOT ≈ {dual_tpot:.1f} ms → {1000/dual_tpot:.1f} tps")
    print(f"  Improvement: {(1 - dual_tpot/single_tpot)*100:.1f}%")

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # === Write both stripe files ===
    print(f"\n=== Writing stripe files ===\n")
    t_write_start = time.perf_counter()

    # Calculate file offsets for each stripe
    current_offset_0 = data_start_offset
    current_offset_1 = data_start_offset
    for g in groups:
        g["file_offset_0"] = current_offset_0
        g["file_offset_1"] = current_offset_1
        current_offset_0 += g["stripe_0_bytes"]
        current_offset_1 += g["stripe_1_bytes"]

    # Streamed weight bytes
    streamed_weight_bytes = sum(layer_sizes.get(l, 0) for l in streamed_layers)

    def write_stripe_file(output_path: Path, stripe_index: int):
        """Write one stripe file."""
        with open(output_path, "wb") as f:
            # --- Header (4096 bytes) ---
            header = bytearray(HEADER_SIZE)
            struct.pack_into("<4s", header, 0, MAGIC)
            struct.pack_into("<I", header, 4, VERSION)  # Version 2
            struct.pack_into("<I", header, 8, num_streamed)
            struct.pack_into("<I", header, 12, num_groups)
            struct.pack_into("<I", header, 16, group_size)
            # offset 20: reserved (was padding in v1)
            struct.pack_into("<Q", header, 24, streamed_weight_bytes)
            file_size = file_0_size if stripe_index == 0 else file_1_size
            struct.pack_into("<Q", header, 32, file_size)
            struct.pack_into("<I", header, 40, SECTOR_SIZE)
            # V2 additions at offset 44:
            struct.pack_into("<H", header, 44, 2)              # num_stripes = 2
            struct.pack_into("<H", header, 46, stripe_index)   # stripe_index (0 or 1)
            # Store full_aligned_bytes for each group (for buffer allocation)
            # Starting at offset 48, up to 42 groups × 8 bytes = 336 bytes (fits in 4096)
            for i, g in enumerate(groups):
                struct.pack_into("<Q", header, 48 + i * 8, g["full_aligned_bytes"])
            f.write(header)

            # --- Group Table (4096 bytes) ---
            # Each entry: file_offset(8) + raw_bytes(8) + aligned_bytes(8) + first_layer(4) + num_layers(4)
            # For stripe files: file_offset and aligned_bytes refer to THIS stripe's portion
            group_table = bytearray(GROUP_TABLE_SIZE)
            for i, g in enumerate(groups):
                base = i * 32
                if stripe_index == 0:
                    struct.pack_into("<Q", group_table, base + 0, g["file_offset_0"])
                    struct.pack_into("<Q", group_table, base + 8, g["stripe_0_bytes"])  # raw = stripe size
                    struct.pack_into("<Q", group_table, base + 16, g["stripe_0_bytes"])  # aligned = stripe size
                else:
                    struct.pack_into("<Q", group_table, base + 0, g["file_offset_1"])
                    struct.pack_into("<Q", group_table, base + 8, g["stripe_1_bytes"])
                    struct.pack_into("<Q", group_table, base + 16, g["stripe_1_bytes"])
                struct.pack_into("<I", group_table, base + 24, g["first_layer"])
                struct.pack_into("<I", group_table, base + 28, g["num_layers"])
            f.write(group_table)

            # --- Per-Layer Table (8192 bytes) --- same for both stripes
            layer_table = bytearray(LAYER_TABLE_SIZE)
            for g in groups:
                offset_in_group = 0
                for layer_idx in range(g["first_layer"], g["first_layer"] + g["num_layers"]):
                    layer_bytes = layer_sizes.get(layer_idx, 0)
                    num_tensors = len(layer_constants[layer_idx])
                    packed_idx = layer_idx - first_streamed
                    base = packed_idx * 24
                    struct.pack_into("<Q", layer_table, base + 0, offset_in_group)
                    struct.pack_into("<Q", layer_table, base + 8, layer_bytes)
                    struct.pack_into("<I", layer_table, base + 16, num_tensors)
                    struct.pack_into("<I", layer_table, base + 20, layer_idx)
                    offset_in_group += layer_bytes
            f.write(layer_table)

            # --- Stripe Data ---
            for g in groups:
                # Pack full group data
                group_data = bytearray()
                for layer_idx in range(g["first_layer"], g["first_layer"] + g["num_layers"]):
                    sorted_layer = sorted(layer_constants[layer_idx], key=lambda x: x[0])
                    for name, op, raw_bytes in sorted_layer:
                        group_data.extend(raw_bytes)

                # Pad to full aligned size
                full_aligned = g["full_aligned_bytes"]
                padding_needed = full_aligned - len(group_data)
                if padding_needed > 0:
                    group_data.extend(b'\x00' * padding_needed)

                # Extract this stripe's portion
                if stripe_index == 0:
                    stripe_data = group_data[:g["stripe_0_bytes"]]
                else:
                    stripe_data = group_data[g["stripe_0_bytes"]:]

                f.write(stripe_data)

        actual_size = os.path.getsize(output_path)
        print(f"  Stripe {stripe_index}: {output_path} → {fmt_size(actual_size)}")

    write_stripe_file(output_0, stripe_index=0)
    write_stripe_file(output_1, stripe_index=1)

    t_write_end = time.perf_counter()
    print(f"\n  Written in {t_write_end - t_write_start:.2f}s")

    # Write metadata JSON
    meta_path = output_0.with_name("dense_weights_streaming_dual.json")
    metadata = {
        "format": "dual-nvme-striping",
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
        "num_stripes": 2,
        "stripe_files": [str(output_0), str(output_1)],
        "streamed_weight_bytes": streamed_weight_bytes,
        "groups": [
            {
                "group_idx": g["group_idx"],
                "first_layer": g["first_layer"],
                "last_layer": g["last_layer"],
                "num_layers": g["num_layers"],
                "raw_bytes": g["raw_bytes"],
                "full_aligned_bytes": g["full_aligned_bytes"],
                "stripe_0_bytes": g["stripe_0_bytes"],
                "stripe_1_bytes": g["stripe_1_bytes"],
                "file_offset_0": g["file_offset_0"],
                "file_offset_1": g["file_offset_1"],
            }
            for g in groups
        ],
        "performance_estimates": {
            "single_nvme_tpot_ms": round(single_tpot, 1),
            "dual_nvme_tpot_ms": round(dual_tpot, 1),
            "improvement_pct": round((1 - dual_tpot / single_tpot) * 100, 1),
        },
    }

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {meta_path}")

    print(f"\n=== Done! ===")
    print(f"  Set environment variables:")
    print(f"    $env:OV_DENSE_STREAM_WEIGHTS = \"{output_0}\"")
    print(f"    $env:OV_DENSE_STREAM_WEIGHTS_2 = \"{output_1}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Pack decoder weights into dual-NVMe striped format"
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
        default=4,
        help="Number of layers per streaming group (default: 4)",
    )
    parser.add_argument(
        "--pin-head",
        type=int,
        default=5,
        help="Number of head decoder layers to pin (default: 5)",
    )
    parser.add_argument(
        "--pin-tail",
        type=int,
        default=5,
        help="Number of tail decoder layers to pin (default: 5)",
    )
    parser.add_argument(
        "--output-0",
        type=Path,
        default=None,
        help="Output path for stripe 0 (NVMe 0, default: model_dir/dense_weights_streaming_0.bin)",
    )
    parser.add_argument(
        "--output-1",
        type=Path,
        default=None,
        help="Output path for stripe 1 (NVMe 1, e.g. D:\\dense_stream\\dense_weights_streaming_1.bin)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze stripe layout without writing files",
    )

    args = parser.parse_args()

    if args.output_0 is None:
        args.output_0 = args.model_dir / "dense_weights_streaming_0.bin"
    if args.output_1 is None:
        args.output_1 = args.model_dir / "dense_weights_streaming_1.bin"

    pack_weights_dual(
        args.model_dir, args.group_size,
        args.output_0, args.output_1,
        pin_head=args.pin_head, pin_tail=args.pin_tail,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
