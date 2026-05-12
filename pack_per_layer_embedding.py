#!/usr/bin/env python3
"""Pack per-layer embedding weights into DirectIO-aligned binary format.

Reads openvino_text_embeddings_per_layer_model.bin and repacks it into a
4K-page-aligned format suitable for DirectIO / DirectStorage random reads.

Input layout:
  offset 0:                weight [262144, 10752] INT8  (row-major)
  offset 2,818,572,288:    scale  [262144, 1]     FP16

Output layout:
  offset 0:       Header (4096 bytes, 1 page)
  offset 4096:    Row 0: [10752 INT8 weight | 2 FP16 scale | 1534 pad] = 12288 bytes
  offset 16384:   Row 1: ...
  ...
  offset 4096 + 262143*12288: Row 262143

Each row is exactly 3 pages (12 KB), enabling a single 4K-aligned DirectIO read
to fetch both weight and scale for one token.

Usage:
  python pack_per_layer_embedding.py --model-dir <model_path> --output <output.bin>
  python pack_per_layer_embedding.py --model-dir <model_path> --output <output.bin> --verify
"""

import argparse
import struct
import os
import sys
import time
import numpy as np
from pathlib import Path


# Constants
MAGIC = b"PLEB"  # Per-Layer EMBedding
VERSION = 1
VOCAB_SIZE = 262144
PER_LAYER_DIM = 10752      # 42 * 256
NUM_LAYERS = 42
LAYER_DIM = 256
WEIGHT_DTYPE_INT8 = 1
SCALE_DTYPE_FP16 = 2
ROW_STRIDE = 12288         # 3 * 4096
HEADER_SIZE = 4096         # 1 page
WEIGHT_ROW_SIZE = 10752    # INT8 bytes per row
SCALE_ROW_SIZE = 2         # FP16 bytes per row
PAD_SIZE = ROW_STRIDE - WEIGHT_ROW_SIZE - SCALE_ROW_SIZE  # 1534
POST_GATHER_SCALE = 16.0   # Gemma4 per-layer model applies ×16 after Gather+dequant

# Special token IDs that get remapped to row 0 (from IR graph Equal comparisons)
SPECIAL_TOKEN_REMAP = {258880: 0, 258884: 0, 258881: 0}


def write_header(f, vocab_size, per_layer_dim, num_layers, layer_dim):
    """Write 4096-byte header."""
    header = bytearray(HEADER_SIZE)
    struct.pack_into("4s", header, 0, MAGIC)
    struct.pack_into("<I", header, 4, VERSION)
    struct.pack_into("<I", header, 8, vocab_size)
    struct.pack_into("<I", header, 12, per_layer_dim)
    struct.pack_into("<I", header, 16, num_layers)
    struct.pack_into("<I", header, 20, layer_dim)
    struct.pack_into("<I", header, 24, WEIGHT_DTYPE_INT8)
    struct.pack_into("<I", header, 28, SCALE_DTYPE_FP16)
    struct.pack_into("<I", header, 32, ROW_STRIDE)
    f.write(header)


def read_header(f):
    """Read and validate header. Returns dict of metadata."""
    header = f.read(HEADER_SIZE)
    if len(header) != HEADER_SIZE:
        raise ValueError("Header too short: %d bytes" % len(header))

    magic = struct.unpack_from("4s", header, 0)[0]
    if magic != MAGIC:
        raise ValueError("Bad magic: %s (expected %s)" % (magic, MAGIC))

    version = struct.unpack_from("<I", header, 4)[0]
    if version != VERSION:
        raise ValueError("Unsupported version: %d" % version)

    return {
        "version": version,
        "vocab_size": struct.unpack_from("<I", header, 8)[0],
        "per_layer_dim": struct.unpack_from("<I", header, 12)[0],
        "num_layers": struct.unpack_from("<I", header, 16)[0],
        "layer_dim": struct.unpack_from("<I", header, 20)[0],
        "weight_dtype": struct.unpack_from("<I", header, 24)[0],
        "scale_dtype": struct.unpack_from("<I", header, 28)[0],
        "row_stride": struct.unpack_from("<I", header, 32)[0],
    }


def pack(model_dir: Path, output_path: Path, verify: bool = False):
    """Pack per-layer embedding into DirectIO-aligned binary."""
    bin_path = model_dir / "openvino_text_embeddings_per_layer_model.bin"
    if not bin_path.exists():
        print("ERROR: %s not found" % bin_path)
        sys.exit(1)

    file_size = bin_path.stat().st_size
    weight_bytes = VOCAB_SIZE * WEIGHT_ROW_SIZE
    scale_bytes = VOCAB_SIZE * SCALE_ROW_SIZE
    print("Input:  %s (%.3f GB)" % (bin_path, file_size / 1e9))
    print("Weight: [%d, %d] INT8 at offset 0, size %d bytes" % (
        VOCAB_SIZE, PER_LAYER_DIM, weight_bytes))
    print("Scale:  [%d, 1] FP16 at offset %d, size %d bytes" % (
        VOCAB_SIZE, weight_bytes, scale_bytes))
    print("Output: %s" % output_path)
    print("Row stride: %d bytes (%d pages)" % (ROW_STRIDE, ROW_STRIDE // 4096))
    expected_output_size = HEADER_SIZE + VOCAB_SIZE * ROW_STRIDE
    print("Expected output: %.3f GB" % (expected_output_size / 1e9))
    print()

    # Read scale tensor entirely (only 512 KB)
    print("Reading scale tensor...")
    with open(bin_path, "rb") as f:
        f.seek(weight_bytes)
        scale_raw = f.read(scale_bytes)
    if len(scale_raw) != scale_bytes:
        print("ERROR: Scale read incomplete: %d / %d" % (len(scale_raw), scale_bytes))
        sys.exit(1)
    scales = np.frombuffer(scale_raw, dtype=np.float16)  # [262144]
    print("  Scale range: [%.6f, %.6f]" % (scales.min(), scales.max()))

    # Process in chunks to limit memory
    CHUNK_ROWS = 8192  # ~85 MB read + ~96 MB write buffer per chunk
    num_chunks = (VOCAB_SIZE + CHUNK_ROWS - 1) // CHUNK_ROWS
    pad_bytes = bytes(PAD_SIZE)

    print("Packing %d rows in %d chunks of %d..." % (VOCAB_SIZE, num_chunks, CHUNK_ROWS))
    t0 = time.time()

    with open(bin_path, "rb") as fin, open(output_path, "wb") as fout:
        # Write header
        write_header(fout, VOCAB_SIZE, PER_LAYER_DIM, NUM_LAYERS, LAYER_DIM)

        for chunk_idx in range(num_chunks):
            row_start = chunk_idx * CHUNK_ROWS
            row_end = min(row_start + CHUNK_ROWS, VOCAB_SIZE)
            n_rows = row_end - row_start

            # Read weight chunk
            fin.seek(row_start * WEIGHT_ROW_SIZE)
            weight_chunk = fin.read(n_rows * WEIGHT_ROW_SIZE)

            # Write interleaved rows
            for i in range(n_rows):
                row_idx = row_start + i
                w_start = i * WEIGHT_ROW_SIZE
                w_data = weight_chunk[w_start:w_start + WEIGHT_ROW_SIZE]
                s_data = scales[row_idx].tobytes()  # 2 bytes FP16
                fout.write(w_data)
                fout.write(s_data)
                fout.write(pad_bytes)

            if (chunk_idx + 1) % 4 == 0 or chunk_idx == num_chunks - 1:
                pct = (row_end / VOCAB_SIZE) * 100
                elapsed = time.time() - t0
                print("  %d / %d rows (%.1f%%) — %.1f s" % (
                    row_end, VOCAB_SIZE, pct, elapsed))

    elapsed = time.time() - t0
    output_size = output_path.stat().st_size
    print()
    print("Done! Output: %.3f GB in %.1f s (%.1f MB/s write)" % (
        output_size / 1e9, elapsed, output_size / 1e6 / elapsed))

    # Validate output size
    if output_size != expected_output_size:
        print("WARNING: Output size mismatch! Expected %d, got %d" % (
            expected_output_size, output_size))
    else:
        print("Output size: ✅ correct (%d bytes)" % output_size)

    if verify:
        verify_repacked(model_dir, output_path)


def lookup_row(f, token_id: int, meta: dict) -> tuple:
    """Read one row from repacked binary. Returns (weight_int8[10752], scale_fp16)."""
    offset = HEADER_SIZE + token_id * meta["row_stride"]
    f.seek(offset)
    row_data = f.read(meta["row_stride"])
    weight = np.frombuffer(row_data[:meta["per_layer_dim"]], dtype=np.int8)
    scale = np.frombuffer(row_data[meta["per_layer_dim"]:meta["per_layer_dim"] + 2],
                          dtype=np.float16)[0]
    return weight, scale


def dequant_row(weight_int8: np.ndarray, scale_fp16) -> np.ndarray:
    """Dequantize: int8 × fp16_scale × POST_GATHER_SCALE → fp32[10752]."""
    return weight_int8.astype(np.float32) * float(scale_fp16) * POST_GATHER_SCALE


def resolve_token_id(token_id: int) -> int:
    """Remap special token IDs; return -1 for out-of-range (→ zeros output)."""
    if token_id in SPECIAL_TOKEN_REMAP:
        return SPECIAL_TOKEN_REMAP[token_id]
    if token_id < 0 or token_id >= VOCAB_SIZE:
        return -1  # out of range → output zeros
    return token_id


def verify_repacked(model_dir: Path, repacked_path: Path, num_samples: int = 20):
    """Verify repacked binary matches original model output."""
    print()
    print("=" * 60)
    print("Verifying repacked binary against original model...")
    print("=" * 60)

    try:
        import openvino as ov
    except ImportError:
        print("WARNING: openvino not installed, skipping model verification")
        return verify_binary_consistency(model_dir, repacked_path, num_samples)

    core = ov.Core()
    xml_path = model_dir / "openvino_text_embeddings_per_layer_model.xml"
    if not xml_path.exists():
        print("WARNING: %s not found, skipping model verification" % xml_path)
        return verify_binary_consistency(model_dir, repacked_path, num_samples)

    # Compile model for CPU (reference)
    print("Compiling per-layer model on CPU for reference...")
    compiled = core.compile_model(str(xml_path), "CPU")
    infer = compiled.create_infer_request()

    # Open repacked binary
    with open(repacked_path, "rb") as f:
        meta = read_header(f)
        print("Header: %s" % meta)

        # Test specific token IDs
        test_ids = [0, 1, 100, 1000, 10000, 50000, 100000, 200000, 262143]
        # Add random samples
        rng = np.random.default_rng(42)
        test_ids.extend(rng.integers(0, VOCAB_SIZE, size=num_samples - len(test_ids)).tolist())
        test_ids = sorted(set(test_ids))[:num_samples]

        max_abs_err = 0.0
        max_rel_err = 0.0
        all_pass = True

        for token_id in test_ids:
            # Reference: compiled model
            input_tensor = np.array([[token_id]], dtype=np.int64)
            infer.set_tensor("input_ids", ov.Tensor(input_tensor))
            infer.infer()
            ref_output = infer.get_output_tensor().data.copy()  # [1, 1, 42, 256] FP32

            # Our implementation: DirectIO read + dequant
            resolved = resolve_token_id(token_id)
            if resolved < 0:
                our_output = np.zeros((1, 1, NUM_LAYERS, LAYER_DIM), dtype=np.float32)
            else:
                weight, scale = lookup_row(f, resolved, meta)
                our_output = dequant_row(weight, scale).reshape(1, 1, NUM_LAYERS, LAYER_DIM)

            # Compare
            abs_err = np.abs(ref_output - our_output).max()
            ref_max = np.abs(ref_output).max()
            rel_err = abs_err / max(ref_max, 1e-8)

            max_abs_err = max(max_abs_err, abs_err)
            max_rel_err = max(max_rel_err, rel_err)

            status = "✅" if abs_err < 0.01 else "❌"
            if abs_err >= 0.01:
                all_pass = False
            print("  token %6d: abs_err=%.6f rel_err=%.6f ref_range=[%.4f,%.4f] %s" % (
                token_id, abs_err, rel_err, ref_output.min(), ref_output.max(), status))

    print()
    print("Max absolute error: %.6f" % max_abs_err)
    print("Max relative error: %.6f" % max_rel_err)
    if all_pass:
        print("✅ All %d samples PASS (abs_err < 0.01)" % len(test_ids))
    else:
        print("❌ Some samples FAILED — investigate dequant differences")
        print("   NOTE: The compiled model may apply additional scaling or special token handling.")
        print("   Check for Multiply nodes after Gather in the IR graph.")


def verify_binary_consistency(model_dir: Path, repacked_path: Path, num_samples: int = 10):
    """Verify repacked binary is consistent with original .bin (no OpenVINO needed)."""
    print("Fallback: verifying raw binary consistency...")

    orig_bin = model_dir / "openvino_text_embeddings_per_layer_model.bin"
    weight_bytes = VOCAB_SIZE * WEIGHT_ROW_SIZE
    scale_offset = weight_bytes

    with open(orig_bin, "rb") as forig, open(repacked_path, "rb") as frep:
        meta = read_header(frep)

        rng = np.random.default_rng(42)
        test_ids = sorted(set([0, 1, 100, 262143] +
                              rng.integers(0, VOCAB_SIZE, size=num_samples).tolist()))

        for token_id in test_ids:
            # Read from original
            forig.seek(token_id * WEIGHT_ROW_SIZE)
            orig_weight = forig.read(WEIGHT_ROW_SIZE)
            forig.seek(scale_offset + token_id * 2)
            orig_scale = forig.read(2)

            # Read from repacked
            weight, scale = lookup_row(frep, token_id, meta)
            rep_weight = weight.tobytes()
            rep_scale = np.float16(scale).tobytes()

            w_match = orig_weight == rep_weight
            s_match = orig_scale == rep_scale

            status = "✅" if (w_match and s_match) else "❌"
            print("  token %6d: weight=%s scale=%s %s" % (
                token_id,
                "match" if w_match else "MISMATCH",
                "match" if s_match else "MISMATCH",
                status))


def main():
    parser = argparse.ArgumentParser(
        description="Pack per-layer embedding for DirectIO access")
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="Path to model directory containing .bin/.xml")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output path for repacked binary")
    parser.add_argument("--verify", action="store_true",
                        help="Verify repacked output against compiled model")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing repacked binary (skip packing)")
    args = parser.parse_args()

    if args.verify_only:
        if not args.output.exists():
            print("ERROR: %s not found" % args.output)
            sys.exit(1)
        verify_repacked(args.model_dir, args.output)
    else:
        pack(args.model_dir, args.output, verify=args.verify)


if __name__ == "__main__":
    main()
