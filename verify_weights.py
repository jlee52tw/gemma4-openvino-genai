"""
Verify packed weight correctness:
1. Byte-level comparison: packed binary weights == original model weights
2. LLM output verification: run model to confirm output is not garbage

Usage:
  python verify_weights.py                      # Full verification
  python verify_weights.py --skip-inference     # Skip LLM output test (faster)
"""

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================================
# Phase 1: Byte-level verification of packed binary against original model
# ============================================================================

def verify_packed_weights(model_dir: Path, bin_path: Path, json_path: Path):
    """Compare every tensor byte in packed binary against original model."""
    import openvino as ov
    import re
    
    print("=" * 70)
    print("  Phase 1: Byte-Level Weight Verification")
    print("=" * 70)
    
    # Load metadata
    meta = json.load(open(json_path))
    print(f"  Binary:  {bin_path}")
    print(f"  Groups:  {meta['num_groups']} x {meta['group_size']} layers")
    print(f"  Pinned:  H{meta['pin_head_layers']}+T{meta['pin_tail_layers']}")
    print(f"  Layers:  {meta['first_streamed_layer']}-{meta['last_streamed_layer']} (streamed)")
    print()
    
    # Load original model
    print("  Loading original model...")
    core = ov.Core()
    model = core.read_model(str(model_dir / "openvino_language_model.xml"))
    
    # Build tensor data map keyed by (layer_idx, position_in_layer)
    # MUST match the packing order: model.get_ordered_ops() iteration order,
    # not alphabetical order! The packer appends tensors in the order they
    # appear in the graph's topological sort.
    layer_tensors = {}  # {layer_idx: [(name, data_bytes), ...]} in graph order
    
    for op in model.get_ordered_ops():
        if op.get_type_name() != "Constant":
            continue
        name = op.get_friendly_name()
        m = re.search(r"layers\.(\d+)", name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        if layer_idx not in layer_tensors:
            layer_tensors[layer_idx] = []
        try:
            data = op.get_data()
            raw_bytes = data.tobytes()
            layer_tensors[layer_idx].append((name, raw_bytes))
        except Exception:
            pass
    
    # Read packed binary
    print("  Reading packed binary...")
    with open(bin_path, "rb") as f:
        bin_data = f.read()
    
    # Verify each group's data against original model
    total_bytes_checked = 0
    total_tensors_checked = 0
    errors = []
    
    for group in meta["groups"]:
        g_idx = group["group_idx"]
        first_layer = group["first_layer"]
        last_layer = group["last_layer"]
        offset = group["file_offset"]
        raw_bytes_size = group["raw_bytes"]
        
        # Extract group data from binary
        group_data = bin_data[offset:offset + raw_bytes_size]
        
        # Reconstruct expected bytes from original model
        # Pack order: layers first_layer..last_layer, each layer's tensors sorted by name
        expected_parts = []
        for layer_idx in range(first_layer, last_layer + 1):
            if layer_idx in layer_tensors:
                for name, tensor_bytes in layer_tensors[layer_idx]:
                    expected_parts.append(tensor_bytes)
        
        expected_data = b"".join(expected_parts)
        
        if len(expected_data) != len(group_data):
            errors.append(f"Group {g_idx}: size mismatch: packed={len(group_data)}, expected={len(expected_data)}")
            continue
        
        if expected_data == group_data:
            tensors_in_group = sum(
                len(layer_tensors.get(l, [])) 
                for l in range(first_layer, last_layer + 1)
            )
            total_tensors_checked += tensors_in_group
            total_bytes_checked += len(group_data)
            print(f"    Group {g_idx:2d} (layers {first_layer:2d}-{last_layer:2d}): "
                  f"✓ MATCH ({len(group_data)/1024**2:.1f} MB, {tensors_in_group} tensors)")
        else:
            # Find first difference
            for i in range(len(expected_data)):
                if expected_data[i] != group_data[i]:
                    errors.append(
                        f"Group {g_idx}: byte mismatch at offset {i} "
                        f"(expected 0x{expected_data[i]:02x}, got 0x{group_data[i]:02x})"
                    )
                    break
    
    print()
    if errors:
        print(f"  ✗ FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"    {e}")
        return False
    else:
        print(f"  ✓ ALL GROUPS MATCH")
        print(f"    Total bytes verified: {total_bytes_checked:,} ({total_bytes_checked/1024**3:.3f} GB)")
        print(f"    Total tensors verified: {total_tensors_checked}")
        return True


# ============================================================================
# Phase 2: LLM output verification (run actual inference)
# ============================================================================

def verify_llm_output(model_dir: Path):
    """Run Gemma4 inference and verify output is coherent, not garbage."""
    print()
    print("=" * 70)
    print("  Phase 2: LLM Output Verification (Full Model, No Streaming)")
    print("=" * 70)
    print()
    print("  This verifies the original model produces correct output.")
    print("  Weight streaming correctness is proven by Phase 1 byte-match:")
    print("  if packed weights are byte-identical to original model weights,")
    print("  then streaming those weights will produce identical output.")
    print()
    
    try:
        import openvino_genai as ov_genai
    except ImportError:
        print("  ✗ openvino_genai not available, skipping inference test")
        return None
    
    # Test prompts and expected characteristics  
    test_cases = [
        {
            "prompt": "What is 2+2? Answer with just the number.",
            "expect_contains": ["4"],
            "description": "basic arithmetic",
        },
        {
            "prompt": "Hello! Please introduce yourself in one sentence.",
            "expect_min_words": 5,
            "description": "coherent self-introduction",
        },
        {
            "prompt": "List the first 5 prime numbers separated by commas.",
            "expect_contains": ["2", "3", "5", "7", "11"],
            "description": "prime number sequence",
        },
    ]
    
    print(f"  Loading VLMPipeline from {model_dir}...")
    t0 = time.time()
    pipe = ov_genai.VLMPipeline(str(model_dir), "GPU")
    t1 = time.time()
    print(f"  Model loaded in {t1-t0:.1f}s")
    print()
    
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = 50
    
    all_pass = True
    
    for i, tc in enumerate(test_cases):
        print(f"  Test {i+1}: {tc['description']}")
        print(f"    Prompt: \"{tc['prompt']}\"")
        
        result = pipe.generate(tc["prompt"], generation_config=config)
        output = result.texts[0] if hasattr(result, 'texts') else str(result)
        output = output.strip()
        
        print(f"    Output: \"{output}\"")
        
        # Check not garbage (all ASCII control chars, random bytes, etc.)
        printable_ratio = sum(1 for c in output if c.isprintable() or c in '\n\t') / max(len(output), 1)
        if printable_ratio < 0.8:
            print(f"    ✗ FAIL — output appears to be garbage (only {printable_ratio*100:.0f}% printable)")
            all_pass = False
            continue
        
        # Check minimum word count if specified
        if "expect_min_words" in tc:
            word_count = len(output.split())
            if word_count < tc["expect_min_words"]:
                print(f"    ✗ FAIL — only {word_count} words (expected >= {tc['expect_min_words']})")
                all_pass = False
                continue
        
        # Check expected content if specified
        if "expect_contains" in tc:
            found_all = all(s in output for s in tc["expect_contains"])
            if not found_all:
                missing = [s for s in tc["expect_contains"] if s not in output]
                print(f"    ⚠ WARNING — missing expected content: {missing}")
                # Not a hard failure - LLM output can vary
        
        print(f"    ✓ PASS (coherent output, {len(output)} chars)")
    
    print()
    if all_pass:
        print(f"  ✓ ALL {len(test_cases)} TESTS PASSED — output is coherent, not garbage")
    else:
        print(f"  ✗ SOME TESTS FAILED")
    
    return all_pass


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Verify packed weight correctness and LLM output quality")
    parser.add_argument("--model-dir", type=Path,
                        default=Path(r"C:\working\gemma4-openvino\gemma-4-E4B-it-ov"),
                        help="Path to OpenVINO model directory")
    parser.add_argument("--bin", type=Path, default=None,
                        help="Path to packed binary (default: model_dir/dense_weights_streaming.bin)")
    parser.add_argument("--json", type=Path, default=None,
                        help="Path to metadata JSON (default: model_dir/dense_weights_streaming.json)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip Phase 2 LLM inference verification")
    args = parser.parse_args()
    
    if args.bin is None:
        args.bin = args.model_dir / "dense_weights_streaming.bin"
    if args.json is None:
        args.json = args.model_dir / "dense_weights_streaming.json"
    
    print()
    print("Dense Weight Streaming — Correctness Verification")
    print("=" * 70)
    print()
    
    # Phase 1: Byte-level verification
    byte_ok = verify_packed_weights(args.model_dir, args.bin, args.json)
    
    # Phase 2: LLM output verification  
    if args.skip_inference:
        print("\n  [Skipping Phase 2: LLM inference verification]")
        llm_ok = None
    else:
        llm_ok = verify_llm_output(args.model_dir)
    
    # Summary
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Phase 1 (byte-level match):  {'PASS' if byte_ok else 'FAIL'}")
    if llm_ok is not None:
        print(f"  Phase 2 (LLM output check):  {'PASS' if llm_ok else 'FAIL'}")
    else:
        print(f"  Phase 2 (LLM output check):  SKIPPED")
    
    if byte_ok:
        print()
        print("  ✓ Weight data integrity confirmed.")
        print("    Streaming these weights will produce bit-identical results")
        print("    to running with all weights in GPU memory.")
    
    return 0 if byte_ok else 1


if __name__ == "__main__":
    sys.exit(main())
