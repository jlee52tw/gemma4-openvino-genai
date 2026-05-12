#!/usr/bin/env python3
"""
Dense Weight Streaming — Phase 1: Blob-Cache Load Time Benchmark
=================================================================
Measures the critical timing data needed for the model splitting decision:

1. Full language model blob-cached compile time (warm load from .cl_cache / CACHE_DIR)
2. Unload time (release compiled model, free GPU memory)
3. Load→unload cycle time (repeated load/unload for average)
4. Memory footprint at each stage

This gives us the baseline numbers. If blob-cached model load is fast enough,
Option B (model splitting) is viable for dense weight streaming.

Usage:
  python measure_load_time.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov
  python measure_load_time.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov --iterations 5
  python measure_load_time.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov --all-models
"""

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import openvino as ov

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_memory_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_peak_memory_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    mem = psutil.Process(os.getpid()).memory_info()
    peak = getattr(mem, "peak_wset", None) or getattr(mem, "peak_pagefile", None)
    if peak is None:
        return mem.rss / (1024 * 1024)
    return peak / (1024 * 1024)


def fmt_time(ms: float) -> str:
    """Format milliseconds for display."""
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms/1000:.3f} s"


def fmt_size(bytes_val: int) -> str:
    """Format bytes for display."""
    if bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    if bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.1f} MB"
    return f"{bytes_val/(1024**3):.2f} GB"


def measure_model_load(core: ov.Core, model_path: str, device: str,
                       iterations: int = 3, label: str = "model") -> dict:
    """Measure blob-cached compile time for a model.
    
    Returns dict with timing statistics.
    """
    model_name = Path(model_path).stem
    xml_path = model_path
    bin_path = model_path.replace(".xml", ".bin")
    
    print(f"\n{'='*70}")
    print(f"  Measuring: {label} ({model_name})")
    print(f"{'='*70}")
    
    # File sizes
    xml_size = os.path.getsize(xml_path) if os.path.exists(xml_path) else 0
    bin_size = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
    print(f"  XML size: {fmt_size(xml_size)}")
    print(f"  BIN size: {fmt_size(bin_size)}")
    
    # Step 1: Read model from XML/BIN (this is CPU-only, no GPU involved)
    mem_before_read = get_memory_mb()
    t0 = time.perf_counter()
    model = core.read_model(xml_path)
    t_read = (time.perf_counter() - t0) * 1000
    mem_after_read = get_memory_mb()
    
    print(f"\n  [read_model] {fmt_time(t_read)}")
    print(f"    Memory: {mem_before_read:.0f} → {mem_after_read:.0f} MB "
          f"(+{mem_after_read - mem_before_read:.0f} MB)")
    print(f"    Inputs:  {len(model.inputs)}, Outputs: {len(model.outputs)}")
    print(f"    Ops: {len(model.get_ordered_ops())}")
    
    # Step 2: First compile (should hit blob cache if CACHE_DIR is set)
    mem_before_compile = get_memory_mb()
    t0 = time.perf_counter()
    compiled = core.compile_model(model, device)
    t_first_compile = (time.perf_counter() - t0) * 1000
    mem_after_compile = get_memory_mb()
    
    print(f"\n  [compile_model #1] {fmt_time(t_first_compile)}")
    print(f"    Memory: {mem_before_compile:.0f} → {mem_after_compile:.0f} MB "
          f"(+{mem_after_compile - mem_before_compile:.0f} MB)")
    
    # Step 3: Unload
    t0 = time.perf_counter()
    del compiled
    gc.collect()
    t_unload = (time.perf_counter() - t0) * 1000
    mem_after_unload = get_memory_mb()
    
    print(f"\n  [unload #1] {fmt_time(t_unload)}")
    print(f"    Memory after unload: {mem_after_unload:.0f} MB "
          f"(freed {mem_after_compile - mem_after_unload:.0f} MB)")
    
    # Step 4: Repeated load/unload cycles (blob cache is now warm)
    compile_times = []
    unload_times = []
    mem_deltas = []
    
    print(f"\n  [Repeated load/unload cycles: {iterations} iterations]")
    for i in range(iterations):
        time.sleep(0.1)  # brief pause between cycles
        gc.collect()
        
        mem_pre = get_memory_mb()
        t0 = time.perf_counter()
        compiled = core.compile_model(model, device)
        t_compile = (time.perf_counter() - t0) * 1000
        mem_post = get_memory_mb()
        
        compile_times.append(t_compile)
        mem_deltas.append(mem_post - mem_pre)
        
        t0 = time.perf_counter()
        del compiled
        gc.collect()
        t_unload_i = (time.perf_counter() - t0) * 1000
        unload_times.append(t_unload_i)
        
        print(f"    Iter {i+1}: compile={fmt_time(t_compile)}, "
              f"unload={fmt_time(t_unload_i)}, "
              f"mem_delta=+{mem_post - mem_pre:.0f} MB")
    
    # Step 5: Measure infer_request creation overhead (within a compile)
    compiled = core.compile_model(model, device)
    t0 = time.perf_counter()
    infer_request = compiled.create_infer_request()
    t_create_req = (time.perf_counter() - t0) * 1000
    print(f"\n  [create_infer_request] {fmt_time(t_create_req)}")
    
    del infer_request
    del compiled
    gc.collect()
    
    # Cleanup
    del model
    gc.collect()
    
    # Summary
    avg_compile = sum(compile_times) / len(compile_times)
    min_compile = min(compile_times)
    max_compile = max(compile_times)
    avg_unload = sum(unload_times) / len(unload_times)
    avg_mem = sum(mem_deltas) / len(mem_deltas)
    
    print(f"\n  {'─'*50}")
    print(f"  SUMMARY for {label}:")
    print(f"    read_model:          {fmt_time(t_read)}")
    print(f"    First compile:       {fmt_time(t_first_compile)}")
    print(f"    Cached compile avg:  {fmt_time(avg_compile)}")
    print(f"    Cached compile min:  {fmt_time(min_compile)}")
    print(f"    Cached compile max:  {fmt_time(max_compile)}")
    print(f"    Unload avg:          {fmt_time(avg_unload)}")
    print(f"    Memory per compile:  +{avg_mem:.0f} MB")
    print(f"    Full cycle avg:      {fmt_time(avg_compile + avg_unload)}")
    print(f"    BIN file size:       {fmt_size(bin_size)}")
    print(f"  {'─'*50}")
    
    return {
        "label": label,
        "model_name": model_name,
        "bin_size_bytes": bin_size,
        "read_model_ms": t_read,
        "first_compile_ms": t_first_compile,
        "compile_times_ms": compile_times,
        "avg_compile_ms": avg_compile,
        "min_compile_ms": min_compile,
        "max_compile_ms": max_compile,
        "unload_times_ms": unload_times,
        "avg_unload_ms": avg_unload,
        "avg_mem_delta_mb": avg_mem,
        "create_request_ms": t_create_req,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Measure OpenVINO blob-cached model load/unload time"
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to the OpenVINO model directory")
    parser.add_argument("--device", default="GPU",
                        help="Target device (default: GPU)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of load/unload cycles (default: 3)")
    parser.add_argument("--all-models", action="store_true",
                        help="Measure all sub-models (embeddings, vision, etc.)")
    parser.add_argument("--cache-dir", default=None,
                        help="Custom CACHE_DIR (default: <model-dir>/model_cache)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable blob cache (measure cold compile time)")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Initialize OpenVINO Core
    core = ov.Core()
    
    # Configure cache
    if not args.no_cache:
        cache_dir = args.cache_dir or str(model_dir / "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        core.set_property(args.device, {"CACHE_DIR": cache_dir})
        print(f"Blob cache dir: {cache_dir}")
    else:
        print("Blob cache: DISABLED (cold compile)")
    
    print(f"Device: {args.device}")
    print(f"Iterations: {args.iterations}")
    print(f"OpenVINO version: {ov.get_version()}")
    if HAS_PSUTIL:
        print(f"Initial RSS: {get_memory_mb():.0f} MB")
    else:
        print("WARNING: psutil not installed, memory tracking disabled")
    
    # Define models to test
    models_to_test = [
        ("language_model", model_dir / "openvino_language_model.xml"),
    ]
    
    if args.all_models:
        optional = [
            ("text_embeddings", model_dir / "openvino_text_embeddings_model.xml"),
            ("text_embed_per_layer", model_dir / "openvino_text_embeddings_per_layer_model.xml"),
            ("vision_embeddings", model_dir / "openvino_vision_embeddings_model.xml"),
            ("tokenizer", model_dir / "openvino_tokenizer.xml"),
            ("detokenizer", model_dir / "openvino_detokenizer.xml"),
        ]
        for label, path in optional:
            if path.exists():
                models_to_test.append((label, path))
    
    # Run measurements
    results = []
    for label, xml_path in models_to_test:
        if not xml_path.exists():
            print(f"\nSKIP: {xml_path} not found")
            continue
        result = measure_model_load(
            core, str(xml_path), args.device,
            iterations=args.iterations, label=label
        )
        results.append(result)
    
    # Final summary table
    print(f"\n{'='*70}")
    print("  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<25} {'BIN Size':>10} {'Cached Compile':>15} "
          f"{'Unload':>10} {'Full Cycle':>12} {'Mem Delta':>10}")
    print(f"  {'─'*25} {'─'*10} {'─'*15} {'─'*10} {'─'*12} {'─'*10}")
    
    for r in results:
        print(f"  {r['label']:<25} "
              f"{fmt_size(r['bin_size_bytes']):>10} "
              f"{fmt_time(r['avg_compile_ms']):>15} "
              f"{fmt_time(r['avg_unload_ms']):>10} "
              f"{fmt_time(r['avg_compile_ms'] + r['avg_unload_ms']):>12} "
              f"+{r['avg_mem_delta_mb']:.0f} MB".rjust(10))
    
    print(f"{'='*70}")
    
    # Weight streaming viability assessment
    if results:
        lang = next((r for r in results if r["label"] == "language_model"), None)
        if lang:
            full_cycle = lang["avg_compile_ms"] + lang["avg_unload_ms"]
            # For 2-way split, each sub-model would be ~half
            # Estimate sub-model compile time (roughly proportional to weight size)
            est_half = full_cycle * 0.6  # conservative: not perfectly linear
            est_6way = full_cycle * 0.25  # 1/6 model, but fixed overhead
            
            print(f"\n  Weight Streaming Viability (language_model):")
            print(f"  Full model load/unload cycle: {fmt_time(full_cycle)}")
            print(f"  Estimated half-model cycle:   {fmt_time(est_half)}")
            print(f"  Estimated 1/6-model cycle:    {fmt_time(est_6way)}")
            
            # TPS estimate for 2-split (2 load/unload per token)
            tps_2split = 1000.0 / (2 * est_half) if est_half > 0 else 0
            tps_6split = 1000.0 / (6 * est_6way) if est_6way > 0 else 0
            
            print(f"\n  Estimated TPS (load overhead only, no compute):")
            print(f"    2-way split: {tps_2split:.1f} tps (overhead only)")
            print(f"    6-way split: {tps_6split:.1f} tps (overhead only)")
            print(f"    Note: Add ~42 ms compute per token for actual TPS")
            
            if full_cycle < 100:
                print(f"\n  ✅ VERDICT: Full-model cycle < 100ms → Option B very viable!")
            elif full_cycle < 500:
                print(f"\n  ⚠️  VERDICT: Full-model cycle {fmt_time(full_cycle)} → "
                      f"Option B marginal, need actual sub-model test")
            else:
                print(f"\n  ❌ VERDICT: Full-model cycle {fmt_time(full_cycle)} → "
                      f"Option B likely too slow, need Option A (DirectStorage)")
    
    if HAS_PSUTIL:
        print(f"\nPeak memory: {get_peak_memory_mb():.0f} MB")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
