#!/usr/bin/env python3
"""
Dense Weight Streaming — Phase 1: Model Splitting Tool
=======================================================
Splits the Gemma4 language model IR into N sub-models at decoder layer
boundaries, then measures blob-cached compile time for each sub-model.

This answers the key Phase 1 question: How fast can we load/unload
individual sub-models from blob cache?

Usage:
  # Split into 2 sub-models (layers 0-20, 21-41)
  python split_language_model.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov

  # Split into 3 sub-models (layers 0-13, 14-27, 28-41)
  python split_language_model.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov --num-splits 3

  # Split and measure compile times
  python split_language_model.py --model-dir C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov --benchmark
"""

import argparse
import gc
import os
import re
import sys
import time
from pathlib import Path

import openvino as ov
from openvino import Model, PartialShape, Type
from openvino._pyopenvino import op as _op

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


NUM_DECODER_LAYERS = 42


def get_memory_mb() -> float:
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def fmt_time(ms: float) -> str:
    if ms < 1000:
        return f"{ms:.1f} ms"
    return f"{ms/1000:.3f} s"


def fmt_size(bytes_val: int) -> str:
    if bytes_val < 1024**2:
        return f"{bytes_val/1024:.1f} KB"
    if bytes_val < 1024**3:
        return f"{bytes_val/(1024**2):.1f} MB"
    return f"{bytes_val/(1024**3):.2f} GB"


def find_layer_boundary_ops(model):
    """Find the output node of each decoder layer's residual connection.
    
    Returns dict mapping layer_index -> (output_op, op_index_in_graph).
    The output op is the Multiply node that produces the hidden state
    after the residual add + optional scaling.
    """
    ops = model.get_ordered_ops()
    layer_outputs = {}
    
    for i, op in enumerate(ops):
        name = op.get_friendly_name()
        # Pattern: __module.model.language_model.layers.{N}/aten::mul_/Multiply
        # This is the final hidden state output of each layer
        match = re.match(
            r'__module\.model\.language_model\.layers\.(\d+)/aten::mul_/Multiply$',
            name
        )
        if match:
            layer_idx = int(match.group(1))
            layer_outputs[layer_idx] = (op, i)
    
    return layer_outputs


def categorize_sinks(model, split_layer):
    """Split KV cache Assign nodes into two groups based on split layer.
    
    Args:
        model: OpenVINO model
        split_layer: First layer in the second sub-model (e.g., 21)
    
    Returns:
        (sinks_a, sinks_b) - sinks for first and second sub-model
    """
    sinks_a = []
    sinks_b = []
    
    for sink in model.get_sinks():
        var_id = sink.get_attributes().get('variable_id', '')
        match = re.search(r'past_key_values\.(\d+)\.', var_id)
        if match:
            layer_idx = int(match.group(1))
            if layer_idx < split_layer:
                sinks_a.append(sink)
            else:
                sinks_b.append(sink)
        else:
            # If can't determine layer, put in model A (conservative)
            sinks_a.append(sink)
    
    return sinks_a, sinks_b


def create_submodel_a(model_path, split_layer):
    """Create sub-model A (first half) from a fresh model load.
    
    Reads model, adds Result at boundary, returns sub-model A.
    Does NOT modify the graph for sub-model B.
    """
    model = ov.Core().read_model(str(model_path))
    layer_outputs = find_layer_boundary_ops(model)
    
    boundary_layer = split_layer - 1
    if boundary_layer not in layer_outputs:
        raise ValueError(f"Cannot find output node for layer {boundary_layer}")
    
    boundary_op, _ = layer_outputs[boundary_layer]
    boundary_output = boundary_op.output(0)
    
    print(f"  Split boundary: layer {boundary_layer} output -> layer {split_layer} input")
    print(f"  Boundary op: {boundary_op.get_friendly_name()}")
    print(f"  Output shape: {boundary_output.get_partial_shape()}")
    
    # Categorize sinks
    sinks_a, _ = categorize_sinks(model, split_layer)
    print(f"  Sinks A: {len(sinks_a)}")
    
    # Add Result at boundary
    result_a = _op.Result(boundary_output)
    result_a.set_friendly_name('hidden_states_out')
    
    params_a = model.get_parameters()
    model_a = Model([result_a], sinks_a, params_a)
    model_a.set_friendly_name(f"language_model_layers_0_to_{boundary_layer}")
    
    return model_a


def create_submodel_b(model_path, split_layer):
    """Create sub-model B (second half) from a fresh model load.
    
    Reads model fresh, modifies graph to replace boundary with Parameter.
    """
    model = ov.Core().read_model(str(model_path))
    layer_outputs = find_layer_boundary_ops(model)
    
    boundary_layer = split_layer - 1
    if boundary_layer not in layer_outputs:
        raise ValueError(f"Cannot find output node for layer {boundary_layer}")
    
    boundary_op, _ = layer_outputs[boundary_layer]
    boundary_output = boundary_op.output(0)
    
    # Categorize sinks
    _, sinks_b = categorize_sinks(model, split_layer)
    print(f"  Sinks B: {len(sinks_b)}")
    
    # Create new Parameter to replace boundary output
    param_hidden = _op.Parameter(Type.f32, PartialShape([-1, -1, 2560]))
    param_hidden.set_friendly_name('hidden_states_in')
    
    # Redirect all consumers of boundary output to new parameter
    target_inputs = list(boundary_output.get_target_inputs())
    for ti in target_inputs:
        ti.replace_source_output(param_hidden.output(0))
    
    # Get original results and parameters
    results_b = model.get_results()
    orig_params = model.get_parameters()
    
    # Build parameter list for sub-model B
    params_b = [param_hidden] + list(orig_params)
    
    model_b = Model(results_b, sinks_b, params_b)
    model_b.set_friendly_name(f"language_model_layers_{split_layer}_to_41")
    
    return model_b


def split_model_at_layer(model_path, split_layer):
    """Split the language model into two sub-models at the given layer boundary.
    
    Uses separate model loads to avoid graph interference between sub-models.
    
    Args:
        model_path: Path to model XML
        split_layer: First layer of the second sub-model (e.g., 21)
    
    Returns:
        (model_a, model_b) - two sub-models
    """
    print(f"\n  Creating sub-model A (layers 0-{split_layer-1})...")
    model_a = create_submodel_a(model_path, split_layer)
    
    print(f"\n  Creating sub-model B (layers {split_layer}-41)...")
    model_b = create_submodel_b(model_path, split_layer)
    
    return model_a, model_b


def split_model_n_way(model_path, num_splits=2):
    """Split the language model into N sub-models.
    
    For N=2: layers 0-20, 21-41
    For N=3: layers 0-13, 14-27, 28-41
    For N=6: layers 0-6, 7-13, 14-20, 21-27, 28-34, 35-41
    For N=7: layers 0-5, 6-11, 12-17, 18-23, 24-29, 30-35, 36-41
    """
    layers_per_split = NUM_DECODER_LAYERS // num_splits
    split_points = [layers_per_split * (i + 1) for i in range(num_splits - 1)]
    
    print(f"\nSplitting model into {num_splits} parts:")
    print(f"  Layers per split: {layers_per_split}")
    print(f"  Split points: {split_points}")
    
    if num_splits == 2:
        model_a, model_b = split_model_at_layer(model_path, split_points[0])
        return [model_a, model_b], split_points
    else:
        # For N>2, only 2-way split is currently supported
        print(f"\n  NOTE: Only 2-way split currently implemented. Using first split point.")
        model_a, model_b = split_model_at_layer(model_path, split_points[0])
        return [model_a, model_b], [split_points[0]]


def save_submodels(sub_models, output_dir, split_points):
    """Save sub-models to disk as XML+BIN."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    for i, model in enumerate(sub_models):
        if i == 0:
            name = f"language_model_part{i}_layers_0_to_{split_points[0]-1}"
        elif i == len(sub_models) - 1:
            name = f"language_model_part{i}_layers_{split_points[-1]}_to_41"
        else:
            name = f"language_model_part{i}_layers_{split_points[i-1]}_to_{split_points[i]-1}"
        
        xml_path = output_dir / f"{name}.xml"
        bin_path = output_dir / f"{name}.bin"
        
        print(f"\n  Saving sub-model {i}: {name}")
        print(f"    Ops: {len(model.get_ordered_ops())}")
        print(f"    Inputs: {len(model.inputs)}")
        print(f"    Outputs: {len(model.outputs)}")
        
        t0 = time.perf_counter()
        ov.save_model(model, str(xml_path))
        t_save = (time.perf_counter() - t0) * 1000
        
        xml_size = xml_path.stat().st_size if xml_path.exists() else 0
        bin_size = bin_path.stat().st_size if bin_path.exists() else 0
        
        print(f"    XML: {fmt_size(xml_size)}")
        print(f"    BIN: {fmt_size(bin_size)}")
        print(f"    Save time: {fmt_time(t_save)}")
        
        saved_paths.append(str(xml_path))
    
    return saved_paths


def benchmark_submodels(saved_paths, device="GPU", iterations=3, cache_dir=None):
    """Measure blob-cached compile time for each sub-model."""
    core = ov.Core()
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        core.set_property(device, {"CACHE_DIR": cache_dir})
        print(f"\n  Cache dir: {cache_dir}")
    
    results = []
    
    for xml_path in saved_paths:
        name = Path(xml_path).stem
        bin_path = xml_path.replace(".xml", ".bin")
        bin_size = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
        
        print(f"\n  {'─'*50}")
        print(f"  Benchmarking: {name}")
        print(f"  BIN size: {fmt_size(bin_size)}")
        
        # Read model
        t0 = time.perf_counter()
        model = core.read_model(xml_path)
        t_read = (time.perf_counter() - t0) * 1000
        print(f"  read_model: {fmt_time(t_read)}")
        
        # First compile (populates cache)
        t0 = time.perf_counter()
        compiled = core.compile_model(model, device)
        t_first = (time.perf_counter() - t0) * 1000
        print(f"  First compile: {fmt_time(t_first)}")
        
        del compiled
        gc.collect()
        time.sleep(0.2)
        
        # Cached compiles
        compile_times = []
        unload_times = []
        for i in range(iterations):
            gc.collect()
            time.sleep(0.1)
            
            t0 = time.perf_counter()
            compiled = core.compile_model(model, device)
            t_compile = (time.perf_counter() - t0) * 1000
            compile_times.append(t_compile)
            
            t0 = time.perf_counter()
            del compiled
            gc.collect()
            t_unload = (time.perf_counter() - t0) * 1000
            unload_times.append(t_unload)
        
        avg_compile = sum(compile_times) / len(compile_times)
        avg_unload = sum(unload_times) / len(unload_times)
        
        print(f"  Cached compile avg: {fmt_time(avg_compile)} "
              f"(min: {fmt_time(min(compile_times))}, max: {fmt_time(max(compile_times))})")
        print(f"  Unload avg: {fmt_time(avg_unload)}")
        print(f"  Full cycle avg: {fmt_time(avg_compile + avg_unload)}")
        
        del model
        gc.collect()
        
        results.append({
            'name': name,
            'bin_size': bin_size,
            'read_ms': t_read,
            'first_compile_ms': t_first,
            'avg_compile_ms': avg_compile,
            'min_compile_ms': min(compile_times),
            'max_compile_ms': max(compile_times),
            'avg_unload_ms': avg_unload,
        })
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SUB-MODEL BENCHMARK SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Name':<45} {'BIN':>8} {'Cached':>10} {'Unload':>8} {'Cycle':>10}")
    print(f"  {'─'*45} {'─'*8} {'─'*10} {'─'*8} {'─'*10}")
    
    total_cycle = 0
    for r in results:
        cycle = r['avg_compile_ms'] + r['avg_unload_ms']
        total_cycle += cycle
        print(f"  {r['name']:<45} "
              f"{fmt_size(r['bin_size']):>8} "
              f"{fmt_time(r['avg_compile_ms']):>10} "
              f"{fmt_time(r['avg_unload_ms']):>8} "
              f"{fmt_time(cycle):>10}")
    
    print(f"\n  Total cycle (all sub-models sequential): {fmt_time(total_cycle)}")
    print(f"  Estimated TPS (streaming overhead only): {1000/total_cycle:.2f} tps" if total_cycle > 0 else "")
    print(f"  + ~42 ms compute per token at 24 tps baseline")
    
    effective_tpot = total_cycle + 42  # ms (compute + load overhead)
    effective_tps = 1000 / effective_tpot if effective_tpot > 0 else 0
    print(f"  Effective TPS estimate: {effective_tps:.2f} tps")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Split Gemma4 language model and benchmark sub-model load times"
    )
    parser.add_argument("--model-dir", required=True,
                        help="Path to the OpenVINO model directory")
    parser.add_argument("--num-splits", type=int, default=2,
                        help="Number of sub-models to split into (default: 2)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for sub-models (default: <model-dir>/split_models)")
    parser.add_argument("--device", default="GPU",
                        help="Target device for benchmarking (default: GPU)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark blob-cached compile time after splitting")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Benchmark iterations (default: 3)")
    parser.add_argument("--skip-save", action="store_true",
                        help="Skip saving (just test splitting logic)")
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    model_path = model_dir / "openvino_language_model.xml"
    
    if not model_path.exists():
        print(f"ERROR: {model_path} not found")
        sys.exit(1)
    
    output_dir = args.output_dir or str(model_dir / "split_models")
    cache_dir = str(Path(output_dir) / "model_cache")
    
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Splits: {args.num_splits}")
    print(f"OpenVINO: {ov.get_version()}")
    
    # Split model
    print(f"\n{'='*70}")
    print(f"  SPLITTING MODEL")
    print(f"{'='*70}")
    
    t0 = time.perf_counter()
    sub_models, split_points = split_model_n_way(model_path, args.num_splits)
    t_split = (time.perf_counter() - t0) * 1000
    
    print(f"\n  Split completed in {fmt_time(t_split)}")
    print(f"  Sub-models: {len(sub_models)}")
    for i, sm in enumerate(sub_models):
        print(f"    Part {i}: {len(sm.get_ordered_ops())} ops, "
              f"{len(sm.inputs)} inputs, {len(sm.outputs)} outputs")
    
    # Save sub-models
    if not args.skip_save:
        print(f"\n{'='*70}")
        print(f"  SAVING SUB-MODELS")
        print(f"{'='*70}")
        
        saved_paths = save_submodels(sub_models, output_dir, split_points)
        print(f"\n  Saved {len(saved_paths)} sub-models to: {output_dir}")
    else:
        print("\n  Skipping save (--skip-save)")
        saved_paths = []
    
    # Free memory before benchmarking
    del sub_models
    gc.collect()
    
    # Benchmark
    if args.benchmark and saved_paths:
        print(f"\n{'='*70}")
        print(f"  BENCHMARKING SUB-MODELS")
        print(f"{'='*70}")
        
        results = benchmark_submodels(
            saved_paths, args.device, args.iterations, cache_dir
        )
    
    print("\nDone.")


if __name__ == "__main__":
    main()
