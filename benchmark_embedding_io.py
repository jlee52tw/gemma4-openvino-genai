#!/usr/bin/env python3
"""
Per-Layer Embedding IO Mode Benchmark
======================================
Compare mmap vs DirectIO per-layer embedding reader across multiple
input prompt lengths: 256, 1K, 2K, 4K, 8K tokens.

Collects: TTFT, TPOT, throughput (tok/s), peak RSS (MB).

Usage:
  # Run all configs (mmap + directio) × (256, 1K, 2K, 4K, 8K tokens)
  python benchmark_embedding_io.py --model-dir C:\working\gemma4-openvino\gemma-4-E4B-it-ov

  # Run only mmap mode
  python benchmark_embedding_io.py --model-dir ... --modes mmap

  # Run specific token lengths
  python benchmark_embedding_io.py --model-dir ... --lengths 256 1024

  # Set output tokens
  python benchmark_embedding_io.py --model-dir ... --max-new-tokens 64
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import openvino_genai as ov_genai

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Long text corpus for generating prompts of different lengths ────────
# This passage covers LLM architecture topics — long enough to tokenize
# into 10K+ tokens when repeated.
CORPUS = """\
Large language models (LLMs) represent a paradigm shift in artificial intelligence, \
enabling machines to understand and generate human language with unprecedented fluency. \
These models are built upon the Transformer architecture, introduced by Vaswani et al. in \
2017, which relies entirely on self-attention mechanisms to capture dependencies between \
tokens regardless of their distance in the input sequence. The key innovation of the \
Transformer is replacing recurrent connections with multi-head self-attention, allowing \
parallel processing of all positions simultaneously.

The self-attention mechanism computes, for each token, three vectors: a query (Q), \
a key (K), and a value (V). The attention weight between any two positions is determined \
by the dot product of the query of one position with the key of the other, scaled by the \
square root of the key dimension. These weights are then applied to the value vectors to \
produce the output representation. Multi-head attention extends this by running multiple \
attention operations in parallel, each with its own learned projection matrices, and \
concatenating their outputs. This allows the model to attend to information from different \
representation subspaces at different positions.

Modern LLMs such as GPT, LLaMA, Gemma, and Mistral typically consist of a decoder-only \
Transformer stack. The decoder uses causal (autoregressive) attention masking, ensuring \
that each position can only attend to earlier positions. This enables efficient left-to-right \
generation during inference. The model architecture typically includes: an embedding layer \
that maps discrete token IDs to continuous vectors; a stack of N transformer decoder layers, \
each containing multi-head attention and feed-forward sub-layers with residual connections \
and layer normalization; and a final output projection that maps hidden states back to the \
vocabulary dimension for next-token prediction.

Training these models requires enormous computational resources. Pre-training involves \
processing hundreds of billions or even trillions of tokens from diverse text corpora. The \
training objective is typically causal language modeling (CLM), where the model learns to \
predict the next token given all preceding tokens. The loss function is the cross-entropy \
between the model's predicted probability distribution over the vocabulary and the actual \
next token. Optimization is performed using variants of stochastic gradient descent, most \
commonly AdamW, with carefully tuned learning rate schedules including warmup and cosine \
decay phases.

Quantization techniques have become essential for deploying large models on resource-constrained \
hardware. INT8 and INT4 quantization reduce the precision of model weights from FP32/FP16 to \
lower bit widths, significantly reducing memory footprint and enabling faster inference on \
hardware with limited memory bandwidth. Group quantization applies per-group scaling factors \
to minimize accuracy loss, with common group sizes of 32, 64, or 128 elements. The OpenVINO \
toolkit supports various quantization formats and optimizes inference across Intel CPUs, GPUs, \
and NPUs through graph compilation and kernel fusion.

Key-value (KV) caching is a critical optimization for autoregressive generation. During the \
decode phase, only the newly generated token needs to be processed through the attention layers, \
while the keys and values from all previous tokens are cached and reused. This reduces the \
computational cost of each decode step from O(n²) to O(n) in sequence length. The KV cache \
memory grows linearly with sequence length and is often the primary memory bottleneck during \
long-context inference. Techniques like sliding window attention, multi-query attention (MQA), \
and grouped-query attention (GQA) help reduce the KV cache memory requirements while \
maintaining model quality.

The inference pipeline for vision-language models (VLMs) extends the text-only architecture \
by incorporating a vision encoder that processes input images into visual tokens. The SigLIP \
vision encoder extracts patch-level features that are then projected into the language model's \
embedding space. These visual tokens are interleaved with text tokens in the input sequence, \
allowing the model to jointly reason over visual and textual information.
"""


def get_memory_mb() -> float:
    """Return current RSS in MB."""
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_peak_memory_mb() -> float:
    """Return peak working set in MB."""
    if not HAS_PSUTIL:
        return 0.0
    mem = psutil.Process(os.getpid()).memory_info()
    peak = getattr(mem, "peak_wset", None) or getattr(mem, "peak_pagefile", None)
    return (peak or mem.rss) / (1024 * 1024)


def build_prompt_for_length(tokenizer, target_tokens: int) -> str:
    """Build a prompt that tokenizes to approximately `target_tokens` tokens.

    Strategy: repeat the CORPUS and trim to exact token count using the
    model's own tokenizer (via openvino_genai.Tokenizer).
    """
    # Over-generate text by repeating the corpus
    repeats = max(1, (target_tokens // 500) + 2)
    long_text = (CORPUS + "\n") * repeats

    # Tokenize
    encoded = tokenizer.encode(long_text)
    token_ids = encoded.input_ids.data.flatten().tolist()

    if len(token_ids) < target_tokens:
        print(f"  Warning: corpus only produced {len(token_ids)} tokens "
              f"(target: {target_tokens}). Using all available.")
        return long_text

    # Trim to exact count and decode back to text
    trimmed_ids = token_ids[:target_tokens]
    trimmed_np = np.array(trimmed_ids, dtype=np.int64).reshape(1, -1)

    import openvino as ov
    trimmed_tensor = ov.Tensor(trimmed_np)
    decoded_list = tokenizer.decode(trimmed_tensor)
    # decode() returns a list of strings (one per batch row)
    decoded = decoded_list[0] if isinstance(decoded_list, list) else str(decoded_list)

    # Verify token count
    verify = tokenizer.encode(decoded).input_ids.data.flatten()
    actual = len(verify)
    print(f"  Built prompt: target={target_tokens}, actual={actual} tokens, "
          f"text_len={len(decoded)} chars")
    return decoded


def run_single_benchmark(pipe, prompt: str, max_new_tokens: int,
                         label: str) -> dict:
    """Run a single generation and collect metrics."""
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens

    rss_before = get_memory_mb()
    t_start = time.perf_counter()

    result = pipe.generate(
        prompt,
        generation_config=config,
    )

    t_elapsed = time.perf_counter() - t_start
    rss_after = get_memory_mb()
    peak = get_peak_memory_mb()

    m = result.perf_metrics
    return {
        "label": label,
        "input_tokens": m.get_num_input_tokens(),
        "output_tokens": m.get_num_generated_tokens(),
        "ttft_ms": round(m.get_ttft().mean, 2),
        "tpot_ms": round(m.get_tpot().mean, 2),
        "throughput_tps": round(m.get_throughput().mean, 2),
        "generate_ms": round(m.get_generate_duration().mean, 2),
        "inference_ms": round(m.get_inference_duration().mean, 2),
        "wall_time_s": round(t_elapsed, 2),
        "rss_before_mb": round(rss_before, 0),
        "rss_after_mb": round(rss_after, 0),
        "peak_rss_mb": round(peak, 0),
    }


def print_results_table(results: list[dict]) -> None:
    """Print results in a formatted table."""
    print()
    print("=" * 110)
    print(f"{'Config':<28} {'InTok':>6} {'OutTok':>6} {'TTFT(ms)':>9} "
          f"{'TPOT(ms)':>9} {'TPS':>7} {'Wall(s)':>8} {'RSS(MB)':>8} {'Peak(MB)':>9}")
    print("-" * 110)
    for r in results:
        print(f"{r['label']:<28} {r['input_tokens']:>6} {r['output_tokens']:>6} "
              f"{r['ttft_ms']:>9.1f} {r['tpot_ms']:>9.1f} {r['throughput_tps']:>7.1f} "
              f"{r['wall_time_s']:>8.1f} {r['rss_after_mb']:>8.0f} {r['peak_rss_mb']:>9.0f}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(
        description="Per-Layer Embedding IO Mode Benchmark (mmap vs DirectIO)")
    parser.add_argument("--model-dir", required=True,
                        help="Path to Gemma4 OpenVINO IR model directory")
    parser.add_argument("--device", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--modes", nargs="+", default=["mmap", "directio"],
                        choices=["mmap", "directio"],
                        help="IO modes to test (default: both)")
    parser.add_argument("--lengths", nargs="+", type=int,
                        default=[256, 1024, 2048, 4096, 8192],
                        help="Input token lengths to test")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Output tokens per run (default: 64)")
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of warmup runs before measurement (default: 0)")
    parser.add_argument("--output-csv", default=None,
                        help="Save results to CSV file")
    parser.add_argument("--output-json", default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: model directory not found: {model_dir}")
        sys.exit(1)

    all_results = []

    for mode in args.modes:
        print(f"\n{'=' * 70}")
        print(f"  IO Mode: {mode.upper()}")
        print(f"{'=' * 70}")

        # Set environment variable for per-layer embedding IO mode
        os.environ["OV_PER_LAYER_DIRECTIO_MODE"] = mode
        print(f"  OV_PER_LAYER_DIRECTIO_MODE = {mode}")

        # Load pipeline with mmap disabled (--no-mmap equivalent)
        print(f"  Loading VLMPipeline (ENABLE_MMAP=NO) ...")
        t0 = time.perf_counter()
        pipe = ov_genai.VLMPipeline(str(model_dir), args.device, ENABLE_MMAP='NO')
        load_s = time.perf_counter() - t0
        print(f"  Pipeline loaded in {load_s:.1f}s")

        rss = get_memory_mb()
        print(f"  RSS after load: {rss:,.0f} MB")

        # Get tokenizer for prompt construction
        tokenizer = pipe.get_tokenizer()

        # Build prompts for all requested lengths
        print(f"\n  Building prompts for lengths: {args.lengths}")
        prompts = {}
        for length in args.lengths:
            prompts[length] = build_prompt_for_length(tokenizer, length)

        # Warmup
        if args.warmup > 0:
            print(f"\n  Running {args.warmup} warmup run(s) ...")
            for i in range(args.warmup):
                short_prompt = prompts[min(args.lengths)]
                _ = run_single_benchmark(
                    pipe, short_prompt, args.max_new_tokens,
                    f"warmup-{i+1}")

        # Benchmark each length
        for length in args.lengths:
            label = f"{mode}/{length}tok"
            print(f"\n  --- {label} ---")
            result = run_single_benchmark(
                pipe, prompts[length], args.max_new_tokens, label)
            all_results.append(result)

            print(f"    TTFT: {result['ttft_ms']:.1f} ms | "
                  f"TPOT: {result['tpot_ms']:.1f} ms | "
                  f"TPS: {result['throughput_tps']:.1f} | "
                  f"RSS: {result['rss_after_mb']:.0f} MB")

        # Release pipeline before next mode
        del pipe
        del tokenizer
        import gc; gc.collect()
        time.sleep(2)  # let GPU memory settle

    # ── Summary ──────────────────────────────────────────────────────
    print_results_table(all_results)

    # ── Save results ─────────────────────────────────────────────────
    if args.output_csv:
        csv_path = Path(args.output_csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_path}")

    if args.output_json:
        json_path = Path(args.output_json)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model_dir": str(model_dir),
                "device": args.device,
                "max_new_tokens": args.max_new_tokens,
                "results": all_results,
            }, f, indent=2)
        print(f"Results saved to {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
