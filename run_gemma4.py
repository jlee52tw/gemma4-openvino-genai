#!/usr/bin/env python3
"""
Gemma 4 — Simple inference with OpenVINO GenAI VLMPipeline
===========================================================
Demonstrates text-only and image+text inference using the
openvino_genai.VLMPipeline API with Gemma 4 models exported
to OpenVINO IR format.  After generation, the built-in
``PerfMetrics`` from openvino.genai are printed (TTFT, TPOT,
throughput, etc.).

Usage:
  # Text-only (default: GPU)
  python run_gemma4.py --model-dir ./gemma-4-E2B-it-ov --prompt "Explain quantum computing."

  # With an image
  python run_gemma4.py --model-dir ./gemma-4-E2B-it-ov --prompt "Describe this image." --image photo.jpg

  # On CPU
  python run_gemma4.py --model-dir ./gemma-4-E2B-it-ov --device CPU --prompt "Hello!"

  # Disable mmap + show memory usage
  python run_gemma4.py --model-dir ./gemma-4-E2B-it-ov --no-mmap --show-memory
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def load_image(path: str) -> ov.Tensor:
    """Load an image from disk and convert to ov.Tensor for VLMPipeline."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)                     # (H, W, 3) uint8
    arr = np.expand_dims(arr, axis=0)       # (1, H, W, 3)
    return ov.Tensor(arr)


def get_memory_mb() -> float:
    """Return current process RSS (Resident Set Size) in MB."""
    if not HAS_PSUTIL:
        return 0.0
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def get_peak_memory_mb() -> float:
    """Return peak working set (Windows) or max RSS (Linux) in MB."""
    if not HAS_PSUTIL:
        return 0.0
    mem = psutil.Process(os.getpid()).memory_info()
    # On Windows, peak_wset is the peak working set size
    peak = getattr(mem, "peak_wset", None) or getattr(mem, "peak_pagefile", None)
    if peak is None:
        return mem.rss / (1024 * 1024)
    return peak / (1024 * 1024)


def print_memory(label: str) -> float:
    """Print memory usage with a label. Returns current RSS in MB."""
    rss = get_memory_mb()
    peak = get_peak_memory_mb()
    if rss > 0:
        print(f"  [{label}]  RSS: {rss:,.0f} MB  |  Peak: {peak:,.0f} MB")
    return rss


def print_perf_metrics(metrics) -> None:
    """Pretty-print the openvino_genai PerfMetrics returned by generate()."""
    sep = "-" * 60
    print(sep)
    print("  OpenVINO GenAI — Performance Metrics")
    print(sep)

    # Token counts
    print(f"  Input tokens          : {metrics.get_num_input_tokens()}")
    print(f"  Generated tokens      : {metrics.get_num_generated_tokens()}")
    print()

    # Latency metrics
    print(f"  Load time             : {metrics.get_load_time():.2f} ms")
    print(f"  TTFT                  : {metrics.get_ttft().mean:.2f} ms")
    print(f"  TPOT                  : {metrics.get_tpot().mean:.2f} ms")
    print()

    # Throughput
    print(f"  Throughput            : {metrics.get_throughput().mean:.2f} tok/s")
    print()

    # Duration breakdown
    print(f"  Generate duration     : {metrics.get_generate_duration().mean:.2f} ms")
    print(f"  Inference duration    : {metrics.get_inference_duration().mean:.2f} ms")
    print(f"  Tokenization duration : {metrics.get_tokenization_duration().mean:.2f} ms")
    print(f"  Detokenization dur.   : {metrics.get_detokenization_duration().mean:.2f} ms")
    print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 inference with OpenVINO GenAI VLMPipeline",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the OpenVINO IR model directory (e.g. gemma-4-E2B-it-ov)",
    )
    parser.add_argument(
        "--device", default="GPU", choices=["CPU", "GPU"],
        help="Inference device (default: GPU)",
    )
    parser.add_argument(
        "--prompt", default="Explain quantum computing in simple terms.",
        help="Text prompt (ignored if --prompt-file is given)",
    )
    parser.add_argument(
        "--prompt-file", default=None,
        help="Path to a text file whose contents are used as the prompt",
    )
    parser.add_argument(
        "--image", default=None,
        help="Optional path to an image file (jpg/png) for multimodal inference",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=256,
        help="Maximum tokens to generate (default: 256)",
    )
    parser.add_argument(
        "--no-mmap", action="store_true",
        help="Disable memory-mapped model loading (copies weights into RAM)",
    )
    parser.add_argument(
        "--show-memory", action="store_true",
        help="Print process memory (RSS / peak) at key stages (requires psutil)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: model directory not found: {model_dir}")
        sys.exit(1)

    # ── Resolve prompt ────────────────────────────────────────────────────
    if args.prompt_file:
        pf = Path(args.prompt_file)
        if not pf.exists():
            print(f"Error: prompt file not found: {pf}")
            sys.exit(1)
        prompt = pf.read_text(encoding="utf-8").strip()
    else:
        prompt = args.prompt

    # ── Load model ──────────────────────────────────────────────────────────
    show_mem = args.show_memory and HAS_PSUTIL
    if args.show_memory and not HAS_PSUTIL:
        print("Warning: --show-memory requires psutil.  pip install psutil")

    if show_mem:
        print()
        print_memory("Before loading")

    print(f"Loading VLMPipeline from {model_dir} on {args.device}...")
    if args.no_mmap:
        print("  (mmap disabled — weights will be copied into RAM)")

    t0 = time.perf_counter()

    if args.no_mmap:
        # Load model components manually into heap memory (no mmap).
        # VLMPipeline second constructor: models dict + tokenizer + config_dir.
        model_names = [
            "language", "text_embeddings", "text_embeddings_per_layer",
            "vision_embeddings",
        ]
        models = {}
        for name in model_names:
            xml_path = model_dir / f"openvino_{name}_model.xml"
            bin_path = model_dir / f"openvino_{name}_model.bin"
            if xml_path.exists() and bin_path.exists():
                xml_str = xml_path.read_text(encoding="utf-8")
                weights = np.fromfile(str(bin_path), dtype=np.uint8)
                models[name] = (xml_str, ov.Tensor(weights))

        # Load tokenizer + detokenizer
        tok_xml   = (model_dir / "openvino_tokenizer.xml").read_text("utf-8")
        tok_bin   = np.fromfile(str(model_dir / "openvino_tokenizer.bin"),   dtype=np.uint8)
        detok_xml = (model_dir / "openvino_detokenizer.xml").read_text("utf-8")
        detok_bin = np.fromfile(str(model_dir / "openvino_detokenizer.bin"), dtype=np.uint8)
        tokenizer = ov_genai.Tokenizer(tok_xml, ov.Tensor(tok_bin),
                                       detok_xml, ov.Tensor(detok_bin))

        pipe = ov_genai.VLMPipeline(
            models, tokenizer, str(model_dir), args.device,
        )
    else:
        pipe = ov_genai.VLMPipeline(str(model_dir), args.device)

    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    if show_mem:
        print_memory("After loading (peak)")

    # ── Generation config ───────────────────────────────────────────────────
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    # ── Streamer (print tokens as they arrive) ──────────────────────────────
    def streamer(subword: str):
        print(subword, end="", flush=True)
        return False  # continue generating

    # ── Prepare image (if provided) ─────────────────────────────────────────
    image_tensor = None
    if args.image:
        image_tensor = load_image(args.image)
        print(f"Image loaded: {args.image}")

    # ── Generate ────────────────────────────────────────────────────────────
    print(f"\nPrompt: {prompt}\n")
    print("Response: ", end="", flush=True)

    if image_tensor is not None:
        result = pipe.generate(
            prompt,
            images=[image_tensor],
            generation_config=config,
            streamer=streamer,
        )
    else:
        result = pipe.generate(
            prompt,
            generation_config=config,
            streamer=streamer,
        )

    print("\n")

    # ── Memory after generation (stabilized) ──────────────────────────────
    if show_mem:
        print_memory("After generation (stabilized)")
        print()

    # ── Performance metrics (from openvino.genai) ───────────────────────────
    print_perf_metrics(result.perf_metrics)


if __name__ == "__main__":
    main()
