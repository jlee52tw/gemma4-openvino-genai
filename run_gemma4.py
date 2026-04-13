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
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image


def load_image(path: str) -> ov.Tensor:
    """Load an image from disk and convert to ov.Tensor for VLMPipeline."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img)                     # (H, W, 3) uint8
    arr = np.expand_dims(arr, axis=0)       # (1, H, W, 3)
    return ov.Tensor(arr)


def print_perf_metrics(metrics) -> None:
    """Pretty-print the openvino_genai PerfMetrics returned by generate()."""

    def fmt_ms(pair) -> str:
        """Format a MeanStdPair (mean ± std) in milliseconds."""
        return f"{pair.mean:.2f} ± {pair.std:.2f} ms"

    def fmt_tok_s(pair) -> str:
        """Format a MeanStdPair as tokens/s."""
        return f"{pair.mean:.2f} ± {pair.std:.2f} tok/s"

    sep = "-" * 60
    print(sep)
    print("  OpenVINO GenAI — Performance Metrics")
    print(sep)

    # Token counts
    num_input  = metrics.get_num_input_tokens()
    num_output = metrics.get_num_generated_tokens()
    print(f"  Input tokens          : {num_input}")
    print(f"  Generated tokens      : {num_output}")
    print()

    # Latency metrics
    print(f"  Load time             : {metrics.get_load_time():.2f} ms")
    print(f"  TTFT                  : {fmt_ms(metrics.get_ttft())}")
    print(f"  TPOT                  : {fmt_ms(metrics.get_tpot())}")
    print(f"  iPOT                  : {fmt_ms(metrics.get_ipot())}")
    print()

    # Throughput
    print(f"  Throughput            : {fmt_tok_s(metrics.get_throughput())}")
    print()

    # Duration breakdown
    print(f"  Generate duration     : {fmt_ms(metrics.get_generate_duration())}")
    print(f"  Inference duration    : {fmt_ms(metrics.get_inference_duration())}")
    print(f"  Tokenization duration : {fmt_ms(metrics.get_tokenization_duration())}")
    print(f"  Detokenization dur.   : {fmt_ms(metrics.get_detokenization_duration())}")
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
    print(f"Loading VLMPipeline from {model_dir} on {args.device}...")
    t0 = time.perf_counter()
    pipe = ov_genai.VLMPipeline(str(model_dir), args.device)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

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

    # ── Performance metrics (from openvino.genai) ───────────────────────────
    print_perf_metrics(result.perf_metrics)


if __name__ == "__main__":
    main()
