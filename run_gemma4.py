#!/usr/bin/env python3
"""
Gemma 4 — Simple inference with OpenVINO GenAI VLMPipeline
===========================================================
Demonstrates text-only and image+text inference using the
openvino_genai.VLMPipeline API with Gemma 4 models exported
to OpenVINO IR format.

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
        help="Text prompt",
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

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"Loading VLMPipeline from {model_dir} on {args.device}...")
    t0 = time.perf_counter()
    pipe = ov_genai.VLMPipeline(str(model_dir), args.device)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # ── Generation config ───────────────────────────────────────────────────
    config = ov_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens

    # ── Streamer (print tokens as they arrive) ──────────────────────────────
    first_token_time = None
    gen_start = None

    def streamer(subword: str):
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.perf_counter()
        print(subword, end="", flush=True)
        return False  # continue generating

    # ── Prepare image (if provided) ─────────────────────────────────────────
    image_tensor = None
    if args.image:
        image_tensor = load_image(args.image)
        print(f"Image loaded: {args.image}")

    # ── Generate ────────────────────────────────────────────────────────────
    print(f"\nPrompt: {args.prompt}\n")
    print("Response: ", end="", flush=True)

    gen_start = time.perf_counter()

    if image_tensor is not None:
        output = pipe.generate(
            args.prompt,
            images=[image_tensor],
            generation_config=config,
            streamer=streamer,
        )
    else:
        output = pipe.generate(
            args.prompt,
            generation_config=config,
            streamer=streamer,
        )

    gen_end = time.perf_counter()

    # ── Summary ─────────────────────────────────────────────────────────────
    total_s = gen_end - gen_start
    ttft_s = (first_token_time - gen_start) if first_token_time else -1
    print(f"\n\n--- Generation completed in {total_s:.2f}s (TTFT: {ttft_s:.3f}s) ---")


if __name__ == "__main__":
    main()
