#!/usr/bin/env python3
"""
Gemma 4 — Benchmark with OpenVINO GenAI VLMPipeline
====================================================
Measures throughput (tokens/s), time-to-first-token (TTFT), and memory
usage for Gemma 4 models on CPU or GPU.

Test matrix: N models × 3 prompt types (short-text, long-text, short-image)
             × configurable warmup + measured runs.

Usage:
  python benchmark.py --help
  python benchmark.py --model-dir ./gemma-4-E2B-it-ov --device GPU
  python benchmark.py --model-dir ./gemma-4-E2B-it-ov ./gemma-4-E4B-it-ov \
                      --device GPU --max-new-tokens 128 --warmup 1 --runs 3 \
                      --output-csv results.csv
"""

import argparse
import csv
import gc
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import psutil
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Constants & prompts
# ─────────────────────────────────────────────────────────────────────────────

SHORT_PROMPT = "Explain quantum computing in simple terms."
TARGET_LONG_TOKENS = 1024

# A long technical passage (>1 024 tokens with Gemma tokenizer) used to
# evaluate prefill performance.  It is trimmed to exactly 1 024 tokens at
# runtime using the model's own tokenizer.
_LONG_RAW_TEXT = """\
The Transformer architecture, introduced in the seminal paper "Attention Is All \
You Need" by Vaswani et al. in 2017, fundamentally changed the landscape of \
natural language processing and, subsequently, many other domains of machine \
learning. At its core, the Transformer relies on a mechanism called \
self-attention, which allows the model to weigh the importance of different \
positions in the input sequence when computing a representation of each \
position. Unlike recurrent neural networks (RNNs) or long short-term memory \
networks (LSTMs), which process sequences step by step, the Transformer \
processes all positions simultaneously, making it highly parallelizable and \
efficient on modern hardware accelerators such as GPUs and TPUs.

The self-attention mechanism operates by computing three vectors for each input \
token: a query vector (Q), a key vector (K), and a value vector (V). The \
attention score between any two positions is computed as the dot product of the \
query at one position with the key at another, scaled by the square root of the \
key dimension. These scores are then passed through a softmax function to \
produce attention weights, which are used to compute a weighted sum of the value \
vectors. This produces the output for each position. In practice, the \
computation is done in matrix form: Attention(Q, K, V) = softmax(QK^T / \
sqrt(d_k)) * V, where d_k is the dimension of the key vectors.

Multi-head attention extends this mechanism by running multiple attention \
operations in parallel, each with its own learned projection matrices. The \
outputs of these parallel heads are concatenated and linearly projected to \
produce the final output. This allows the model to attend to information from \
different representation subspaces at different positions, greatly increasing \
the expressiveness of the attention mechanism. A typical large language model \
might use anywhere from 8 to 128 attention heads, depending on its size.

The Transformer encoder consists of a stack of identical layers, each containing \
a multi-head self-attention sub-layer followed by a position-wise feed-forward \
network. Each sub-layer is surrounded by a residual connection and layer \
normalization. The feed-forward network consists of two linear transformations \
with a nonlinear activation function (typically ReLU or GELU) in between. The \
decoder has a similar structure but includes an additional cross-attention \
sub-layer that attends to the encoder output, and the self-attention in the \
decoder is masked to prevent positions from attending to subsequent positions, \
maintaining the autoregressive property.

Positional encoding is a critical component since the Transformer has no \
inherent notion of sequence order. The original Transformer used sinusoidal \
positional encodings, where each dimension of the positional encoding \
corresponds to a sinusoid with a different frequency. Modern variants often use \
learned positional embeddings or rotary position embeddings (RoPE), which encode \
relative positions through rotation matrices applied to the query and key \
vectors. RoPE has become the standard in most large language models including \
LLaMA, Gemma, and Mistral families, as it provides better generalization to \
longer sequences than the original approach.

The training process for large language models based on the Transformer involves \
pre-training on massive text corpora using a language modeling objective. Causal \
language models predict the next token given all previous tokens, which \
naturally supports autoregressive text generation at inference time. The loss \
function is the cross-entropy between the predicted probability distribution \
over the vocabulary and the actual next token. The vocabulary itself is \
typically constructed using subword tokenization methods such as Byte Pair \
Encoding (BPE), WordPiece, or SentencePiece, which balance between \
character-level and word-level representations. Gemma models, for instance, use \
a SentencePiece tokenizer with a vocabulary size of 262,144 tokens, which is \
significantly larger than many other model families.

Optimization during training typically uses the AdamW optimizer with a learning \
rate schedule that includes a warmup phase followed by cosine decay. Gradient \
accumulation is used to simulate larger effective batch sizes when memory is \
limited, and gradient clipping prevents exploding gradients during training. \
Mixed precision training, using bfloat16 or float16 representations, reduces \
memory usage and increases throughput on hardware that supports it. Distributed \
training across multiple devices is essential for large models and involves \
techniques such as data parallelism, tensor parallelism, and pipeline \
parallelism, or combinations thereof.

After pre-training, models undergo supervised fine-tuning (SFT) on curated \
instruction-following datasets, followed by reinforcement learning from human \
feedback (RLHF) or direct preference optimization (DPO). The SFT stage teaches \
the model to follow instructions and produce helpful responses, while the \
alignment stage refines the model's behavior to better match human preferences \
and safety requirements. This multi-stage training pipeline has become standard \
practice for building instruction-following language models.

At inference time, text generation involves repeatedly sampling from the model's \
predicted probability distribution over the next token. Various decoding \
strategies are used, including greedy decoding (always selecting the \
highest-probability token), beam search (maintaining multiple candidate \
sequences), top-k sampling (restricting sampling to the k most likely tokens), \
top-p (nucleus) sampling (restricting to the smallest set of tokens whose \
cumulative probability exceeds a threshold p), and temperature scaling (dividing \
logits by a temperature parameter to control randomness). The choice of decoding \
strategy significantly affects the quality, diversity, and coherence of the \
generated text. Key-value caching is used to avoid redundant computation during \
autoregressive generation, storing the key and value projections from previous \
timesteps so they do not need to be recomputed. This is essential for efficient \
generation, especially for long sequences.

Modern multimodal models extend the Transformer architecture to handle inputs \
beyond text. Vision language models (VLMs) incorporate a vision encoder — \
typically a Vision Transformer (ViT) — that processes images into a sequence of \
patch embeddings. These visual tokens are then combined with text token \
embeddings and fed into the language model. The vision encoder divides an input \
image into fixed-size patches (commonly 16x16 or 14x14 pixels), linearly \
projects each patch into an embedding vector, and processes these through \
multiple Transformer layers. Some architectures use cross-attention layers to \
fuse visual and textual information, while others simply concatenate visual and \
text tokens in the input sequence. The Gemma 4 family uses a SigLIP-based \
vision encoder with per-layer text embeddings, where visual features are \
injected at specific layers of the language model rather than only at the input \
layer, allowing for more nuanced multimodal reasoning throughout the network \
depth.

Quantization is a key technique for deploying large models efficiently. \
Post-training quantization reduces the precision of model weights from floating \
point (typically bfloat16 or float32) to lower-precision formats such as INT8 \
or INT4. Group-wise quantization, where a small group of weights shares a \
single scale factor, preserves more accuracy than per-tensor or per-channel \
quantization at very low bit widths. The asymmetric variant allows for a \
zero-point offset in addition to the scale, which can better represent \
non-symmetric weight distributions. Mixed-precision quantization applies \
different bit widths to different parts of the model based on their sensitivity \
to precision loss, typically keeping embedding layers and attention mechanisms \
at higher precision while aggressively quantizing feed-forward layers. The NNCF \
(Neural Network Compression Framework) and OpenVINO toolkits provide automated \
quantization pipelines that can compress models while maintaining accuracy \
within acceptable bounds.

Mixture of Experts (MoE) is another scaling technique that has gained popularity \
in recent large language models. In an MoE layer, multiple feed-forward "expert" \
sub-networks exist, and a gating network (router) selects a subset of experts \
to process each token. This allows the total parameter count to be much larger \
than the parameters active for any single token, providing better capacity \
without a proportional increase in computation. The Gemma 4 26B-A4B model, for \
example, has 25.2 billion total parameters but only 3.8 billion active \
parameters per token, using 128 experts with a top-8 routing strategy. The \
router is typically a simple linear layer that produces logits over all experts, \
and the top-k experts (with the highest router logits) are selected. Load \
balancing loss terms are added during training to ensure that tokens are \
distributed evenly across experts, preventing the collapse phenomenon where only \
a few experts receive most of the traffic.

Hardware acceleration plays a crucial role in making inference practical for \
large models. Intel's integrated GPUs in recent processors can accelerate model \
inference through the OpenVINO toolkit, which optimizes models for Intel \
hardware. The toolkit converts models to an Intermediate Representation (IR) \
format and applies hardware-specific optimizations including operation fusion, \
memory layout transformation, and automatic selection of optimal kernel \
implementations.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Result data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    model_name: str = ""
    prompt_type: str = ""
    device: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    max_new_tokens: int = 0
    ttft_s: float = -1.0
    total_s: float = 0.0
    tokens_per_sec: float = 0.0
    peak_rss_gb: float = 0.0
    status: str = "OK"
    error_msg: str = ""
    warmup_runs: int = 0
    test_runs: int = 0
    run_index: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


def make_test_image(save_path: Path) -> "ov.Tensor":
    """Create a small synthetic test image and return an ov.Tensor."""
    import openvino as ov
    if not save_path.exists():
        img = Image.new("RGB", (336, 336), color=(100, 150, 200))
        pixels = np.array(img)
        for i in range(0, 336, 48):
            pixels[i:i+24, :, 0] = 200
            pixels[:, i:i+24, 2] = 50
        Image.fromarray(pixels).save(str(save_path))
    img = Image.open(str(save_path)).convert("RGB")
    arr = np.expand_dims(np.array(img), axis=0)  # (1, H, W, 3)
    return ov.Tensor(arr)


def build_long_prompt(model_dir: str) -> str:
    """Build a ~1 024-token prompt, trimming with the model's own tokenizer."""
    import openvino_genai as ov_genai
    tokenizer = ov_genai.Tokenizer(model_dir)
    full_ids = tokenizer.encode(_LONG_RAW_TEXT).input_ids.data.flatten().tolist()
    print(f"  Raw long text: {len(full_ids)} tokens")
    if len(full_ids) < TARGET_LONG_TOKENS:
        raise ValueError(f"Raw text too short: {len(full_ids)} < {TARGET_LONG_TOKENS}")
    trimmed_text = tokenizer.decode(full_ids[:TARGET_LONG_TOKENS])
    verify = tokenizer.encode(trimmed_text).input_ids.data.flatten().tolist()
    print(f"  Trimmed long prompt: {len(verify)} tokens (target: {TARGET_LONG_TOKENS})")
    return trimmed_text


def format_table(results: List[BenchResult]) -> str:
    """Pretty-print results as an ASCII table."""
    if not results:
        return "No results."
    headers = ["Model", "Prompt", "Device", "InTok", "OutTok",
               "TTFT(s)", "Total(s)", "Tok/s", "RSS(GB)", "Status"]
    rows = []
    for r in results:
        rows.append([
            r.model_name, r.prompt_type, r.device,
            str(r.input_tokens), str(r.output_tokens),
            f"{r.ttft_s:.3f}" if r.ttft_s >= 0 else "N/A",
            f"{r.total_s:.2f}" if r.total_s > 0 else "0.00",
            f"{r.tokens_per_sec:.2f}",
            f"{r.peak_rss_gb:.2f}",
            r.status,
        ])
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    lines = [sep, hdr, sep]
    for row in rows:
        lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")
    lines.append(sep)
    return "\n".join(lines)


def save_csv(results: List[BenchResult], path: str):
    if not results:
        return
    fields = list(asdict(results[0]).keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"\nResults saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# VLMPipeline benchmark core
# ─────────────────────────────────────────────────────────────────────────────

def bench_vlm(
    model_dir: str,
    model_name: str,
    device: str,
    prompt: str,
    prompt_type: str,
    input_tokens: int,
    max_new_tokens: int,
    warmup_runs: int,
    test_runs: int,
    image=None,
) -> List[BenchResult]:
    """Run warmup + measured runs for one model × prompt combination."""
    import openvino_genai as ov_genai
    results: List[BenchResult] = []

    print(f"  Loading VLMPipeline: {model_name} on {device}...")
    t0 = time.perf_counter()
    pipe = ov_genai.VLMPipeline(str(model_dir), device)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s, RSS: {get_rss_gb():.2f} GB")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens

    # Warmup
    for i in range(warmup_runs):
        print(f"    Warmup {i+1}/{warmup_runs}...")
        if image is not None:
            pipe.generate(prompt, images=[image], generation_config=config)
        else:
            pipe.generate(prompt, generation_config=config)

    # Measured runs
    for run_idx in range(test_runs):
        gc.collect()
        rss_before = get_rss_gb()
        first_token_time = None
        token_count = 0

        def streamer_callback(subword):
            nonlocal first_token_time, token_count
            token_count += 1
            if first_token_time is None:
                first_token_time = time.perf_counter()
            return False

        gen_start = time.perf_counter()
        if image is not None:
            pipe.generate(prompt, images=[image],
                          generation_config=config, streamer=streamer_callback)
        else:
            pipe.generate(prompt, generation_config=config,
                          streamer=streamer_callback)
        gen_end = time.perf_counter()

        total_s = gen_end - gen_start
        ttft_s = (first_token_time - gen_start) if first_token_time else -1.0
        peak_rss = max(rss_before, get_rss_gb())
        out_tok = token_count if token_count > 0 else 1
        tok_s = out_tok / total_s if total_s > 0 else 0

        results.append(BenchResult(
            model_name=model_name, prompt_type=prompt_type, device=device,
            input_tokens=input_tokens, output_tokens=out_tok,
            max_new_tokens=max_new_tokens, ttft_s=ttft_s, total_s=total_s,
            tokens_per_sec=tok_s, peak_rss_gb=peak_rss,
            warmup_runs=warmup_runs, test_runs=test_runs, run_index=run_idx,
        ))
        print(f"    Run {run_idx+1}/{test_runs}: {out_tok} tokens in "
              f"{total_s:.2f}s ({tok_s:.2f} tok/s), "
              f"TTFT={ttft_s:.3f}s, RSS={peak_rss:.2f}GB")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 VLMPipeline benchmark with OpenVINO GenAI",
    )
    parser.add_argument(
        "--model-dir", nargs="+", required=True,
        help="One or more paths to OpenVINO IR model directories",
    )
    parser.add_argument("--device", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    print(f"Gemma 4 VLMPipeline Benchmark — {time.strftime('%Y-%m-%d %H:%M:%S')}")

    import openvino_genai as ov_genai
    print(f"openvino_genai version: "
          f"{ov_genai.__version__ if hasattr(ov_genai, '__version__') else 'unknown'}")

    # Validate model dirs
    model_dirs = []
    for d in args.model_dir:
        p = Path(d)
        if not p.exists():
            print(f"Error: directory not found: {d}")
            sys.exit(1)
        model_dirs.append(p)

    ref_dir = str(model_dirs[0])

    # ── Prompt preparation ──────────────────────────────────────────────────
    print(f"\n{'='*75}\n  PROMPT VALIDATION\n{'='*75}")
    tokenizer = ov_genai.Tokenizer(ref_dir)
    short_tokens = len(tokenizer.encode(SHORT_PROMPT).input_ids.data.flatten().tolist())
    long_prompt = build_long_prompt(ref_dir)
    long_tokens = len(tokenizer.encode(long_prompt).input_ids.data.flatten().tolist())
    print(f"  SHORT prompt: {short_tokens} tokens")
    print(f"  LONG  prompt: {long_tokens} tokens")

    test_image = make_test_image(Path("test_image.jpg"))

    prompts = [
        (SHORT_PROMPT, "short-text",  short_tokens, None),
        (long_prompt,  "long-text",   long_tokens,  None),
        (SHORT_PROMPT, "short-image", short_tokens,  test_image),
    ]

    # ── Run benchmarks ──────────────────────────────────────────────────────
    all_results: List[BenchResult] = []

    for model_path in model_dirs:
        model_name = model_path.name
        for prompt_text, prompt_type, in_tok, img in prompts:
            print(f"\n{'='*75}\n  {model_name} | {prompt_type} | {args.device}\n{'='*75}")
            try:
                res = bench_vlm(
                    str(model_path), model_name, args.device,
                    prompt_text, prompt_type, in_tok,
                    args.max_new_tokens, args.warmup, args.runs,
                    image=img,
                )
                all_results.extend(res)
                for r in res:
                    print(f"  [OK] {r.tokens_per_sec:.2f} tok/s")
            except Exception as e:
                traceback.print_exc()
                all_results.append(BenchResult(
                    model_name=model_name, prompt_type=prompt_type,
                    device=args.device, input_tokens=in_tok,
                    max_new_tokens=args.max_new_tokens,
                    status="ERROR", error_msg=str(e),
                ))
                print(f"  [ERR] {e}")
            gc.collect()

    # ── Results ─────────────────────────────────────────────────────────────
    print(f"\n{'='*75}\n  RESULTS\n{'='*75}")
    print(format_table(all_results))
    if args.output_csv:
        save_csv(all_results, args.output_csv)
    ok = sum(1 for r in all_results if r.status == "OK")
    err = sum(1 for r in all_results if r.status != "OK")
    print(f"\nTotal: {len(all_results)} runs | OK: {ok} | ERROR: {err}")


if __name__ == "__main__":
    main()
