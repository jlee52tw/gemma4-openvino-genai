#!/usr/bin/env python3
"""
WWB accuracy validation using the official built-in TextEvaluator
=================================================================
Uses whowhatbench.TextEvaluator with its default 27 English prompts
(who/what style factual questions) to compare:
  - Ground Truth (GT) : HuggingFace original model (FP32, CPU)
  - Target            : OpenVINO INT4 model via openvino.genai VLMPipeline (GPU)

The TextEvaluator computes:
  - Similarity  : cosine similarity of sentence embeddings
                   (sentence-transformers/all-mpnet-base-v2)
  - Divergency  : FDT (First Divergent Token), SDT (Sum of Divergent Tokens)

Usage:
  # Step 1 – Generate ground truth from HF model (slow, CPU)
  python run_wwb_builtin.py --step gt --hf-model ./gemma-4-E4B-it-hf

  # Step 2 – Generate target answers from OV INT4 model (fast, GPU)
  python run_wwb_builtin.py --step target --ov-model ./gemma-4-E4B-it-ov

  # Step 3 – Score (compare GT vs Target)
  python run_wwb_builtin.py --step score --hf-model ./gemma-4-E4B-it-hf

  # All steps at once
  python run_wwb_builtin.py --step all --hf-model ./gemma-4-E4B-it-hf --ov-model ./gemma-4-E4B-it-ov

Prerequisites:
  pip install transformers torch sentence-transformers openvino-genai whowhatbench
"""

import argparse
import os
import sys
import time

import pandas as pd

# ── Defaults ─────────────────────────────────────────────────────────────────
DEFAULT_GT_CSV     = "wwb_builtin_gt.csv"
DEFAULT_TARGET_CSV = "wwb_builtin_target.csv"
DEFAULT_MAX_TOKENS = 128
DEFAULT_DEVICE     = "GPU"


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 : Ground Truth — HF model on CPU
# ─────────────────────────────────────────────────────────────────────────────
def generate_gt(hf_model_dir: str, gt_csv: str, max_new_tokens: int,
                num_samples: int | None):
    """Use TextEvaluator with the HF model to produce GT answers."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from whowhatbench import TextEvaluator

    print(f"\n{'='*60}")
    print(f"  Step 1 : Ground Truth  (HF model, CPU)")
    print(f"  Model  : {hf_model_dir}")
    print(f"  Prompts: built-in 27 English who/what prompts")
    print(f"  max_new_tokens: {max_new_tokens}")
    print(f"{'='*60}\n")

    # Load HF model + processor
    print("Loading HF model (this may take a few minutes)...")
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(hf_model_dir)
    tokenizer = processor.tokenizer
    model = AutoModelForImageTextToText.from_pretrained(
        hf_model_dir, torch_dtype=torch.float32, device_map="cpu",
    )
    model.eval()
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # Custom generation function for Gemma4 (uses chat template via processor)
    def gemma4_gen_answer(model, tokenizer, prompt, max_new_tokens,
                          crop_question, use_chat_template=False,
                          empty_adapters=False, num_assistant_tokens=0,
                          assistant_confidence_threshold=0.0):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        input_text = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = processor(text=input_text, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        generated_ids = outputs[:, inputs["input_ids"].shape[-1]:]
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Create evaluator — this generates GT answers using built-in prompts
    evaluator = TextEvaluator(
        base_model=model,
        tokenizer=tokenizer,
        metrics=["similarity"],
        max_new_tokens=max_new_tokens,
        num_samples=num_samples,
        language="en",
        gen_answer_fn=gemma4_gen_answer,
    )

    # Save GT
    evaluator.dump_gt(gt_csv)
    n = len(evaluator.gt_data)
    print(f"\nGround truth saved to {gt_csv}  ({n} prompts)")
    print("Sample prompts:")
    for i, row in evaluator.gt_data.head(3).iterrows():
        print(f"  [{i+1}] {row['prompts']}")
        print(f"       → {row['answers'][:100]}...")

    return evaluator


# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 : Target — OpenVINO GenAI VLMPipeline on GPU
# ─────────────────────────────────────────────────────────────────────────────
def generate_target(ov_model_dir: str, gt_csv: str, target_csv: str,
                    max_new_tokens: int, device: str):
    """Generate target answers using OV GenAI VLMPipeline."""
    import openvino_genai as ov_genai

    if not os.path.exists(gt_csv):
        print(f"Error: {gt_csv} not found. Run --step gt first.")
        sys.exit(1)

    gt_data = pd.read_csv(gt_csv, keep_default_na=False)
    prompts = gt_data["prompts"].tolist()

    print(f"\n{'='*60}")
    print(f"  Step 2 : Target  (GenAI VLMPipeline, {device})")
    print(f"  Model  : {ov_model_dir}")
    print(f"  Prompts: {len(prompts)}, max_new_tokens: {max_new_tokens}")
    print(f"{'='*60}\n")

    print("Loading VLMPipeline...")
    t0 = time.perf_counter()
    pipe = ov_genai.VLMPipeline(ov_model_dir, device)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    config = ov_genai.GenerationConfig()
    config.max_new_tokens = max_new_tokens

    answers = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...", end=" ", flush=True)
        t1 = time.perf_counter()
        result = pipe.generate(prompt, generation_config=config)
        answer = result.texts[0] if hasattr(result, "texts") else str(result)
        tok_s = result.perf_metrics.get_throughput().mean
        elapsed = time.perf_counter() - t1
        print(f"({elapsed:.1f}s, {tok_s:.1f} tok/s)")
        answers.append(answer)

    # Save target CSV in the same format as GT
    df = pd.DataFrame({"prompts": prompts, "answers": answers})
    df["language"] = "en"
    df["prompt_length_type"] = "short"
    df.to_csv(target_csv, index=False)
    print(f"\nTarget answers saved to {target_csv}")


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 : Score — compare GT vs Target using TextEvaluator
# ─────────────────────────────────────────────────────────────────────────────
def compute_score(hf_model_dir: str, gt_csv: str, target_csv: str):
    """Use TextEvaluator.score() to compare GT and Target CSVs."""
    from transformers import AutoTokenizer
    from whowhatbench import TextEvaluator

    if not os.path.exists(gt_csv):
        print(f"Error: {gt_csv} not found. Run --step gt first.")
        sys.exit(1)
    if not os.path.exists(target_csv):
        print(f"Error: {target_csv} not found. Run --step target first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Step 3 : Scoring")
    print(f"  GT     : {gt_csv}")
    print(f"  Target : {target_csv}")
    print(f"  Metrics: similarity + divergency")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir)

    evaluator = TextEvaluator(
        gt_data=gt_csv,
        tokenizer=tokenizer,
        metrics=["similarity", "divergency"],
    )

    per_prompt, summary = evaluator.score(target_csv)

    # ── Print per-prompt results ─────────────────────────────────────────
    print(f"\n{'─'*100}")
    print(f"  {'#':>3}  {'Sim':>7}  {'FDT':>5}  {'FDT_n':>7}  {'SDT':>5}  {'SDT_n':>7}  Prompt")
    print(f"{'─'*100}")
    cmp = evaluator.last_cmp
    for i, row in cmp.iterrows():
        sim = row.get("similarity", float("nan"))
        fdt = row.get("FDT", float("nan"))
        fdt_n = row.get("FDT norm", float("nan"))
        sdt = row.get("SDT", float("nan"))
        sdt_n = row.get("SDT norm", float("nan"))
        prompt = row["prompt"][:45]
        print(f"  {i+1:>3}  {sim:>7.4f}  {fdt:>5.0f}  {fdt_n:>7.4f}  {sdt:>5.0f}  {sdt_n:>7.4f}  {prompt}...")
    print(f"{'─'*100}")

    # ── Print aggregate ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  AGGREGATE METRICS  ({len(cmp)} prompts)")
    print(f"{'='*60}")
    for col in summary.columns:
        val = summary[col].values[0]
        print(f"  {col:<20s} : {val:.4f}")
    print(f"{'='*60}")

    sim = summary["similarity"].values[0]
    if sim >= 0.97:
        verdict = "Excellent INT4 quantization quality (>= 0.97)"
    elif sim >= 0.95:
        verdict = "Good INT4 quantization quality (>= 0.95)"
    elif sim >= 0.90:
        verdict = "Acceptable INT4 quantization quality (>= 0.90)"
    else:
        verdict = "Poor INT4 quantization quality (< 0.90)"
    print(f"\n  Verdict: {verdict}")

    # ── Worst examples ───────────────────────────────────────────────────
    worst = evaluator.worst_examples(top_k=5, metric="similarity")
    print(f"\n{'─'*60}")
    print(f"  Worst 5 examples by similarity:")
    print(f"{'─'*60}")
    for rank, row in enumerate(worst):
        print(f"\n  #{rank+1} (sim={row['similarity']:.4f}):")
        print(f"    Prompt : {row['prompt']}")
        print(f"    GT     : {row['source_model'][:120]}...")
        print(f"    INT4   : {row['optimized_model'][:120]}...")

    return sim


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="WWB accuracy validation using built-in TextEvaluator (27 prompts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Example:
  python run_wwb_builtin.py --step gt     --hf-model ./gemma-4-E4B-it-hf
  python run_wwb_builtin.py --step target --ov-model ./gemma-4-E4B-it-ov
  python run_wwb_builtin.py --step score  --hf-model ./gemma-4-E4B-it-hf
""",
    )
    parser.add_argument("--step", choices=["gt", "target", "score", "all"],
                        required=True)
    parser.add_argument("--hf-model", default=None,
                        help="Path to original HF model dir (needed for gt, score)")
    parser.add_argument("--ov-model", default=None,
                        help="Path to OV INT4 model dir (needed for target)")
    parser.add_argument("--gt-csv", default=DEFAULT_GT_CSV)
    parser.add_argument("--target-csv", default=DEFAULT_TARGET_CSV)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Limit number of prompts (default: all 27)")
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=["CPU", "GPU"])
    args = parser.parse_args()

    if args.step in ("gt", "all"):
        if not args.hf_model:
            parser.error("--hf-model is required for --step gt")
        generate_gt(args.hf_model, args.gt_csv, args.max_new_tokens,
                     args.num_samples)

    if args.step in ("target", "all"):
        if not args.ov_model:
            parser.error("--ov-model is required for --step target")
        generate_target(args.ov_model, args.gt_csv, args.target_csv,
                        args.max_new_tokens, args.device)

    if args.step in ("score", "all"):
        if not args.hf_model:
            parser.error("--hf-model is required for --step score (for tokenizer)")
        compute_score(args.hf_model, args.gt_csv, args.target_csv)


if __name__ == "__main__":
    main()
