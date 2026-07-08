#!/usr/bin/env python3
"""
Gemma 4 — Extended KPI Benchmark  (OV GenAI 0706 + PagedAttention)
====================================================================
Full metrics across multiple input-token lengths.

Metrics per run
  TTFT        time-to-first-token            (ms)
  prefill     input_tokens / TTFT            (tok/s)
  TPOT        time-per-output-token          (ms)   = (total - TTFT) / (out-1)
  decode_tps  (out - 1) / (total - TTFT)    (tok/s)
  e2e_tps     out / total                    (tok/s)
  peak_rss    process RSS                    (GB)

Input lengths tested : 512 / 1024 / 2048 / 4096 tokens  (configurable)
Output tokens        : 128  (configurable)
Warmup               : 1    (configurable)

Usage — source setupvars.ps1 first so 0706 DLLs are found:
  & C:\\...\\openvino_genai_windows_2026.3.0.0.dev20260706_x86_64\\setupvars.ps1 -python_version 3.12
  python benchmark_kpi2.py --model-dir models/qat-johnson models/nonqat-johnson \\
      --output-csv results_kpi_0706.csv
"""

import argparse
import csv
import gc
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import psutil

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_rss_gb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# System / iGPU info
# ─────────────────────────────────────────────────────────────────────────────

def collect_system_info(device: str = "GPU") -> Dict[str, str]:
    info: Dict[str, str] = {}
    info["date"]   = time.strftime("%Y-%m-%d %H:%M:%S")
    info["os"]     = platform.platform()
    info["cpu"]    = platform.processor() or platform.machine()
    try:
        import cpuinfo  # optional: pip install py-cpuinfo
        info["cpu"] = cpuinfo.get_cpu_info().get("brand_raw", info["cpu"])
    except Exception:
        pass
    info["ram_gb"] = f"{psutil.virtual_memory().total / (1024**3):.1f}"

    try:
        import openvino as ov
        core = ov.Core()
        info["ov_runtime"] = ov.__version__
        try:
            import openvino_genai as og
            info["ov_genai"] = og.__version__
        except Exception:
            info["ov_genai"] = "unknown"
        if device.upper() == "GPU":
            props = {
                "gpu_name":     "FULL_DEVICE_NAME",
                "gpu_mem_gb":   "GPU_DEVICE_TOTAL_MEM_SIZE",
                "gpu_driver":   "GPU_DRIVER_VERSION",
                "gpu_eu_count": "GPU_EXECUTION_UNITS_COUNT",
                "gpu_uarch":    "GPU_UARCH_VERSION",
            }
            for label, key in props.items():
                try:
                    val = core.get_property("GPU", key)
                    if "mem" in label.lower():
                        val = f"{int(val) / (1024**3):.2f} GB"
                    info[label] = str(val)
                except Exception:
                    info[label] = "n/a"
    except Exception as e:
        info["ov_error"] = str(e)
    return info


def print_system_info(info: Dict[str, str]) -> None:
    print("\n" + "=" * 70)
    print("  SYSTEM INFO")
    print("=" * 70)
    for k, v in info.items():
        print(f"  {k:<22}: {v}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder — target a specific raw-token count
# ─────────────────────────────────────────────────────────────────────────────

_FILLER = (
    "The Transformer architecture introduced in 2017 fundamentally changed "
    "natural language processing by replacing recurrent networks with "
    "self-attention mechanisms allowing fully parallel sequence processing. "
    "Modern large language models such as Gemma LLaMA and Mistral all use "
    "variants of this design with rotary positional embeddings grouped-query "
    "attention and SwiGLU feed-forward layers to achieve state-of-the-art "
    "language understanding and generation across diverse tasks. Hardware "
    "accelerators including GPUs and NPUs exploit high arithmetic intensity of "
    "matrix multiplications while quantization techniques reduce model size and "
    "memory bandwidth requirements for efficient on-device inference. "
    "Speculative decoding uses a smaller draft model to propose candidate tokens "
    "that the large model then verifies in parallel improving decode throughput "
    "at the cost of additional memory and modest draft model inference overhead. "
    "PagedAttention manages KV-cache memory as fixed-size blocks analogous to "
    "virtual memory pages which eliminates fragmentation and enables continuous "
    "batching of requests with variable sequence lengths on GPU hardware. "
)


def build_prompt(hf_tokenizer, target_tokens: int, chat_overhead: int = 35) -> str:
    """
    Build a raw user-text string that should produce ~target_tokens total
    input tokens (including chat-template overhead) for the model.
    """
    user_target = max(64, target_tokens - chat_overhead)
    # Repeat filler until long enough
    src = _FILLER * (user_target // len(_FILLER.split()) + 20)
    ids = hf_tokenizer.encode(src, add_special_tokens=False)
    trimmed = hf_tokenizer.decode(ids[:user_target], skip_special_tokens=True)
    return trimmed


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KpiResult:
    model_name: str          = ""
    target_input_tokens: int = 0
    actual_input_tokens: int = 0   # from OV tokenizer (raw prompt, no template)
    actual_output_tokens: int = 0  # from perf_metrics
    max_new_tokens: int      = 128
    device: str              = "GPU"
    run_index: int           = 0
    warmup_runs: int         = 1
    paged_attention: bool    = True
    # ── Timing (from OV GenAI perf_metrics) ──
    ttft_ms: float      = 0.0   # time to first token (ms)
    tpot_ms: float      = 0.0   # ms per output token (OV native)
    prefill_tps: float  = 0.0   # input_tok / ttft_s
    decode_tps: float   = 0.0   # (out-1) / (total_s - ttft_s)
    e2e_tps: float      = 0.0   # out / total_s  (OV native throughput)
    total_ms: float     = 0.0   # total generate (ms, OV native)
    tokenize_ms: float  = 0.0   # tokenization time (ms, OV native)
    # ── Memory ──
    peak_rss_gb: float  = 0.0
    # ── Status ──
    status: str   = "OK"
    error_msg: str = ""


CSV_FIELDS = [
    "model_name", "target_input_tokens", "actual_input_tokens",
    "actual_output_tokens", "max_new_tokens", "device",
    "run_index", "warmup_runs", "paged_attention",
    "ttft_ms", "prefill_tps", "tpot_ms", "decode_tps", "e2e_tps",
    "total_ms", "tokenize_ms", "peak_rss_gb",
    "status", "error_msg",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark — one model, all prompt lengths
# ─────────────────────────────────────────────────────────────────────────────

def bench_model(
    model_dir: str,
    model_name: str,
    device: str,
    prompts: Dict[int, str],          # {target: prompt_text}
    raw_input_tokens: Dict[int, int], # {target: measured_raw_tok}
    max_new_tokens: int,
    warmup_runs: int,
    test_runs: int,
    use_pa: bool = True,
) -> List[KpiResult]:
    import openvino_genai as ov_genai

    results: List[KpiResult] = []
    pa_tag = " [PA]" if use_pa else ""
    print(f"\n{'='*70}")
    print(f"  Model : {model_name}")
    print(f"  Device: {device}{pa_tag}")
    print(f"{'='*70}")

    t0 = time.perf_counter()
    if use_pa and device.upper() != "NPU":
        sched = ov_genai.SchedulerConfig()
        sched.enable_prefix_caching = False
        sched.max_num_batched_tokens = sys.maxsize
        pipe = ov_genai.VLMPipeline(str(model_dir), device,
                                     scheduler_config=sched)
    else:
        pipe = ov_genai.VLMPipeline(str(model_dir), device)
    load_s = time.perf_counter() - t0
    print(f"  Loaded in {load_s:.1f}s   RSS: {get_rss_gb():.2f} GB")

    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = max_new_tokens

    for target in sorted(prompts.keys()):
        text = prompts[target]
        raw_in = raw_input_tokens.get(target, target)
        print(f"\n  ── target={target}tok (raw={raw_in}tok) ──")

        # Warmup
        for wi in range(warmup_runs):
            print(f"    warmup {wi+1}/{warmup_runs} ...", flush=True)
            pipe.generate(text, generation_config=gen_cfg)
        gc.collect()

        # Measured runs
        for ri in range(test_runs):
            rss0 = get_rss_gb()
            res = pipe.generate(text, generation_config=gen_cfg)
            peak_rss = max(rss0, get_rss_gb())
            pm = res.perf_metrics

            out_tok   = pm.get_num_generated_tokens()
            ttft_ms   = pm.get_ttft().mean
            tpot_ms   = pm.get_tpot().mean
            e2e_tps   = pm.get_throughput().mean
            total_ms  = pm.get_generate_duration().mean
            tok_ms    = pm.get_tokenization_duration().mean

            # Derived
            ttft_s   = ttft_ms / 1000.0
            total_s  = total_ms / 1000.0
            prefill  = raw_in / ttft_s if ttft_s > 0 else 0.0
            decode_d = total_s - ttft_s
            decode   = (out_tok - 1) / decode_d if (decode_d > 1e-6 and out_tok > 1) else 0.0

            r = KpiResult(
                model_name=model_name,
                target_input_tokens=target,
                actual_input_tokens=raw_in,
                actual_output_tokens=out_tok,
                max_new_tokens=max_new_tokens,
                device=device,
                run_index=ri,
                warmup_runs=warmup_runs,
                paged_attention=use_pa,
                ttft_ms=round(ttft_ms, 1),
                tpot_ms=round(tpot_ms, 2),
                prefill_tps=round(prefill, 1),
                decode_tps=round(decode, 1),
                e2e_tps=round(e2e_tps, 2),
                total_ms=round(total_ms, 1),
                tokenize_ms=round(tok_ms, 2),
                peak_rss_gb=round(peak_rss, 3),
            )
            results.append(r)

            print(f"    R{ri+1}: out={out_tok}  "
                  f"TTFT={ttft_ms:.0f}ms  "
                  f"prefill={prefill:.0f}tps  "
                  f"TPOT={tpot_ms:.1f}ms  "
                  f"decode={decode:.1f}tps  "
                  f"E2E={e2e_tps:.1f}tps  "
                  f"RSS={peak_rss:.2f}GB")

        gc.collect()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[KpiResult]) -> None:
    from collections import defaultdict

    # Use warm runs (run_index >= 1), fall back to all if warmup=0
    warm = [r for r in results if r.status == "OK" and r.run_index >= 1]
    if not warm:
        warm = [r for r in results if r.status == "OK"]

    grp: Dict = defaultdict(list)
    for r in warm:
        grp[(r.model_name, r.target_input_tokens)].append(r)

    models  = list(dict.fromkeys(r.model_name for r in results))
    targets = sorted({r.target_input_tokens for r in results})

    W = 95
    print("\n" + "=" * W)
    print("  Warm-avg KPI (run_index ≥ 1)  —  Input lengths × Models")
    print("=" * W)
    hdr = (f"  {'Model':<36} {'InTok':>6} {'TTFT_ms':>8} {'Pre_tps':>8} "
           f"{'TPOT_ms':>8} {'Dec_tps':>8} {'E2E_tps':>8} {'RSS_GB':>7}")
    print(hdr)
    print("-" * W)

    for m in models:
        for t in targets:
            rows = grp.get((m, t), [])
            if not rows:
                continue
            def avg(attr: str) -> float:
                vals = [getattr(r, attr) for r in rows if getattr(r, attr, 0) > 0]
                return sum(vals) / len(vals) if vals else 0.0
            shortm = ("..." + m[-33:]) if len(m) > 36 else m
            print(f"  {shortm:<36} {t:>6}  "
                  f"{avg('ttft_ms'):>8.0f}  "
                  f"{avg('prefill_tps'):>8.0f}  "
                  f"{avg('tpot_ms'):>8.1f}  "
                  f"{avg('decode_tps'):>8.1f}  "
                  f"{avg('e2e_tps'):>8.1f}  "
                  f"{avg('peak_rss_gb'):>7.2f}")
        print()
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gemma 4 extended KPI benchmark (0706 runtime + PA)",
    )
    parser.add_argument("--model-dir", nargs="+", required=True)
    parser.add_argument("--device", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--input-lengths", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096])
    parser.add_argument("--no-paged-attention", action="store_true")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    use_pa = not args.no_paged_attention

    # ── System info ──────────────────────────────────────────────────────────
    sys_info = collect_system_info(args.device)
    print_system_info(sys_info)

    import openvino_genai as ov_genai
    print(f"openvino_genai : {ov_genai.__version__}")
    print(f"PagedAttention : {'ON (SchedulerConfig)' if use_pa else 'OFF'}")
    print(f"Input lengths  : {args.input_lengths}")
    print(f"Output tokens  : {args.max_new_tokens}  warmup={args.warmup}  runs={args.runs}")

    # ── Validate dirs ─────────────────────────────────────────────────────────
    model_dirs = []
    for d in args.model_dir:
        p = Path(d)
        if not p.exists():
            print(f"ERROR: not found: {d}"); sys.exit(1)
        model_dirs.append(p)

    # ── Build prompts ─────────────────────────────────────────────────────────
    print(f"\nBuilding prompts for {args.input_lengths} ...")
    try:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(str(model_dirs[0]))
        use_hf = True
        print(f"  HF tokenizer loaded from {model_dirs[0].name}")
    except Exception as e:
        use_hf = False
        print(f"  WARNING: HF tokenizer unavailable ({e}), using char estimate")

    ov_tok = ov_genai.Tokenizer(str(model_dirs[0]))
    prompts: Dict[int, str] = {}
    raw_tokens: Dict[int, int] = {}

    for tgt in args.input_lengths:
        if use_hf:
            text = build_prompt(hf_tok, tgt)
        else:
            n = int(tgt * 3.8)
            text = (_FILLER * (n // len(_FILLER) + 5))[:n]
        enc = ov_tok.encode(text)
        n_raw = int(enc.input_ids.get_shape()[1])
        prompts[tgt]    = text
        raw_tokens[tgt] = n_raw
        print(f"  target={tgt:5d}  raw_tokens={n_raw:5d}  chars={len(text):6d}")

    # ── Benchmark ─────────────────────────────────────────────────────────────
    all_results: List[KpiResult] = []

    for mdir in model_dirs:
        mname = mdir.name
        try:
            res = bench_model(
                str(mdir), mname, args.device,
                prompts, raw_tokens,
                args.max_new_tokens, args.warmup, args.runs,
                use_pa=use_pa,
            )
            all_results.extend(res)
        except Exception as e:
            traceback.print_exc()
            for t in args.input_lengths:
                all_results.append(KpiResult(
                    model_name=mname, target_input_tokens=t,
                    actual_input_tokens=raw_tokens.get(t, t),
                    device=args.device, max_new_tokens=args.max_new_tokens,
                    paged_attention=use_pa, status="ERROR",
                    error_msg=str(e)[:200],
                ))
        gc.collect()

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(all_results)

    # ── CSV ───────────────────────────────────────────────────────────────────
    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            # Comment block with system info
            f.write("# KPI Benchmark — Gemma 4\n")
            for k, v in sys_info.items():
                f.write(f"# {k}: {v}\n")
            f.write(f"# ov_genai: {ov_genai.__version__}\n")
            f.write(f"# paged_attention: {use_pa}\n")
            f.write(f"# input_lengths: {args.input_lengths}\n")
            f.write(f"# max_new_tokens: {args.max_new_tokens}\n")
            f.write(f"# warmup: {args.warmup}  runs: {args.runs}\n")
            f.write("#\n")
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in all_results:
                w.writerow(asdict(r))
        print(f"\nResults saved: {args.output_csv}")

    ok  = sum(1 for r in all_results if r.status == "OK")
    err = sum(1 for r in all_results if r.status != "OK")
    print(f"\nTotal: {len(all_results)} runs  |  OK: {ok}  |  ERROR: {err}")


if __name__ == "__main__":
    main()
