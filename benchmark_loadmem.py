#!/usr/bin/env python3
"""
Load-time + Memory KPI Benchmark  (OV GenAI 0706 + PagedAttention)
===================================================================
Metrics captured:

  * OV load_time             — from perf_metrics.load_time (built-in OV GenAI
                               metric, ms).  Reported on the FIRST generate()
                               call after each cold/warm load.
  * COLD load wall-clock     — pipeline constructor time including GPU compile
                               (1st load, compiles kernel cache from scratch).
  * WARM load wall-clock     — pipeline constructor time reading compiled-blob
                               cache (2nd load — what matters for app restarts).
  * PEAK load RSS            — max RSS sampled by background thread during the
                               warm load phase.
  * SUSTAINED infer RSS      — median RSS while generate() runs (stable working
                               set during inference).
  * PEAK infer RSS           — max RSS during generate() phase.
  * OS peak working set      — Windows lifetime process peak (memory_info()
                               .peak_wset) — catches the brief mmap spike even
                               if the sampler misses it.

Both ENABLE_MMAP=YES (default) and ENABLE_MMAP=NO are measured and labelled
in the CSV so you can see:
  mmap=YES:  lower steady-state RSS (OS reclaims file-backed pages after GPU
             upload), but brief transient peak during loading that may be
             missed by a slow sampler.
  mmap=NO:   higher but stable RSS (anonymous heap, OS won't reclaim), easier
             to measure but represents worst-case memory usage.

Works for BOTH:
  * VLM models (Gemma-4 E4B, image-text-to-text)  -> VLMPipeline
  * plain LLMs (Llama-3.1-8B, text-generation)     -> LLMPipeline
  (auto-detected from presence of openvino_vision_embeddings_model.xml)

LLMPipeline note: passing input as a list ["text"] forces DecodedResults
return (with perf_metrics). Passing a bare string with num_return_sequences=1
(default) returns a plain str — no perf_metrics accessible.

Usage (source setupvars.ps1 first so 0706 DLLs are found):
  python benchmark_loadmem.py \\
      --model-dir models/gemma-4-E4B-it-qat-ov-int4-gs64-johnson \\
                  models/Llama-3.1-8B-Instruct-ov-int4-gs64-johnson \\
      --input-lengths 512 1024 2048 \\
      --output-csv results_loadmem_0706.csv
"""

import argparse
import csv
import gc
import platform
import shutil
import statistics
import sys
import threading
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import psutil


# ─────────────────────────────────────────────────────────────────────────────
# Background memory sampler — the only reliable way to get a true peak
# ─────────────────────────────────────────────────────────────────────────────

class MemoryMonitor:
    """Samples process RSS in a background thread at a fixed interval."""

    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.proc = psutil.Process()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[int] = []

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self.proc.memory_info().rss)
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self.samples = [self.proc.memory_info().rss]
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    # ── stats (GB) ──
    def peak_gb(self) -> float:
        return (max(self.samples) / 1024 ** 3) if self.samples else 0.0

    def median_gb(self) -> float:
        return (statistics.median(self.samples) / 1024 ** 3) if self.samples else 0.0

    def mean_gb(self) -> float:
        return (sum(self.samples) / len(self.samples) / 1024 ** 3) if self.samples else 0.0

    def min_gb(self) -> float:
        return (min(self.samples) / 1024 ** 3) if self.samples else 0.0


def os_peak_wset_gb() -> float:
    """Windows-tracked lifetime peak working set (independent of our sampler)."""
    try:
        mi = psutil.Process().memory_info()
        pw = getattr(mi, "peak_wset", None)
        if pw is not None:
            return pw / 1024 ** 3
    except Exception:
        pass
    return 0.0


def cur_rss_gb() -> float:
    return psutil.Process().memory_info().rss / 1024 ** 3


# ─────────────────────────────────────────────────────────────────────────────
# System / iGPU info
# ─────────────────────────────────────────────────────────────────────────────

def collect_system_info(device: str = "GPU") -> Dict[str, str]:
    info: Dict[str, str] = {}
    info["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["os"] = platform.platform()
    info["cpu"] = platform.processor() or platform.machine()
    try:
        import cpuinfo
        info["cpu"] = cpuinfo.get_cpu_info().get("brand_raw", info["cpu"])
    except Exception:
        pass
    info["ram_gb"] = f"{psutil.virtual_memory().total / (1024 ** 3):.1f}"
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
                "gpu_name": "FULL_DEVICE_NAME",
                "gpu_mem_gb": "GPU_DEVICE_TOTAL_MEM_SIZE",
                "gpu_driver": "GPU_DRIVER_VERSION",
            }
            for label, key in props.items():
                try:
                    val = core.get_property("GPU", key)
                    if "mem" in label.lower():
                        val = f"{int(val) / (1024 ** 3):.2f} GB"
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
# Prompt builder
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
    user_target = max(64, target_tokens - chat_overhead)
    src = _FILLER * (user_target // len(_FILLER.split()) + 20)
    ids = hf_tokenizer.encode(src, add_special_tokens=False)
    return hf_tokenizer.decode(ids[:user_target], skip_special_tokens=True)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoadMemResult:
    model_name: str = ""
    pipeline_type: str = ""            # "VLM" | "LLM"
    device: str = "GPU"
    paged_attention: bool = True
    mmap: bool = True                  # ENABLE_MMAP setting
    # ── Load ──
    load_cold_s: float = 0.0           # 1st load wall-clock (compile + write cache)
    load_warm_s: float = 0.0           # 2nd load wall-clock (read compiled-blob cache)
    ov_load_time_ms: float = 0.0       # perf_metrics.load_time from OV GenAI (ms)
    peak_load_rss_gb: float = 0.0      # max RSS sampled during warm load
    # ── Inference throughput ──
    target_input_tokens: int = 0
    actual_input_tokens: int = 0
    actual_output_tokens: int = 0
    max_new_tokens: int = 128
    run_index: int = 0
    ttft_ms: float = 0.0
    prefill_tps: float = 0.0
    tpot_ms: float = 0.0
    decode_tps: float = 0.0
    e2e_tps: float = 0.0
    total_ms: float = 0.0
    # ── Inference memory ──
    sustained_rss_gb: float = 0.0      # median RSS while generate() runs
    peak_infer_rss_gb: float = 0.0     # max RSS during the inference phase
    os_peak_wset_gb: float = 0.0       # Windows lifetime peak working set
    # ── Status ──
    status: str = "OK"
    error_msg: str = ""


CSV_FIELDS = [
    "model_name", "pipeline_type", "device", "paged_attention", "mmap",
    "load_cold_s", "load_warm_s", "ov_load_time_ms", "peak_load_rss_gb",
    "target_input_tokens", "actual_input_tokens", "actual_output_tokens",
    "max_new_tokens", "run_index",
    "ttft_ms", "prefill_tps", "tpot_ms", "decode_tps", "e2e_tps", "total_ms",
    "sustained_rss_gb", "peak_infer_rss_gb", "os_peak_wset_gb",
    "status", "error_msg",
]


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def detect_pipeline_type(model_dir: Path) -> str:
    if (model_dir / "openvino_vision_embeddings_model.xml").exists():
        return "VLM"
    return "LLM"


def make_pipeline(ov_genai, model_dir: str, device: str, ptype: str,
                  use_pa: bool, cache_dir: Optional[str], use_mmap: bool = True):
    """Construct a VLM or LLM pipeline with optional PA + compile-cache + mmap."""
    kwargs: Dict = {}
    if cache_dir:
        kwargs["CACHE_DIR"] = cache_dir
    # ENABLE_MMAP: 'YES' (default) uses file-backed mmap — OS reclaims pages
    # after GPU upload so RSS drops; 'NO' keeps weights in anonymous heap.
    kwargs["ENABLE_MMAP"] = "YES" if use_mmap else "NO"

    if use_pa and device.upper() != "NPU":
        sched = ov_genai.SchedulerConfig()
        sched.enable_prefix_caching = False
        sched.max_num_batched_tokens = sys.maxsize
        kwargs["scheduler_config"] = sched

    cls = ov_genai.VLMPipeline if ptype == "VLM" else ov_genai.LLMPipeline
    return cls(str(model_dir), device, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Core benchmark — one model
# ─────────────────────────────────────────────────────────────────────────────

def _generate(pipe, ptype: str, text: str, gen_cfg):
    """Call generate() and return DecodedResults for both VLM and LLM pipelines.

    LLMPipeline.generate(str, ...) with num_return_sequences==1 returns a plain
    str (no perf_metrics).  Wrapping in a list forces DecodedResults return.
    VLMPipeline always returns VLMDecodedResults regardless.
    """
    if ptype == "LLM":
        return pipe.generate([text], generation_config=gen_cfg)
    else:
        return pipe.generate(text, generation_config=gen_cfg)


def bench_model(
    model_dir: Path,
    device: str,
    prompts: Dict[int, str],
    raw_input_tokens: Dict[int, int],
    max_new_tokens: int,
    warmup_runs: int,
    test_runs: int,
    use_pa: bool,
    cache_root: Path,
    mmap_modes: List[bool],
    pre_warmup_runs: int = 0,
) -> List[LoadMemResult]:
    import openvino_genai as ov_genai

    mname = model_dir.name
    ptype = detect_pipeline_type(model_dir)
    results: List[LoadMemResult] = []

    print(f"\n{'=' * 70}")
    print(f"  Model : {mname}")
    print(f"  Type  : {ptype}   Device: {device}   PA: {use_pa}")
    print(f"  mmap modes to test: {['ON' if m else 'OFF' for m in mmap_modes]}")
    print(f"{'=' * 70}")

    gen_cfg = ov_genai.GenerationConfig()
    gen_cfg.max_new_tokens = max_new_tokens

    for use_mmap in mmap_modes:
        mmap_tag = "mmap=ON " if use_mmap else "mmap=OFF"

        # Fresh cache dir so each mmap mode gets its own compiled blob.
        cache_dir = cache_root / f"cache_{mname}_{'mmap' if use_mmap else 'nommap'}"
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # ── LOAD #1 (cold): compile + write cache ─────────────────────────
        print(f"\n  [{mmap_tag}] load 1/2 cold (compile + write cache) ...", flush=True)
        t0 = time.perf_counter()
        pipe = make_pipeline(ov_genai, str(model_dir), device, ptype,
                             use_pa, str(cache_dir), use_mmap)
        load_cold_s = time.perf_counter() - t0
        print(f"    cold load : {load_cold_s:6.1f}s   RSS={cur_rss_gb():.2f}GB")
        del pipe
        gc.collect()
        time.sleep(2.0)

        # ── LOAD #2 (warm): read compiled-blob cache ───────────────────────
        print(f"  [{mmap_tag}] load 2/2 warm (cached) — measuring peak RSS ...",
              flush=True)
        mon = MemoryMonitor(interval=0.005)  # 5ms poll — better mmap spike capture
        mon.start()
        t0 = time.perf_counter()
        pipe = make_pipeline(ov_genai, str(model_dir), device, ptype,
                             use_pa, str(cache_dir), use_mmap)
        load_warm_s = time.perf_counter() - t0
        mon.stop()
        peak_load_rss_gb = mon.peak_gb()
        print(f"    warm load : {load_warm_s:6.1f}s   "
              f"peak_load_RSS={peak_load_rss_gb:.2f}GB "
              f"os_peak={os_peak_wset_gb():.2f}GB "
              f"(speedup x{load_cold_s / load_warm_s:.1f})")

        # ── Read OV built-in load_time from first generate call ────────────
        # perf_metrics.load_time is populated by OV GenAI internally and
        # reflects the pipeline's own measurement of model load duration (ms).
        print(f"  [{mmap_tag}] warmup run to capture ov_load_time ...", flush=True)
        first_target = sorted(prompts.keys())[0]
        first_res = _generate(pipe, ptype, prompts[first_target], gen_cfg)
        ov_load_ms = 0.0
        try:
            # get_load_time() returns a plain float (ms) in OV GenAI 0706
            ov_load_ms = float(first_res.perf_metrics.get_load_time())
        except Exception:
            pass
        print(f"    ov_load_time = {ov_load_ms:.1f} ms")

        # ── GPU PRE-HEAT: extra generate calls to ramp iGPU frequency ─────
        # iGPU (Arc) can sit at reduced frequency after an idle period.
        # Running N short inferences before measurement forces the driver
        # to ramp to boost frequency before any timed run.
        if pre_warmup_runs > 0:
            preheat_target = sorted(prompts.keys())[0]
            preheat_text = prompts[preheat_target]
            print(f"  [{mmap_tag}] GPU pre-heat: {pre_warmup_runs} runs @ {preheat_target}tok to ramp iGPU freq ...", flush=True)
            t_heat = time.perf_counter()
            for hi in range(pre_warmup_runs):
                _generate(pipe, ptype, preheat_text, gen_cfg)
            heat_s = time.perf_counter() - t_heat
            print(f"    pre-heat done in {heat_s:.1f}s ({heat_s/pre_warmup_runs*1000:.0f}ms/call avg)")
            gc.collect()

        # ── INFERENCE ─────────────────────────────────────────────────────
        for target in sorted(prompts.keys()):
            text = prompts[target]
            raw_in = raw_input_tokens.get(target, target)
            print(f"\n  [{mmap_tag}] target={target}tok (raw={raw_in}tok)")

            for wi in range(warmup_runs):
                print(f"    warmup {wi + 1}/{warmup_runs} ...", flush=True)
                _generate(pipe, ptype, text, gen_cfg)
            gc.collect()

            for ri in range(test_runs):
                infer_mon = MemoryMonitor(interval=0.005)
                infer_mon.start()
                res = _generate(pipe, ptype, text, gen_cfg)
                infer_mon.stop()

                pm = res.perf_metrics
                out_tok = pm.get_num_generated_tokens()
                ttft_ms = pm.get_ttft().mean
                tpot_ms = pm.get_tpot().mean
                e2e_tps = pm.get_throughput().mean
                total_ms = pm.get_generate_duration().mean

                ttft_s = ttft_ms / 1000.0
                total_s = total_ms / 1000.0
                prefill = raw_in / ttft_s if ttft_s > 0 else 0.0
                decode_d = total_s - ttft_s
                decode = (out_tok - 1) / decode_d if (decode_d > 1e-6 and out_tok > 1) else 0.0

                r = LoadMemResult(
                    model_name=mname,
                    pipeline_type=ptype,
                    device=device,
                    paged_attention=use_pa,
                    mmap=use_mmap,
                    load_cold_s=round(load_cold_s, 2),
                    load_warm_s=round(load_warm_s, 2),
                    ov_load_time_ms=round(ov_load_ms, 1),
                    peak_load_rss_gb=round(peak_load_rss_gb, 3),
                    target_input_tokens=target,
                    actual_input_tokens=raw_in,
                    actual_output_tokens=out_tok,
                    max_new_tokens=max_new_tokens,
                    run_index=ri,
                    ttft_ms=round(ttft_ms, 1),
                    prefill_tps=round(prefill, 1),
                    tpot_ms=round(tpot_ms, 2),
                    decode_tps=round(decode, 1),
                    e2e_tps=round(e2e_tps, 2),
                    total_ms=round(total_ms, 1),
                    sustained_rss_gb=round(infer_mon.median_gb(), 3),
                    peak_infer_rss_gb=round(infer_mon.peak_gb(), 3),
                    os_peak_wset_gb=round(os_peak_wset_gb(), 3),
                )
                results.append(r)
                print(f"    R{ri + 1}: out={out_tok}  TTFT={ttft_ms:.0f}ms  "
                      f"prefill={prefill:.0f}tps  decode={decode:.1f}tps  "
                      f"sustain={r.sustained_rss_gb:.2f}GB  "
                      f"peak_infer={r.peak_infer_rss_gb:.2f}GB")
            gc.collect()

        del pipe
        gc.collect()
        time.sleep(3.0)   # let OV release file handles before rmtree
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except Exception:
            pass
        time.sleep(2.0)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[LoadMemResult]) -> None:
    from collections import defaultdict
    warm = [r for r in results if r.status == "OK" and r.run_index >= 1]
    if not warm:
        warm = [r for r in results if r.status == "OK"]

    # unique (model, mmap) combos in original order
    model_mmap_keys = list(dict.fromkeys((r.model_name, r.mmap) for r in results))
    targets = sorted({r.target_input_tokens for r in results if r.target_input_tokens})

    # ── Load-time table ───────────────────────────────────────────────────────
    W = 116
    print("\n" + "=" * W)
    print("  LOAD-TIME & PEAK MEMORY  (warm/cached load)")
    print("  ov_load_time = perf_metrics.load_time from OV GenAI (ms)")
    print("  peak_load_RSS = max sampled by 5ms background thread during warm load")
    print("  os_peak_wset = Windows lifetime process peak (catches brief mmap spike)")
    print("=" * W)
    print(f"  {'Model':<44} {'mmap':>5} {'Cold_s':>7} {'Warm_s':>7} "
          f"{'OVload_ms':>10} {'PkLoadRSS':>10} {'OSpeak_GB':>10}")
    print("-" * W)
    for m, mm in model_mmap_keys:
        rows = [r for r in results if r.model_name == m and r.mmap == mm and r.status == "OK"]
        if not rows:
            continue
        r0 = rows[0]
        os_peak = max((r.os_peak_wset_gb for r in rows), default=0.0)
        shortm = m if len(m) <= 44 else "..." + m[-41:]
        mm_tag = "ON " if mm else "OFF"
        print(f"  {shortm:<44} {mm_tag:>5} {r0.load_cold_s:>7.1f} {r0.load_warm_s:>7.1f} "
              f"{r0.ov_load_time_ms:>10.1f} {r0.peak_load_rss_gb:>10.2f} {os_peak:>10.2f}")
    print("=" * W)

    # ── Inference KPI table ───────────────────────────────────────────────────
    grp: Dict = defaultdict(list)
    for r in warm:
        grp[(r.model_name, r.mmap, r.target_input_tokens)].append(r)

    print("\n" + "=" * W)
    print("  INFERENCE KPI + SUSTAINED MEMORY  (warm runs, run_index >= 1)")
    print("=" * W)
    print(f"  {'Model':<40} {'mmap':>5} {'InTok':>6} {'TTFT_ms':>8} {'Pre_tps':>8} "
          f"{'Dec_tps':>8} {'Sustain_GB':>10} {'PkInf_GB':>9}")
    print("-" * W)
    for m, mm in model_mmap_keys:
        for t in targets:
            rows = grp.get((m, mm, t), [])
            if not rows:
                continue

            def avg(a: str) -> float:
                vals = [getattr(r, a) for r in rows if getattr(r, a, 0) > 0]
                return sum(vals) / len(vals) if vals else 0.0

            shortm = m if len(m) <= 40 else "..." + m[-37:]
            mm_tag = "ON " if mm else "OFF"
            print(f"  {shortm:<40} {mm_tag:>5} {t:>6} {avg('ttft_ms'):>8.0f} "
                  f"{avg('prefill_tps'):>8.0f} {avg('decode_tps'):>8.1f} "
                  f"{avg('sustained_rss_gb'):>10.2f} {avg('peak_infer_rss_gb'):>9.2f}")
        print()
    print("=" * W)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load-time + memory KPI benchmark (0706 runtime + PA)")
    parser.add_argument("--model-dir", nargs="+", required=True)
    parser.add_argument("--device", default="GPU", choices=["CPU", "GPU"])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--pre-warmup-runs", type=int, default=0,
                        help="Extra generate calls before measurement to ramp iGPU "
                             "frequency (default 0). Use 10-20 for fresh/idle iGPU.")
    parser.add_argument("--input-lengths", nargs="+", type=int, default=[512, 1024, 2048])
    parser.add_argument("--no-paged-attention", action="store_true")
    parser.add_argument("--cache-root", default="ov_cache_tmp")
    parser.add_argument("--mmap", nargs="+", choices=["on", "off"], default=["on", "off"],
                        help="mmap modes to test: 'on' (default) and/or 'off'. "
                             "Default tests both.")
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    use_pa = not args.no_paged_attention
    mmap_modes = [m == "on" for m in args.mmap]  # [True] / [False] / [True, False]

    sys_info = collect_system_info(args.device)
    print_system_info(sys_info)

    import openvino_genai as ov_genai
    print(f"openvino_genai : {ov_genai.__version__}")
    print(f"PagedAttention : {'ON' if use_pa else 'OFF'}")
    print(f"mmap modes     : {['ON' if m else 'OFF' for m in mmap_modes]}")
    print(f"Input lengths  : {args.input_lengths}")
    print(f"Output tokens  : {args.max_new_tokens}  warmup={args.warmup}  runs={args.runs}  pre_warmup={args.pre_warmup_runs}")

    model_dirs = []
    for d in args.model_dir:
        p = Path(d)
        if not p.exists():
            print(f"ERROR: not found: {d}")
            sys.exit(1)
        model_dirs.append(p)

    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    all_results: List[LoadMemResult] = []
    for mdir in model_dirs:
        # Build prompts with THIS model's own tokenizer for accurate token counts.
        print(f"\nBuilding prompts for {mdir.name} ...")
        try:
            from transformers import AutoTokenizer
            hf_tok = AutoTokenizer.from_pretrained(str(mdir))
            use_hf = True
        except Exception as e:
            use_hf = False
            print(f"  WARNING: HF tokenizer unavailable ({e}); char estimate")
        ov_tok = ov_genai.Tokenizer(str(mdir))
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
            prompts[tgt] = text
            raw_tokens[tgt] = n_raw
            print(f"  target={tgt:5d}  raw_tokens={n_raw:5d}")

        try:
            res = bench_model(
                mdir, args.device, prompts, raw_tokens,
                args.max_new_tokens, args.warmup, args.runs,
                use_pa, cache_root, mmap_modes,
                pre_warmup_runs=args.pre_warmup_runs,
            )
            all_results.extend(res)
        except Exception as e:
            traceback.print_exc()
            all_results.append(LoadMemResult(
                model_name=mdir.name, device=args.device,
                paged_attention=use_pa, mmap=True, status="ERROR",
                error_msg=str(e)[:200],
            ))
        gc.collect()

    print_summary(all_results)

    if args.output_csv:
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            f.write("# Load-time + Memory KPI Benchmark\n")
            for k, v in sys_info.items():
                f.write(f"# {k}: {v}\n")
            f.write(f"# ov_genai: {ov_genai.__version__}\n")
            f.write(f"# paged_attention: {use_pa}\n")
            f.write(f"# mmap_modes: {['ON' if m else 'OFF' for m in mmap_modes]}\n")
            f.write(f"# input_lengths: {args.input_lengths}\n")
            f.write(f"# max_new_tokens: {args.max_new_tokens}\n")
            f.write(f"# warmup: {args.warmup}  runs: {args.runs}\n")
            f.write("# ov_load_time_ms = perf_metrics.load_time from OV GenAI (built-in, ms)\n")
            f.write("# load_warm_s = 2nd (cached) load wall-clock; peak_load_rss_gb = max RSS sampled at 5ms during it\n")
            f.write("# mmap=ON: file-backed pages, OS reclaims after GPU upload => brief transient peak\n")
            f.write("# mmap=OFF: anonymous heap, stays resident => stable but ~2x higher RSS\n")
            f.write("# sustained_rss_gb = median RSS during generate(); peak_infer_rss_gb = max\n")
            f.write("#\n")
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in all_results:
                w.writerow(asdict(r))
        print(f"\nResults saved: {args.output_csv}")

    # Cleanup temp cache root
    shutil.rmtree(cache_root, ignore_errors=True)

    ok = sum(1 for r in all_results if r.status == "OK")
    err = sum(1 for r in all_results if r.status != "OK")
    print(f"\nTotal: {len(all_results)} rows  |  OK: {ok}  |  ERROR: {err}")


if __name__ == "__main__":
    main()
