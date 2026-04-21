#!/usr/bin/env python3
"""
Gemma 4 E4B — KPI Benchmark
============================
Measures key performance indicators for Gemma-4-e4b-it (INT4) via
VLMPipeline on 16 GB / 32 GB configurations.

KPIs measured:
  - Model size on disk (GB)
  - Cache creation time (s) + peak memory during cache creation (GB)
  - Per-scenario (varying input token counts):
      * Input tokens (#)
      * Output tokens (#)
      * Prefill Speed (tokens/s)  = input_tokens / TTFT
      * Output TPS (tokens/s)     = output_tokens / (total_time - TTFT)
      * Total peak memory (GB)    = process peak working set
      * Private memory (GB)       = process private bytes (Task Manager "Details" view)
  - Model load time (s)
  - mmap on / off comparison

Test methodology:
  - Input ~467, ~1058, ~2075 tokens
  - Output until model stops or reaches max_new_tokens

Usage:
  python benchmark_kpi.py --model-dir ./gemma-4-E4B-it-ov --device GPU --prompt-file prompt.txt
  python benchmark_kpi.py --model-dir ./gemma-4-E4B-it-ov --device GPU --no-mmap --prompt-file p1.txt p2.txt
  python benchmark_kpi.py --model-dir ./gemma-4-E4B-it-ov --device GPU --prompt-file prompt.txt --max-new-tokens 300

Notes:
  - Run on a clean boot for best memory measurement accuracy.
  - Close other applications to minimize memory noise.
  - The script flushes the model cache directory before cache-creation timing.
"""

import argparse
import ctypes
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import psutil
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# GPU Memory via Windows PDH (Performance Data Helper)
# ─────────────────────────────────────────────────────────────────────────────

_GPU_MEM_AVAILABLE = False
try:
    if sys.platform == "win32":
        import ctypes.wintypes
        _pdh = ctypes.windll.pdh
        _GPU_MEM_AVAILABLE = True
except Exception:
    pass


def get_gpu_memory_gb() -> float:
    """
    Try to read Intel iGPU dedicated + shared memory usage via Windows
    Performance Counters.  Falls back to 0.0 if unavailable.
    """
    if not _GPU_MEM_AVAILABLE:
        return 0.0
    try:
        # Use WMI as fallback — simpler than raw PDH API
        import subprocess
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage').CounterSamples | "
                "Where-Object { $_.InstanceName -like '*python*' } | "
                "Measure-Object -Property CookedValue -Sum | "
                "Select-Object -ExpandProperty Sum"
            ],
            capture_output=True, text=True, timeout=5,
        )
        dedicated = float(result.stdout.strip() or "0") / (1024 ** 3)

        result2 = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-Counter '\\GPU Process Memory(*)\\Shared Usage').CounterSamples | "
                "Where-Object { $_.InstanceName -like '*python*' } | "
                "Measure-Object -Property CookedValue -Sum | "
                "Select-Object -ExpandProperty Sum"
            ],
            capture_output=True, text=True, timeout=5,
        )
        shared = float(result2.stdout.strip() or "0") / (1024 ** 3)
        return dedicated + shared
    except Exception:
        return 0.0


def get_gpu_memory_gb_wmi() -> float:
    """Alternative: Use Windows Task Manager-style GPU memory query."""
    if sys.platform != "win32":
        return 0.0
    try:
        import subprocess
        # Query total GPU adapter memory usage (not per-process but total)
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage').CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | "
                "Select-Object -ExpandProperty Sum"
            ],
            capture_output=True, text=True, timeout=5,
        )
        dedicated = float(result.stdout.strip() or "0") / (1024 ** 3)

        result2 = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage').CounterSamples | "
                "Measure-Object -Property CookedValue -Sum | "
                "Select-Object -ExpandProperty Sum"
            ],
            capture_output=True, text=True, timeout=5,
        )
        shared = float(result2.stdout.strip() or "0") / (1024 ** 3)
        return dedicated + shared
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Process memory helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_peak_working_set_gb() -> float:
    """Peak working set (Windows) or max RSS (Linux) in GB."""
    mem = psutil.Process(os.getpid()).memory_info()
    peak = getattr(mem, "peak_wset", None) or mem.rss
    return peak / (1024 ** 3)


def get_rss_gb() -> float:
    """Current RSS in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)


def reset_peak_working_set():
    """Reset the peak working set counter on Windows so we can measure
    peak memory for a specific phase (e.g., cache creation vs inference)."""
    if sys.platform == "win32":
        try:
            import ctypes
            k32 = ctypes.windll.kernel32
            psapi = ctypes.windll.psapi
            handle = k32.GetCurrentProcess()
            # SetProcessWorkingSetSize with -1, -1 trims the working set
            # and resets the peak working set counter
            psapi.EmptyWorkingSet(handle)
            # Alternative: K32EmptyWorkingSet
        except Exception:
            pass




# ─────────────────────────────────────────────────────────────────────────────
# Result data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    prefill_speed_tps: float = 0.0   # input_tokens / TTFT_seconds
    output_tps: float = 0.0          # output_tokens / decode_seconds
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0             # 1000 / output_tps
    total_peak_memory_gb: float = 0.0
    private_memory_gb: float = 0.0   # process private bytes
    total_time_s: float = 0.0


@dataclass
class CacheInfo:
    cache_creation_time_s: float = 0.0
    cache_peak_memory_gb: float = 0.0
    cache_size_gb: float = 0.0


@dataclass
class FullReport:
    model_name: str = ""
    model_dir: str = ""
    model_size_gb: float = 0.0
    device: str = ""
    mmap_mode: str = ""
    cache_dir: str = ""
    system_memory_gb: float = 0.0
    ov_version: str = ""
    genai_version: str = ""
    model_load_time_s: float = 0.0
    cache_info: CacheInfo = field(default_factory=CacheInfo)
    scenarios: List[ScenarioResult] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Model size calculation
# ─────────────────────────────────────────────────────────────────────────────

def get_model_size_gb(model_dir: Path) -> float:
    """Sum all .bin files in the model directory."""
    total = sum(f.stat().st_size for f in model_dir.glob("*.bin"))
    return total / (1024 ** 3)


def get_cache_size_gb(model_dir: Path) -> float:
    """Sum the OV cache directory size (model_cache/ or similar)."""
    # Check common cache locations
    cache_dirs = [
        model_dir / "cache",
        model_dir / "model_cache",
    ]
    total_size = 0
    found_cache_dir = False
    for cd in cache_dirs:
        if cd.exists():
            found_cache_dir = True
            total_size += sum(f.stat().st_size for f in cd.rglob("*") if f.is_file())
    # Only count top-level .blob files if no cache directory was found
    if not found_cache_dir:
        total_size = sum(f.stat().st_size for f in model_dir.glob("*.blob"))
    return total_size / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt loader
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt_file(filepath: str) -> str:
    """Read prompt text from an external file."""
    p = Path(filepath)
    if not p.exists():
        print(f"  Error: prompt file not found: {filepath}")
        sys.exit(1)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        print(f"  Error: prompt file is empty: {filepath}")
        sys.exit(1)
    return text


def tokenize_prompt(tokenizer, prompt: str) -> int:
    """Return the number of tokens in the prompt."""
    return len(tokenizer.encode(prompt).input_ids.data.flatten().tolist())


# ─────────────────────────────────────────────────────────────────────────────
# Cache management
# ─────────────────────────────────────────────────────────────────────────────

def find_and_clear_cache(model_dir: Path) -> None:
    """Remove OV compiled model cache so we can time fresh cache creation."""
    patterns = ["*.blob", "cache/", "model_cache/"]
    cleared = False
    for p in model_dir.glob("*.blob"):
        p.unlink()
        cleared = True
    for subdir_name in ["cache", "model_cache"]:
        subdir = model_dir / subdir_name
        if subdir.exists():
            shutil.rmtree(subdir)
            cleared = True
    if cleared:
        print("  Cleared existing model cache files.")
    else:
        print("  No existing cache files found.")


def measure_cache_creation(
    model_dir: Path, device: str, no_mmap: bool, cache_dir: str = None
) -> Tuple[CacheInfo, float]:
    """
    Load the model for the first time (cache creation), measure time and
    peak memory.  Returns (CacheInfo, cache_size_gb_after).
    """
    import openvino as ov
    import openvino_genai as ov_genai

    print("\n--- Cache Creation Measurement ---")
    find_and_clear_cache(model_dir)

    gc.collect()
    reset_peak_working_set()
    time.sleep(1)  # let memory settle

    rss_before = get_rss_gb()
    peak_before = get_peak_working_set_gb()
    print(f"  Memory before load: RSS={rss_before:.2f} GB, Peak={peak_before:.2f} GB")

    t0 = time.perf_counter()

    kwargs = {}
    if no_mmap:
        kwargs['ENABLE_MMAP'] = 'NO'
    if cache_dir:
        kwargs['CACHE_DIR'] = cache_dir

    if kwargs:
        pipe = ov_genai.VLMPipeline(str(model_dir), device, **kwargs)
    else:
        pipe = ov_genai.VLMPipeline(str(model_dir), device)

    cache_time = time.perf_counter() - t0

    peak_after = get_peak_working_set_gb()
    rss_after = get_rss_gb()

    print(f"  Cache creation time: {cache_time:.1f}s")
    print(f"  Memory after load: RSS={rss_after:.2f} GB, Peak={peak_after:.2f} GB")

    cache_size = get_cache_size_gb(model_dir)
    if cache_dir:
        cd = Path(cache_dir)
        if cd.exists():
            cache_size += sum(f.stat().st_size for f in cd.rglob('*') if f.is_file()) / (1024 ** 3)
    print(f"  Cache size on disk: {cache_size:.2f} GB")

    info = CacheInfo(
        cache_creation_time_s=cache_time,
        cache_peak_memory_gb=peak_after,
        cache_size_gb=cache_size,
    )

    # Delete the pipeline to free memory before scenario runs
    del pipe
    gc.collect()
    time.sleep(2)

    return info, cache_size


# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario inference measurement
# ─────────────────────────────────────────────────────────────────────────────

def run_scenario(
    pipe,
    prompt: str,
    scenario_name: str,
    input_tokens: int,
    max_new_tokens: int,
) -> ScenarioResult:
    """Run one inference scenario and collect KPIs."""
    import openvino_genai as ov_genai

    print(f"\n  --- Scenario: {scenario_name} (input ~{input_tokens} tokens) ---")

    config = ov_genai.GenerationConfig()
    if max_new_tokens > 0:
        config.max_new_tokens = max_new_tokens

    gc.collect()
    time.sleep(1)

    # Reset peak so we measure only this scenario's peak
    reset_peak_working_set()
    time.sleep(0.5)

    first_token_time = None
    token_count = 0

    def streamer_callback(subword):
        nonlocal first_token_time, token_count
        token_count += 1
        if first_token_time is None:
            first_token_time = time.perf_counter()
            print("\n    [Output] ", end="", flush=True)
        print(subword, end="", flush=True)
        return False

    gen_start = time.perf_counter()
    result = pipe.generate(
        prompt, generation_config=config, streamer=streamer_callback
    )
    gen_end = time.perf_counter()

    total_time = gen_end - gen_start
    output_tokens = token_count
    print("\n")  # newline after streamed output

    # Compute TTFT
    if first_token_time is not None:
        ttft_s = first_token_time - gen_start
    else:
        ttft_s = total_time  # fallback

    ttft_ms = ttft_s * 1000.0

    # Prefill speed = input_tokens / TTFT_seconds
    prefill_speed = input_tokens / ttft_s if ttft_s > 0 else 0.0

    # Decode time = total - TTFT
    decode_time = total_time - ttft_s
    # Output TPS = output_tokens / decode_time (excluding first token time)
    # Note: first token is generated at TTFT, remaining (output_tokens-1) in decode_time
    if decode_time > 0 and output_tokens > 1:
        output_tps = (output_tokens - 1) / decode_time
    elif output_tokens == 1:
        output_tps = 1.0 / total_time if total_time > 0 else 0.0
    else:
        output_tps = 0.0

    # TPOT = 1000 / output_tps (ms per token)
    tpot_ms = (1000.0 / output_tps) if output_tps > 0 else 0.0

    # Memory
    total_peak = get_peak_working_set_gb()
    private_mem = psutil.Process(os.getpid()).memory_info().private / (1024 ** 3)

    # Also try to get metrics from PerfMetrics if available
    try:
        perf = result.perf_metrics
        # Use PerfMetrics TTFT/TPOT if available (more accurate)
        pm_ttft = perf.get_ttft().mean  # in ms
        pm_tpot = perf.get_tpot().mean  # in ms
        pm_throughput = perf.get_throughput().mean  # tok/s
        pm_in_tok = perf.get_num_input_tokens()
        pm_out_tok = perf.get_num_generated_tokens()
        print(f"    PerfMetrics: TTFT={pm_ttft:.1f}ms, TPOT={pm_tpot:.1f}ms, "
              f"Throughput={pm_throughput:.1f} tok/s")
        print(f"    PerfMetrics: InTok={pm_in_tok}, OutTok={pm_out_tok}")

        # Override with PerfMetrics values (more accurate)
        ttft_ms = pm_ttft
        ttft_s = pm_ttft / 1000.0
        prefill_speed = pm_in_tok / ttft_s if ttft_s > 0 else 0.0
        input_tokens = pm_in_tok
        output_tokens = pm_out_tok

        # Recalculate output TPS from PerfMetrics TPOT
        if pm_tpot > 0:
            output_tps = 1000.0 / pm_tpot
        tpot_ms = pm_tpot

    except Exception as e:
        print(f"    (PerfMetrics not available: {e})")

    sr = ScenarioResult(
        scenario=scenario_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prefill_speed_tps=prefill_speed,
        output_tps=output_tps,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        total_peak_memory_gb=total_peak,
        private_memory_gb=private_mem,
        total_time_s=total_time,
    )

    print(f"    Input tokens     : {sr.input_tokens}")
    print(f"    Output tokens    : {sr.output_tokens}")
    print(f"    TTFT             : {sr.ttft_ms:.1f} ms")
    print(f"    Prefill Speed    : {sr.prefill_speed_tps:.1f} tokens/s")
    print(f"    Output TPS       : {sr.output_tps:.1f} tokens/s")
    print(f"    TPOT             : {sr.tpot_ms:.1f} ms")
    print(f"    Total peak mem   : {sr.total_peak_memory_gb:.2f} GB")
    print(f"    Private mem      : {sr.private_memory_gb:.2f} GB")
    print(f"    Total time       : {sr.total_time_s:.2f} s")

    return sr


# ─────────────────────────────────────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────────────────────────────────────

def print_report(report: FullReport):
    """Print the full KPI report."""
    sep = "=" * 72
    print(f"\n{sep}")
    print("  KPI BENCHMARK REPORT")
    print(sep)
    print(f"  Model          : {report.model_name}")
    print(f"  Model dir      : {report.model_dir}")
    print(f"  Model size     : {report.model_size_gb:.2f} GB")
    print(f"  Device         : {report.device}")
    print(f"  mmap           : {report.mmap_mode}")
    if report.cache_dir:
        print(f"  CACHE_DIR      : {report.cache_dir}")
    print(f"  System memory  : {report.system_memory_gb:.1f} GB")
    print(f"  OpenVINO       : {report.ov_version}")
    print(f"  openvino-genai : {report.genai_version}")
    print()
    print(f"  Model load time: {report.model_load_time_s:.1f} s")
    print(f"  Cache size     : {report.cache_info.cache_size_gb:.2f} GB")
    print(f"  Cache time     : {report.cache_info.cache_creation_time_s:.1f} s")
    print(f"  Cache peak mem : {report.cache_info.cache_peak_memory_gb:.2f} GB")
    print()

    # Table header
    hdr = (
        f"  {'Input tokens':>13} {'Output tokens':>14} "
        f"{'Prefill (t/s)':>14} {'Output TPS':>11} "
        f"{'Total peak (GB)':>16} {'Private (GB)':>13}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in report.scenarios:
        print(
            f"  {s.input_tokens:>13} {s.output_tokens:>14} "
            f"{s.prefill_speed_tps:>14.1f} {s.output_tps:>11.1f} "
            f"{s.total_peak_memory_gb:>16.1f} {s.private_memory_gb:>13.1f}"
        )

    print()
    # Also print TTFT / TPOT details
    scn_w = max(8, max(len(s.scenario) for s in report.scenarios))  # dynamic width
    print(f"  {'Scenario':>{scn_w}} {'TTFT (ms)':>10} {'TPOT (ms)':>10} "
          f"{'Prefill (t/s)':>14} {'Output TPS':>11}")
    print("  " + "-" * (scn_w + 10 + 10 + 14 + 11 + 4))
    for s in report.scenarios:
        print(
            f"  {s.scenario:>{scn_w}} {s.ttft_ms:>10.1f} {s.tpot_ms:>10.1f} "
            f"{s.prefill_speed_tps:>14.1f} {s.output_tps:>11.1f}"
        )

    print(sep)


def save_report_json(report: FullReport, path: str):
    """Save report as JSON for later comparison."""
    import dataclasses

    def to_dict(obj):
        if dataclasses.is_dataclass(obj):
            return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
        elif isinstance(obj, list):
            return [to_dict(x) for x in obj]
        return obj

    with open(path, "w") as f:
        json.dump(to_dict(report), f, indent=2)
    print(f"  Report saved to: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gemma 4 E4B — KPI Benchmark",
    )
    parser.add_argument(
        "--model-dir", required=True,
        help="Path to the OpenVINO IR model directory (e.g. gemma-4-E4B-it-ov)",
    )
    parser.add_argument(
        "--device", default="GPU", choices=["CPU", "GPU"],
        help="Inference device (default: GPU)",
    )
    parser.add_argument(
        "--prompt-file", nargs="+", required=True,
        help="Path(s) to prompt text file(s). Each file is one scenario.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=0,
        help="Max output tokens per scenario (default: 0 = no limit, generate until EOS)",
    )
    parser.add_argument(
        "--no-mmap", action="store_true",
        help="Disable memory-mapped model loading",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Set CACHE_DIR for GPU compiled model cache (reduces peak memory on 2nd+ load)",
    )
    parser.add_argument(
        "--skip-cache-measurement", action="store_true",
        help="Skip cache creation measurement (use if cache already exists)",
    )
    parser.add_argument(
        "--warmup", type=int, default=1,
        help="Number of warmup runs before measurement (default: 1)",
    )
    parser.add_argument(
        "--output-json", default=None,
        help="Save results to JSON file (e.g. results_16gb_mmap.json)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        print(f"Error: model directory not found: {model_dir}")
        sys.exit(1)

    # ── System info ────────────────────────────────────────────────────────
    import openvino as ov
    import openvino_genai as ov_genai

    sys_mem_gb = psutil.virtual_memory().total / (1024 ** 3)
    ov_ver = ov.__version__
    genai_ver = getattr(ov_genai, "__version__", "unknown")

    print(f"{'='*72}")
    print(f"  Gemma 4 E4B — KPI Benchmark")
    print(f"{'='*72}")
    print(f"  Date           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  System memory  : {sys_mem_gb:.1f} GB")
    print(f"  Model dir      : {model_dir}")
    print(f"  Device         : {args.device}")
    print(f"  mmap           : {'OFF (--no-mmap)' if args.no_mmap else 'ON (default)'}")
    print(f"  CACHE_DIR      : {args.cache_dir or 'not set'}")
    print(f"  OpenVINO       : {ov_ver}")
    print(f"  openvino-genai : {genai_ver}")
    print(f"  Prompt files   : {args.prompt_file}")
    print(f"  Max new tokens : {args.max_new_tokens if args.max_new_tokens > 0 else 'no limit'}")
    print(f"  Warmup runs    : {args.warmup}")

    model_size = get_model_size_gb(model_dir)
    print(f"  Model size     : {model_size:.2f} GB")

    report = FullReport(
        model_name=model_dir.name,
        model_dir=str(model_dir),
        model_size_gb=model_size,
        device=args.device,
        mmap_mode="OFF" if args.no_mmap else "ON",
        cache_dir=args.cache_dir or "",
        system_memory_gb=sys_mem_gb,
        ov_version=ov_ver,
        genai_version=genai_ver,
    )

    # ── Step 1: Cache creation measurement ─────────────────────────────────
    if not args.skip_cache_measurement:
        cache_info, _ = measure_cache_creation(model_dir, args.device, args.no_mmap, args.cache_dir)
        report.cache_info = cache_info
    else:
        print("\n  Skipping cache creation measurement.")
        report.cache_info = CacheInfo(
            cache_size_gb=get_cache_size_gb(model_dir)
        )

    # ── Step 2: Load model for scenario runs ────────────────────────────────
    print(f"\n--- Loading model for scenario runs ---")
    gc.collect()
    time.sleep(2)

    t0 = time.perf_counter()
    load_kwargs = {}
    if args.no_mmap:
        load_kwargs['ENABLE_MMAP'] = 'NO'
    if args.cache_dir:
        load_kwargs['CACHE_DIR'] = args.cache_dir

    if load_kwargs:
        pipe = ov_genai.VLMPipeline(str(model_dir), args.device, **load_kwargs)
    else:
        pipe = ov_genai.VLMPipeline(str(model_dir), args.device)

    load_time = time.perf_counter() - t0
    report.model_load_time_s = load_time
    print(f"  Model loaded in {load_time:.1f}s (cache reuse), RSS={get_rss_gb():.2f} GB")

    # ── Step 3: Load prompts from files ──────────────────────────────────────
    print(f"\n--- Loading prompts from files ---")
    tokenizer_ref = ov_genai.Tokenizer(str(model_dir))

    scenario_prompts = []
    for pf in args.prompt_file:
        prompt = load_prompt_file(pf)
        actual = tokenize_prompt(tokenizer_ref, prompt)
        name = Path(pf).stem  # filename without extension as scenario name
        scenario_prompts.append((name, prompt, actual))
        print(f"  {name}: {actual} input tokens  ({pf})")

    # ── Step 4: Warmup ──────────────────────────────────────────────────────
    if args.warmup > 0:
        print(f"\n--- Warmup ({args.warmup} runs with shortest prompt) ---")
        warmup_prompt = scenario_prompts[0][1]
        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 16  # short warmup
        for i in range(args.warmup):
            print(f"  Warmup {i+1}/{args.warmup}...")
            pipe.generate(warmup_prompt, generation_config=config)
        print("  Warmup complete.")

    # ── Step 5: Run scenarios ───────────────────────────────────────────────
    print(f"\n--- Running {len(scenario_prompts)} scenarios ---")

    for scenario_name, prompt, actual_tokens in scenario_prompts:
        sr = run_scenario(
            pipe, prompt, scenario_name, actual_tokens, args.max_new_tokens
        )
        report.scenarios.append(sr)

    # ── Step 6: Print final report ──────────────────────────────────────────
    print_report(report)

    if args.output_json:
        save_report_json(report, args.output_json)

    # ── Auto-generate comparison filename if not specified ──────────────
    if not args.output_json:
        mem_tag = f"{int(sys_mem_gb)}gb"
        mmap_tag = "nommap" if args.no_mmap else "mmap"
        cache_tag = "_cached" if args.cache_dir else ""
        auto_path = f"results_{mem_tag}_{mmap_tag}{cache_tag}_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_report_json(report, auto_path)


if __name__ == "__main__":
    main()
