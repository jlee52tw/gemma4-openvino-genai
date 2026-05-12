# DirectStorage vs ReadFile vs mmap — MoE Weight I/O Benchmark

Standalone benchmark for comparing I/O paths when loading MoE expert weights from a
single-file OTD `.bin` (MOEW format). Simulates the real MoE inference workload: for each
token, a router selects top-K experts per layer and their weights must be loaded from NVMe.

## I/O Methods Tested

| # | Method | `CreateFile` Flags | Cache | Description |
|---|--------|-------------------|-------|-------------|
| 1 | **ReadFile (Buffered)** | `FILE_FLAG_RANDOM_ACCESS` | OS page cache | Standard Win32 `ReadFile` — reads go through the Windows filesystem cache. Fast on warm runs (data served from RAM), slow on cold start. |
| 2 | **ReadFile (Direct I/O)** | `FILE_FLAG_NO_BUFFERING \| FILE_FLAG_RANDOM_ACCESS` | Bypassed | Bypasses OS page cache — every read goes to NVMe. Requires sector-aligned offsets/sizes, so reads into an aligned temp buffer then `memcpy` to destination. Consistent latency but slower than buffered on warm runs. |
| 3 | **mmap** | N/A (`CreateFileMapping` + `MapViewOfFile`) | OS page cache | Maps the entire file into virtual address space, then `memcpy` per expert. OS handles page faults and caching. Fastest warm, slowest cold (page fault overhead). |
| 4 | **DirectStorage (Batched)** | N/A (DS API) | BypassIO (kernel bypass) | All 192 read requests enqueued to a single DS queue, one `Submit()`, one fence wait. Bypasses the entire kernel I/O stack via BypassIO — NVMe commands go directly from user-mode to the drive. |
| 5 | **DirectStorage (Parallel)** | N/A (DS API) | BypassIO (kernel bypass) | Same as batched but uses N per-thread queues. Each thread enqueues and submits its portion independently. |

### ReadFile (Buffered) vs ReadFile (Direct I/O) — Key Differences

#### The I/O Path

| Aspect | ReadFile (Buffered) | ReadFile (Direct I/O) |
|--------|--------------------|-----------------------|
| `CreateFile` flags | `FILE_FLAG_RANDOM_ACCESS` | `FILE_FLAG_NO_BUFFERING \| FILE_FLAG_RANDOM_ACCESS` |
| OS page cache | ✅ Goes **through** Windows filesystem cache | ❌ **Bypasses** cache entirely |
| Alignment | None required — any offset, any size | Must be **sector-aligned** (512B offset, 512B-multiple size) |
| Buffer | Reads directly to destination `buffer` | Reads into `_aligned_malloc` temp buffer, then `memcpy` to destination |
| Cold start | Slow (~424 ms) — must populate cache | Faster (~259 ms) — no cache to warm up |
| Warm state | Fast (~157 ms @ 7.0 GB/s) — reads from RAM cache | Slower (~238 ms @ 4.7 GB/s) — **always hits disk**, never from RAM |
| Extra copy | 0 copies (kernel → user buffer directly) | 1 extra `memcpy` per read (aligned tmp → final dest) |
| I/O path | App → Kernel (NTFS) → Cache Manager → return | App → Kernel (NTFS) → Storage Driver → NVMe → return |

#### Why "Direct I/O" is slower on warm runs

With **Buffered** ReadFile, after warmup iteration 0 loads the data into the Windows page
cache, iterations 1–10 read from **RAM** (~7 GB/s). The OS caches the file pages and serves
them from memory.

With **Direct I/O** (`FILE_FLAG_NO_BUFFERING`), **every iteration goes to the NVMe** — the
OS cache is bypassed entirely. Plus there's an extra `memcpy` because Direct I/O requires
sector-aligned offsets/sizes, so the code reads into a temporary aligned buffer then copies
the actual data region out:

```
  Direct I/O:  NVMe → sector-aligned temp buffer → memcpy → final destination buffer
  vs.
  Buffered:    NVMe/cache → final destination buffer (direct)
```

## Per-Iteration Workload Breakdown

With `--layers 24 --topk 4 --seed 42` on the 20B model (`moe_weights_otd.bin`):

### Expert Selection (one-time, same for ALL methods and ALL iterations)

Using seed=42, the benchmark generates random top-4 expert indices per layer:

- **24 layers × 4 experts/layer = 96 expert reads**
- Each expert has **2 components** read from the file:
  - `up_weight`: 7.91 MB (8,294,400 bytes) at offset `expert_base`
  - `down_weight`: 3.96 MB (4,147,200 bytes) at offset `expert_base + up_weight + up_scale + up_bias`
- **Per expert**: 11.87 MB of data read (up + down)
- **Per layer**: 4 × 11.87 = **47.46 MB**
- **Per iteration total**: 96 × 11.87 = **1,139.06 MB (1.11 GB)**
- **Total ReadFile/DS calls per iteration**: 96 experts × 2 components = **192 individual reads**

### What happens in ONE iteration

```
For each of 96 experts (spread across 4 threads in parallel):
  ├── Read #1:  up_weight   = 7.91 MB from file offset [expert_base]
  └── Read #2:  down_weight = 3.96 MB from file offset [expert_base + 9.35 MB gap]

Total: 192 ReadFile() calls → 1,139 MB read → into 1,284 MB VirtualAlloc buffer
```

The reads are **scattered** across a 10.04 GB file — expert indices are random, so reads
land at non-sequential offsets across different layers. This is a **random I/O** workload,
not sequential.

### Per method, what differs per iteration

| Method | Opens | Calls/iter | Alignment | Cache |
|--------|-------|-----------|-----------|-------|
| **ReadFile (Buffered)** | 1 `HANDLE` | 192 `ReadFile` w/ `OVERLAPPED` | None | Uses OS cache |
| **Direct I/O** | 1 `HANDLE` | 192 `ReadFile` w/ `OVERLAPPED` | Sector-aligned (512B) | Bypasses cache |
| **mmap** | 1 mapping over whole 10 GB file | 192 `memcpy` from mapped view | None | Uses OS cache + page faults |
| **DS Batched** | 1 DS file, 1 queue | 192 DS requests enqueued → 1 `Submit` → 1 fence wait | None | BypassIO (kernel bypass) |
| **DS Parallel** | 1 DS file, 4 queues | 192 requests split across 4 queues → 4 `Submit` → 4 fence waits | None | BypassIO (kernel bypass) |

## Prerequisites

- **Windows 10+** for ReadFile / Direct I/O / mmap
- **Windows 11 22H2+** (build 22621+) for DirectStorage BypassIO
- **NVMe SSD** with BypassIO support (inbox `stornvme` driver, not Intel RST)
- **Visual Studio 2022** (or 2019) with C++ desktop workload
- **CMake 3.20+**
- **DirectStorage SDK** — auto-downloaded during cmake (see below)

### Check BypassIO Support

```powershell
# Run in Admin PowerShell
fsutil bypassIo state C:

# Expected for supported system:
#   BypassIo on "C:" is currently supported.
#   Storage Type:   NVMe
#   Driver:         stornvme
```

## Build

The DirectStorage SDK (2.8 MB) is **automatically downloaded** from nuget.org on first
cmake configure. No manual setup needed.

```powershell
cd ds_perf_test

# Configure — auto-downloads DirectStorage SDK on first run
cmake -B build -G "Visual Studio 17 2022" -A x64

# Build Release
cmake --build build --config Release
```

The executable will be at: `build\Release\ds_perf_test.exe`
(with `dstorage.dll` and `dstoragecore.dll` copied alongside it)

### Build Behind a Corporate Proxy

If behind a proxy (e.g., Intel lab), set the proxy env vars **before** cmake:

```powershell
$env:https_proxy = "http://proxy-dmz.intel.com:912"
$env:http_proxy  = "http://proxy-dmz.intel.com:912"
$env:no_proxy    = ".intel.com,intel.com,localhost,127.0.0.1"

cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

CMake tries 3 download methods in order: `file(DOWNLOAD)` → PowerShell `WebClient` → `curl.exe`.

### Build with a Manually-Specified SDK Path

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64 ^
      -DDIRECTSTORAGE_SDK_PATH="C:\path\to\DirectStorage.1.3.0\native"
cmake --build build --config Release
```

### Build Without DirectStorage (ReadFile / Direct I/O / mmap only)

If the SDK download fails or the system lacks DS support, the build proceeds
automatically — DS benchmarks are excluded, all other methods still work.

### SDK Discovery Priority

| Priority | Location | When |
|----------|----------|------|
| 1 | `-DDIRECTSTORAGE_SDK_PATH=<path>` | User explicitly specifies |
| 2 | `build/directstorage_sdk/...` | Auto-downloaded (cached from previous build) |
| 3 | `../packages/Microsoft.Direct3D.DirectStorage.1.3.0/native` | Sibling NuGet packages dir |
| 4 | Auto-download from nuget.org | First build on a new system |

## Usage

```powershell
ds_perf_test.exe --file <path_to_moe_weights.bin> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--file <path>` | *required* | Path to MoE weights `.bin` file (MOEW format) |
| `--layers <N>` | all | Number of layers to test (1 to num_layers in header) |
| `--topk <K>` | 4 | Random experts per layer to load |
| `--iterations <N>` | 10 | Benchmark iterations (timed, used for statistics) |
| `--warmup <N>` | 2 | Warmup iterations (not included in stats) |
| `--threads <N>` | 4 | Thread count / DS queue count |
| `--seed <N>` | 42 | Random seed for expert selection |
| `--method <m>` | all | `readfile`, `directio`, `mmap`, `ds_batched`, `ds_parallel`, `ds` (both DS), `all` |
| `--purge-cache` | off | Try to purge page cache between iterations (admin recommended) |

### Examples

```powershell
# Run all 5 methods on 20B model, top-4 experts, all 24 layers, 10 iters + 2 warmup
ds_perf_test.exe --file C:\ov_models\gpt-oss-20b-slim-v2\moe_weights_otd.bin

# DirectStorage only, 8 layers, top-8 experts
ds_perf_test.exe --file C:\ov_models\gpt-oss-20b-slim-v2\moe_weights_otd.bin ^
    --method ds --layers 8 --topk 8

# Compare mmap vs DirectStorage with more iterations
ds_perf_test.exe --file C:\ov_models\gpt-oss-20b-slim-v2\moe_weights_otd.bin ^
    --method all --iterations 20 --warmup 5

# Cold-cache test (run as Administrator for best effect)
ds_perf_test.exe --file C:\ov_models\gpt-oss-20b-slim-v2\moe_weights_otd.bin ^
    --purge-cache --iterations 5

# Single layer, top-2, 50 iterations (micro-benchmark)
ds_perf_test.exe --file C:\ov_models\gpt-oss-20b-slim-v2\moe_weights_otd.bin ^
    --layers 1 --topk 2 --iterations 50

# 120B model (60 GB file, 24 layers × 128 experts)
ds_perf_test.exe --file C:\ov_models\gpt-oss-120b-slim-v2\moe_weights_otd.bin ^
    --layers 24 --topk 8
```

## Output

The benchmark produces:

1. **File header dump** — layer count, expert sizes, data layout, total data region
2. **Benchmark config** — top-K, layers, total reads, threads, data per iteration
3. **Full-pass results** — per-iteration timing with min/avg/median/max/GB/s for each method
4. **Per-layer breakdown** — timing per layer to identify hot spots or layer-dependent behavior
5. **Summary** — file, configuration, seed

### Sample Output (24 layers × top-4, 12xe system)

```
============================================================
  FULL-PASS RESULTS (all 24 layers x top-4 = 96 experts)
============================================================

--- ReadFile (Buffered, 4 threads) ---
    [warmup  0] 424.03 ms  (2.62 GB/s)
    [warmup  1] 161.43 ms  (6.89 GB/s)
    [iter    2] 158.82 ms  (7.00 GB/s)
    ...
  ReadFile        min=  155.75 ms  avg=  158.48 ms  median=  157.98 ms  |  7.04 GB/s

--- ReadFile (Direct I/O, FILE_FLAG_NO_BUFFERING, 4 threads) ---
    [warmup  0] 259.27 ms  (4.29 GB/s)
    [warmup  1] 242.70 ms  (4.58 GB/s)
    [iter    2] 252.57 ms  (4.40 GB/s)
    ...
  DirectIO        min=  227.45 ms  avg=  236.43 ms  median=  238.12 ms  |  4.67 GB/s

--- Memory-Mapped I/O (MapViewOfFile + memcpy, 4 threads) ---
    [warmup  0] 766.33 ms  (1.45 GB/s)    ← cold: page faults
    [warmup  1] 151.19 ms  (7.36 GB/s)    ← warm: from RAM
    ...
  mmap            min=  150.33 ms  avg=  153.71 ms  median=  154.08 ms  |  7.22 GB/s

--- DirectStorage BATCHED ---
    [warmup  0] 176.78 ms  (6.29 GB/s)
    [iter    2] 168.14 ms  (6.62 GB/s)
    ...
  DS-Batched      min=  168.01 ms  avg=  168.31 ms  median=  168.20 ms  |  6.61 GB/s

--- DirectStorage PARALLEL (4 queues) ---
    ...
  DS-Parallel     min=  168.25 ms  avg=  168.63 ms  median=  168.45 ms  |  6.60 GB/s
```

### Interpreting Results

| Method | Warm Median | GB/s | Why |
|--------|-----------|------|-----|
| **mmap** | 154 ms | 7.22 | Warm cache — data in RAM, `memcpy` only |
| **ReadFile (Buffered)** | 158 ms | 7.04 | Warm cache — similar to mmap but through ReadFile API |
| **DS Batched** | 168 ms | 6.61 | Always from NVMe via BypassIO — consistent, no cache dependency |
| **DS Parallel** | 168 ms | 6.60 | Same as batched — NVMe is the bottleneck, not queue count |
| **Direct I/O** | 238 ms | 4.67 | Always from NVMe but through full kernel I/O stack + extra `memcpy` |

**Key insight:** Buffered and mmap win warm-state because they serve from **RAM**. DS and
Direct I/O always hit disk — but DS does it faster because BypassIO skips the kernel
filesystem stack, while Direct I/O still goes through the full
`NTFS → Storage Driver → NVMe` path. On cold start, DS is **4.3× faster** than mmap
(177 ms vs 766 ms).

## File Format (MOEW)

The `.bin` file uses the MOEW header format:

```
[Header: 128 bytes]
  magic[4]    = "MOEW"
  version     = 1
  num_layers, num_experts_per_layer
  expert_up_weight_size, expert_down_weight_size
  expert_up/down_scale_size, expert_up/down_bias_size
  data_offset = 128

[Data region: layer × experts × components]
  Layer 0: Expert 0 [up_w | up_scale | up_bias | down_w | down_scale | down_bias]
           Expert 1 [...]
           ...
           Expert 31 [...]
  Layer 1: Expert 0 [...]
           ...
  ...
  Layer 23: Expert 31 [...]
```

For the 20B model: 24 layers × 32 experts × 13.38 MB/expert = 10.04 GB data region.

## Notes

- The buffer is allocated via `VirtualAlloc` (page-aligned) to simulate USM host memory.
- DirectStorage tests create/destroy the factory per iteration; in production, the factory persists.
- For fair cold-cache comparison, use `--purge-cache` with admin privileges.
- On systems without DirectStorage runtime (e.g., 4xe PTL), DS tests report `E_NOTIMPL` and skip gracefully.
- The benchmark reads only `up_weight` and `down_weight` per expert (not scales/biases) — matching the real inference hot path.
- See [DIRECTSTORAGE_DEPENDENCY.md](DIRECTSTORAGE_DEPENDENCY.md) for detailed SDK setup, proxy config, and troubleshooting.
