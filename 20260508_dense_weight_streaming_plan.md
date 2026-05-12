# Dense Model Weight Streaming — Feasibility Plan
**Date:** 2026-05-08
**Model:** Gemma-4-E4B-it (INT4), 42 layers, 6.05 GB
**Target:** Minimize memory footprint for 8 GB system iGPU inference
**Reference:** [MoE OTD DirectStorage](https://github.com/jlee52tw/openvino/blob/moe-otd-pr-squash/moe_cpp/20260326_how_to_intercept_moe_expert_topk_for_otd.md)

---

## 1. Baseline Performance (from README.md)

**Hardware:** Intel PTL 12Xe iGPU, 16 GB LPDDR5
**Software:** OpenVINO 2026.2.0.dev + openvino.genai PR #3644

| Scenario | Output TPS | TPOT (ms) | TTFT (s) | Peak RSS (GB) |
|---|---:|---:|---:|---:|
| short-text | **24.0** | **41.7** | 0.30 | 7.1 |
| long-text (1024 in) | **17.4** | **57.5** | 0.88 | 7.9 |
| short-image | 19.7 | 50.8 | 0.54 | 8.3 |

### Per-Layer Timing (decode phase)

| Scenario | TPOT | Per-layer | Per 7-layer group |
|---|---:|---:|---:|
| short-text (24 tps) | 41.7 ms | ~1.0 ms | ~7.0 ms |
| long-text (17.4 tps) | 57.5 ms | ~1.4 ms | ~9.6 ms |

---

## 2. Model Weight Breakdown

### Per-Layer Weight Size (INT4, group_size=64, asymmetric)

| Component | Shape | INT4+scale Size |
|---|---|---:|
| Q proj | 2560×2048 | ~2.8 MB |
| K proj (sliding) | 2560×512 | ~0.7 MB |
| V proj (sliding) | 2560×512 | ~0.7 MB |
| O proj | 2048×2560 | ~2.8 MB |
| Gate proj | 2560×10240 | ~14.1 MB |
| Up proj | 2560×10240 | ~14.1 MB |
| Down proj | 10240×2560 | ~14.1 MB |
| LayerNorm ×2 | 2560 | ~0.01 MB |
| **Per layer total** | | **~49.5 MB** |
| **42 layers total** | | **~2.09 GB** |

### Non-Decoder Components (must stay resident)

| Component | Estimated Size |
|---|---:|
| Token embedding (262144×2560) | ~1.3 GB |
| Per-layer token embedding | ~0.13 GB |
| Vision encoder (16 layers) | ~0.15 GB |
| Audio encoder (12 layers) | ~0.08 GB |
| Other (LM head, norms) | ~0.1 GB |
| **Non-decoder total** | **~1.8 GB** |
| OpenVINO runtime + KV cache + activations | **~2.0-3.0 GB** |

### 8 GB System Memory Budget

| Component | Memory |
|---|---:|
| OS + system | ~2.0 GB |
| Non-decoder model weights (resident) | ~1.8 GB |
| OpenVINO runtime overhead | ~0.8 GB |
| KV cache (512 sliding window) | ~0.3 GB |
| Activations / intermediates | ~0.3 GB |
| **Available for decoder layer weights** | **~2.8 GB** |
| Decoder layers total | 2.09 GB |
| **Fits?** | **✅ YES (tight)** |

---

## 3. NVMe Gen5×4 Bandwidth Analysis

| Parameter | Value |
|---|---|
| PCIe 5.0 ×4 raw bandwidth | 15.75 GB/s |
| NVMe Gen5×4 sustained sequential read | ~12 GB/s |
| DirectStorage BypassIO (zero-copy) | ~12-14 GB/s |

### Can IO Be Hidden Behind GPU Compute? (Decode Phase)

```
Per 7-layer group:
  GPU compute time:  ~7.0 ms (at 24 tps)
  NVMe IO time:      350 MB / 12 GB/s = ~29.2 ms
  IO / compute ratio: 4.2x  ❌  IO is 4x slower than compute

Double-buffer pipeline (6 groups of 7 layers):
  Token TPOT = 6 × max(29.2, 7.0) = ~175 ms → ~5.7 tps
  vs baseline 41.7 ms → 24 tps
  Effective slowdown: ~4.2x
```

### Theoretical TPS vs Weight Budget (8 GB System)

| Weight Budget | Resident % | Swap/token | IO time | TPS |
|---:|---:|---:|---:|---:|
| All 2.09 GB | 100% | 0 | 0 | **24 tps** (baseline) |
| 1.5 GB budget | 72% | ~0.6 GB | 50 ms | **~11 tps** |
| 1.0 GB budget | 48% | ~1.1 GB | 92 ms | **~7 tps** |
| 0.5 GB budget | 24% | ~1.6 GB | 133 ms | **~5 tps** |
| 0.35 GB (2 groups) | 17% | ~1.7 GB | 146 ms | **~5 tps** |

> **Conclusion:** On 8 GB system with ~1.5 GB available for decoder layers
> (keeping ~28 layers resident, swapping ~14 layers), achievable target is
> ~11 tps. Full streaming (all layers swapped) gives ~5-6 tps.

---

## 4. Approach Decision

### ❌ Option C — MMAP (Deleted)
Creates OS memory pressure. `attach_or_copy_data()` copies mmap→USM, doubling
memory usage. On 8 GB system causes pagefile thrashing. **Not viable.**

### ❌ Option D — MMAP-backed USM (Deleted)
Essentially still mmap. GPU page fault behavior undefined on iGPU. DirectStorage
is strictly superior for NVMe→memory with zero-copy and no OS page cache
pressure. **Not viable. Use DirectStorage instead.**

### 🏆 Option B — GenAI-Level Model Splitting (DO FIRST)

**Concept:** Split 42-layer language model into N sub-models. Load/unload
sub-models sequentially, passing hidden states between them via USM buffer.

**Inter-group handover data:**

| Data | Decode (1 token) | Prefill (1024 tokens) |
|---|---:|---:|
| Hidden states [B, seq, 2560] FP16 | 5 KB | 5 MB |
| Attention mask | ~bytes | ~4 KB |
| Position IDs | ~bytes | ~4 KB |
| KV cache (persists in host memory) | ~300 MB | ~300 MB |
| **Transfer overhead** | **< 0.1 ms** | **< 1 ms** |

> **Hidden state handover is negligible.** The real question is model load/unload
> time with blob cache.

**Core test (ignore first-time compilation — one-time cost):**

| What to measure | Why |
|---|---|
| Blob-cached sub-model load time | This determines per-token overhead |
| Blob-cached sub-model unload time | Memory release speed |
| Full round-trip: load→infer 1 token→unload | End-to-end per-group cost |
| Weightless blob + direct weight buffer write | Fastest possible reload |

**Expected results to determine viability:**

| Per sub-model load time | Groups/token | Total overhead | Effective TPS | Verdict |
|---:|---:|---:|---:|---|
| < 50 ms | 6 | < 300 ms | ~3 tps | Marginal |
| < 100 ms | 6 | < 600 ms | ~1.6 tps | Too slow alone |
| < 20 ms | 6 | < 120 ms | ~6 tps | Usable |
| < 10 ms | 6 | < 60 ms | ~10 tps | Good |

> If blob-cache load is > 100 ms per sub-model, Option B alone is insufficient
> and must be combined with Option A (DirectStorage) for weight buffer swap.

### Option A — Direct I/O + GPU Plugin Weight Streaming (PHASE 2)

**Concept:** Use Win32 Direct I/O (`FILE_FLAG_NO_BUFFERING`) to stream decoder
weights from NVMe directly into USM host buffers. GPU reads weights from the
same LPDDR5 memory (iGPU shares system memory). Zero-copy end-to-end.

**Why Direct I/O instead of DirectStorage?**

IO benchmark (§8) showed Direct I/O achieves **11.2 GB/s** vs DS **8.5 GB/s**
for dense sequential reads. DS's BypassIO advantage (eliminating kernel
overhead) only helps random small-IO workloads like MoE. For dense sequential
reads, kernel read-ahead optimization gives Direct I/O the edge.

**Data path (zero-copy):**
```
NVMe SSD →(DMA)→ LPDDR5 (USM host buffer) →(GPU kernel read)→ iGPU compute
               ReadFile(NO_BUFFERING)          same physical memory
```

**Reuse from MoE OTD concepts:**

| MoE OTD Component | Dense Adaptation |
|---|---|
| `allocate_buffers()` USM host | 2 ping-pong buffers × N-layer size |
| `load_experts()` with offset calc | `load_layer_group(n)` sequential layout |
| `preload_hot_experts()` | `preload_resident_layers()` |
| Tiered LRU eviction | Not needed (sequential, always evict oldest) |
| `slot_mapping` + indirect addressing | Not needed (fixed layer order) |
| DirectStorage init (4 queues) | **Replaced by** Direct I/O file handle |

**Simpler than MoE OTD because:**
- Access pattern is **always sequential** (layer 0→41), not random
- No router decision needed — all layers always execute
- No slot mapping needed — just ping-pong buffer A/B
- Uses simple Win32 ReadFile API instead of DirectStorage SDK
- Multi-threaded OVERLAPPED reads for +7% throughput

---

## 5. Implementation Plan

### Phase 1: Option B — Model Splitting & Load Time Measurement ✅ DONE

**Goal:** Get real data on blob-cached sub-model load/unload time.

- [x] **Step 1:** Write Python tool to split `openvino_language_model.xml/bin`
  into 2 sub-models (21 layers each) → `split_language_model.py`
- [x] **Step 2:** Compile both sub-models to GPU with `CACHE_DIR` (one-time)
- [x] **Step 3:** Measure blob-cached load time per sub-model (repeated loads)
  → `measure_load_time.py`
- [x] **Step 4:** Measure unload (release compiled_model, memory freed)

**Decision gate result:** Load time ~1.0-1.2 s per sub-model >> 200 ms
→ **Option B alone is NOT viable. Proceeding to Option A.**

### Phase 2: Option A — Direct I/O Weight Streaming (2-4 weeks) ← ACTIVE

**Phase 1 confirmed load time too high. Now implementing Option A with Direct I/O.**

- [x] **Step 1:** Design `DenseWeightStreamingManager` class
  → `cpp/dense_weight_streaming_manager.hpp` + `.cpp`
- [x] **Step 2:** Create `dense_weights_streaming.bin` packer tool
  → `pack_dense_weights.py` (per-layer contiguous layout, sector-aligned)
- [x] **Step 3:** Implement double-buffer manager (ping-pong USM allocations)
  - Buffer A: current layer group being computed by GPU
  - Buffer B: next layer group being loaded from NVMe via Direct I/O
- [x] **Step 3b:** IO benchmark → Direct I/O (11.2 GB/s) > DirectStorage (8.5 GB/s)
  → **Decision: use Direct I/O API** (`FILE_FLAG_NO_BUFFERING`, multi-threaded)
- [x] **Step 4:** Update `DenseWeightStreamingManager` to use Direct I/O
  - Replace DirectStorage code with `ReadFile` + `FILE_FLAG_NO_BUFFERING`
  - Multi-threaded OVERLAPPED reads (4 threads per group)
  - Read directly into USM host buffer (page-aligned, zero-copy to GPU)
- [x] **Step 5:** Implement `swap_weight_pointers()` GPU plugin integration
  - `parse_layer_index_from_name()` — parse "layers.N" from OV constant names
  - `build_weight_mapping()` — scan cldnn::network primitives, build tensor→primitive mapping
  - `build_weight_mapping_from_json()` — read JSON metadata + cross-reference with network
  - `swap_weight_pointers()` — compute USM offsets, `engine.attach_memory()`, `force_set_output_memory()`, `set_arguments()`
  - Standalone stub path compiles clean; GPU integration path ready for `primitive_inst.h` patch
  - **Required OV patch:** add `force_set_output_memory()` to `primitive_inst.h` (1 line)
- [x] **Step 6:** Add layer-group boundary synchronization
  - `execute_streamed_decode()` — full per-token pipeline orchestrator
  - `wait_for_gpu()` — GPU fence via `cldnn::stream::finish()`
  - `build_group_exec_order()` — partition `_exec_order` into per-group primitive lists
  - `TokenPipelineStats` — per-token timing breakdown (io_wait, swap, gpu, gpu_fence per group)
  - `GroupComputeCallback` — pluggable GPU compute function for testing
  - Pipeline: cold-load G0 → [swap → GPU + prefetch G1] → [fence → IO wait → swap → GPU + prefetch] → ...
  - `force_set_output_memory()` patch applied to OpenVINO `primitive_inst.h`
  - **OpenVINO GPU plugin build verified:** `openvino_intel_gpu_plugin.dll` built successfully (Release, VS2022)
  - Standalone `dense_weight_streaming_manager.cpp` compiles clean (MSVC C++17)
- [x] **Step 7:** Pipeline Benchmark (H5+T5 Hybrid)
- [ ] **Step 8:** Tune group size (1 layer vs multi-layer groups)

**Key architecture (H5+T5 Hybrid Pinning):**
```
┌─────────────────────────────────────────────────────┐
│  compile_model() — ONE TIME                         │
│  (or load from .blob cache — 2.3s startup)          │
│  → Compiled graph with weight buffer pointers       │
└─────────────────────────────────────────────────────┘
          ↓ (at runtime, per token)
┌─────────────────────────────────────────────────────┐
│  H5+T5 Three-Phase Pipeline (per token decode):     │
│                                                     │
│  Phase 1: HEAD (pinned, layers 0-4)                 │
│    GPU compute 5 layers (~5 ms), no IO              │
│                                                     │
│  Phase 2: STREAMED MIDDLE (layers 5-36)             │
│    Cold-load layer 5 → Buffer A                     │
│    GPU compute layer 5 (Buffer A) +                 │
│      async prefetch layer 6 → Buffer B              │
│    Swap A↔B, GPU layer 6 + prefetch layer 7 → A    │
│    ... repeat 32 single-layer groups ...             │
│    [4 IO threads, ReadFile NO_BUFFERING, 11.1 GB/s] │
│                                                     │
│  Phase 3: TAIL (pinned, layers 37-41)               │
│    GPU compute 5 layers (~5 ms), no IO              │
│                                                     │
│  Data path: NVMe→DMA→LPDDR5(USM)→iGPU (zero-copy)  │
└─────────────────────────────────────────────────────┘
```

---

## 6. Key Differences: Dense Streaming vs MoE OTD

| Aspect | MoE OTD | Dense Weight Streaming |
|---|---|---|
| Per-token IO | ~50 MB (4/128 experts) | 0.6-2 GB (all layers) |
| Access pattern | Random (per expert) | Sequential (layer 0→41) |
| Cache hit rate | ~80-98% (hot pinning) | 0% (all layers needed every token) |
| IO hideable? | ✅ (IO < compute) | ❌ (IO >> compute) |
| Indirection needed? | Yes (slot_mapping) | No (fixed order) |
| IO API | DirectStorage (BypassIO) | **Direct I/O** (FILE_FLAG_NO_BUFFERING) |
| IO read size | ~12 MB per expert | ~350 MB per group |
| Prefetch strategy | LRU + hot pinning | Double-buffer sequential |

---

## 7. Success Criteria

| Metric | 8 GB Target | Notes |
|---|---:|---|
| Peak memory during inference | < 7.0 GB | Down from ~9 GB |
| Output TPS (short-text) | ≥ 5 tps | Acceptable 5x slowdown from 24 tps |
| Output TPS (long-text) | ≥ 4 tps | From 17.4 tps baseline |
| Correctness | Bit-exact | Output must match full model |
| First-time compile (one-time) | < 5 min | Acceptable for one-time setup |
| Cached model load (runtime) | < 30 s | Total startup with streaming |

---

## 8. IO Benchmark Results (Measured 2026-05-08)

### Test Setup

- **File:** `dense_weights_streaming.bin` (1.981 GB, 6 groups × 7 layers)
- **IO chunk size:** 64 MB (unified for all methods)
- **Buffer:** VirtualAlloc page-aligned (341.6 MB, largest group)
- **Warmup:** 3 iterations | **Measured:** 10 iterations
- **Benchmark:** `cpp/benchmark_directstorage_io.cpp` v3

### Results Summary

| # | Method | GB/s | Median (ms) | TPOT serial | TPS serial | TPS overlap |
|---|---|---:|---:|---:|---:|---:|
| 3 | **Direct I/O, 4 threads** | **11.19** | **176.5** | **218.5** | **4.6** | **5.7** |
| 2 | Direct I/O, 1 thread | 10.38 | 188.9 | 230.9 | 4.3 | 5.3 |
| 6 | DS single queue, 341MB staging | 8.58 | 231.4 | 273.4 | 3.7 | 4.3 |
| 4 | DS single queue, 64MB chunks | 8.50 | 233.1 | 275.1 | 3.6 | 4.3 |
| 5 | DS 4 queues, parallel | 7.90 | 249.7 | 291.7 | 3.4 | 4.0 |
| 1 | ReadFile buffered | 7.14 | 276.6 | 318.6 | 3.1 | 3.6 |

> **TPOT serial** = IO median + 42ms GPU compute (no overlap)
> **TPS overlap** = 1000 / max(IO median, 42ms) (ideal double-buffer)

### Per-Group Breakdown (Direct I/O, 4 threads — best method)

| Group | Size | Median | GB/s |
|---|---:|---:|---:|
| 0 | 341.6 MB | 29.7 ms | 11.26 |
| 1 | 341.6 MB | 30.2 ms | 11.15 |
| 2 | 341.6 MB | 29.7 ms | 11.27 |
| 3 | 336.2 MB | 29.6 ms | 11.10 |
| 4 | 330.8 MB | 28.9 ms | 11.13 |
| 5 | 336.2 MB | 29.1 ms | 11.24 |

### Why Direct I/O Beats DirectStorage for Dense Sequential IO

| Factor | Direct I/O | DirectStorage |
|---|---|---|
| Kernel sequential optimization | ✅ `FILE_FLAG_SEQUENTIAL_SCAN` enables read-ahead | ❌ BypassIO skips kernel, loses read-ahead |
| Per-request kernel overhead | Small — only ~36 ReadFile calls per iteration | N/A — kernel bypassed |
| BypassIO benefit | N/A | Small savings for few large requests |
| Multi-thread scaling | ✅ +7% with 4 threads (OVERLAPPED reads) | ❌ Multi-queue adds overhead |
| **Net result** | **11.2 GB/s** | **8.5 GB/s** |

> **Key insight:** DS's BypassIO advantage is eliminating per-request kernel
> overhead — significant for MoE's 192 random small reads (DS 6.6 > DirectIO
> 4.7 GB/s), but negligible for dense's 36 large sequential reads where
> kernel read-ahead optimization dominates.

### Updated Performance Estimates

Based on actual measured IO throughput (Direct I/O 4T, 11.2 GB/s):

| Scenario | IO time | GPU time | TPOT | TPS |
|---|---:|---:|---:|---:|
| Serial (IO + GPU) | 176.5 ms | 42 ms | 218.5 ms | **4.6** |
| Ideal overlap (double-buffer) | 176.5 ms | 42 ms | 176.5 ms | **5.7** |
| Partial resident (50% layers) | 88 ms | 42 ms | 130 ms | **7.7** |
| Baseline (all resident) | 0 | 42 ms | 42 ms | **24** |

### IO Method Decision

**Direct I/O (`FILE_FLAG_NO_BUFFERING`) is the preferred method** for dense
weight streaming. DirectStorage is better suited for MoE-style random small IO.

---

## 9. Next Steps (Updated after IO Benchmark)

1. ~~Run Phase 1 Step 1-3~~ ✅ Done — blob cache load ~1s per sub-model
2. ~~Decide Option B alone vs B+A~~ ✅ Decision: Option A required
3. ~~Implement DenseWeightStreamingManager class~~ ✅ Done (hpp + cpp)
4. ~~Create pack_dense_weights.py~~ ✅ Done (packs model→streaming.bin)
5. ~~Run IO benchmark~~ ✅ Done — Direct I/O 11.2 GB/s > DS 8.5 GB/s
6. ~~IO API decision~~ ✅ **Direct I/O** (not DirectStorage) — kernel read-ahead
   optimization beats BypassIO for sequential large reads
7. **NOW:** Update `DenseWeightStreamingManager` to use Direct I/O API
   - Replace DirectStorage with ReadFile + FILE_FLAG_NO_BUFFERING
   - Multi-threaded OVERLAPPED reads (4 threads per group)
   - Direct write to USM host buffer (zero-copy to GPU)
8. **NEXT:** Implement `swap_weight_pointers()` in GPU plugin
9. **NEXT:** Add layer-group boundary synchronization
10. **Parallel:** Verify actual memory usage on 8 GB system

---

## 10. Phase 1 Results (Measured 2026-05-08)

### Full Model Blob-Cache Load Time

| Model | BIN Size | Cached Compile | Unload | Throughput |
|---|---:|---:|---:|---:|
| language_model (full) | 2.62 GB | **2.27 s** | 53 ms | 1.15 GB/s |
| text_embeddings | 640 MB | 435 ms | 6 ms | 1.47 GB/s |
| text_embed_per_layer | 2.63 GB | 2.07 s | 8 ms | 1.27 GB/s |
| vision_embeddings | 162 MB | 203 ms | 19 ms | 0.80 GB/s |

### Split Model (2-way) Blob-Cache Load Time

| Sub-model | Layers | BIN Size | Cached Compile | Unload | Cycle |
|---|---|---:|---:|---:|---:|
| Part 0 | 0-20 (21 layers) | 1.02 GB | **1.03 s** | 29 ms | 1.06 s |
| Part 1 | 21-41 (21 layers) | 1.62 GB | **1.22 s** | 20 ms | 1.24 s |
| **Total cycle** | | | | | **2.30 s** |

### Key Findings

1. **Blob-cache compile_model throughput: ~1.0-1.5 GB/s effective**
   - Limited by USM allocation + memcpy, NOT by NVMe bandwidth
   - NVMe Gen5x4 raw: 12 GB/s, but compile_model achieves only 1.2 GB/s
   - The overhead is in memory management, not I/O

2. **Per-token streaming with compile_model: 0.43 tps**
   - 2 sub-models × ~1.1s load = 2.2s per token (just load overhead)
   - Add 42ms compute → effective **0.43 tps** — completely unacceptable

3. **Unload is fast: 20-53 ms**
   - Memory release is not the bottleneck
   - The bottleneck is purely on the LOAD side

4. **Sub-model sizes (2-way split):**
   - Part 0 (layers 0-20): 1.02 GB — includes all 42 KV cache sinks for layers 0-20
   - Part 1 (layers 21-41): 1.62 GB — includes 6 KV cache sinks for layers 21-23
   - Only 24 layers (0-23) have KV cache; layers 24-41 have none

### Decision: Option B Alone is NOT Viable

**Option B (model splitting + per-token compile_model) is 100% ruled out.**

The compile_model API has ~1s overhead per sub-model due to:
- Weight buffer allocation (USM host/device memory)
- Weight data copy (memcpy from blob→USM at ~1.2 GB/s)
- Graph state initialization

**Must proceed to Option A (DirectStorage weight buffer swap) which operates
at the raw USM buffer level, bypassing compile_model entirely.**

### Revised Approach: Direct USM Buffer Swap

Instead of compile_model per token, the viable approach is:
1. **Compile ONCE** with placeholder/minimal weights (get compiled graph cached)
2. **At runtime:** Swap weight USM buffers directly via DirectStorage DMA
3. **Target throughput:** NVMe 12 GB/s raw → 338 MB per 7-layer group in ~27.5 ms
4. **Key API:** Reuse MoE OTD's `get_arguments()` pattern to redirect weight pointers

This is fundamentally Option A from the original plan, confirmed by data.

### Updated Data from pack_dense_weights.py (Dry-Run, 2026-05-08)

| Group | Layers | Constants | Size | IO@12GB/s |
|---|---|---:|---:|---:|
| Group 0 | 0-6 | 231 | 341.6 MB | 27.8 ms |
| Group 1 | 7-13 | 231 | 341.6 MB | 27.8 ms |
| Group 2 | 14-20 | 231 | 341.6 MB | 27.8 ms |
| Group 3 | 21-27 | 201 | 336.2 MB | 27.4 ms |
| Group 4 | 28-34 | 168 | 330.8 MB | 26.9 ms |
| Group 5 | 35-41 | 168 | 336.2 MB | 27.4 ms |
| **Total** | 42 layers | **1224** | **1.981 GB** | **165 ms** |

**Performance estimates (6 groups × 7 layers):**
- Non-overlapped (serial IO+compute): TPOT=207 ms → **4.8 tps**
- Double-buffer (IO overlaps GPU): TPOT=172 ms → **5.8 tps** ✅ Meets ≥5 tps target
- Baseline (all weights resident): TPOT=42 ms → **24 tps**

---

## 11. Step 7 — Pipeline Benchmark Results (H5+T5 Hybrid, Measured 2026-05-10)

### Test Setup

- **Binary:** `dense_weights_streaming.bin` (1.515 GB, 32 groups × 1 layer)
- **Strategy:** H5+T5 — Pin layers 0-4 (239.2 MB) + layers 37-41 (237.9 MB)
- **Streamed:** Layers 5-36 (32 groups, each 46.5-54.6 MB)
- **IO:** Direct I/O (`FILE_FLAG_NO_BUFFERING`), 4 threads
- **GPU sim:** 0.99 ms per layer (spin-wait, from baseline 41.7 ms / 42 layers)
- **Benchmark:** `cpp/benchmark_pipeline.cpp` (5 iterations, median)

### IO Baseline

| Metric | Value |
|---|---:|
| Sequential IO (all 32 groups) | **136.9 ms** |
| IO throughput | **11.07 GB/s** |
| Per-group IO (typical) | 3.9-5.1 ms |
| Pipeline IO overhead | 2.1% (vs sequential) |

### Pipeline Results (Simulated GPU)

| Metric | Value |
|---|---:|
| **Total TPOT** | **145.3 ms** |
| **Estimated TPS** | **6.88 tps** |
| vs baseline 24 tps | **28.7% throughput** |
| Streaming penalty | **3.5× slowdown** |
| Bottleneck | **IO-bound** |

### Pipeline Breakdown (Last Iteration)

| Phase | Time |
|---|---:|
| Pinned HEAD GPU (5 layers) | 1.0 ms |
| First load (cold, group 0) | 5.0 ms |
| Total IO wait (32 groups) | 107.6 ms |
| Total streamed GPU (32 groups) | 31.8 ms |
| Swap overhead | 0.00 ms |
| GPU fence overhead | 0.00 ms |
| Pinned TAIL GPU (5 layers) | 1.0 ms |
| **IO/GPU overlap ratio** | **81%** |

### IO/GPU Overlap Analysis

| Metric | Value |
|---|---:|
| IO-only pipeline | 138.5 ms |
| GPU compute total | 31.8 ms |
| Expected serial (no overlap) | 170.3 ms |
| Actual with GPU | 144.7 ms |
| Time saved by overlap | **25.6 ms** |
| **Overlap ratio** | **81%** |

### IO Thread Scaling

| Threads | IO Only (ms) | IO GB/s | Pipeline (ms) | Est. TPS |
|---:|---:|---:|---:|---:|
| 1 | 142.1 | 10.66 | 152.1 | 6.58 |
| 2 | 150.2 | 10.08 | 152.2 | 6.57 |
| **4** | **155.4** | **9.75** | **146.2** | **6.84** |
| 8 | 173.1 | 8.75 | 154.5 | 6.47 |

> 4 threads optimal for pipeline (best overlap despite slightly lower raw IO).

### 8 GB System Memory Prediction

| Component | Memory |
|---|---:|
| Non-decoder weights (embed+vision+lm_head) | 1.48 GB |
| Pinned decoder (10 layers, H5+T5) | 0.47 GB |
| Streaming double-buffer (2 × 55 MB) | 0.11 GB |
| KV cache + runtime | 1.20 GB |
| **Predicted RSS** | **3.26 GB** |
| **Fits in 8 GB** | **✅ YES (large margin)** |

### Comparison: H5+T5 vs Original 6×7 Layout

| Metric | 6×7 groups (old) | H5+T5 32×1 (new) | Change |
|---|---:|---:|---|
| Streamed weight | 1.981 GB | 1.515 GB | -24% |
| IO per token | 176.5 ms | 136.9 ms | -22% |
| Buffer memory | 684 MB (2×342) | 110 MB (2×55) | -84% |
| Pipeline TPOT | ~182 ms (est) | 145.3 ms | -20% |
| TPS | ~5.5 (est) | **6.88** | +25% |
| IO/GPU overlap | ~23% | **81%** | +252% |

> H5+T5 with 1-layer groups achieves much better IO/GPU overlap because
> each group's GPU compute (1 ms) is small enough to fully hide behind the
> next group's IO (3.9-5.1 ms). The finer granularity enables continuous
> double-buffering without GPU stalls.

### Key Findings

1. **6.88 tps exceeds the ≥5 tps target** with significant margin
2. **81% IO/GPU overlap** — double-buffer pipeline is very effective with 1-layer groups
3. **3.26 GB predicted RSS** — leaves ~4.7 GB free on 8 GB system (room for longer KV cache)
4. **IO is the bottleneck** — 136.9 ms IO vs 31.8 ms GPU (4.3:1 ratio)
5. **Further optimization:** Larger group sizes could reduce IO overhead but at the cost of overlap ratio; current 1-layer groups are near-optimal for this IO/GPU ratio

**Per-layer structure:**
- Layers 0-23: 33 constants each (~47.8 MB, 54.6 MB for global-attention layers 5/11/17/23)
- Layers 24-41: 24 constants each (~46.5 MB, 51.9 MB for layers 29/35/41)
- Difference: layers 0-23 include KV cache sink-related constants
