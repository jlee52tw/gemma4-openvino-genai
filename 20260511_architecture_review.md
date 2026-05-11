# Dense Weight Streaming — Architecture Review
**Date:** 2026-05-11  
**Author:** jlee52tw  
**Status:** Phase 1 complete (streamed execution path validated), Phase 2 pending (actual weight swap)

---

## 1. Problem Statement

**Goal:** Run Gemma-4-E4B-it (INT4, 6.05 GB) on an **8 GB system** with Intel iGPU.

**Challenge:** The standard OpenVINO GPU plugin loads ALL model weights into USM (Unified Shared Memory) at compile time. On an 8 GB system, the memory budget is too tight to hold the full model + KV cache + runtime overhead simultaneously.

**Solution:** Keep head/tail decoder layers pinned in memory, stream the middle 32 layers from NVMe SSD per-token via Direct I/O double-buffered pipeline. The NVMe Gen5×4 provides ~12 GB/s sequential read bandwidth, and iGPU USM shares the same LPDDR5 physical memory — enabling zero-copy NVMe→GPU data path.

---

## 2. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        User Application                         │
│   pipe = VLMPipeline(model_dir, 'GPU')                         │
│   out = pipe.generate("What is 2+2?", config)                  │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    OpenVINO GenAI Runtime                        │
│   VLMPipeline: tokenize → vision encoder → language model →     │
│                 detokenize                                       │
│   Language model uses PagedAttention (dynamic shapes)           │
└──────────────┬───────────────────────────────────────────────────┘
               │ compile_model() + infer_request.infer()
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                    OpenVINO GPU Plugin                           │
│   network::execute_impl()                                       │
│     ├─ try_init_dense_streaming() [lazy, once]                  │
│     ├─ if has_dense_weight_streaming():                         │
│     │    └─ execute_impl_streamed()  ← OUR CODE                │
│     └─ else: normal execution path                              │
└──────────────┬───────────────────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────────────────┐
│              DenseWeightStreamingManager                         │
│   - Reads dense_weights_streaming.bin (Direct I/O, 4 threads)  │
│   - USM double-buffer (2 × 55 MB)                              │
│   - Per-primitive group annotation                              │
│   - Group transition detection → weight swap trigger            │
└──────────────┬───────────────────────────────────────────────────┘
               │ NVMe → LPDDR5 (USM) → iGPU
               ▼
┌──────────────────────────────────────────────────────────────────┐
│                   Intel iGPU (12 Xe EUs)                        │
│   Executes GPU kernels reading weights from USM buffers         │
│   Same physical LPDDR5 memory — zero-copy from NVMe DMA        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. End-to-End Sequence Flow

### 3.1. Model Loading (one-time)

```
User: pipe = VLMPipeline(model_dir, 'GPU')
  │
  ▼
GenAI: load openvino_model.xml + .bin
  │
  ▼
GPU Plugin: compile_model()
  ├── Parse IR → build GPU kernel graph
  ├── Allocate USM memory for ALL weight tensors
  ├── Copy weights from .bin → USM buffers
  ├── Build execution order (_exec_order: topological sort)
  └── Cache as .blob (subsequent loads skip compilation)
  │
  Result: 3 compiled networks
    Network 1: Vision encoder       (19 primitives)
    Network 2: Small sub-network    (4 primitives)
    Network 3: Language model       (2549 primitives) ← decoder
```

### 3.2. First Token Generation (lazy streaming init)

```
User: pipe.generate("What is 2+2?", config)
  │
  ▼
GenAI: tokenize → prepare input_ids
  │
  ▼
GPU Plugin: network::execute_impl(events)
  │
  ├── [ONCE] try_init_dense_streaming()
  │     ├── Check OV_DENSE_STREAM_WEIGHTS env var
  │     ├── Skip small networks (< 200 primitives)
  │     ├── Read dense_weights_streaming.bin header
  │     │     → 32 groups × 1 layer, H5+T5 (layers 5-36)
  │     ├── Allocate USM double-buffer (2 × 55 MB)
  │     ├── Open 4 Direct I/O file handles
  │     ├── build_weight_mapping()
  │     │     → Scan 3787 primitives, find 448 streamed constants
  │     │     → Map each constant to packed binary offset
  │     └── build_group_exec_order()
  │           → Per-primitive group annotation (parallel to _exec_order)
  │           → Pre(-3) / HEAD(-2) / Group(0-31) / TAIL(-1) / Post(32)
  │
  ├── has_dense_weight_streaming() → true
  │
  └── execute_impl_streamed(events)
        ├── set_arguments()       [same as normal path]
        ├── for each prim in _exec_order:
        │     ├── group = m_exec_group_assignment[idx]
        │     ├── if entering new group:
        │     │     [Phase 2: load_group → swap_weight_pointers]
        │     ├── prim->reset_events()
        │     ├── prim->prepare_primitive()
        │     ├── prim->execute()
        │     └── flush every 16 prims (dynamic model)
        └── final flush + reset_flags
```

### 3.3. Subsequent Tokens (decode loop)

```
For each output token:
  │
  GPU Plugin: network::execute_impl(events)
  │
  ├── m_dense_streaming_checked == true (skip init)
  ├── has_dense_weight_streaming() → true
  │
  └── execute_impl_streamed(events)
        ├── set_arguments()
        └── Linear iteration over _exec_order (2549 prims):
              │
              │  [Pre-decoder: 6 prims]
              │    embeddings, position encoding
              │
              │  [Pinned HEAD: 555 prims — layers 0-4]
              │    ✅ Weights in original USM (no streaming)
              │    Full-attention layer 0 + sliding-window layers 1-4
              │
              │  [Group 0: 70 prims — layer 5]
              │    → Phase 2: load_group(0), swap_weight_pointers(0)
              │    self_attn (Q/K/V/O proj) + MLP (gate/up/down proj)
              │
              │  [Group 1-31: 40-66 prims each — layers 6-36]
              │    → Phase 2: detect transition, prefetch(g+1),
              │               wait_for_load(g), swap_weight_pointers(g)
              │               Overlap: GPU(g) || NVMe→USM(g+1)
              │
              │  [Pinned TAIL: 204 prims — layers 37-41]
              │    ✅ Weights in original USM (no streaming)
              │
              │  [Post-decoder: 10 prims]
              │    final norm → lm_head → logits
              │
              ▼
        Output: next token logits → sampling → token ID
```

### 3.4. Final Output

```
GenAI decode loop:
  Repeat section 3.3 for each token until:
    - max_new_tokens reached, OR
    - EOS token generated
  │
  ▼
Detokenize token IDs → output string
  │
  ▼
User: "Tokyo"
```

---

## 4. Key Component: Execution Order & Group Annotation

### 4.1. Why Not Group-Based Execution?

**Discovery (this session):** The GPU compiler's `_exec_order` is a **topological sort**, not a layer-sequential order. The compiler interleaves operations from different layers for optimal memory reuse and kernel scheduling.

Example from actual _exec_order:
```
idx 7:  layers.0.input_layernorm/ReduceMean      ← Layer 0
idx 8:  layers.0.input_layernorm/Multiply         ← Layer 0
idx 9:  layers.0.self_attn.q_proj/MatMul          ← Layer 0
...
idx 223: layers.5.self_attn/Unsqueeze_1           ← Layer 5 (INTERLEAVED!)
idx 224: layers.5.self_attn/Unsqueeze             ← Layer 5
idx 225: ScatterUpdate_76822                       ← No layer (shared op)
```

**Problem:** Our original design extracted primitives into per-group lists and concatenated them. This produced 2532 order mismatches out of 2549 primitives — causing crashes because primitives depended on outputs from primitives now in different groups.

### 4.2. Solution: Linear + Annotation

Instead of reordering, we:
1. **Keep `_exec_order` intact** — iterate in the GPU compiler's topological order
2. **Annotate each primitive** with a group ID via `m_exec_group_assignment[]`
3. **Detect group transitions** on-the-fly during execution

```
m_exec_group_assignment[]:
  [0..5]     = -3 (PRE_DECODER)
  [6..560]   = -2 (PINNED_HEAD, layers 0-4 + shared ops)
  [561..630] =  0 (Group 0 = layer 5 + shared ops)
  [631..696] =  1 (Group 1 = layer 6 + shared ops)
  ...
  [2338..2538] = -1 (PINNED_TAIL, layers 37-41 + shared ops)
  [2539..2548] = 32 (POST_DECODER)
```

Non-layer primitives (e.g., `Convert_73477`, `rotary_emb/Unsqueeze`) **inherit** the group of the most recently seen decoder layer primitive. This ensures proper data dependencies are maintained.

---

## 5. Key Changes Summary

### 5.1. Files in This Repository (workspace)

| File | Purpose | Key Details |
|---|---|---|
| `pack_dense_weights.py` | Pack decoder weights → `.bin` | H5+T5 hybrid, sector-aligned, layer table uses packed index (`packed_idx = layer_idx - first_streamed`) |
| `cpp/dense_weight_streaming_manager.hpp` | Manager class header | `DenseWeightStreamingManager`, file format structs, `GroupComputeCallback`, exec slot constants, `m_exec_group_assignment` |
| `cpp/dense_weight_streaming_manager.cpp` | Manager implementation | Direct I/O (4 threads), USM double-buffer, `build_weight_mapping()` with try-catch for `_optimized_` IDs, `build_group_exec_order()` 2-pass algorithm with current-layer tracking, `swap_weight_pointers()`, `execute_streamed_decode()` pipeline |

### 5.2. Patches to OpenVINO GPU Plugin

| File | Change | Lines Added |
|---|---|---:|
| `network.hpp` | Declare `m_dense_streaming`, `try_init_dense_streaming()`, `execute_impl_streamed()`, `has_dense_weight_streaming()` | ~40 |
| `network.cpp` | Lazy init in `execute_impl()`, streamed execution path with linear + group annotation, `try_init_dense_streaming()` reads env var + initializes manager | ~120 |
| `primitive_inst.h` | Add `force_set_output_memory()` method for weight pointer swapping | ~16 |
| `CMakeLists.txt` | `OV_DENSE_WEIGHT_STREAMING_ENABLED` define, link `winmm.lib`, add source files | ~7 |

### 5.3. Explanation of Each Change

#### `try_init_dense_streaming()` — Lazy Initialization
- **Why lazy?** VLMPipeline creates the network at `compile_model()` time, but we need the compiled network's `_exec_order` to build group assignments. The first call to `execute_impl()` triggers initialization.
- **Why env var?** `OV_DENSE_STREAM_WEIGHTS` lets users opt-in without modifying OpenVINO config APIs. Clean fallback: if env var is not set, zero overhead.
- **Network size guard:** Skip networks with < 200 primitives (vision encoder has 19, language model has 2549). Prevents unnecessary USM allocation on non-decoder networks.
- **try-catch around `get_primitive()`:** Some IDs (e.g., `_optimized_`) exist in `get_all_primitive_ids()` but throw when accessed — a VLMPipeline/PagedAttention internal detail.

#### `execute_impl_streamed()` — Group-Annotated Linear Execution
- **Identical to normal path:** Same `set_arguments()`, same per-primitive `reset_events()` + `prepare_primitive()` + `execute()`, same dynamic flushing every 16 prims. This is why Phase 1 has **zero performance overhead**.
- **Group transition detection:** `m_exec_group_assignment[idx]` tracks which group each primitive belongs to. When a new group is entered (`group >= 0 && group != last_streamed_group`), this is where Phase 2 will insert weight swap logic.

#### `build_weight_mapping()` — Network ↔ Binary Cross-Reference
- Scans all primitives in the compiled network via `get_all_primitive_ids()`
- Identifies 1238 total constants → 588 decoder constants → 448 in streamed range (layers 5-36)
- Maps each constant to its offset in the packed binary file
- Uses `parse_layer_index_from_name()` to extract layer index from primitive ID strings (e.g., `__module.model.language_model.layers.12.mlp.gate_proj.weight` → layer 12)

#### `build_group_exec_order()` — 2-Pass Group Assignment
- **Pass 1:** Scan `_exec_order` to find first/last decoder primitive positions
- **Pass 2:** Iterate `_exec_order`, tracking `current_layer`. Decoder primitives update `current_layer`; non-decoder primitives inherit it. Map each primitive to a group slot.
- Builds both `m_group_exec_order[]` (per-group primitive lists) and `m_exec_group_assignment[]` (per-primitive annotation)

#### `force_set_output_memory()` — Weight Pointer Swapping
- Added to `primitive_inst` class to allow replacing a data primitive's output memory at runtime
- In Phase 2: `swap_weight_pointers()` will call this for each weight constant in a group, pointing it to the correct offset within the streaming USM buffer
- Zero-copy: the USM buffer is the same physical LPDDR5 memory the GPU reads from

#### `pack_dense_weights.py` — Binary Packing Tool
- Reads the original OpenVINO model's `.bin` file
- Extracts constants for layers 5-36 (32 middle layers)
- Packs them into sector-aligned groups (1 layer per group)
- Header + group table + per-layer table + data = 1.515 GB total
- Layer table uses **packed index** (0-31), not model index (5-36) — critical for C++ reader alignment

---

## 6. Performance Analysis

### 6.1. Why Phase 1 Has Zero Overhead

| Aspect | Normal Path | Streamed Path (Phase 1) |
|---|---|---|
| Loop structure | `for (inst : _exec_order)` | `for (inst : _exec_order)` — **identical** |
| `set_arguments()` | Called once | Called once |
| Per-primitive work | `reset_events → prepare → execute` | `reset_events → prepare → execute` |
| Dynamic flushing | Every 16 prims | Every 16 prims |
| Extra work | None | Group transition check (`int32_t` comparison) |

The only difference is a single `int32_t` comparison per primitive — negligible at ~2549 iterations vs 41.7ms per token.

### 6.2. Measured Performance

| Metric | Normal Path | Streamed Path (Phase 1) |
|---|---:|---:|
| TPS (50 tokens) | 23.4 | 23.4 |
| Time (50 tokens) | 2.14s | 2.13s |
| Per-token latency | ~42.8ms | ~42.6ms |
| **Overhead** | — | **0%** |

### 6.3. Phase 2 Expected Performance

Phase 2 will add per-group weight loading from NVMe. Expected overhead per token:

| Operation | Time | Notes |
|---|---:|---|
| GPU compute per layer | ~1.0 ms | Baseline measurement |
| NVMe read per layer (~48 MB) | ~4.0 ms | 12 GB/s ÷ 48 MB |
| **Overlap efficiency** | | GPU(g) ‖ NVMe(g+1) |
| GPU(g): 1.0 ms | | Computing current group |
| NVMe(g+1): 4.0 ms | | Loading next group |
| **Per-group cost:** | **~4.0 ms** | Bottleneck: NVMe IO |
| **32 groups total:** | **~128 ms** | Streamed middle layers |
| Pinned HEAD (5 layers) | ~5.0 ms | No IO, full GPU speed |
| Pinned TAIL (5 layers) | ~5.0 ms | No IO, full GPU speed |
| Pre/post decoder | ~1.0 ms | Embeddings, LM head |
| **Total per token:** | **~139 ms** | |
| **Expected TPS:** | **~7.2 tps** | vs 24.0 tps baseline |

Memory savings: streamed layers' USM can be deallocated after compile, saving ~1.5 GB.

---

## 7. Streaming Data Flow (Phase 2 — Future)

```
Per-Token Decode: Double-Buffer Pipeline
═══════════════════════════════════════════

Time ──────────────────────────────────────────────────────►

HEAD layers (pinned, ~5ms):
  GPU: ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Group 0 (cold start):
  IO:  ████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Load G0→Buf[0]
  GPU: ░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  Execute G0

Group 1 (pipelined):
  IO:  ░░░░░░░░░░░░░░░░████████████████░░░░░░░░░░░░░░░░░░░  Prefetch G1→Buf[1]
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████░░░░░░░░░░░░░░  Execute G1

Group 2:
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████████░░░  Load G2→Buf[0]
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████  Execute G2

  ... repeat for 32 groups ...

TAIL layers (pinned, ~5ms):
  GPU: ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░████████████
  IO:  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░

Legend: ████ = active, ░░░░ = idle
Buffer ping-pong: Buf[0] ↔ Buf[1] alternates each group
```

### Zero-Copy Data Path (iGPU advantage)

```
NVMe SSD ──(DMA)──► LPDDR5 Physical Memory ◄──(GPU Read)── Intel iGPU
                     │                                        │
                     │         USM (Unified Shared Memory)    │
                     │         Same physical address space    │
                     │                                        │
                     ├── ReadFile(FILE_FLAG_NO_BUFFERING)     │
                     │   Direct I/O → No OS page cache copy   │
                     │                                        │
                     └── GPU kernels read directly from       │
                         the same physical pages ─────────────┘
```

Unlike discrete GPUs that need PCIe transfers (CPU→VRAM), Intel iGPU shares the same LPDDR5 physical memory. This means:
1. NVMe DMA writes data to LPDDR5
2. GPU reads from the exact same physical pages
3. **Zero extra copy** — NVMe→GPU is truly one-hop

---

## 8. File & Build Structure

### 8.1. Repository Files

```
gemma4-openvino-genai/
├── pack_dense_weights.py           # Weight packing tool
├── cpp/
│   ├── dense_weight_streaming_manager.hpp  # Manager header
│   ├── dense_weight_streaming_manager.cpp  # Manager implementation (1800+ lines)
│   └── ...
├── temp/
│   ├── dense_weights_streaming.bin # Packed weights (1.515 GB)
│   ├── dense_weights_streaming.json # Tensor metadata
│   ├── pip_original/              # Backup of original DLLs
│   └── test_streaming*.log        # Test logs
└── deploy_streaming_to_openvino.ps1  # Deployment script
```

### 8.2. OpenVINO GPU Plugin Patches (4 files)

```
openvino/src/plugins/intel_gpu/
├── include/intel_gpu/graph/
│   └── network.hpp                    # +40 lines (declarations)
└── src/graph/
    ├── network.cpp                    # +120 lines (exec path + init)
    ├── include/
    │   ├── primitive_inst.h           # +16 lines (force_set_output_memory)
    │   └── dense_weight_streaming_manager.hpp  # deployed from workspace
    ├── dense_weight_streaming_manager.cpp      # deployed from workspace
    └── CMakeLists.txt                 # +7 lines (define + link + sources)
```

### 8.3. Build & Deploy Workflow

```bash
# 1. Build from pip's exact commit (detached HEAD at 9c4a2eb9ad3)
cd openvino
git checkout 9c4a2eb9ad3
git submodule update --init --recursive

# 2. Configure (GPU only, matching pip version)
set CI_BUILD_NUMBER=2026.2.0-21571-9c4a2eb9ad3
cmake -B build -G "Visual Studio 17 2022" ^
  -DENABLE_INTEL_GPU=ON -DENABLE_INTEL_CPU=OFF -DBUILD_SHARED_LIBS=ON

# 3. Build only the GPU plugin
cmake --build build --target openvino_intel_gpu_plugin --config Release

# 4. Deploy: swap ONLY the GPU plugin DLL (keep pip's openvino.dll)
copy bin\intel64\Release\openvino_intel_gpu_plugin.dll ^
     %PYTHON%\Lib\site-packages\openvino\libs\

# 5. Run with streaming
set OV_DENSE_STREAM_WEIGHTS=temp\dense_weights_streaming.bin
python run_gemma4.py
```

---

## 9. Network Primitive Statistics

```
VLMPipeline creates 3 GPU networks:
  Network 1:    19 primitives → vision encoder    (SKIPPED)
  Network 2:     4 primitives → small sub-network (SKIPPED)
  Network 3: 2,549 primitives → language decoder  (STREAMING)

Language decoder breakdown:
  Total primitives:    2,549
  Total constants:     1,238
  Decoder constants:     588  (in layers.0-41)
  Streamed constants:    448  (in layers 5-36, 14 per layer)
  
  Group assignment:
    Pre-decoder:           6  primitives
    Pinned HEAD (L0-4):  555  primitives (layers 0-4 + shared ops)
    Group 0  (L5):        70  primitives
    Group 1  (L6):        66  primitives
    ...
    Group 17 (L22):       64  primitives
    Group 18 (L23):       64  primitives
    Group 19 (L24):       40  primitives  ← fewer (no KV cache ops)
    ...
    Group 31 (L36):       40  primitives
    Pinned TAIL (L37-41):204  primitives (layers 37-41 + shared ops)
    Post-decoder:         10  primitives

  NOTE: Groups 0-18 (layers 5-23) have ~60-70 prims because these layers
  have KV cache (sliding window attention). Groups 19-31 (layers 24-36)
  have ~40 prims because they use shared KV cache from full-attention layers.
```

---

## 10. Phase 2 — Next Steps

### 10.1. Enable Weight Swapping

At each group transition detected in `execute_impl_streamed()`:

```cpp
if (group >= 0 && group != last_streamed_group) {
    // NEW in Phase 2:
    if (last_streamed_group >= 0) {
        get_stream().flush();       // GPU fence: previous group done
    }
    wait_for_load(group);           // IO fence: NVMe→USM complete
    swap_weight_pointers(group, this);  // Redirect data primitives
    if (group + 1 < num_groups) {
        prefetch_next_group(group + 1); // Async: start loading next
    }
    last_streamed_group = group;
}
```

### 10.2. Validate Weight Data Correctness

Before swapping, verify that the packed binary data byte-for-byte matches the original model weights:
1. For each weight tensor in a group, compare `USM_original[tensor]` vs `packed_binary[offset]`
2. If mismatch → data alignment or packing order bug
3. Use `verify_weights.py` (already exists) for Python-side verification

### 10.3. Memory Deallocation

After streaming init succeeds, **deallocate the original USM memory** for streamed layers (5-36):
- This is the core memory savings (~1.5 GB freed)
- Pinned layers (0-4, 37-41) keep their original USM memory
- All 32 middle groups read from the two 55 MB streaming buffers

### 10.4. Performance Optimization

1. **Group size tuning:** Current 1-layer/group means 32 group transitions per token. Larger groups (e.g., 4 layers) reduce IO round trips but increase buffer size.
2. **Prefetch depth:** Current design prefetches 1 group ahead. Consider 2-group prefetch.
3. **IO/GPU overlap measurement:** Instrument group transitions to measure actual overlap efficiency.

### 10.5. 8 GB System Validation

- [ ] Boot with 8 GB memory configuration
- [ ] Measure Peak RSS with streaming enabled
- [ ] Verify no OOM during inference
- [ ] Measure actual TPS on memory-constrained system

---

## 11. Risk Assessment

| Risk | Impact | Mitigation |
|---|---|---|
| Weight data format mismatch | Corrupt output | `verify_weights.py` byte-level comparison |
| Topological sort dependency across groups | Crash / wrong output | Linear execution preserves compiler order |
| USM deallocation of in-use memory | Crash | Fence GPU before dealloc, verify no references |
| NVMe thermal throttling (sustained load) | TPS drop | Monitor SSD temperature, add cooldown |
| Dynamic shape recompilation invalidates group assignment | Wrong mapping | Re-run `build_group_exec_order()` on shape change |

---

## 12. Summary

**What we built (Phase 1):**
- A research prototype that hooks into OpenVINO's GPU plugin execution path
- Validates streaming infrastructure: file format, weight mapping, group annotation, execution routing
- **Zero performance overhead** — the streamed path is functionally identical to the normal path
- Confirmed: 33 group transitions detected per token, 2549 primitives executed correctly

**What Phase 2 will add:**
- Actual NVMe→USM weight loading at group transitions
- weight pointer swapping via `force_set_output_memory()`
- Memory deallocation of streamed layers (~1.5 GB savings)
- Double-buffered IO/GPU overlap pipeline

**Expected outcome:**
- 8 GB system: inference becomes possible (currently OOM)
- 16 GB system: ~7 tps (from 24 tps) with 1.5 GB memory saved
- Trade-off: **3.3× speed reduction for 1.5 GB memory savings**
