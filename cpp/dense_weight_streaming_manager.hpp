// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Dense Weight Streaming Manager
// ================================
// Adapted from MoE OTD's moe_expert_weight_manager.hpp for dense (non-MoE)
// models. Provides Direct I/O (FILE_FLAG_NO_BUFFERING) based NVMe→USM
// double-buffer streaming for decoder layer weights, enabling inference
// on memory-constrained systems.
//
// IO Method: Win32 Async ReadFile with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED
//   - Bypasses OS page cache → no memory pressure
//   - Reads directly into USM host buffer (page-aligned → sector-aligned)
//   - True async I/O: ReadFile returns immediately, NVMe DMA runs in background
//   - Multiple overlapped reads across file handles for max NVMe queue depth
//   - Zero-copy: NVMe DMA → LPDDR5 (USM) → iGPU kernel read
//   - GPU compute and NVMe DMA run concurrently (both DMA, no CPU bottleneck)
//   - LPDDR5 8533MHz bandwidth (>100 GB/s) >> NVMe (~12 GB/s) + iGPU (~50 GB/s)
//
// Why Async ReadFile instead of multi-threaded sync ReadFile?
//   - No thread creation/destruction overhead per group transition
//   - Native kernel-level async → NVMe controller DMA without CPU involvement
//   - GPU and IO truly independent: no thread synchronization latency
//   - Event-based completion: WaitForMultipleObjects for zero-overhead fence
//
// Key differences from MoE OTD:
//   - Sequential access pattern (no LRU/slot management needed)
//   - Double-buffer ping-pong instead of LRU cache
//   - Loads entire layer groups (~350 MB) instead of individual experts (~12 MB)
//   - No routing/expert selection — all layers always execute
//
// Architecture:
//   1. Model compiled ONCE (all weights loaded into USM initially)
//   2. At runtime: free excess weight memory, allocate 2 streaming buffers
//   3. Per-token decode loop (async IO pipeline):
//      - Issue async ReadFile for group N → Buffer[active] (non-blocking)
//      - Wait for async IO completion
//      - Swap weight pointers + set kernel arguments
//      - Issue async ReadFile for group N+1 → Buffer[inactive] (non-blocking)
//      - Execute group N on GPU (concurrent with N+1's NVMe DMA)
//      - Repeat for all groups
//
// Data path (zero-copy on iGPU):
//   NVMe SSD →(DMA)→ LPDDR5 (USM host buffer) →(GPU read)→ iGPU compute
//              Async ReadFile(OVERLAPPED)         same physical memory
//
// Usage:
//   DenseWeightStreamingManager manager(engine, config);
//   manager.initialize("dense_weights_streaming.bin");
//
//   // Per-token decode:
//   for (int group = 0; group < manager.num_groups(); ++group) {
//       manager.load_group(group);      // Direct I/O NVMe→USM
//       manager.wait_for_load(group);   // Wait for async IO completion
//       // Execute layers... (GPU reads from manager.get_weight_buffer(group))
//       manager.prefetch_next_group(group + 1);  // Overlap with GPU
//   }

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Forward declarations (avoid including full OpenVINO headers)
// NOTE: struct/class must match actual type declarations for MSVC ABI
namespace cldnn {
class engine;
class stream;
struct memory;     // struct in intel_gpu/runtime/memory.hpp
using memory_ptr = std::shared_ptr<memory>;
}  // namespace cldnn

namespace ov::intel_gpu {

// ============================================================================
// File Format Structures
// ============================================================================

/// @brief Magic number for dense weight streaming binary file
static constexpr char DNSW_MAGIC[4] = {'D', 'N', 'S', 'W'};
static constexpr uint32_t DNSW_VERSION = 2;  // V2: dual-NVMe striping support
static constexpr size_t SECTOR_SIZE = 4096;

/// @brief File header (padded to 4096 bytes for sector alignment)
#pragma pack(push, 1)
struct DenseWeightsFileHeader {
    char magic[4];              // "DNSW"
    uint32_t version;           // 1
    uint32_t num_layers;        // 42
    uint32_t num_groups;        // e.g., 6 (for group_size=7)
    uint32_t group_size;        // layers per group (e.g., 7)
    uint32_t reserved_0;       // padding
    uint64_t total_weight_bytes;  // sum of all layer weights
    uint64_t total_file_size;     // total file size
    uint32_t sector_size;       // 4096
    // ... rest padded to SECTOR_SIZE
};

/// @brief Group table entry (32 bytes each)
struct GroupTableEntry {
    uint64_t file_offset;       // Absolute file offset of group data
    uint64_t raw_bytes;         // Actual weight data size
    uint64_t aligned_bytes;     // Sector-aligned size (for Direct I/O)
    uint32_t first_layer;       // First layer index in this group
    uint32_t num_layers;        // Number of layers in this group
};

/// @brief Per-layer table entry (24 bytes each)
struct LayerTableEntry {
    uint64_t offset_in_group;   // Byte offset within its group data
    uint64_t size_bytes;        // Total weight bytes for this layer
    uint32_t num_tensors;       // Number of weight tensors
    uint32_t layer_idx;         // Layer index (0-41)
};
#pragma pack(pop)

// ============================================================================
// Weight-to-Primitive Mapping (for swap_weight_pointers integration)
// ============================================================================

/// @brief Describes one weight tensor's location within the packed binary
/// and its corresponding primitive in the compiled cldnn::network.
///
/// Built at initialization by reading the JSON metadata file produced by
/// pack_dense_weights.py and cross-referencing with the compiled network's
/// primitive IDs.
struct WeightTensorMapping {
    uint32_t layer_idx;          // Packed layer index (0..num_layers-1)
    std::string tensor_name;     // OV constant name or JSON tensor name
    std::string primitive_id;    // primitive_id in compiled cldnn::network (data primitive)
    uint64_t offset_in_layer;   // Byte offset within the layer's packed data
    uint64_t size_bytes;         // Tensor size in bytes

    // FC weight cache integration (for dynamic FullyConnectedCompressed):
    // When is_fc_weight==true, the tensor is a weight dependency of a dynamic FC.
    // swap_weight_pointers() must replace the FC's _reordered_weights_cache entry
    // AND the dep data primitive's output memory (for update_weights correctness).
    bool is_fc_weight = false;          // true = swap via FC weight cache
    std::string fc_primitive_id;        // FC primitive's ID (for update_weights_cache)
};
// ============================================================================
// Callback & Pipeline Types (Step 6 — boundary synchronization)
// ============================================================================

/// @brief Callback for GPU compute of one layer group
/// @param group_idx Group index (0 to num_groups-1)
/// @param first_layer First decoder layer index in this group
/// @param num_layers Number of layers in this group
/// @return true if compute succeeded
///
/// In real integration: iterates the group's primitives in _exec_order
/// and calls inst->execute() for each.
/// In standalone testing: can be a no-op, a sleep(), or a simulation.
using GroupComputeCallback = std::function<bool(uint32_t group_idx,
                                                uint32_t first_layer,
                                                uint32_t num_layers)>;

/// @brief Per-token pipeline timing breakdown
///
/// Captures detailed timing for each phase of the streamed decode pipeline:
///   load → swap → GPU compute → GPU fence → prefetch overlap → repeat
///
/// Used to diagnose bottlenecks:
///   - IO-bound: io_wait_ms >> gpu_ms (need faster NVMe or smaller groups)
///   - GPU-bound: gpu_ms >> io_wait_ms (ideal — IO fully hidden)
///   - Swap overhead: swap_ms too high (too many weight tensors per group)
struct TokenPipelineStats {
    // Aggregate timing
    double total_token_ms = 0.0;       // Wall time for full decode (all groups)
    double first_load_ms = 0.0;        // Cold start: group 0 synchronous IO
    double total_io_wait_ms = 0.0;     // Sum of all IO fence waits
    double total_gpu_ms = 0.0;         // Sum of all GPU compute times
    double total_swap_ms = 0.0;        // Sum of all pointer swap times
    double total_gpu_fence_ms = 0.0;   // Sum of all GPU fence waits
    double pipeline_efficiency = 0.0;  // 1.0 = perfect overlap, 0.0 = fully serial
    
    // Hybrid pinning timing (pinned layers run at full GPU speed, no IO)
    double pinned_head_gpu_ms = 0.0;   // GPU time for pinned head layers
    double pinned_tail_gpu_ms = 0.0;   // GPU time for pinned tail layers

    // Per-group breakdown
    struct GroupTiming {
        double io_wait_ms = 0.0;   // Time waiting for IO fence
        double swap_ms = 0.0;      // swap_weight_pointers() + set_arguments()
        double gpu_ms = 0.0;       // GPU compute callback duration
        double gpu_fence_ms = 0.0; // stream.finish() after compute
    };
    std::vector<GroupTiming> groups;

    void reset(uint32_t num_groups) {
        total_token_ms = first_load_ms = total_io_wait_ms = 0.0;
        total_gpu_ms = total_swap_ms = total_gpu_fence_ms = 0.0;
        pipeline_efficiency = 0.0;
        pinned_head_gpu_ms = pinned_tail_gpu_ms = 0.0;
        groups.clear();
        groups.resize(num_groups);
    }

    /// @brief Compute aggregate stats from per-group data
    void aggregate() {
        total_io_wait_ms = total_gpu_ms = total_swap_ms = total_gpu_fence_ms = 0.0;
        for (const auto& g : groups) {
            total_io_wait_ms += g.io_wait_ms;
            total_gpu_ms += g.gpu_ms;
            total_swap_ms += g.swap_ms;
            total_gpu_fence_ms += g.gpu_fence_ms;
        }
        // Pipeline efficiency: ratio of (sum of parts) to (wall time)
        // Perfect overlap → sum > wall → efficiency approaches 1.0
        double serial_sum = total_io_wait_ms + total_gpu_ms + total_swap_ms + total_gpu_fence_ms;
        if (total_token_ms > 0 && serial_sum > 0) {
            pipeline_efficiency = 1.0 - (total_token_ms / serial_sum);
            if (pipeline_efficiency < 0) pipeline_efficiency = 0.0;
        }
    }

    void print() const;
};

// ============================================================================
// Configuration
// ============================================================================

/// @brief Configuration for dense weight streaming
struct DenseStreamingConfig {
    /// Path to dense_weights_streaming.bin
    std::string weights_file_path;
    
    /// Number of layer groups (default: determined from file header)
    uint32_t num_groups = 0;
    
    /// Number of IO handles for concurrent async reads (1-8, default: 4)
    /// Each handle issues its own async ReadFile for a portion of the data.
    /// More handles = more NVMe queue depth = higher throughput.
    uint32_t num_io_threads = 4;
    
    /// Whether to lock USM buffers in physical memory (VirtualLock)
    bool lock_memory = false;
    
    /// Enable timing statistics
    bool enable_timing = false;
    
    /// Enable debug logging
    bool debug_logging = false;
    
    /// Path for debug log file (empty = stderr). Set via OV_DENSE_STREAM_LOG_FILE
    std::string debug_log_path;
    
    /// Maximum memory budget for streaming buffers (bytes, 0 = auto)
    uint64_t max_buffer_bytes = 0;
    
    /// Number of streaming buffers (2 = double-buffer, 3 = triple-buffer for
    /// multi-group prefetch pipeline). Set via OV_DENSE_STREAM_NUM_BUFFERS.
    uint32_t num_buffers = 2;
    
    // ====================================================================
    // Hybrid Pinning Strategy (H5+T5 default)
    // ====================================================================
    // Pinned layers stay in original USM memory from compile_model() and
    // run at full GPU speed (~0.99 ms/layer). Only the middle (streamed)
    // layers are loaded from NVMe per token.
    //
    // Default H5+T5: pin layers 0-4 and 37-41, stream layers 5-36.
    //   Pinned decoder:  480 MB (10 layers)
    //   Streamed:       1548 MB (32 layers)
    //   Buffer (2×55):   110 MB
    //   Est TPOT:       ~144 ms → 7.0 tps
    
    /// Number of head decoder layers to keep permanently pinned (default: 5)
    uint32_t pin_head_layers = 5;
    
    /// Number of tail decoder layers to keep permanently pinned (default: 5)
    uint32_t pin_tail_layers = 5;
    
    /// Total decoder layers in the model (default: 42 for Gemma4)
    uint32_t total_decoder_layers = 42;
    
    /// Disable IO/GPU overlap (prefetch). When true, each group is loaded
    /// synchronously before GPU compute — no pipelining. Set via
    /// OV_DENSE_STREAM_NO_PREFETCH=1. Useful for v1-vs-v2 comparison.
    bool no_prefetch = false;

    // ====================================================================
    // Dual-NVMe Striping (2× bandwidth)
    // ====================================================================
    // When weights_file_path_2 is set, the manager reads from 2 NVMe files
    // simultaneously (group-half striping). Each group's data is split:
    //   - File 0 (weights_file_path): first half of each group
    //   - File 1 (weights_file_path_2): second half of each group
    // IO handles are split evenly: handles[0..N/2-1] → file 0,
    //   handles[N/2..N-1] → file 1.
    // Combined bandwidth: ~24 GB/s (2× single NVMe).
    //
    // Set via: OV_DENSE_STREAM_WEIGHTS_2=D:\path\to\stripe_1.bin

    /// Path to second NVMe stripe file (empty = single-NVMe mode)
    std::string weights_file_path_2;

    /// @brief Check if dual-NVMe mode is active
    bool is_dual_nvme() const { return !weights_file_path_2.empty(); }
    
    /// @brief First streamed layer index: pin_head_layers
    uint32_t first_streamed_layer() const { return pin_head_layers; }
    
    /// @brief Last streamed layer index (inclusive): total - pin_tail - 1
    uint32_t last_streamed_layer() const {
        return total_decoder_layers - pin_tail_layers - 1;
    }
    
    /// @brief Number of streamed (non-pinned) layers
    uint32_t num_streamed_layers() const {
        return total_decoder_layers - pin_head_layers - pin_tail_layers;
    }
    
    /// @brief Check if a layer is pinned (not streamed)
    bool is_pinned(uint32_t layer_idx) const {
        return layer_idx < pin_head_layers ||
               layer_idx >= (total_decoder_layers - pin_tail_layers);
    }
    
    /// Read from environment variables (OV_DENSE_STREAM_*)
    void read_from_env();
};

// ============================================================================
// Per-Token Timing Record (for debug analysis)
// ============================================================================

/// @brief Timing breakdown for one token's streamed execution
struct TokenTimingRecord {
    uint32_t token_idx = 0;
    double total_ms = 0.0;         // Wall-clock for entire execute_impl_streamed
    double flush_ms = 0.0;         // GPU fence (get_stream().flush)
    double load_ms = 0.0;          // NVMe IO wait (wait_for_load / load_group)
    double swap_ms = 0.0;          // swap_weight_pointers (create_subbuffer + force_set)
    double set_args_ms = 0.0;      // set_arguments (rebind kernel args)
    double gpu_compute_ms = 0.0;   // GPU execution (total - flush - load - swap - set_args)
    uint32_t group_transitions = 0;
    uint32_t prims_executed = 0;
};

// ============================================================================
// Timing Statistics
// ============================================================================

/// @brief Performance statistics for weight streaming
struct DenseStreamingStats {
    // Per-group timing (last token)
    std::vector<double> group_load_time_ms;   // Direct I/O read time
    std::vector<double> group_gpu_time_ms;    // GPU compute time (if tracked)
    
    // Aggregate stats
    std::atomic<uint64_t> total_loads{0};
    std::atomic<uint64_t> total_bytes_loaded{0};
    std::atomic<double> total_load_time_ms{0};
    std::atomic<double> peak_throughput_gbps{0};
    
    // Pipeline efficiency
    double overlap_ratio = 0.0;  // How much IO overlaps with GPU (0-1)
    
    void reset();
    void print_summary() const;
};

// ============================================================================
// Main Class
// ============================================================================

/// @brief Manages Direct I/O based weight streaming for dense decoder models
///
/// Provides double-buffered NVMe→USM streaming, allowing GPU inference on
/// models larger than available memory by loading layer groups on-demand.
/// Uses Win32 Async ReadFile with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED
/// for zero-copy reads directly into USM host buffers.
class DenseWeightStreamingManager {
public:
    /// @brief Construct manager (does not initialize — call initialize() next)
    /// @param engine Reference to cldnn GPU engine (for USM allocation)
    /// @param config Streaming configuration
    explicit DenseWeightStreamingManager(cldnn::engine& engine,
                                          const DenseStreamingConfig& config);
    
    ~DenseWeightStreamingManager();
    
    // Non-copyable, non-movable (owns GPU resources)
    DenseWeightStreamingManager(const DenseWeightStreamingManager&) = delete;
    DenseWeightStreamingManager& operator=(const DenseWeightStreamingManager&) = delete;

    // ========================================================================
    // Initialization
    // ========================================================================
    
    /// @brief Initialize: open file, read header, allocate double-buffers, init Direct I/O
    /// @return true if initialization succeeded
    bool initialize();
    
    /// @brief Check if initialized successfully
    bool is_initialized() const { return m_initialized; }
    
    /// @brief Get number of layer groups
    uint32_t num_groups() const { return m_header.num_groups; }
    
    /// @brief Get layers per group
    uint32_t group_size() const { return m_header.group_size; }
    
    /// @brief Get number of streaming buffers (2=double, 3=triple, etc.)
    int get_num_buffers() const { return m_num_buffers; }
    
    /// @brief Get total model weight size
    uint64_t total_weight_bytes() const { return m_header.total_weight_bytes; }
    
    /// @brief Get size of a specific group (bytes)
    uint64_t group_bytes(uint32_t group_idx) const;

    // ========================================================================
    // Runtime: Weight Loading (called per-token decode)
    // ========================================================================
    
    /// @brief Load a layer group's weights from NVMe into the active buffer
    /// @param group_idx Group index (0 to num_groups-1)
    /// @return true if load initiated successfully
    ///
    /// Uses Async ReadFile (FILE_FLAG_OVERLAPPED) for non-blocking I/O.
    /// Issues async reads and returns immediately. Call wait_for_load()
    /// to wait for completion.
    bool load_group(uint32_t group_idx);
    
    /// @brief Wait for a group load to complete
    /// @param group_idx Group index that was previously submitted via load_group()
    /// @return true if load completed without errors
    bool wait_for_load(uint32_t group_idx);
    
    /// @brief Prefetch next group while current group is being computed
    /// @param next_group_idx Group to prefetch (loads into inactive buffer)
    /// @return true if prefetch initiated
    ///
    /// This is the key to the double-buffer pipeline:
    /// Issues async ReadFile (non-blocking) for the next group into the
    /// OTHER buffer. NVMe DMA runs concurrently with GPU compute.
    /// No overlap conflict because they use different memory regions.
    bool prefetch_next_group(uint32_t next_group_idx);
    
    /// @brief Check if a prefetch has completed (non-blocking)
    bool is_prefetch_complete(uint32_t group_idx) const;
    
    /// @brief Load group synchronously (blocking) — simpler API for testing
    /// @param group_idx Group to load
    /// @return true if load succeeded
    bool load_group_sync(uint32_t group_idx);

    // ========================================================================
    // Buffer Access (GPU kernel reads weights from here)
    // ========================================================================
    
    /// @brief Get the USM buffer pointer for a loaded group's weights
    /// @param group_idx Group index (must have been loaded & waited)
    /// @return Pointer to USM host memory containing the group's weights
    ///
    /// The returned pointer is valid until the next load into the same buffer slot.
    /// GPU kernels should read weights from this pointer.
    void* get_group_buffer_ptr(uint32_t group_idx) const;
    
    /// @brief Get cldnn::memory for a loaded group (for OpenVINO internal use)
    /// @param group_idx Group index
    cldnn::memory_ptr get_group_memory(uint32_t group_idx) const;
    
    /// @brief Get buffer for a specific layer within a loaded group
    /// @param layer_idx Absolute layer index (0-41)
    /// @return Pointer to that layer's weight data within the group buffer
    ///
    /// Requires that the layer's group has been loaded.
    void* get_layer_buffer_ptr(uint32_t layer_idx) const;
    
    /// @brief Get the active buffer index (0 or 1 for double-buffer)
    int active_buffer_idx() const { return m_active_buffer; }

    // ========================================================================
    // Weight Pointer Swapping (Integration with compiled model)
    // ========================================================================
    
    /// @brief Build weight-to-primitive mapping by scanning the compiled network
    /// @param network Pointer to cldnn::network
    /// @return true if mapping was built successfully
    ///
    /// Must be called ONCE after model compilation (or blob cache load).
    /// Scans all primitive instances, identifies data (constant) nodes matching
    /// decoder layer weight naming patterns, and records their primitive_ids
    /// and layouts for use by swap_weight_pointers().
    bool build_weight_mapping(void* network);
    
    /// @brief Build weight-to-primitive mapping from JSON metadata file
    /// @param json_metadata_path Path to the JSON file from pack_dense_weights.py
    /// @param network Pointer to cldnn::network (for primitive lookup)
    /// @return true if mapping was built successfully
    ///
    /// Reads per-tensor metadata (name, size, dtype, shape) from the JSON file,
    /// then matches tensor names to primitive_ids in the compiled network.
    bool build_weight_mapping_from_json(const std::string& json_metadata_path,
                                         void* network);
    
    /// @brief Get the weight mapping table (for debugging)
    const std::vector<WeightTensorMapping>& get_weight_mappings() const {
        return m_weight_mappings;
    }
    
    /// @brief Check if weight mapping is ready
    bool has_weight_mapping() const { return !m_weight_mappings.empty(); }

    /// @brief Swap weight argument pointers in the compiled network
    /// @param group_idx Group whose weights to activate
    /// @param network Pointer to the internal cldnn::network
    /// @return true if swap succeeded
    ///
    /// Implementation:
    ///   1. For each weight tensor in this group's layers:
    ///      a. Compute pointer: group_buffer + layer_offset + tensor_offset
    ///      b. Create memory::ptr via engine.attach_memory(layout, ptr)
    ///      c. Swap data_inst._outputs[0] with the new memory
    ///   2. Re-set kernel arguments: network.set_arguments()
    ///
    /// This is the "get_arguments()" pattern from MoE OTD:
    /// Modifies the compiled graph's data node pointers to reference
    /// the freshly-loaded buffer instead of the original weight memory.
    ///
    /// Requires:
    ///   - build_weight_mapping() or build_weight_mapping_from_json() called first
    ///   - wait_for_load() completed for this group
    ///   - Must be called BEFORE GPU execution of this group's layers
    bool swap_weight_pointers(uint32_t group_idx, void* network);

    // ========================================================================
    // Orchestrated Decode Pipeline (Step 6 — boundary synchronization)
    // ========================================================================
    
    /// @brief Execute one full streamed token decode across all layer groups
    /// @param compute_fn Callback for GPU compute of each group's layers
    /// @param network Pointer to cldnn::network (for swap_weight_pointers)
    /// @param stream_ptr Pointer to cldnn::stream for GPU fence (nullptr = skip GPU fence)
    /// @return true if all groups completed successfully
    ///
    /// Three-phase pipeline with hybrid pinning (H5+T5 default):
    ///
    /// Phase 1: Pinned HEAD (layers 0-4) — pure GPU, no IO
    ///   [GPU head layers][GPU fence]
    ///
    /// Phase 2: Streamed MIDDLE (layers 5-36) — IO+GPU double-buffer pipeline
    ///   Group 0 (cold):  [== IO load 0 ==][swap][GPU 0]
    ///                                           [IO prefetch 1]
    ///   Group 1..N-1:    [fence][IO wait][swap][GPU][prefetch]
    ///   Last group:      [fence][IO wait][swap][GPU][fence]
    ///
    /// Phase 3: Pinned TAIL (layers 37-41) — pure GPU, no IO
    ///   [GPU tail layers][GPU fence]
    ///
    /// H5+T5 estimated per-token latency (1-layer groups, 11.2 GB/s):
    ///   Head GPU:     5 x 0.99 =   5.0 ms
    ///   Stream IO:   32 x 4.2  = 134.0 ms (IO-bound)
    ///   Tail GPU:     5 x 0.99 =   5.0 ms
    ///   Total: ~144 ms -> ~7.0 tps
    bool execute_streamed_decode(const GroupComputeCallback& compute_fn,
                                  void* network = nullptr,
                                  void* stream_ptr = nullptr);
    
    /// @brief Wait for GPU to finish all enqueued work (GPU fence)
    /// @param stream_ptr Pointer to cldnn::stream (calls stream.finish())
    ///
    /// Must be called after GPU compute and BEFORE swapping weight pointers,
    /// to ensure the GPU is done reading from the current buffer.
    void wait_for_gpu(void* stream_ptr);
    
    /// @brief Build per-group execution order from the compiled network
    /// @param network Pointer to cldnn::network
    /// @return true if partition succeeded
    ///
    /// Partitions network exec_order into per-group primitive lists (H5+T5 aware).
    /// Layout (ng + 4 slots):
    ///   [0..ng-1]  = streamed decoder groups (middle layers)
    ///   [ng]       = pinned HEAD decoder layers
    ///   [ng+1]     = pinned TAIL decoder layers
    ///   [ng+2]     = pre-decoder (embeddings, vision encoder, etc.)
    ///   [ng+3]     = post-decoder (final norm, lm_head, logits)
    bool build_group_exec_order(void* network);
    
    /// @brief Group index constants for exec order slots
    uint32_t exec_slot_pinned_head() const { return m_header.num_groups; }
    uint32_t exec_slot_pinned_tail() const { return m_header.num_groups + 1; }
    uint32_t exec_slot_pre_decoder() const { return m_header.num_groups + 2; }
    uint32_t exec_slot_post_decoder() const { return m_header.num_groups + 3; }
    
    /// @brief Get the primitive list for a group (after build_group_exec_order)
    /// @param group_idx Group index (0..ng-1 for streamed, ng/ng+1/ng+2/ng+3 for special)
    /// @return Vector of primitive_inst pointers to execute for this group
    const std::vector<void*>& get_group_exec_primitives(uint32_t group_idx) const;
    
    /// @brief Get per-primitive group assignment (parallel to _exec_order)
    /// @return Vector of group indices for each position in _exec_order
    const std::vector<int32_t>& exec_group_assignments() const { return m_exec_group_assignment; }
    
    /// @brief Group assignment constants for special regions
    static constexpr int32_t GROUP_PRE_DECODER = -3;
    static constexpr int32_t GROUP_PINNED_HEAD = -2;
    static constexpr int32_t GROUP_PINNED_TAIL = -1;
    // 0..ng-1 = streamed middle groups
    // ng = post-decoder
    
    /// @brief Get last token's pipeline statistics
    const TokenPipelineStats& get_last_token_stats() const { return m_last_token_stats; }
    
    /// @brief Get pinning configuration
    uint32_t pin_head_layers() const { return m_config.pin_head_layers; }
    uint32_t pin_tail_layers() const { return m_config.pin_tail_layers; }
    uint32_t num_streamed_layers() const { return m_config.num_streamed_layers(); }
    bool is_layer_pinned(uint32_t layer_idx) const { return m_config.is_pinned(layer_idx); }
    
    /// @brief Check if debug logging is enabled (for network.cpp per-group logging)
    bool debug_enabled() const { return m_config.debug_logging; }
    
    /// @brief Check if prefetch (IO/GPU overlap) is disabled (v1 sequential mode)
    bool is_prefetch_disabled() const { return m_config.no_prefetch; }

    // ========================================================================
    // Statistics & Debug
    // ========================================================================
    
    /// @brief Get streaming performance statistics
    const DenseStreamingStats& get_stats() const { return m_stats; }
    
    /// @brief Reset statistics
    void reset_stats() { m_stats.reset(); }
    
    /// @brief Print current status
    void print_status() const;

private:
    // ========================================================================
    // Private: Initialization helpers
    // ========================================================================
    
    /// @brief Read and validate file header
    bool read_file_header();
    
    /// @brief Read group table and layer table from file
    bool read_tables();
    
    /// @brief Read group table from second stripe file (dual-NVMe)
    bool read_tables_file2();
    
    /// @brief Allocate double-buffer USM host memory
    bool allocate_buffers();
    
    /// @brief Initialize Async I/O file handles and event resources
    bool initialize_direct_io();
    
    /// @brief Cleanup IO resources
    void shutdown_io();

    // ========================================================================
    // Private: I/O methods
    // ========================================================================
    
    /// @brief Load via Async ReadFile (FILE_FLAG_OVERLAPPED)
    /// @param file_offset Absolute offset in dense_weights_streaming.bin
    /// @param size Bytes to read
    /// @param dest_ptr Destination USM host pointer (must be page-aligned)
    /// @param buffer_idx Buffer slot (0 or 1) for tracking
    /// @return true if all async IO operations completed successfully
    ///
    /// Synchronous wrapper: issues async ReadFile calls then waits.
    /// Reads directly into dest_ptr with zero-copy. The destination pointer
    /// (USM host buffer) is page-aligned (4096 bytes), which satisfies
    /// FILE_FLAG_NO_BUFFERING's sector-alignment requirement.
    bool load_direct_io(uint64_t file_offset, uint64_t size,
                        void* dest_ptr, uint32_t buffer_idx);
    
    /// @brief Issue async ReadFile calls (non-blocking, returns immediately)
    /// @param file_offset Absolute offset in file
    /// @param size Bytes to read (must be sector-aligned)
    /// @param dest_ptr Destination USM host pointer (must be page-aligned)
    /// @return true if all async IO operations were submitted successfully
    ///
    /// Splits the read across num_io_handles file handles, each with its own
    /// async ReadFile + OVERLAPPED + Event. Returns immediately — the NVMe
    /// controller performs DMA in the background.
    bool start_async_load(uint64_t file_offset, uint64_t size, void* dest_ptr);
    
    /// @brief Issue async ReadFile calls across 2 NVMe files (dual-NVMe mode)
    /// @param group_idx Group index — used to look up per-file offsets/sizes
    /// @param dest_ptr Destination USM host pointer (contiguous buffer)
    /// @return true if all async IO operations were submitted successfully
    ///
    /// Reads first half from file 0 into dest[0..size_0), second half from
    /// file 1 into dest[size_0..size_0+size_1). Both reads run in parallel
    /// using separate file handles for each NVMe, achieving ~2x bandwidth.
    bool start_async_load_dual(uint32_t group_idx, void* dest_ptr);
    
    /// @brief Wait for all outstanding async ReadFile operations to complete
    /// @return true if all operations completed successfully
    ///
    /// Uses WaitForMultipleObjects to efficiently wait for all async reads.
    /// After return, the destination buffer contains the loaded data.
    bool wait_async_load();
    
    /// @brief Check if async load has completed (non-blocking)
    /// @return true if no async ops in flight or all have completed
    bool is_async_load_complete() const;

    // ========================================================================
    // Private: Members
    // ========================================================================
    
    cldnn::engine& m_engine;
    DenseStreamingConfig m_config;
    DenseWeightsFileHeader m_header{};
    
    bool m_initialized = false;
    
    // File tables (read once at init)
    std::vector<GroupTableEntry> m_group_table;
    std::vector<LayerTableEntry> m_layer_table;
    
    // Multi-buffer USM allocations (2-4 buffers)
    // One buffer is "active" (GPU reads from it), others are available for
    // prefetch. With 3+ buffers, multiple groups can be prefetched ahead.
    static constexpr int MAX_BUFFERS = 4;
    int m_num_buffers = 2;  // Actual count (from config)
    cldnn::memory_ptr m_buffers[MAX_BUFFERS];  // USM host allocations
    void* m_buffer_ptrs[MAX_BUFFERS] = {};
    uint64_t m_buffer_sizes[MAX_BUFFERS] = {};
    int m_active_buffer = 0;  // Index of buffer currently being read by GPU
    
    // Track which group is loaded in each buffer (-1 = empty)
    int32_t m_buffer_group[MAX_BUFFERS] = {-1, -1, -1, -1};
    
    /// @brief Find a free (non-active) buffer for loading. Returns buffer index.
    int find_free_buffer(uint32_t exclude_group = UINT32_MAX) const;
    
    // Direct I/O file handles (one per IO handle for concurrent async reads)
    // Opened with FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED
    std::vector<void*> m_io_handles;    // HANDLE[], file 0 (or all in single mode)
    std::vector<void*> m_io_handles_2;  // HANDLE[], file 1 (dual-NVMe only)
    size_t m_sector_size = SECTOR_SIZE;

    // Dual-NVMe: per-file group table (file_offset and aligned_bytes differ per stripe)
    std::vector<GroupTableEntry> m_group_table_2;  // Group table from stripe file 1
    // Full group aligned sizes (for buffer allocation, sum of both stripes)
    std::vector<uint64_t> m_full_group_aligned_bytes;
    
    // Async I/O state (FILE_FLAG_OVERLAPPED based)
    // Replaces std::thread prefetch with native kernel-level async I/O.
    // Each file handle gets its own OVERLAPPED struct + manual-reset Event.
    // NVMe DMA runs in background, GPU and IO are truly independent.
    static constexpr uint32_t MAX_ASYNC_OPS = 32;
    std::vector<void*> m_io_events;      // HANDLE[], manual-reset events
    void* m_overlapped_storage = nullptr; // Array of OVERLAPPED structs (opaque)
    uint32_t m_async_num_ops = 0;        // Number of async ReadFile ops in flight
    int m_async_target_buffer = -1;      // Buffer slot being async-loaded into
    uint32_t m_async_target_group = UINT32_MAX;  // Group being async-loaded
    std::atomic<bool> m_async_in_progress{false};
    std::chrono::high_resolution_clock::time_point m_async_start_time;
    
    // Statistics
    DenseStreamingStats m_stats;
    
    // Weight-to-primitive mapping (built by build_weight_mapping*)
    // Maps each weight tensor in the binary file to its primitive_id in the
    // compiled network. Used by swap_weight_pointers() to redirect each data
    // primitive's output to the correct offset within our streaming buffer.
    std::vector<WeightTensorMapping> m_weight_mappings;
    
    // Per-group execution order (built by build_group_exec_order)
    // Each entry is a list of primitive_inst* for that group's layers.
    // Used by execute_streamed_decode() for per-group partial execution.
    std::vector<std::vector<void*>> m_group_exec_order;
    
    // Per-primitive group assignment (parallel to _exec_order, built by
    // build_group_exec_order). Each entry is the group index:
    //   -3 = pre-decoder
    //   -2 = pinned HEAD
    //   -1 = pinned TAIL
    //   0..ng-1 = streamed middle groups
    //   ng = post-decoder (stored as m_header.num_groups)
    // Used by execute_impl_streamed() for group transition detection.
    std::vector<int32_t> m_exec_group_assignment;
    
    // Empty vector returned by get_group_exec_primitives on error
    static const std::vector<void*> s_empty_exec_order;
    
    // Pipeline timing (last token)
    TokenPipelineStats m_last_token_stats;
    
    // Debug log file (nullptr = use stderr)
    FILE* m_debug_log = nullptr;
    
    // Per-token timing records (for post-analysis)
    std::vector<TokenTimingRecord> m_token_timings;
    uint32_t m_token_counter = 0;

public:
    /// @brief Get debug log file handle (nullptr → stderr)
    FILE* debug_log_file() const { return m_debug_log ? m_debug_log : stderr; }
    
    /// @brief Number of token timing records collected so far
    size_t token_timings_count() const { return m_token_timings.size(); }
    
    /// @brief Record timing for one token
    void record_token_timing(const TokenTimingRecord& record);
    
    /// @brief Write all token timings to log file + summary
    void flush_token_timings();

private:
    
    // Synchronization
    std::mutex m_load_mutex;  // Protects load operations
    uint32_t m_prefetch_group_idx = UINT32_MAX;
};

}  // namespace ov::intel_gpu
