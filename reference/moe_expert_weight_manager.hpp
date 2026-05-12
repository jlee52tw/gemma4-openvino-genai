// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Source: https://github.com/jlee52tw/openvino/blob/moe-otd-pr-squash/
//         src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_expert_weight_manager.hpp

#pragma once

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/layout.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <memory>
#include <mutex>
#include <chrono>
#include <atomic>
#include <thread>
#include <queue>
#include <future>
#include <condition_variable>
#include <functional>

namespace ov::intel_gpu::ocl {

/// @brief Simple thread pool for parallel expert loading
/// Pre-allocates threads to avoid creation overhead per call
/// Controlled by OV_MOE_THREAD_POOL_SIZE environment variable (default: 4)
/// Each worker thread has a unique index (0 to num_threads-1) for DirectStorage queue assignment
class ExpertLoadingThreadPool {
public:
    /// @brief Construct thread pool with specified number of worker threads
    /// @param num_threads Number of worker threads (default: 4)
    explicit ExpertLoadingThreadPool(size_t num_threads = 4);
    
    /// @brief Destructor - signals workers to stop and joins all threads
    ~ExpertLoadingThreadPool();
    
    // Non-copyable, non-movable
    ExpertLoadingThreadPool(const ExpertLoadingThreadPool&) = delete;
    ExpertLoadingThreadPool& operator=(const ExpertLoadingThreadPool&) = delete;
    
    /// @brief Submit a task to the thread pool
    /// @param task Callable to execute
    /// @return std::future to wait for completion
    std::future<void> submit(std::function<void()> task);
    
    /// @brief Submit a task that receives the worker's thread index
    /// @param task Callable that takes size_t thread_idx parameter
    /// @return std::future to wait for completion
    /// @note thread_idx is in range [0, num_threads-1] and can be used as DirectStorage queue index
    std::future<void> submit_with_thread_idx(std::function<void(size_t)> task);
    
    /// @brief Get number of worker threads
    size_t size() const { return m_workers.size(); }
    
    /// @brief Check if parallel loading is enabled
    static bool is_parallel_load_enabled();
    
    /// @brief Get configured thread pool size from environment
    static size_t get_configured_size();
    
private:
    std::vector<std::thread> m_workers;
    
    // Task queue: each task is a pair of (callable, needs_thread_idx)
    // If needs_thread_idx is true, we call the stored function with the worker's index
    struct TaskWrapper {
        std::packaged_task<void()> simple_task;
        std::function<void(size_t)> indexed_task;
        bool needs_thread_idx = false;
    };
    std::queue<TaskWrapper> m_tasks;
    
    std::mutex m_queue_mutex;
    std::condition_variable m_condition;
    std::atomic<bool> m_stop{false};
    
    void worker_thread(size_t thread_idx);
};

/// @brief Expert profiling statistics for distribution analysis
/// Controlled by OV_MOE_PROFILE_EXPERTS=1 environment variable
struct ExpertProfileStats {
    // Per-expert hit counts (indexed by global_expert_id = layer * 32 + local_id)
    // Using unique_ptr because std::atomic is not copyable (cannot use vector::resize)
    std::unique_ptr<std::atomic<uint64_t>[]> expert_hit_count;
    size_t expert_hit_count_size = 0;
    
    // Per-layer statistics
    std::unique_ptr<std::atomic<uint64_t>[]> layer_total_selections;
    
    // Temporal locality tracking
    std::unordered_set<int32_t> prev_token_experts;   // Previous token's experts
    std::unordered_set<int32_t> current_token_experts; // Current token's experts
    std::atomic<uint64_t> consecutive_hits{0};  // Expert reused from previous token
    std::atomic<uint64_t> total_selections{0};  // Total expert selections
    
    // Token-level tracking
    std::atomic<uint64_t> total_tokens{0};
    
    // Initialization flag
    bool initialized = false;
    uint32_t num_layers = 0;
    uint32_t num_experts_per_layer = 0;
    
    void initialize(uint32_t layers, uint32_t experts_per_layer) {
        num_layers = layers;
        num_experts_per_layer = experts_per_layer;
        size_t total_experts = static_cast<size_t>(layers) * experts_per_layer;
        expert_hit_count_size = total_experts;
        expert_hit_count = std::make_unique<std::atomic<uint64_t>[]>(total_experts);
        for (size_t i = 0; i < total_experts; ++i) {
            expert_hit_count[i].store(0);
        }
        layer_total_selections = std::make_unique<std::atomic<uint64_t>[]>(layers);
        for (size_t i = 0; i < layers; ++i) {
            layer_total_selections[i].store(0);
        }
        initialized = true;
    }
    
    void record_selection(int32_t global_expert_id, uint32_t layer_idx) {
        if (!initialized) return;
        if (global_expert_id < 0 || static_cast<size_t>(global_expert_id) >= expert_hit_count_size) return;
        
        expert_hit_count[global_expert_id]++;
        layer_total_selections[layer_idx]++;
        total_selections++;
        
        // Track for temporal locality
        current_token_experts.insert(global_expert_id);
        if (prev_token_experts.count(global_expert_id) > 0) {
            consecutive_hits++;
        }
    }
    
    void end_token() {
        if (!initialized) return;
        total_tokens++;
        prev_token_experts = std::move(current_token_experts);
        current_token_experts.clear();
    }
};

/// @brief Timing statistics for MoE OTD performance analysis
/// Controlled by OV_MOE_TIMING=1 environment variable
struct MoETimingStats {
    // Per-call statistics (reset after each print)
    std::atomic<uint64_t> mmap_read_count{0};          // Number of mmap reads
    std::atomic<uint64_t> mmap_read_bytes{0};          // Total bytes read via mmap
    std::atomic<double> mmap_read_time_us{0};          // Total mmap read time (microseconds)
    
    std::atomic<uint64_t> gpu_copy_count{0};           // Number of GPU copies
    std::atomic<uint64_t> gpu_copy_bytes{0};           // Total bytes copied to GPU
    std::atomic<double> gpu_copy_time_us{0};           // Total GPU copy time (microseconds)
    
    std::atomic<uint64_t> cache_hits{0};               // Cache hit count
    std::atomic<uint64_t> cache_misses{0};             // Cache miss count
    
    // Per-layer timing (indexed by layer_idx)
    std::vector<double> layer_load_time_us;            // Time to load experts per layer
    std::vector<size_t> layer_experts_loaded;          // Number of experts loaded per layer
    
    // Aggregate statistics (never reset)
    std::atomic<uint64_t> total_tokens{0};             // Total tokens processed
    std::atomic<double> total_io_time_us{0};           // Total I/O time across all tokens
    
    void reset_per_token() {
        mmap_read_count = 0;
        mmap_read_bytes = 0;
        mmap_read_time_us = 0;
        gpu_copy_count = 0;
        gpu_copy_bytes = 0;
        gpu_copy_time_us = 0;
        cache_hits = 0;
        cache_misses = 0;
        layer_load_time_us.clear();
        layer_experts_loaded.clear();
    }
    
    void init_layers(size_t num_layers) {
        layer_load_time_us.resize(num_layers, 0.0);
        layer_experts_loaded.resize(num_layers, 0);
    }
};

/// @brief File header for MoE weights binary file
struct MoEWeightsFileHeader {
    char magic[4] = {'M', 'O', 'E', 'W'};  // Magic number "MOEW"
    uint32_t version = 1;
    uint32_t num_layers = 0;
    uint32_t num_experts_per_layer = 0;
    uint64_t expert_up_weight_size = 0;      // Size of up-projection weight per expert (bytes)
    uint64_t expert_down_weight_size = 0;    // Size of down-projection weight per expert (bytes)
    uint64_t expert_up_scale_size = 0;       // Size of up-projection scale per expert (bytes)
    uint64_t expert_down_scale_size = 0;     // Size of down-projection scale per expert (bytes)
    uint64_t expert_up_bias_size = 0;        // Size of up-projection bias per expert (bytes)
    uint64_t expert_down_bias_size = 0;      // Size of down-projection bias per expert (bytes)
    uint64_t data_offset = 0;                // Offset to the start of weight data
    uint64_t reserved[7] = {0};              // Reserved for future use (7 x uint64 = 56 bytes, total header = 128 bytes)
};

/// @brief Describes weight tensor layout for one expert
struct ExpertWeightDesc {
    size_t offset = 0;      // Offset in the file
    size_t size = 0;        // Size in bytes
    cldnn::layout layout;   // Tensor layout
};

/// @brief Layer-level weight information
struct LayerWeightInfo {
    uint32_t layer_idx = 0;
    uint32_t num_experts = 0;
    std::vector<ExpertWeightDesc> up_weights;      // Per-expert up-projection weights
    std::vector<ExpertWeightDesc> down_weights;    // Per-expert down-projection weights
    std::vector<ExpertWeightDesc> up_scales;       // Per-expert up-projection scales
    std::vector<ExpertWeightDesc> down_scales;     // Per-expert down-projection scales
    std::vector<ExpertWeightDesc> up_biases;       // Per-expert up-projection biases
    std::vector<ExpertWeightDesc> down_biases;     // Per-expert down-projection biases
};

/// @brief Configuration for MoEExpertWeightManager
struct MoEOTDConfig {
    std::string weights_path;          // Path to the weights file on disk (SSD)
    int64_t resident_experts = 0;      // Number of experts to keep in GPU memory
    bool enabled = false;              // Whether OTD is enabled
};

/// @brief Manages MoE expert weights with Offload-To-Disk (OTD) capability
///
/// This class manages the loading and caching of MoE expert weights from
/// disk (SSD) to GPU memory. It maintains a buffer in GPU memory that can
/// hold a limited number of experts, and dynamically loads/unloads experts
/// based on runtime requirements.
class MoEExpertWeightManager {
public:
    /// @brief Construct a new MoE Expert Weight Manager
    /// @param engine GPU engine for memory allocation
    /// @param config OTD configuration
    MoEExpertWeightManager(cldnn::engine& engine, const MoEOTDConfig& config);
    
    /// @brief Destructor
    ~MoEExpertWeightManager();

    /// @brief Initialize the manager by reading file header and allocating buffers
    /// @return true if initialization succeeded
    bool initialize();

    /// @brief Check if OTD is enabled and properly initialized
    bool is_enabled() const { return m_initialized && m_config.enabled; }

    /// @brief Load specified experts into GPU memory buffer
    /// @param layer_idx The layer index (0-based)
    /// @param expert_ids List of expert IDs to load
    /// @param stream GPU stream for async operations
    /// @param is_up_projection true for up-projection, false for down-projection
    void load_experts(uint32_t layer_idx,
                      const std::vector<int32_t>& expert_ids,
                      cldnn::stream& stream,
                      bool is_up_projection);

    /// @brief Get the GPU memory buffer containing loaded expert weights
    /// @param is_up_projection true for up-projection, false for down-projection
    /// @return Pointer to GPU memory
    cldnn::memory::ptr get_weight_buffer(bool is_up_projection) const;

    /// @brief Get the GPU memory buffer view for a specific layer
    /// @param layer_idx The layer index (0-based)
    /// @param is_up_projection true for up-projection, false for down-projection
    /// @return Pointer to GPU memory subbuffer starting at the layer's offset
    /// @note This returns a subbuffer view that starts at layer_idx * 32 * expert_size,
    ///       allowing the kernel to access experts using local expert IDs (0-31)
    cldnn::memory::ptr get_weight_buffer_for_layer(uint32_t layer_idx, bool is_up_projection) const;

    /// @brief Get/create compact weight buffer for kernel execution
    /// @param is_up_projection true for up-projection, false for down-projection
    /// @return Pointer to compact buffer (num_experts_per_layer * expert_size bytes)
    /// @note Shared across all kernel calls; caller copies needed experts into it
    cldnn::memory::ptr get_or_create_compact_weight_buffer(bool is_up_projection);

    /// @brief Get the GPU memory buffer containing loaded expert scales
    cldnn::memory::ptr get_scale_buffer(bool is_up_projection) const;

    /// @brief Get the GPU memory buffer containing loaded expert biases
    cldnn::memory::ptr get_bias_buffer(bool is_up_projection) const;

    /// @brief Get mapping from buffer slot index to expert ID
    /// @param is_up_projection true for up-projection, false for down-projection
    const std::vector<int32_t>& get_slot_to_expert_mapping(bool is_up_projection) const;

    /// @brief Get mapping from expert ID to buffer slot index
    /// @param is_up_projection true for up-projection, false for down-projection
    const std::unordered_map<int32_t, int32_t>& get_expert_to_slot_mapping(bool is_up_projection) const;

    /// @brief Get the number of experts currently loaded
    size_t get_num_loaded_experts(bool is_up_projection) const;

    /// @brief Get the file header information
    const MoEWeightsFileHeader& get_file_header() const { return m_header; }

    /// @brief Get layer weight information
    const LayerWeightInfo& get_layer_info(uint32_t layer_idx) const;

    /// @brief Set layer weight information (used during file creation)
    void set_layer_info(uint32_t layer_idx, const LayerWeightInfo& info);

    /// @brief Check if timing is enabled (OV_MOE_TIMING=1)
    static bool is_timing_enabled();
    
    /// @brief Get timing statistics (for external access)
    MoETimingStats& get_timing_stats() { return m_timing_stats; }
    const MoETimingStats& get_timing_stats() const { return m_timing_stats; }
    
    /// @brief Print timing summary for current token
    void print_timing_summary(bool is_prefill = false);
    
    /// @brief Reset per-token timing statistics
    void reset_timing_stats() { m_timing_stats.reset_per_token(); }
    
    /// @brief Record start of a new token (for decode phase timing)
    void begin_token_timing();
    
    /// @brief Record end of token and print summary
    void end_token_timing(bool is_prefill = false);

    /// @brief Check if expert profiling is enabled (OV_MOE_PROFILE_EXPERTS=1)
    static bool is_profiling_enabled();
    
    /// @brief Initialize DirectStorage for faster I/O (Windows 11+ with NVMe BypassIO)
    /// @return true if DirectStorage initialized successfully
    bool initialize_directstorage();
    
    /// @brief Load expert weights via DirectStorage (bypasses OS file cache)
    /// @param file_offset Offset in the weights file
    /// @param size Size of data to read
    /// @param dest_ptr Destination pointer (GPU usm_host memory)
    /// @param bytes_loaded_out Atomic counter for bytes loaded
    /// @param queue_idx Index of the DirectStorage queue to use (0 to NUM_DS_QUEUES-1)
    /// @return true if read succeeded
    bool load_expert_via_directstorage(uint64_t file_offset, size_t size,
                                       void* dest_ptr, std::atomic<size_t>& bytes_loaded_out,
                                       size_t queue_idx);
    
    /// @brief Record expert selection for profiling
    /// @param global_expert_id Global expert ID (layer * 32 + local_id)
    /// @param layer_idx Layer index
    void record_expert_selection(int32_t global_expert_id, uint32_t layer_idx);
    
    /// @brief Signal end of token for temporal locality tracking
    void end_profiling_token();
    
    /// @brief Export profile data to JSON file
    /// @param path Output file path
    void export_profile_to_json(const std::string& path);
    
    /// @brief Get profiling statistics (for external access)
    ExpertProfileStats& get_profile_stats() { return m_profile_stats; }
    const ExpertProfileStats& get_profile_stats() const { return m_profile_stats; }

    /// @brief Check if hot expert pinning is enabled (OV_MOE_PIN_HOT_EXPERTS, default: enabled)
    static bool is_hot_expert_pinning_enabled();
    
    /// @brief Preload hot experts into their dedicated buffer slots at initialization
    /// This eliminates SSD reads for hot experts on the first inference pass.
    /// Only preloads experts whose layers have dedicated slots (layers 0 to dedicated_layers-1).
    void preload_hot_experts();
    
    /// @brief Check if an expert is in the hot expert list (377 profiled experts)
    bool is_hot_expert(int32_t global_expert_id) const;
    
    /// @brief Check if an expert is pinned (top 128 highest-priority experts)
    bool is_pinned_expert(int32_t global_expert_id) const;
    
    /// @brief Get hot expert preload statistics
    size_t get_hot_experts_preloaded() const { return m_hot_experts_preloaded_count; }

private:
    /// @brief Read file header and validate
    bool read_file_header();

    /// @brief Calculate buffer sizes and allocate GPU memory
    bool allocate_buffers();

    /// @brief Load a single expert's weights from disk to host buffer
    void load_expert_to_host(uint32_t layer_idx, int32_t expert_id, bool is_up_projection);

    /// @brief Copy host buffer to GPU memory slot
    void copy_to_gpu_slot(int32_t slot_idx, bool is_up_projection, cldnn::stream& stream);

    /// @brief Load expert directly from mmap to GPU buffer (skip staging buffer)
    /// @return true if direct copy succeeded, false if fallback to staging is needed
    bool load_expert_direct_to_gpu(uint32_t layer_idx, int32_t global_expert_id, 
                                   int32_t slot_idx, bool is_up_projection);

    /// @brief Load expert directly (no timing) - for parallel loading with external wall-clock measurement
    /// @param bytes_loaded_out Atomic counter to report bytes loaded
    /// @param ds_queue_idx DirectStorage queue index for this thread (0 to NUM_DS_QUEUES-1)
    /// @return true if direct copy succeeded, false if fallback needed
    bool load_expert_direct_to_gpu_no_timing(uint32_t layer_idx, int32_t global_expert_id,
                                              int32_t slot_idx, bool is_up_projection,
                                              std::atomic<size_t>& bytes_loaded_out,
                                              size_t ds_queue_idx = 0);
    
    /// @brief Load expert using Direct I/O (FILE_FLAG_NO_BUFFERING) for static layers
    /// This bypasses OS page cache, data flows directly: SSD -> USM buffer
    /// @param layer_idx Layer index (must be < m_dedicated_layers)
    /// @param global_expert_id Global expert ID
    /// @param slot_idx Target slot in GPU buffer
    /// @param is_up_projection true for up-projection weights
    /// @return true if direct I/O succeeded
    bool load_expert_via_direct_io(uint32_t layer_idx, int32_t global_expert_id,
                                   int32_t slot_idx, bool is_up_projection);

    /// @brief Find an available slot (or evict least recently used)
    int32_t find_available_slot(bool is_up_projection);

    cldnn::engine& m_engine;
    MoEOTDConfig m_config;
    MoEWeightsFileHeader m_header;
    
    bool m_initialized = false;
    std::unique_ptr<std::ifstream> m_weights_file;
    std::mutex m_mutex;

    // Memory-mapped file support (Windows) - for DYNAMIC layers (11-23) only
    void* m_mmap_handle = nullptr;      // HANDLE for CreateFileMapping
    void* m_mmap_view = nullptr;        // Pointer adjusted to actual data start (may differ from raw view)
    void* m_mmap_view_base = nullptr;   // Raw pointer from MapViewOfFile (for UnmapViewOfFile)
    size_t m_mmap_size = 0;             // Size of mapped region (dynamic layers only)
    size_t m_mmap_file_offset = 0;      // File offset where mmap starts (layer 11 offset)
    void* m_file_handle = nullptr;      // HANDLE for CreateFile (mmap use)
    
    // Direct I/O support (Windows) - for STATIC layers (0-10)
    // FILE_FLAG_NO_BUFFERING bypasses OS page cache: SSD -> USM directly
    void* m_direct_io_handle = nullptr; // HANDLE with FILE_FLAG_NO_BUFFERING
    size_t m_sector_size = 4096;        // Sector size for alignment (default 4KB)
    size_t m_dedicated_layers = 0;      // Number of layers using direct I/O (0-10 = 11 layers)
    std::vector<uint8_t> m_aligned_read_buffer;  // Sector-aligned buffer for ReadFile
    mutable std::mutex m_direct_io_mutex;        // Protects m_aligned_read_buffer and direct I/O reads

    // Per-layer weight information
    std::vector<LayerWeightInfo> m_layer_infos;

    // GPU memory buffers for up-projection
    cldnn::memory::ptr m_up_weight_buffer;
    cldnn::memory::ptr m_up_scale_buffer;
    cldnn::memory::ptr m_up_bias_buffer;

    // GPU memory buffers for down-projection
    cldnn::memory::ptr m_down_weight_buffer;
    cldnn::memory::ptr m_down_scale_buffer;
    cldnn::memory::ptr m_down_bias_buffer;

    // Compact weight buffers for kernel execution (num_experts_per_layer experts each)
    // Shared across all MoEGemmImpl instances since kernels execute sequentially.
    // Each kernel call packs needed experts into this contiguous buffer, avoiding
    // scattered access across the large full weight buffer.
    cldnn::memory::ptr m_compact_up_weight_buffer;
    cldnn::memory::ptr m_compact_down_weight_buffer;

    // Buffer sizes (for VirtualLock/VirtualUnlock)
    size_t m_up_weight_buffer_size = 0;
    size_t m_down_weight_buffer_size = 0;
    size_t m_up_scale_buffer_size = 0;
    size_t m_down_scale_buffer_size = 0;

    // Host staging buffers for async loading
    std::vector<uint8_t> m_host_staging_buffer;

    // Slot management for up-projection
    std::vector<int32_t> m_up_slot_to_expert;     // slot_idx -> expert_id (-1 if empty)
    std::unordered_map<int32_t, int32_t> m_up_expert_to_slot;  // expert_id -> slot_idx
    std::vector<uint64_t> m_up_slot_access_time;  // For LRU eviction

    // Slot management for down-projection
    std::vector<int32_t> m_down_slot_to_expert;
    std::unordered_map<int32_t, int32_t> m_down_expert_to_slot;
    std::vector<uint64_t> m_down_slot_access_time;

    // Access time counter for LRU
    uint64_t m_access_counter = 0;
    
    // DirectStorage support (Windows 11+ with NVMe BypassIO)
    // Enabled via OV_MOE_USE_DIRECTSTORAGE=1 environment variable
    // Provides faster I/O by bypassing OS file cache and using direct NVMe access
    // All pointers stored as void* for ABI compatibility;
    // cast in .cpp when OPENVINO_USE_DIRECTSTORAGE defined
    //
    // Per-thread queue architecture (eliminates mutex serialization):
    // - Create NUM_DS_QUEUES independent queues (one per thread pool worker)
    // - Each worker thread uses its own queue/status_array pair (no locking needed)
    // - Queue index passed via thread_local or task parameter
    static constexpr size_t NUM_DS_QUEUES = 4;  // Match thread pool size
    
    bool m_directstorage_initialized = false;
    void* m_ds_factory = nullptr;                          // IDStorageFactory*
    void* m_ds_queues[NUM_DS_QUEUES] = {nullptr};          // IDStorageQueue* per thread
    void* m_ds_status_arrays[NUM_DS_QUEUES] = {nullptr};   // IDStorageStatusArray* per thread
    void* m_ds_file = nullptr;                             // IDStorageFile* (shared, thread-safe for reads)
    void* m_ds_fence_events[NUM_DS_QUEUES] = {nullptr};    // HANDLE per thread for completion signaling
    // Note: No mutex needed - each thread uses its own queue exclusively
    
    // DirectStorage statistics
    std::atomic<uint64_t> m_ds_read_count{0};
    std::atomic<uint64_t> m_ds_read_bytes{0};
    std::atomic<double> m_ds_read_time_us{0};
    
    // Timing statistics (enabled by OV_MOE_TIMING=1)
    MoETimingStats m_timing_stats;
    std::chrono::high_resolution_clock::time_point m_token_start_time;
    
    // Expert profiling statistics (enabled by OV_MOE_PROFILE_EXPERTS=1)
    ExpertProfileStats m_profile_stats;
    mutable std::mutex m_profile_mutex;  // Protects prev_token_experts operations
    
    // Thread pool for parallel expert loading (created once at init)
    // Controlled by OV_MOE_PARALLEL_LOAD=1 (default enabled) and OV_MOE_THREAD_POOL_SIZE=4
    std::unique_ptr<ExpertLoadingThreadPool> m_thread_pool;
    
    // Note: With hybrid Direct I/O architecture, static layers (0-10) use FILE_FLAG_NO_BUFFERING
    // which bypasses OS page cache entirely, eliminating Standby List pollution.
    // Only dynamic layers (11-23) use mmap for efficient random access with LRU caching.
    
    // Hot Expert Pinning (OV_MOE_PIN_HOT_EXPERTS=1, default enabled)
    // Pre-loads profiled hot experts into their dedicated buffer slots at initialization.
    // Provides 100% cache hits for dedicated-layer hot experts from the first token.
    // Top 128 experts are "pinned" (highest priority), remaining 249 are "hot" (high priority).
    std::unordered_set<int32_t> m_hot_expert_set;       // All 377 hot expert global IDs (O(1) lookup)
    std::unordered_set<int32_t> m_pinned_expert_set;    // Top 128 highest-frequency experts (O(1) lookup)
    bool m_hot_experts_initialized = false;              // Hot expert sets have been built
    size_t m_hot_experts_preloaded_count = 0;            // Number of experts preloaded at init
};

}  // namespace ov::intel_gpu::ocl
