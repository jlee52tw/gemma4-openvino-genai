// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Dense Weight Streaming Manager — Implementation
// =================================================
// Async I/O (FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) based NVMe→USM
// double-buffer streaming for dense decoder models. Uses Event-based async
// ReadFile for true GPU/IO concurrency — NVMe DMA and iGPU DMA run in
// parallel on LPDDR5 (>100 GB/s bandwidth >> ~12+50 GB/s combined demand).
//
// IO design: Async ReadFile + FILE_FLAG_OVERLAPPED
//   - ReadFile returns immediately, NVMe controller performs DMA in background
//   - WaitForMultipleObjects for zero-overhead completion fence
//   - No thread creation/destruction per group — pure kernel-level async
//   - GPU compute and NVMe DMA are truly independent (no sync overhead)

#include "dense_weight_streaming_manager.hpp"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <unordered_set>

// Windows headers for Direct I/O
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// OpenVINO GPU plugin internals
#ifdef OV_DENSE_WEIGHT_STREAMING_ENABLED
// In-tree build: use real OpenVINO runtime headers
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "primitive_inst.h"
#define OPENVINO_GPU_RUNTIME_AVAILABLE
#else
// Standalone build: use shared stubs
#include "cldnn_stubs.hpp"
#endif

namespace ov::intel_gpu {

// Helper: atomic add for double (C++17 doesn't have atomic<double>::fetch_add)
static void atomic_add_double(std::atomic<double>& target, double value) {
    double current = target.load(std::memory_order_relaxed);
    while (!target.compare_exchange_weak(current, current + value,
                                          std::memory_order_relaxed)) {}
}

// ============================================================================
// DenseStreamingConfig
// ============================================================================

void DenseStreamingConfig::read_from_env() {
    // Read configuration from environment variables
    if (const char* v = std::getenv("OV_DENSE_STREAM_FILE"))
        weights_file_path = v;
    if (const char* v = std::getenv("OV_DENSE_STREAM_IO_THREADS"))
        num_io_threads = std::clamp(static_cast<uint32_t>(std::stoul(v)), 1u, 8u);
    if (const char* v = std::getenv("OV_DENSE_STREAM_LOCK_MEMORY"))
        lock_memory = (std::string(v) == "1");
    if (const char* v = std::getenv("OV_DENSE_STREAM_TIMING"))
        enable_timing = (std::string(v) == "1");
    if (const char* v = std::getenv("OV_DENSE_STREAM_DEBUG"))
        debug_logging = (std::string(v) == "1");
    if (const char* v = std::getenv("OV_DENSE_STREAM_LOG_FILE"))
        debug_log_path = v;
    if (const char* v = std::getenv("OV_DENSE_STREAM_MAX_BUFFER_GB")) {
        double gb = std::stod(v);
        max_buffer_bytes = static_cast<uint64_t>(gb * 1024.0 * 1024.0 * 1024.0);
    }
    if (const char* v = std::getenv("OV_DENSE_STREAM_NUM_BUFFERS"))
        num_buffers = std::clamp(static_cast<uint32_t>(std::stoul(v)), 2u, 4u);
    // Hybrid pinning: number of head/tail layers to keep permanently in memory
    if (const char* v = std::getenv("OV_DENSE_STREAM_PIN_HEAD"))
        pin_head_layers = std::clamp(static_cast<uint32_t>(std::stoul(v)), 0u, total_decoder_layers);
    if (const char* v = std::getenv("OV_DENSE_STREAM_PIN_TAIL"))
        pin_tail_layers = std::clamp(static_cast<uint32_t>(std::stoul(v)), 0u, total_decoder_layers);
    if (const char* v = std::getenv("OV_DENSE_STREAM_TOTAL_LAYERS"))
        total_decoder_layers = static_cast<uint32_t>(std::stoul(v));
}

// ============================================================================
// DenseStreamingStats
// ============================================================================

void DenseStreamingStats::reset() {
    group_load_time_ms.clear();
    group_gpu_time_ms.clear();
    total_loads.store(0);
    total_bytes_loaded.store(0);
    total_load_time_ms.store(0);
    peak_throughput_gbps.store(0);
    overlap_ratio = 0.0;
}

void DenseStreamingStats::print_summary() const {
    uint64_t loads = total_loads.load();
    if (loads == 0) {
        std::cout << "[DenseStreaming] No loads recorded.\n";
        return;
    }
    
    double total_ms = total_load_time_ms.load();
    uint64_t total_bytes = total_bytes_loaded.load();
    double avg_throughput_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_ms / 1000.0);
    
    std::cout << "[DenseStreaming] Stats:\n"
              << "  Total loads: " << loads << "\n"
              << "  Total bytes: " << (total_bytes / (1024.0 * 1024.0)) << " MB\n"
              << "  Total time: " << total_ms << " ms\n"
              << "  Avg throughput: " << avg_throughput_gbps << " GB/s\n"
              << "  Peak throughput: " << peak_throughput_gbps.load() << " GB/s\n"
              << "  IO/GPU overlap: " << (overlap_ratio * 100.0) << "%\n";
}

// ============================================================================
// Token Timing Recording
// ============================================================================

void DenseWeightStreamingManager::record_token_timing(const TokenTimingRecord& record) {
    m_token_timings.push_back(record);
}

void DenseWeightStreamingManager::flush_token_timings() {
    FILE* f = m_debug_log ? m_debug_log : stderr;
    
    if (m_token_timings.empty()) return;
    
    fprintf(f, "\n============================================================\n");
    fprintf(f, "  Dense Weight Streaming — Per-Token Timing Summary\n");
    fprintf(f, "  Total tokens: %zu\n", m_token_timings.size());
    fprintf(f, "============================================================\n");
    fprintf(f, "%6s %9s %8s %8s %8s %8s %8s %6s\n",
            "Token", "Total_ms", "Load_ms", "Swap_ms", "Args_ms",
            "Flush_ms", "GPU_ms", "Groups");
    
    // Aggregates
    double sum_total = 0, sum_load = 0, sum_swap = 0;
    double sum_args = 0, sum_flush = 0, sum_gpu = 0;
    
    for (const auto& t : m_token_timings) {
        fprintf(f, "%6u %9.2f %8.2f %8.2f %8.2f %8.2f %8.2f %6u\n",
                t.token_idx, t.total_ms, t.load_ms, t.swap_ms,
                t.set_args_ms, t.flush_ms, t.gpu_compute_ms, t.group_transitions);
        sum_total += t.total_ms;
        sum_load += t.load_ms;
        sum_swap += t.swap_ms;
        sum_args += t.set_args_ms;
        sum_flush += t.flush_ms;
        sum_gpu += t.gpu_compute_ms;
    }
    
    size_t n = m_token_timings.size();
    fprintf(f, "------------------------------------------------------------\n");
    fprintf(f, "%6s %9.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n",
            "AVG", sum_total/n, sum_load/n, sum_swap/n,
            sum_args/n, sum_flush/n, sum_gpu/n);
    fprintf(f, "%6s %9.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n",
            "TOTAL", sum_total, sum_load, sum_swap,
            sum_args, sum_flush, sum_gpu);
    fprintf(f, "------------------------------------------------------------\n");
    fprintf(f, "  Avg TPOT: %.2f ms → %.2f tok/s\n",
            sum_total / n, 1000.0 * n / sum_total);
    fprintf(f, "  Overhead breakdown (avg per token):\n");
    fprintf(f, "    NVMe load:     %6.2f ms (%5.1f%%)\n", sum_load/n, 100.0*sum_load/sum_total);
    fprintf(f, "    Swap pointers: %6.2f ms (%5.1f%%)\n", sum_swap/n, 100.0*sum_swap/sum_total);
    fprintf(f, "    set_arguments: %6.2f ms (%5.1f%%)\n", sum_args/n, 100.0*sum_args/sum_total);
    fprintf(f, "    GPU fence:     %6.2f ms (%5.1f%%)\n", sum_flush/n, 100.0*sum_flush/sum_total);
    fprintf(f, "    GPU compute:   %6.2f ms (%5.1f%%)\n", sum_gpu/n, 100.0*sum_gpu/sum_total);
    fprintf(f, "============================================================\n\n");
    fflush(f);
    
    if (m_debug_log) {
        // Also print brief summary to stderr
        fprintf(stderr, "[DenseStreaming] Token timing logged to: %s\n",
                m_config.debug_log_path.c_str());
        fprintf(stderr, "[DenseStreaming] %zu tokens, avg %.2f ms/tok (%.2f tok/s)\n",
                n, sum_total/n, 1000.0*n/sum_total);
        fprintf(stderr, "[DenseStreaming] Breakdown: load=%.1f%% swap=%.1f%% "
                "args=%.1f%% flush=%.1f%% gpu=%.1f%%\n",
                100.0*sum_load/sum_total, 100.0*sum_swap/sum_total,
                100.0*sum_args/sum_total, 100.0*sum_flush/sum_total,
                100.0*sum_gpu/sum_total);
    }
}

// ============================================================================
// DenseWeightStreamingManager — Constructor / Destructor
// ============================================================================

DenseWeightStreamingManager::DenseWeightStreamingManager(cldnn::engine& engine,
                                                           const DenseStreamingConfig& config)
    : m_engine(engine)
    , m_config(config) {
    // Open debug log file if specified
    if (!m_config.debug_log_path.empty()) {
        m_debug_log = fopen(m_config.debug_log_path.c_str(), "w");
        if (m_debug_log) {
            fprintf(m_debug_log, "=== Dense Weight Streaming Debug Log ===\n");
            fprintf(m_debug_log, "Timestamp: %s\n", __TIMESTAMP__);
            fflush(m_debug_log);
        } else {
            std::cerr << "[DenseStreaming] WARNING: Cannot open log file: "
                      << m_config.debug_log_path << ", using stderr\n";
        }
    }
}

DenseWeightStreamingManager::~DenseWeightStreamingManager() {
    // Flush token timing summary before shutdown
    if (!m_token_timings.empty()) {
        flush_token_timings();
    }
    
    // Close debug log file
    if (m_debug_log) {
        fclose(m_debug_log);
        m_debug_log = nullptr;
    }
    
    // Wait for any in-flight async IO
    if (m_async_in_progress.load()) {
        wait_async_load();
    }
    
    // Close all IO handles
    shutdown_io();
    
    // Free USM buffers
    for (int i = 0; i < m_num_buffers; ++i) {
#ifdef OPENVINO_GPU_RUNTIME_AVAILABLE
        // Real integration: release the cldnn::memory handle (shared_ptr)
        m_buffers[i].reset();
#else
        // Standalone: VirtualFree the manually allocated buffer
        if (m_buffer_ptrs[i]) {
            m_engine.free_usm_host(m_buffer_ptrs[i]);
        }
#endif
        m_buffer_ptrs[i] = nullptr;
    }
}

// ============================================================================
// Initialization
// ============================================================================

bool DenseWeightStreamingManager::initialize() {
    if (m_config.weights_file_path.empty()) {
        std::cerr << "[DenseStreaming] ERROR: No weights file path specified.\n";
        return false;
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Initializing with file: " 
                  << m_config.weights_file_path << "\n"
                  << "  IO threads: " << m_config.num_io_threads << "\n";
    }
    
    // Step 1: Read file header
    if (!read_file_header()) {
        std::cerr << "[DenseStreaming] ERROR: Failed to read file header.\n";
        return false;
    }
    
    // Step 2: Read group and layer tables
    if (!read_tables()) {
        std::cerr << "[DenseStreaming] ERROR: Failed to read tables.\n";
        return false;
    }
    
    // Step 3: Allocate double-buffer USM memory
    if (!allocate_buffers()) {
        std::cerr << "[DenseStreaming] ERROR: Failed to allocate buffers.\n";
        return false;
    }
    
    // Step 4: Open Direct I/O file handles (one per IO thread)
    if (!initialize_direct_io()) {
        std::cerr << "[DenseStreaming] ERROR: Failed to initialize Direct I/O.\n";
        return false;
    }
    
    // Initialize stats
    m_stats.group_load_time_ms.resize(m_header.num_groups, 0.0);
    m_stats.group_gpu_time_ms.resize(m_header.num_groups, 0.0);
    
    m_initialized = true;
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Initialized successfully:\n"
                  << "  Groups: " << m_header.num_groups << " x " 
                  << m_header.group_size << " layers\n"
                  << "  Total weights: " << (m_header.total_weight_bytes / (1024.0*1024.0))
                  << " MB\n"
                  << "  Buffer size: " << (m_buffer_sizes[0] / (1024.0*1024.0)) << " MB x 2\n"
                  << "  IO method: Direct I/O (" << m_config.num_io_threads << " threads)\n"
                  << "  Sector size: " << m_sector_size << "\n"
                  << "  Hybrid pinning: H" << m_config.pin_head_layers
                  << "+T" << m_config.pin_tail_layers << " (stream "
                  << m_config.num_streamed_layers() << " middle layers)\n";
    }
    
    return true;
}

bool DenseWeightStreamingManager::read_file_header() {
    std::ifstream file(m_config.weights_file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "[DenseStreaming] Cannot open file: " << m_config.weights_file_path << "\n";
        return false;
    }
    
    // Read header
    file.read(reinterpret_cast<char*>(&m_header), sizeof(m_header));
    if (!file) {
        std::cerr << "[DenseStreaming] Failed to read header.\n";
        return false;
    }
    
    // Validate magic
    if (std::memcmp(m_header.magic, DNSW_MAGIC, 4) != 0) {
        std::cerr << "[DenseStreaming] Invalid magic number in header.\n";
        return false;
    }
    
    // Validate version
    if (m_header.version != DNSW_VERSION) {
        std::cerr << "[DenseStreaming] Unsupported version: " << m_header.version 
                  << " (expected " << DNSW_VERSION << ")\n";
        return false;
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Header: " << m_header.num_layers << " layers, "
                  << m_header.num_groups << " groups, "
                  << m_header.group_size << " layers/group\n";
    }
    
    return true;
}

bool DenseWeightStreamingManager::read_tables() {
    std::ifstream file(m_config.weights_file_path, std::ios::binary);
    if (!file.is_open()) return false;
    
    // Seek past header to group table
    file.seekg(SECTOR_SIZE);  // Header is 4096 bytes
    
    // Read group table entries
    m_group_table.resize(m_header.num_groups);
    for (uint32_t i = 0; i < m_header.num_groups; ++i) {
        file.read(reinterpret_cast<char*>(&m_group_table[i]), sizeof(GroupTableEntry));
        if (!file) {
            std::cerr << "[DenseStreaming] Failed reading group table entry " << i << "\n";
            return false;
        }
    }
    
    // Seek to layer table (after group table sector)
    file.seekg(SECTOR_SIZE + SECTOR_SIZE);  // Header + GroupTable
    
    // Read layer table entries
    m_layer_table.resize(m_header.num_layers);
    for (uint32_t i = 0; i < m_header.num_layers; ++i) {
        file.read(reinterpret_cast<char*>(&m_layer_table[i]), sizeof(LayerTableEntry));
        if (!file) {
            std::cerr << "[DenseStreaming] Failed reading layer table entry " << i << "\n";
            return false;
        }
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Group table loaded (" << m_group_table.size() << " entries)\n";
        for (const auto& g : m_group_table) {
            std::cout << "  Group: layers " << g.first_layer << "-" 
                      << (g.first_layer + g.num_layers - 1)
                      << "  offset=" << g.file_offset 
                      << "  size=" << (g.raw_bytes / (1024.0*1024.0)) << " MB\n";
        }
    }
    
    return true;
}

bool DenseWeightStreamingManager::allocate_buffers() {
    // Find the largest group (determines buffer size)
    uint64_t max_group_bytes = 0;
    for (const auto& g : m_group_table) {
        max_group_bytes = std::max(max_group_bytes, g.aligned_bytes);
    }
    
    if (max_group_bytes == 0) {
        std::cerr << "[DenseStreaming] ERROR: All groups have zero size.\n";
        return false;
    }
    
    // Apply memory budget limit if configured
    if (m_config.max_buffer_bytes > 0 && max_group_bytes > m_config.max_buffer_bytes) {
        std::cerr << "[DenseStreaming] WARNING: Largest group (" 
                  << (max_group_bytes / (1024.0*1024.0)) << " MB) exceeds budget ("
                  << (m_config.max_buffer_bytes / (1024.0*1024.0)) << " MB)\n";
        // Still allocate — can't reduce group size at runtime
    }
    
    // Allocate buffers (2-4, configurable)
    m_num_buffers = static_cast<int>(std::min(m_config.num_buffers,
                                               static_cast<uint32_t>(MAX_BUFFERS)));
    for (int i = 0; i < m_num_buffers; ++i) {
        m_buffer_sizes[i] = max_group_bytes;

#ifdef OPENVINO_GPU_RUNTIME_AVAILABLE
        // Real integration: allocate via cldnn::engine → USM host memory
        // Create a flat layout (u8 × max_group_bytes) for raw byte buffer.
        cldnn::layout flat_layout(
            ov::PartialShape{1, static_cast<int64_t>(max_group_bytes)},
            cldnn::data_types::u8,
            cldnn::format::bfyx);
        m_buffers[i] = m_engine.allocate_memory(
            flat_layout, cldnn::allocation_type::usm_host, false);
        if (!m_buffers[i]) {
            std::cerr << "[DenseStreaming] ERROR: Failed to allocate USM buffer " << i
                      << " (" << (max_group_bytes / (1024.0*1024.0)) << " MB)\n";
            return false;
        }
        m_buffer_ptrs[i] = m_buffers[i]->buffer_ptr();
#else
        // Standalone: VirtualAlloc-based allocation
        m_buffer_ptrs[i] = m_engine.allocate_usm_host(max_group_bytes);
#endif

        if (!m_buffer_ptrs[i]) {
            std::cerr << "[DenseStreaming] ERROR: Failed to allocate USM buffer " << i
                      << " (" << (max_group_bytes / (1024.0*1024.0)) << " MB)\n";
            return false;
        }
        
        // Zero-fill (optional, helps with debugging)
        std::memset(m_buffer_ptrs[i], 0, max_group_bytes);
        
#ifdef _WIN32
        // Lock in physical memory if requested (prevent pagefile swap)
        if (m_config.lock_memory) {
            if (!VirtualLock(m_buffer_ptrs[i], max_group_bytes)) {
                std::cerr << "[DenseStreaming] WARNING: VirtualLock failed for buffer " << i
                          << " (error " << GetLastError() << "). Continuing without lock.\n";
            }
        }
#endif
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Allocated " << m_num_buffers << " × " 
                  << (max_group_bytes / (1024.0*1024.0)) << " MB USM buffers\n";
    }
    
    return true;
}

// ============================================================================
// Direct I/O Initialization (replaces DirectStorage)
// ============================================================================

bool DenseWeightStreamingManager::initialize_direct_io() {
#ifdef _WIN32
    // Get disk sector size for alignment verification
    std::wstring wide_path(m_config.weights_file_path.begin(), 
                           m_config.weights_file_path.end());
    
    DWORD sectors_per_cluster, bytes_per_sector, free_clusters, total_clusters;
    wchar_t root_path[4] = {wide_path[0], L':', L'\\', L'\0'};
    if (GetDiskFreeSpaceW(root_path, &sectors_per_cluster, &bytes_per_sector,
                          &free_clusters, &total_clusters)) {
        m_sector_size = bytes_per_sector;
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Disk sector size: " << m_sector_size << " bytes\n";
    }
    
    // Open one file handle per IO slot for concurrent async reads.
    // FILE_FLAG_NO_BUFFERING: bypass OS page cache (no memory pressure on 8 GB systems)
    // FILE_FLAG_OVERLAPPED: enable true async I/O (ReadFile returns immediately,
    //   NVMe controller performs DMA in background, Event signals completion)
    // This replaces the previous multi-threaded sync ReadFile approach.
    uint32_t num_handles = m_config.num_io_threads;
    m_io_handles.resize(num_handles, nullptr);
    
    for (uint32_t i = 0; i < num_handles; ++i) {
        HANDLE hFile = CreateFileW(
            wide_path.c_str(),
            GENERIC_READ,
            FILE_SHARE_READ,
            nullptr,
            OPEN_EXISTING,
            FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
            nullptr);
        
        if (hFile == INVALID_HANDLE_VALUE) {
            std::cerr << "[DenseStreaming] ERROR: Cannot open file handle #" << i
                      << " for Async I/O. Error: " << GetLastError() << "\n";
            shutdown_io();
            return false;
        }
        m_io_handles[i] = hFile;
    }
    
    // Create manual-reset events for async I/O completion notification.
    // Each event is paired with one OVERLAPPED struct for one async ReadFile.
    // Manual-reset events: must be explicitly reset after wait (vs auto-reset).
    m_io_events.resize(MAX_ASYNC_OPS, nullptr);
    for (uint32_t i = 0; i < MAX_ASYNC_OPS; ++i) {
        HANDLE hEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
        if (!hEvent) {
            std::cerr << "[DenseStreaming] ERROR: Cannot create event #" << i
                      << ". Error: " << GetLastError() << "\n";
            shutdown_io();
            return false;
        }
        m_io_events[i] = hEvent;
    }
    
    // Allocate OVERLAPPED structs (opaque storage, sizeof(OVERLAPPED) = 32 on x64)
    m_overlapped_storage = new OVERLAPPED[MAX_ASYNC_OPS];
    memset(m_overlapped_storage, 0, sizeof(OVERLAPPED) * MAX_ASYNC_OPS);
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Opened " << num_handles
                  << " Async I/O file handles + " << MAX_ASYNC_OPS << " events\n";
    }
    
    return true;
    
#else
    // Linux: open file descriptor (Direct I/O with O_DIRECT)
    // TODO: Implement O_DIRECT + io_uring for Linux
    std::cerr << "[DenseStreaming] Direct I/O not yet implemented on this platform.\n";
    return false;
#endif
}

void DenseWeightStreamingManager::shutdown_io() {
#ifdef _WIN32
    // Wait for any in-flight async IO before closing handles
    if (m_async_in_progress.load()) {
        wait_async_load();
    }
    
    // Close file handles
    for (auto& handle : m_io_handles) {
        if (handle && handle != INVALID_HANDLE_VALUE) {
            CloseHandle(static_cast<HANDLE>(handle));
        }
    }
    m_io_handles.clear();
    
    // Close event handles
    for (auto& evt : m_io_events) {
        if (evt) {
            CloseHandle(static_cast<HANDLE>(evt));
        }
    }
    m_io_events.clear();
    
    // Free OVERLAPPED storage
    if (m_overlapped_storage) {
        delete[] static_cast<OVERLAPPED*>(m_overlapped_storage);
        m_overlapped_storage = nullptr;
    }
    
    m_async_num_ops = 0;
    m_async_in_progress.store(false);
#endif
}

// ============================================================================
// Runtime: Buffer Management
// ============================================================================

int DenseWeightStreamingManager::find_free_buffer(uint32_t exclude_group) const {
    // Priority: prefer empty buffer (-1), then oldest non-active non-excluded
    int best = -1;
    for (int i = 0; i < m_num_buffers; ++i) {
        if (i == m_active_buffer) continue;  // don't touch GPU's active buffer
        if (m_buffer_group[i] == -1) return i;  // empty → immediate use
        if (exclude_group != UINT32_MAX && 
            m_buffer_group[i] == static_cast<int32_t>(exclude_group)) continue;
        best = i;  // reusable (overwrite stale data)
    }
    // Fallback: if all non-active buffers hold needed groups, pick any non-active
    if (best < 0) {
        for (int i = 0; i < m_num_buffers; ++i) {
            if (i != m_active_buffer) { best = i; break; }
        }
    }
    return best;
}

// ============================================================================
// Runtime: Weight Loading
// ============================================================================

uint64_t DenseWeightStreamingManager::group_bytes(uint32_t group_idx) const {
    if (group_idx >= m_group_table.size()) return 0;
    return m_group_table[group_idx].raw_bytes;
}

bool DenseWeightStreamingManager::load_group(uint32_t group_idx) {
    if (!m_initialized || group_idx >= m_header.num_groups) return false;
    
    // Wait for any ongoing async IO to complete first
    if (m_async_in_progress.load()) {
        wait_async_load();
    }
    
    // Check if this group is already loaded in one of the buffers
    for (int i = 0; i < m_num_buffers; ++i) {
        if (m_buffer_group[i] == static_cast<int32_t>(group_idx)) {
            if (m_config.debug_logging) {
                std::cout << "[DenseStreaming] load_group(" << group_idx 
                          << ") — already in buffer[" << i << "], skipping IO\n";
            }
            return true;  // Already loaded, wait_for_load will just swap
        }
    }
    
    // Determine which buffer to load into (find a free non-active buffer)
    int load_buffer = find_free_buffer();
    
    const auto& group = m_group_table[group_idx];
    void* dest = m_buffer_ptrs[load_buffer];
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] load_group(" << group_idx 
                  << ") → buffer[" << load_buffer << "]"
                  << "  offset=" << group.file_offset 
                  << "  size=" << (group.aligned_bytes / (1024.0*1024.0)) << " MB\n";
    }
    
    // Synchronous async I/O: issue + wait (for cold start / forced sync load)
    auto t0 = std::chrono::high_resolution_clock::now();
    m_async_target_buffer = load_buffer;
    m_async_target_group = group_idx;
    bool success = start_async_load(group.file_offset, group.aligned_bytes, dest);
    if (success) {
        success = wait_async_load();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    if (success) {
        m_buffer_group[load_buffer] = static_cast<int32_t>(group_idx);
        
        // Record timing
        if (m_config.enable_timing) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            m_stats.total_loads.fetch_add(1);
            m_stats.total_bytes_loaded.fetch_add(group.aligned_bytes);
            atomic_add_double(m_stats.total_load_time_ms, elapsed_ms);
            
            if (group_idx < m_stats.group_load_time_ms.size()) {
                m_stats.group_load_time_ms[group_idx] = elapsed_ms;
            }
            
            double throughput_gbps = (group.aligned_bytes / (1024.0*1024.0*1024.0)) 
                                   / (elapsed_ms / 1000.0);
            double current_peak = m_stats.peak_throughput_gbps.load();
            while (throughput_gbps > current_peak) {
                m_stats.peak_throughput_gbps.compare_exchange_weak(current_peak, throughput_gbps);
            }
            
            if (m_config.debug_logging) {
                std::cout << "[DenseStreaming] load_group(" << group_idx 
                          << ") completed: " << elapsed_ms << " ms, "
                          << throughput_gbps << " GB/s\n";
            }
        }
    }
    
    return success;
}

bool DenseWeightStreamingManager::wait_for_load(uint32_t group_idx) {
    if (!m_initialized) return false;
    
    // Wait for async IO if this is the prefetched group
    if (m_async_in_progress.load() && m_async_target_group == group_idx) {
        if (!wait_async_load()) {
            std::cerr << "[DenseStreaming] wait_for_load(" << group_idx 
                      << "): async IO failed!\n";
            return false;
        }
    }
    
    // Find which buffer has this group
    int target_buffer = -1;
    for (int i = 0; i < m_num_buffers; ++i) {
        if (m_buffer_group[i] == static_cast<int32_t>(group_idx)) {
            target_buffer = i;
            break;
        }
    }
    
    if (target_buffer < 0) {
        std::cerr << "[DenseStreaming] wait_for_load(" << group_idx 
                  << "): group not loaded in any buffer!\n";
        return false;
    }
    
    // Swap: make this buffer active (GPU reads from it)
    m_active_buffer = target_buffer;
    
    return true;
}

bool DenseWeightStreamingManager::prefetch_next_group(uint32_t next_group_idx) {
    if (!m_initialized || next_group_idx >= m_header.num_groups) return false;
    
    // Wait for any previous async IO to complete
    if (m_async_in_progress.load()) {
        wait_async_load();
    }
    
    // Check if already loaded
    for (int i = 0; i < m_num_buffers; ++i) {
        if (m_buffer_group[i] == static_cast<int32_t>(next_group_idx)) {
            if (m_config.debug_logging) {
                std::cout << "[DenseStreaming] prefetch_next_group(" << next_group_idx
                          << ") — already in buffer[" << i << "], skipping\n";
            }
            return true;
        }
    }
    
    m_prefetch_group_idx = next_group_idx;
    
    // Load into a free buffer (does not conflict with active buffer GPU reads)
    int prefetch_buffer = find_free_buffer();
    if (prefetch_buffer < 0) {
        std::cerr << "[DenseStreaming] prefetch: no free buffer available!\n";
        return false;
    }
    const auto& group = m_group_table[next_group_idx];
    void* dest = m_buffer_ptrs[prefetch_buffer];
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] prefetch_next_group(" << next_group_idx
                  << ") → buffer[" << prefetch_buffer << "] (async IO)\n";
    }
    
    // Issue async ReadFile calls — returns IMMEDIATELY.
    // NVMe controller performs DMA in background while GPU computes.
    // No thread creation/destruction overhead.
    m_async_target_buffer = prefetch_buffer;
    m_async_target_group = next_group_idx;
    m_async_start_time = std::chrono::high_resolution_clock::now();
    
    bool ok = start_async_load(group.file_offset, group.aligned_bytes, dest);
    if (!ok) {
        std::cerr << "[DenseStreaming] prefetch: start_async_load failed!\n";
        m_async_in_progress.store(false);
        return false;
    }
    
    return true;
}

bool DenseWeightStreamingManager::is_prefetch_complete(uint32_t group_idx) const {
    if (!m_async_in_progress.load()) return true;  // No async IO in flight
    return is_async_load_complete();
}

bool DenseWeightStreamingManager::load_group_sync(uint32_t group_idx) {
    if (!load_group(group_idx)) return false;
    return wait_for_load(group_idx);
}

// ============================================================================
// Buffer Access
// ============================================================================

void* DenseWeightStreamingManager::get_group_buffer_ptr(uint32_t group_idx) const {
    // Find which buffer holds this group
    for (int i = 0; i < m_num_buffers; ++i) {
        if (m_buffer_group[i] == static_cast<int32_t>(group_idx)) {
            return m_buffer_ptrs[i];
        }
    }
    return nullptr;  // Group not loaded
}

cldnn::memory_ptr DenseWeightStreamingManager::get_group_memory(uint32_t group_idx) const {
    // In real integration, return the cldnn::memory::ptr for the buffer
    // For now, return nullptr (stub)
    return nullptr;
}

void* DenseWeightStreamingManager::get_layer_buffer_ptr(uint32_t layer_idx) const {
    if (layer_idx >= m_header.num_layers) return nullptr;
    
    // Find which group this layer belongs to
    uint32_t group_idx = layer_idx / m_header.group_size;
    
    // Get the group's buffer
    void* group_ptr = get_group_buffer_ptr(group_idx);
    if (!group_ptr) return nullptr;
    
    // Add offset within group to get layer's data
    uint64_t offset = m_layer_table[layer_idx].offset_in_group;
    return static_cast<uint8_t*>(group_ptr) + offset;
}

// ============================================================================
// Weight Pointer Swapping — Integration with OpenVINO compiled model
// ============================================================================
//
// Architecture Overview:
// =====================
// In a compiled cldnn::network, weights are stored as "data" primitive instances.
// Each data primitive has _outputs[0] = memory::ptr pointing to USM memory
// containing the weight tensor data.
//
// When a compute primitive (e.g. fully_connected) executes:
//   1. get_arguments() collects input_memory_ptr(i) for all dependencies
//   2. input_memory_ptr(i) → dep_memory_ptr(i) → data_inst._outputs[0]
//   3. stream.set_arguments() calls clSetKernelArgMemPointerINTEL() with the USM ptr
//   4. GPU kernel reads weights from that USM address
//
// To swap weight pointers, we:
//   1. Create a memory::ptr wrapping our streaming buffer sub-region
//      via engine.attach_memory(layout, ptr) → simple_attached_memory
//   2. Replace data_inst._outputs[0] with the new memory
//   3. Call network.set_arguments() to re-bind all kernel arguments
//
// IMPORTANT: primitive_inst::set_output_memory() does NOT work for weights
// because data nodes have is_constant()==true, and the method COPIES data
// instead of swapping the pointer. We need direct _outputs[0] assignment.
//
// Required OpenVINO source modification (one line in primitive_inst.h):
//   Add to public section of class primitive_inst:
//     void force_set_output_memory(memory::ptr mem, size_t idx = 0) {
//         _outputs[idx] = std::move(mem);
//     }
//
// This is the same pattern used by MoE OTD for expert weight swapping,
// but applied to entire decoder layers instead of individual experts.

// Helper: parse "layers.N" from an OpenVINO constant node name
// Returns layer index (0-41) or -1 if not a decoder layer constant
static int32_t parse_layer_index_from_name(const std::string& name) {
    // Pattern: "...layers.N..." where N is the layer index
    // Examples:
    //   "__module.model.layers.5.self_attn.q_proj.weight"
    //   "model.layers.12.mlp.gate_proj.weight"
    //   "aten::_to_copy/layers.0/self_attn/Constant_42"
    
    // Search for "layers." followed by digits
    const std::string marker = "layers.";
    size_t pos = name.find(marker);
    if (pos == std::string::npos) return -1;
    
    pos += marker.length();
    if (pos >= name.size() || !std::isdigit(name[pos])) return -1;
    
    int32_t layer_idx = 0;
    while (pos < name.size() && std::isdigit(name[pos])) {
        layer_idx = layer_idx * 10 + (name[pos] - '0');
        ++pos;
    }
    
    // Validate range (Gemma4 has 42 decoder layers)
    if (layer_idx < 0 || layer_idx >= 42) return -1;
    
    return layer_idx;
}

bool DenseWeightStreamingManager::build_weight_mapping(void* network_ptr) {
#ifdef OPENVINO_GPU_RUNTIME_AVAILABLE
    // ====================================================================
    // REAL INTEGRATION: with actual cldnn::network
    // ====================================================================
    auto* net = static_cast<cldnn::network*>(network_ptr);
    m_weight_mappings.clear();
    
    // Determine the streamed layer range from the group table
    // group_table[0].first_layer is the first model layer index (e.g. 5)
    // m_header.num_layers is how many layers are in the packed file (e.g. 32)
    const uint32_t first_streamed = m_group_table[0].first_layer;
    const uint32_t last_streamed = first_streamed + m_header.num_layers - 1;
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] build_weight_mapping: scanning network\n"
                  << "  Streamed range: model layers " << first_streamed 
                  << "-" << last_streamed << " (" << m_header.num_layers << " layers)\n";
    }
    
    // Iterate all primitives in the compiled network
    auto all_ids = net->get_all_primitive_ids();
    
    if (m_config.debug_logging) {
        std::cout << "  Total primitives: " << all_ids.size() << "\n";
    }
    
    // Group tensors by packed layer index (0..num_layers-1) to compute offset_in_layer
    struct TensorInfo {
        std::string name;
        std::string prim_id;
        uint64_t size_bytes;
    };
    std::vector<std::vector<TensorInfo>> per_layer_tensors(m_header.num_layers);
    
    uint32_t total_constants = 0;
    uint32_t decoder_constants = 0;
    uint32_t streamed_constants = 0;
    
    for (const auto& id : all_ids) {
        // Some primitive IDs (e.g. "_optimized_") exist in the ID list
        // but throw when accessed via get_primitive(). Protect against this.
        cldnn::primitive_inst* prim = nullptr;
        try {
            prim = net->get_primitive(id).get();
        } catch (...) {
            continue;
        }
        if (!prim) continue;
        
        // Check if this is a data (constant) primitive
        // data primitives are constants with is_constant()==true and no inputs
        if (!prim->is_constant()) continue;
        if (prim->inputs_memory_count() > 0) continue;
        ++total_constants;
        
        // Parse layer index from name
        int32_t layer_idx = parse_layer_index_from_name(id);
        if (layer_idx < 0) continue;
        ++decoder_constants;
        
        // Only include layers in the streamed range
        if (layer_idx < static_cast<int32_t>(first_streamed) || 
            layer_idx > static_cast<int32_t>(last_streamed)) {
            continue;
        }
        ++streamed_constants;
        
        // Get memory info
        auto mem = prim->output_memory_ptr(0);
        if (!mem) continue;
        
        TensorInfo info;
        info.name = id;
        info.prim_id = id;
        info.size_bytes = mem->size();
        
        // Map model layer index to packed index: layer 5 -> packed 0, etc.
        uint32_t packed_idx = static_cast<uint32_t>(layer_idx) - first_streamed;
        per_layer_tensors[packed_idx].push_back(std::move(info));
    }
    
    if (m_config.debug_logging) {
        std::cout << "  Constants found: " << total_constants << " total, "
                  << decoder_constants << " decoder, "
                  << streamed_constants << " in streamed range\n";
        // Enhanced debug: dump FC dependencies + large unnamed constants
        {
            std::string dbg_path = m_config.weights_file_path;
            auto dpos = dbg_path.rfind('\\');
            if (dpos == std::string::npos) dpos = dbg_path.rfind('/');
            if (dpos != std::string::npos) dbg_path = dbg_path.substr(0, dpos+1);
            else dbg_path = "";
            dbg_path += "debug_primitives.txt";
            FILE* dbg = fopen(dbg_path.c_str(), "w");
            if (dbg) {
                // Section 1: per-layer data primitives (existing)
                for (uint32_t l = 0; l < std::min<uint32_t>(2, m_header.num_layers); ++l) {
                    fprintf(dbg, "--- Packed layer %u (model layer %u) ---\n", l, first_streamed + l);
                    for (const auto& t : per_layer_tensors[l]) {
                        fprintf(dbg, "  %s (%llu bytes)\n", t.name.c_str(), (unsigned long long)t.size_bytes);
                    }
                }
                
                // Section 2: ALL constants > 100KB that DON'T have "layers." in name
                fprintf(dbg, "\n=== Large constants WITHOUT 'layers.' in name (>100KB) ===\n");
                uint64_t unnamed_total = 0;
                uint32_t unnamed_count = 0;
                for (const auto& id : all_ids) {
                    cldnn::primitive_inst* prim = nullptr;
                    try { prim = net->get_primitive(id).get(); } catch (...) { continue; }
                    if (!prim || !prim->is_constant() || prim->inputs_memory_count() > 0) continue;
                    if (id.find("layers.") != std::string::npos) continue; // skip ones we already found
                    auto mem = prim->output_memory_ptr(0);
                    if (!mem) continue;
                    uint64_t sz = mem->size();
                    if (sz >= 100*1024) {
                        fprintf(dbg, "  [%llu bytes = %.2f MB] %s\n", 
                                (unsigned long long)sz, sz / (1024.0*1024.0), id.c_str());
                        unnamed_total += sz;
                        ++unnamed_count;
                    }
                }
                fprintf(dbg, "  Total: %u constants, %llu bytes = %.1f MB\n",
                        unnamed_count, (unsigned long long)unnamed_total, unnamed_total / (1024.0*1024.0));
                
                // Section 3: Large constants WITH "layers." but inputs_memory_count > 0 
                fprintf(dbg, "\n=== Constants with 'layers.' but inputs_memory_count > 0 ===\n");
                for (const auto& id : all_ids) {
                    if (id.find("layers.") == std::string::npos) continue;
                    cldnn::primitive_inst* prim = nullptr;
                    try { prim = net->get_primitive(id).get(); } catch (...) { continue; }
                    if (!prim || !prim->is_constant()) continue;
                    if (prim->inputs_memory_count() == 0) continue; // already counted
                    auto mem = prim->output_memory_ptr(0);
                    uint64_t sz = mem ? mem->size() : 0;
                    fprintf(dbg, "  [inputs=%zu, %llu bytes] %s\n", 
                            prim->inputs_memory_count(), (unsigned long long)sz, id.c_str());
                }
                
                // Section 4: FC dependencies for layer 5
                fprintf(dbg, "\n=== FC primitive dependencies for layers.5 ===\n");
                std::string fc_marker = "fullyconnectedcompressed:__module.model.language_model.layers.5.";
                for (const auto& id : all_ids) {
                    if (id.find(fc_marker) == std::string::npos) continue;
                    cldnn::primitive_inst* fc = nullptr;
                    try { fc = net->get_primitive(id).get(); } catch (...) { continue; }
                    if (!fc) continue;
                    fprintf(dbg, "\nFC: %s\n", id.c_str());
                    fprintf(dbg, "  is_dynamic: %d\n", fc->is_dynamic() ? 1 : 0);
                    auto& deps = fc->dependencies();
                    fprintf(dbg, "  dependencies count: %zu\n", deps.size());
                    for (size_t d = 0; d < deps.size(); ++d) {
                        auto* dep_inst = deps[d].first;
                        int32_t dep_out_idx = deps[d].second;
                        auto dep_mem = dep_inst->output_memory_ptr(dep_out_idx);
                        uint64_t dep_sz = dep_mem ? dep_mem->size() : 0;
                        fprintf(dbg, "  dep[%zu]: (out_idx=%d) [%llu bytes = %.2f MB] is_const=%d inputs=%zu id=%s\n",
                                d, dep_out_idx, (unsigned long long)dep_sz, dep_sz/(1024.0*1024.0),
                                dep_inst->is_constant() ? 1 : 0,
                                dep_inst->inputs_memory_count(),
                                dep_inst->id().c_str());
                    }
                }
                
                fclose(dbg);
                std::cout << "  Debug written to: " << dbg_path << "\n";
            }
        }
    }
    
    // Build mapping with computed offsets
    // The offset_in_layer must match the packing order in pack_dense_weights.py
    // pack_dense_weights.py iterates layer_constants[layer_idx] in the order
    // returned by find_layer_constants(), which uses the OV model's natural
    // constant iteration order (op graph traversal order).
    //
    // IMPORTANT: The tensor order in the network may differ from the packing
    // order. To ensure correctness, we should either:
    //   a) Sort by name in both packer and here, OR
    //   b) Use the JSON metadata file which records the exact packing order
    //
    // For robustness, prefer build_weight_mapping_from_json() which uses
    // the authoritative packing order from the JSON metadata.
    
    for (uint32_t layer = 0; layer < m_header.num_layers; ++layer) {
        // Sort by name for deterministic ordering
        auto& tensors = per_layer_tensors[layer];
        std::sort(tensors.begin(), tensors.end(),
                  [](const TensorInfo& a, const TensorInfo& b) {
                      return a.name < b.name;
                  });
        
        uint64_t offset = 0;
        for (const auto& t : tensors) {
            WeightTensorMapping mapping;
            mapping.layer_idx = layer;
            mapping.tensor_name = t.name;
            mapping.primitive_id = t.prim_id;
            mapping.offset_in_layer = offset;
            mapping.size_bytes = t.size_bytes;
            m_weight_mappings.push_back(std::move(mapping));
            
            offset += t.size_bytes;
        }
    }

    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Built weight mapping: "
                  << m_weight_mappings.size() << " tensors across "
                  << m_header.num_layers << " layers\n";
    }
    
    return !m_weight_mappings.empty();
    
#else
    // ====================================================================
    // STANDALONE STUB: simulate mapping for testing
    // ====================================================================
    (void)network_ptr;
    m_weight_mappings.clear();
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] build_weight_mapping: "
                  << "standalone mode — generating simulated mapping\n";
    }
    
    // Generate simulated mapping from layer table
    for (uint32_t layer = 0; layer < m_header.num_layers; ++layer) {
        const auto& layer_info = m_layer_table[layer];
        uint64_t offset = 0;
        
        // Create one mapping entry per tensor in the layer
        for (uint32_t t = 0; t < layer_info.num_tensors; ++t) {
            WeightTensorMapping mapping;
            mapping.layer_idx = layer;
            mapping.tensor_name = "model.layers." + std::to_string(layer) 
                                + ".tensor_" + std::to_string(t);
            mapping.primitive_id = mapping.tensor_name;
            mapping.offset_in_layer = offset;
            // Divide layer size evenly among tensors (approximation)
            mapping.size_bytes = layer_info.size_bytes / layer_info.num_tensors;
            offset += mapping.size_bytes;
            
            m_weight_mappings.push_back(std::move(mapping));
        }
    }
    
    std::cout << "[DenseStreaming] Simulated mapping: " 
              << m_weight_mappings.size() << " tensors\n";
    return !m_weight_mappings.empty();
#endif
}

bool DenseWeightStreamingManager::build_weight_mapping_from_json(
    const std::string& json_metadata_path, void* network_ptr) {
    // ====================================================================
    // FC-SCAN APPROACH: Build weight mapping from FC primitive dependencies
    // ====================================================================
    //
    // Strategy:
    //   1. Parse JSON to build per-layer offset table (tensor name → offset)
    //   2. Scan compiled network for fullyconnectedcompressed primitives
    //   3. For each dynamic FC in streamed layers:
    //      - Extract component name from FC ID (e.g. "mlp.gate_proj")
    //      - Find dep[1] (weight), dep[2] (scale), dep[3] (zp)
    //      - Match dep sizes against JSON tensor sizes
    //      - If match: create mapping entry with correct byte offset
    //
    // Why FC-scan instead of data-primitive-scan?
    //   - INT4 weight deps have names like "Constant_135971" (no "layers.")
    //   - scale/zp deps have reorder names that don't match JSON names
    //   - Only the FC primitive itself has "layers.N" in its name
    //   - Dynamic FCs use _reordered_weights_cache, not dep_memory_ptr(1)
    //     directly, so we need to update the cache too
    
    // Step 1: Parse JSON for per-layer tensor offset tables
    std::ifstream json_file(json_metadata_path);
    if (!json_file.is_open()) {
        std::cerr << "[DenseStreaming] Cannot open JSON metadata: " 
                  << json_metadata_path << "\n";
        return false;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(json_file)),
                              std::istreambuf_iterator<char>());
    json_file.close();
    
    const uint32_t first_streamed = m_group_table[0].first_layer;
    
    // Build lookup: per packed layer, map partial name → (offset, size)
    // e.g. "mlp.gate_proj.weight" → (offset=72, size=13107200)
    struct TensorOffset {
        uint64_t offset_in_layer;
        uint64_t size_bytes;
        std::string full_name;
    };
    // Key: partial name after "layers.N." (e.g. "mlp.gate_proj.weight")
    // Value: offset info
    std::vector<std::unordered_map<std::string, TensorOffset>> layer_offset_tables(
        m_header.num_layers);
    
    for (uint32_t layer = 0; layer < m_header.num_layers; ++layer) {
        uint32_t model_layer = first_streamed + layer;
        std::string layer_marker = "\"layer_idx\": " + std::to_string(model_layer) + ",";
        size_t layer_pos = json_content.find(layer_marker);
        if (layer_pos == std::string::npos) continue;
        
        size_t tensors_pos = json_content.find("\"tensors\":", layer_pos);
        if (tensors_pos == std::string::npos) continue;
        size_t array_start = json_content.find('[', tensors_pos);
        if (array_start == std::string::npos) continue;
        
        uint64_t offset_in_layer = 0;
        size_t search_pos = array_start;
        
        // Prefix to strip: "self.model.language_model.layers.N."
        std::string prefix = "self.model.language_model.layers." 
                           + std::to_string(model_layer) + ".";
        
        while (true) {
            size_t obj_start = json_content.find('{', search_pos);
            if (obj_start == std::string::npos) break;
            size_t next_bracket = json_content.find(']', search_pos);
            if (next_bracket != std::string::npos && next_bracket < obj_start) break;
            size_t obj_end = json_content.find('}', obj_start);
            if (obj_end == std::string::npos) break;
            
            std::string obj = json_content.substr(obj_start, obj_end - obj_start + 1);
            
            std::string tensor_name;
            size_t name_pos = obj.find("\"name\":");
            if (name_pos != std::string::npos) {
                size_t q1 = obj.find('"', name_pos + 7);
                size_t q2 = obj.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    tensor_name = obj.substr(q1 + 1, q2 - q1 - 1);
            }
            
            uint64_t size_bytes = 0;
            size_t sz_pos = obj.find("\"size_bytes\":");
            if (sz_pos != std::string::npos) {
                size_t ns = sz_pos + 13;
                while (ns < obj.size() && !std::isdigit(obj[ns])) ++ns;
                if (ns < obj.size()) size_bytes = std::stoull(obj.substr(ns));
            }
            
            if (!tensor_name.empty() && size_bytes > 0) {
                // Extract partial name (after "self.model.language_model.layers.N.")
                std::string partial;
                if (tensor_name.find(prefix) == 0) {
                    partial = tensor_name.substr(prefix.size());
                }
                if (!partial.empty()) {
                    layer_offset_tables[layer][partial] = {offset_in_layer, size_bytes, tensor_name};
                }
                offset_in_layer += size_bytes;
            }
            search_pos = obj_end + 1;
        }
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] JSON offset tables built: "
                  << layer_offset_tables[0].size() << " entries for layer 0\n";
    }
    
    // Step 2: Scan FC primitives and build mapping
    m_weight_mappings.clear();
    
#ifdef OPENVINO_GPU_RUNTIME_AVAILABLE
    if (!network_ptr) return false;
    auto* net = static_cast<cldnn::network*>(network_ptr);
    auto all_ids = net->get_all_primitive_ids();
    
    uint32_t fc_found = 0, weight_mapped = 0, scale_mapped = 0, zp_mapped = 0;
    uint32_t weight_skipped = 0, scale_skipped = 0;
    
    // Pattern: "fullyconnectedcompressed:__module.model.language_model.layers.N."
    const std::string fc_prefix = "fullyconnectedcompressed:__module.model.language_model.layers.";
    
    for (const auto& id : all_ids) {
        if (id.find(fc_prefix) != 0) continue;  // Not an FC primitive
        
        // Parse layer index from FC name
        size_t layers_dot = id.find("layers.");
        if (layers_dot == std::string::npos) continue;
        size_t num_start = layers_dot + 7;
        int32_t model_layer = 0;
        size_t p = num_start;
        while (p < id.size() && std::isdigit(id[p])) {
            model_layer = model_layer * 10 + (id[p] - '0');
            ++p;
        }
        
        // Check streamed range
        if (model_layer < static_cast<int32_t>(first_streamed) ||
            model_layer > static_cast<int32_t>(first_streamed + m_header.num_layers - 1))
            continue;
        
        uint32_t packed_layer = static_cast<uint32_t>(model_layer) - first_streamed;
        
        // Extract component: text between "layers.N." and "/" 
        // e.g. "layers.5.mlp.gate_proj/ov_ext..." → "mlp.gate_proj"
        if (p >= id.size() || id[p] != '.') continue;
        ++p;  // skip the dot after layer number
        size_t slash = id.find('/', p);
        if (slash == std::string::npos) continue;
        std::string component = id.substr(p, slash - p);
        
        // Get FC primitive instance
        cldnn::primitive_inst* fc_inst = nullptr;
        try { fc_inst = net->get_primitive(id).get(); } catch (...) { continue; }
        if (!fc_inst || !fc_inst->is_dynamic()) continue;
        
        auto& deps = fc_inst->dependencies();
        if (deps.size() < 4) continue;  // Need at least: input, weight, scale, zp
        
        ++fc_found;
        
        auto& offset_table = layer_offset_tables[packed_layer];
        
        // === dep[1]: Weight ===
        auto* weight_dep = deps[1].first;
        auto weight_mem = weight_dep->output_memory_ptr(deps[1].second);
        uint64_t weight_size = weight_mem ? weight_mem->size() : 0;
        
        std::string weight_key = component + ".weight";
        auto wit = offset_table.find(weight_key);
        if (wit != offset_table.end() && wit->second.size_bytes == weight_size) {
            WeightTensorMapping mapping;
            mapping.layer_idx = packed_layer;
            mapping.tensor_name = wit->second.full_name;
            mapping.primitive_id = weight_dep->id();
            mapping.offset_in_layer = wit->second.offset_in_layer;
            mapping.size_bytes = weight_size;
            mapping.is_fc_weight = true;
            mapping.fc_primitive_id = id;
            m_weight_mappings.push_back(std::move(mapping));
            ++weight_mapped;
        } else {
            ++weight_skipped;
            if (m_config.debug_logging && packed_layer == 0) {
                std::cout << "[DenseStreaming] Skip weight: " << component
                          << " dep_size=" << weight_size;
                if (wit != offset_table.end())
                    std::cout << " json_size=" << wit->second.size_bytes;
                else
                    std::cout << " (no JSON match for '" << weight_key << "')";
                std::cout << "\n";
            }
        }
        
        // === dep[2] (Scale) and dep[3] (ZP): NOT mapped ===
        // Scales and zero points go through propagate_constants reorder
        // in the GPU compiler. Their compiled format differs from the raw
        // IR data in our binary file. Only unreordered FC weights (dep[1])
        // can be safely swapped from the binary.
        // Scale/ZP primitives remain pinned in GPU memory (small: ~5 MB/layer).
    }
    
    if (m_config.debug_logging) {
        uint64_t total_bytes = 0;
        for (const auto& m : m_weight_mappings) total_bytes += m.size_bytes;
        std::cout << "[DenseStreaming] FC scan results:\n"
                  << "  FC primitives found: " << fc_found << "\n"
                  << "  Weights mapped: " << weight_mapped 
                  << " (skipped: " << weight_skipped << ")\n"
                  << "  Scales mapped:  " << scale_mapped
                  << " (skipped: " << scale_skipped << ")\n"
                  << "  ZPs mapped:     " << zp_mapped << "\n"
                  << "  Total mapping:  " << m_weight_mappings.size() 
                  << " tensors, " << (total_bytes / (1024.0*1024.0)) << " MB\n";

        // Debug: dump mapping to file
        std::string dbg_path = m_config.weights_file_path;
        auto dp = dbg_path.rfind('.');
        if (dp != std::string::npos) dbg_path = dbg_path.substr(0, dp);
        dbg_path += "_fc_mapping.txt";
        FILE* dbg = fopen(dbg_path.c_str(), "w");
        if (dbg) {
            fprintf(dbg, "=== FC Weight Mapping (%zu entries) ===\n", m_weight_mappings.size());
            for (const auto& m : m_weight_mappings) {
                fprintf(dbg, "  layer=%u %s offset=%llu size=%llu %s prim=%s fc=%s\n",
                    m.layer_idx, m.tensor_name.c_str(),
                    (unsigned long long)m.offset_in_layer,
                    (unsigned long long)m.size_bytes,
                    m.is_fc_weight ? "[FC_WEIGHT]" : "[DATA]",
                    m.primitive_id.c_str(),
                    m.fc_primitive_id.c_str());
            }
            fclose(dbg);
            std::cout << "[DenseStreaming] FC mapping written to: " << dbg_path << "\n";
        }
    }
    
#else
    (void)network_ptr;
#endif
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] Built weight mapping from JSON: "
                  << m_weight_mappings.size() << " tensors\n";
        
        // Print first few entries
        for (size_t i = 0; i < std::min<size_t>(5, m_weight_mappings.size()); ++i) {
            const auto& m = m_weight_mappings[i];
            std::cout << "  [" << m.layer_idx << "] " << m.tensor_name
                      << " @ offset " << m.offset_in_layer 
                      << " (" << m.size_bytes << " bytes)"
                      << (m.is_fc_weight ? " [FC_WEIGHT]" : "") << "\n";
        }
        if (m_weight_mappings.size() > 5) {
            std::cout << "  ... and " << (m_weight_mappings.size() - 5) << " more\n";
        }
    }
    
    return !m_weight_mappings.empty();
}

bool DenseWeightStreamingManager::swap_weight_pointers(uint32_t group_idx, void* network_ptr) {
    if (!m_initialized) return false;
    if (group_idx >= m_header.num_groups) return false;
    
    void* group_ptr = get_group_buffer_ptr(group_idx);
    if (!group_ptr) {
        std::cerr << "[DenseStreaming] swap_weight_pointers: group " << group_idx 
                  << " not loaded!\n";
        return false;
    }
    
    const auto& group = m_group_table[group_idx];

#ifdef OPENVINO_GPU_RUNTIME_AVAILABLE
    // ====================================================================
    // REAL INTEGRATION: swap data primitive outputs in cldnn::network
    // ====================================================================
    //
    // Prerequisites:
    //   1. build_weight_mapping() or build_weight_mapping_from_json() called
    //   2. wait_for_load(group_idx) completed
    //   3. primitive_inst.h has force_set_output_memory() added (see comment above)
    //
    // Process:
    //   For each weight tensor in this group's layers:
    //     1. Compute USM pointer: group_buffer + layer_offset + tensor_offset
    //     2. Create memory::ptr wrapping that pointer (zero-copy)
    //     3. Replace data_inst._outputs[0]
    //   Then: re-set kernel arguments for all affected compute primitives
    
    auto* net = static_cast<cldnn::network*>(network_ptr);
    uint32_t swapped = 0;
    
    // Find which buffer holds this group (for create_subbuffer)
    int buf_idx = -1;
    for (int i = 0; i < m_num_buffers; ++i) {
        if (m_buffer_group[i] == static_cast<int32_t>(group_idx)) {
            buf_idx = i;
            break;
        }
    }
    if (buf_idx < 0 || !m_buffers[buf_idx]) {
        std::cerr << "[DenseStreaming] swap_weight_pointers: group " << group_idx 
                  << " not found in any buffer!\n";
        return false;
    }
    
    // Convert group's model layer range to packed layer range
    // group.first_layer is model index (e.g. 5), mapping.layer_idx is packed (0-based)
    const uint32_t first_streamed = m_group_table[0].first_layer;
    const uint32_t packed_first = group.first_layer - first_streamed;
    const uint32_t packed_end = packed_first + group.num_layers;
    
    for (const auto& mapping : m_weight_mappings) {
        // Only process tensors belonging to this group's layers (packed indices)
        if (mapping.layer_idx < packed_first ||
            mapping.layer_idx >= packed_end) {
            continue;
        }
        
        // Calculate the byte offset for this tensor within our streaming buffer
        uint64_t layer_offset = m_layer_table[mapping.layer_idx].offset_in_group;
        uint64_t tensor_byte_offset = layer_offset + mapping.offset_in_layer;
        
        // Get the data primitive (dep[1] for FC weights, or scale/zp data prim)
        auto data_inst = net->get_primitive(mapping.primitive_id);
        if (!data_inst) {
            if (m_config.debug_logging) {
                std::cerr << "[DenseStreaming] WARNING: primitive '" 
                          << mapping.primitive_id << "' not found\n";
            }
            continue;
        }
        
        auto original_layout = data_inst->output_memory_ptr(0)->get_layout();
        
        // Create a USM sub-buffer from our streaming buffer (zero-copy).
        auto new_mem = m_engine.create_subbuffer(
            *m_buffers[buf_idx], original_layout, 
            static_cast<size_t>(tensor_byte_offset));
        
        if (mapping.is_fc_weight) {
            // ============================================================
            // FC WEIGHT PATH: replace both dep[1] output AND weight cache
            // ============================================================
            //
            // Dynamic FC primitives use _reordered_weights_cache for
            // weights_memory(), NOT dep_memory_ptr(1) directly.
            // We must update BOTH:
            //   1. dep[1]'s output (so update_weights() sees new data)
            //   2. The FC's cache entry (so weights_memory() returns new data)
            
            // Step 1: Replace dep[1] data primitive's output
            data_inst->force_set_output_memory(new_mem, 0);
            
            // Step 2: Replace FC's weight cache entry
            auto fc_inst = net->get_primitive(mapping.fc_primitive_id);
            if (fc_inst) {
                auto weights_layout_opt = fc_inst->get_impl_params()->weights_layout;
                if (weights_layout_opt.has_value()) {
                    fc_inst->update_weights_cache(weights_layout_opt.value(), new_mem);
                } else {
                    // weights_layout not set yet — just rely on dep[1] replacement
                    // update_weights() will pick up the new dep[1] on next call
                }
            }
            ++swapped;
        } else {
            // ============================================================
            // DATA PRIMITIVE PATH: scale/zp — direct output replacement
            // ============================================================
            data_inst->force_set_output_memory(std::move(new_mem), 0);
            ++swapped;
        }
    }
    
    // NOTE: Kernel argument rebinding is NOT done here.
    // The caller (execute_impl_streamed) must set _reset_arguments = true
    // and call network::set_arguments() after swap_weight_pointers() returns.
    // This is because set_arguments() uses a _reset_arguments guard that
    // prevents re-entry after the initial call unless explicitly reset.
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] swap_weight_pointers(group=" << group_idx 
                  << "): swapped " << swapped << " tensors, "
                  << "layers " << group.first_layer << "-" 
                  << (group.first_layer + group.num_layers - 1) << "\n";
    }
    
    return swapped > 0;
    
#else
    // ====================================================================
    // STANDALONE STUB: verify mapping and simulate swap
    // ====================================================================
    (void)network_ptr;
    
    if (m_weight_mappings.empty()) {
        if (m_config.debug_logging) {
            std::cout << "[DenseStreaming] swap_weight_pointers: "
                      << "no mapping built — call build_weight_mapping() first\n";
        }
        // Still return true for basic testing without mapping
        return true;
    }
    
    uint32_t tensor_count = 0;
    uint64_t total_bytes = 0;
    
    // Convert group's model layer range to packed layer range
    const uint32_t first_streamed = m_group_table[0].first_layer;
    const uint32_t packed_first = group.first_layer - first_streamed;
    const uint32_t packed_end = packed_first + group.num_layers;
    
    for (const auto& mapping : m_weight_mappings) {
        if (mapping.layer_idx < packed_first ||
            mapping.layer_idx >= packed_end) {
            continue;
        }
        
        // Verify the pointer is within our buffer bounds
        uint64_t layer_offset = m_layer_table[mapping.layer_idx].offset_in_group;
        uint64_t tensor_end = layer_offset + mapping.offset_in_layer + mapping.size_bytes;
        
        if (tensor_end > m_buffer_sizes[0]) {
            std::cerr << "[DenseStreaming] ERROR: tensor '" << mapping.tensor_name
                      << "' exceeds buffer bounds! offset=" 
                      << (layer_offset + mapping.offset_in_layer)
                      << " + size=" << mapping.size_bytes 
                      << " > buffer=" << m_buffer_sizes[0] << "\n";
            return false;
        }
        
        ++tensor_count;
        total_bytes += mapping.size_bytes;
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] swap_weight_pointers(group=" << group_idx 
                  << "): verified " << tensor_count << " tensors ("
                  << (total_bytes / (1024.0*1024.0)) << " MB), "
                  << "layers " << group.first_layer << "-" 
                  << (group.first_layer + group.num_layers - 1) << "\n";
    }
    
    return true;
#endif
}

// ============================================================================
// Orchestrated Decode Pipeline (Step 6 — boundary synchronization)
// ============================================================================
//
// The streamed decode pipeline orchestrates the interaction between two
// asynchronous DMA subsystems for each token generation step:
//
//   1. NVMe Async I/O — async ReadFile (FILE_FLAG_OVERLAPPED) DMA reads
//   2. Pointer Swap   — update weight buffer pointers in compiled graph (~0.1 ms)
//   3. GPU Compute    — execute decoder layers on iGPU (~7 ms per group)
//
// Both NVMe DMA and iGPU DMA run independently on LPDDR5 bus:
//   - LPDDR5 8533MHz bandwidth: >100 GB/s
//   - NVMe demand: ~12 GB/s, iGPU demand: ~50 GB/s
//   - Combined: ~62 GB/s << 100 GB/s → no bandwidth contention
//
// Synchronization points (fences):
//   - IO fence:  wait_async_load()   — WaitForMultipleObjects (zero CPU overhead)
//   - GPU fence: stream.finish()     — ensures GPU is done reading old buffer
//
// Key invariant: GPU must never read from a buffer that IO is writing to.
// The double-buffer design and fence ordering guarantee this:
//
//   Buffer[active]   → GPU reads from here
//   Buffer[inactive] → Async IO writes here (NVMe DMA)
//
//   After GPU fence + IO fence:
//     - GPU is done with Buffer[active]
//     - IO has filled Buffer[inactive] with new data
//     - Swap: active ↔ inactive
//     - Update weight pointers to reference new active buffer
//     - Issue async ReadFile for next group (returns immediately)
//     - Start GPU compute (concurrent with NVMe DMA)
//
// Timeline (6 groups, 338 MB/group, 11.2 GB/s IO, 7 ms/group GPU):
//   With true GPU/IO overlap (async ReadFile, no thread sync overhead):
//
//   IO:   [===== G0 30ms =====][G1 30ms][G2 30ms][G3 30ms][G4 30ms][G5 30ms]
//   GPU:  <wait>               [G0 7ms ][G1 7ms ][G2 7ms ][G3 7ms ][G4 7ms ][G5 7ms]
//   Swap:                      [0.1ms]  [0.1ms]  [0.1ms]  [0.1ms]  [0.1ms]  [0.1ms]
//
//   Total ≈ 30ms (cold G0) + 5 × max(30, 7+0.2)ms = ~180 ms → ~5.5 tps
//   (vs 24 tps baseline with all weights in RAM — 4.4× penalty from IO)
//   With smaller groups (H5+T5, 1-layer): max overlap → approaching ~7 tps

// Static member
const std::vector<void*> DenseWeightStreamingManager::s_empty_exec_order;

void TokenPipelineStats::print() const {
    std::cout << "=== Token Pipeline Stats ===\n"
              << "  Total token time:   " << total_token_ms << " ms\n"
              << "  Pinned head GPU:    " << pinned_head_gpu_ms << " ms\n"
              << "  First load (cold):  " << first_load_ms << " ms\n"
              << "  IO wait total:      " << total_io_wait_ms << " ms\n"
              << "  GPU compute total:  " << total_gpu_ms << " ms\n"
              << "  Pointer swap total: " << total_swap_ms << " ms\n"
              << "  GPU fence total:    " << total_gpu_fence_ms << " ms\n"
              << "  Pinned tail GPU:    " << pinned_tail_gpu_ms << " ms\n"
              << "  Pipeline efficiency:" << (pipeline_efficiency * 100.0) << "%\n"
              << "  Streamed group breakdown:\n";
    for (size_t i = 0; i < groups.size(); ++i) {
        const auto& g = groups[i];
        std::cout << "    Group " << i << ": "
                  << "io_wait=" << g.io_wait_ms << " ms, "
                  << "swap=" << g.swap_ms << " ms, "
                  << "gpu=" << g.gpu_ms << " ms, "
                  << "fence=" << g.gpu_fence_ms << " ms\n";
    }
}

bool DenseWeightStreamingManager::execute_streamed_decode(
    const GroupComputeCallback& compute_fn,
    void* network,
    void* stream_ptr) {
    
    if (!m_initialized) {
        std::cerr << "[DenseStreaming] execute_streamed_decode: not initialized!\n";
        return false;
    }
    
    const uint32_t ng = m_header.num_groups;
    m_last_token_stats.reset(ng);
    
    auto token_t0 = std::chrono::high_resolution_clock::now();
    
    // ====================================================================
    // Phase 1: Pinned HEAD layers — pure GPU, no IO needed
    // ====================================================================
    // These layers (0..pin_head-1) stay in original compile_model() USM memory.
    // They run at full GPU speed (~0.99 ms/layer) with zero IO overhead.
    if (m_config.pin_head_layers > 0) {
        auto head_t0 = std::chrono::high_resolution_clock::now();
        
        // Execute all pinned head layers as a single batch.
        // No swap_weight_pointers needed — weights are still in original memory.
        bool ok = compute_fn(UINT32_MAX,  // special group_idx: pinned head
                              0,           // first_layer = 0
                              m_config.pin_head_layers);
        if (!ok) {
            std::cerr << "[DenseStreaming] GPU compute failed for pinned head layers\n";
            return false;
        }
        
        // GPU fence: ensure head layers complete before streaming starts
        wait_for_gpu(stream_ptr);
        
        auto head_t1 = std::chrono::high_resolution_clock::now();
        m_last_token_stats.pinned_head_gpu_ms = 
            std::chrono::duration<double, std::milli>(head_t1 - head_t0).count();
        
        if (m_config.debug_logging) {
            std::cout << "[Pipeline] Pinned HEAD layers 0-" << (m_config.pin_head_layers - 1)
                      << ": " << m_last_token_stats.pinned_head_gpu_ms << " ms\n";
        }
    }
    
    // ====================================================================
    // Phase 2: Streamed MIDDLE layers — IO+GPU double-buffer pipeline
    // ====================================================================
    // These layers (pin_head..total-pin_tail-1) are loaded from NVMe per token.
    // Uses the same double-buffer pipeline as before, but only for middle groups.
    if (ng > 0) {
        // --- Group 0 (cold start): synchronous load ---
        {
            auto io_t0 = std::chrono::high_resolution_clock::now();
            
            if (!load_group(0)) {
                std::cerr << "[DenseStreaming] Failed to load group 0\n";
                return false;
            }
            if (!wait_for_load(0)) {
                std::cerr << "[DenseStreaming] Failed to wait for group 0\n";
                return false;
            }
            
            auto io_t1 = std::chrono::high_resolution_clock::now();
            double cold_ms = std::chrono::duration<double, std::milli>(io_t1 - io_t0).count();
            m_last_token_stats.first_load_ms = cold_ms;
            m_last_token_stats.groups[0].io_wait_ms = cold_ms;
            
            if (m_config.debug_logging) {
                std::cout << "[Pipeline] Group 0 cold load: " << cold_ms << " ms\n";
            }
        }
        
        // Swap weight pointers for group 0
        {
            auto swap_t0 = std::chrono::high_resolution_clock::now();
            swap_weight_pointers(0, network);
            auto swap_t1 = std::chrono::high_resolution_clock::now();
            m_last_token_stats.groups[0].swap_ms = 
                std::chrono::duration<double, std::milli>(swap_t1 - swap_t0).count();
        }
        
        // Start prefetch for group 1 BEFORE GPU compute of group 0
        if (ng > 1) {
            prefetch_next_group(1);
        }
        
        // GPU compute group 0
        {
            auto gpu_t0 = std::chrono::high_resolution_clock::now();
            
            bool ok = compute_fn(0, m_group_table[0].first_layer,
                                  m_group_table[0].num_layers);
            if (!ok) {
                std::cerr << "[DenseStreaming] GPU compute failed for group 0\n";
                return false;
            }
            
            auto gpu_t1 = std::chrono::high_resolution_clock::now();
            m_last_token_stats.groups[0].gpu_ms = 
                std::chrono::duration<double, std::milli>(gpu_t1 - gpu_t0).count();
        }
        
        // --- Groups 1..N-1: pipelined ---
        for (uint32_t g = 1; g < ng; ++g) {
            // GPU Fence: wait for previous group's compute to finish
            {
                auto fence_t0 = std::chrono::high_resolution_clock::now();
                wait_for_gpu(stream_ptr);
                auto fence_t1 = std::chrono::high_resolution_clock::now();
                m_last_token_stats.groups[g - 1].gpu_fence_ms = 
                    std::chrono::duration<double, std::milli>(fence_t1 - fence_t0).count();
            }
            
            // IO Fence: wait for this group's prefetch
            {
                auto io_t0 = std::chrono::high_resolution_clock::now();
                if (!wait_for_load(g)) {
                    std::cerr << "[DenseStreaming] Failed to wait for group " << g << "\n";
                    return false;
                }
                auto io_t1 = std::chrono::high_resolution_clock::now();
                m_last_token_stats.groups[g].io_wait_ms = 
                    std::chrono::duration<double, std::milli>(io_t1 - io_t0).count();
            }
            
            // Swap weight pointers
            {
                auto swap_t0 = std::chrono::high_resolution_clock::now();
                swap_weight_pointers(g, network);
                auto swap_t1 = std::chrono::high_resolution_clock::now();
                m_last_token_stats.groups[g].swap_ms = 
                    std::chrono::duration<double, std::milli>(swap_t1 - swap_t0).count();
            }
            
            // Prefetch next group
            if (g + 1 < ng) {
                prefetch_next_group(g + 1);
            }
            
            // GPU compute this group
            {
                auto gpu_t0 = std::chrono::high_resolution_clock::now();
                
                bool ok = compute_fn(g, m_group_table[g].first_layer,
                                      m_group_table[g].num_layers);
                if (!ok) {
                    std::cerr << "[DenseStreaming] GPU compute failed for group " << g << "\n";
                    return false;
                }
                
                auto gpu_t1 = std::chrono::high_resolution_clock::now();
                m_last_token_stats.groups[g].gpu_ms = 
                    std::chrono::duration<double, std::milli>(gpu_t1 - gpu_t0).count();
            }
        }
        
        // Final GPU fence for last streamed group
        {
            auto fence_t0 = std::chrono::high_resolution_clock::now();
            wait_for_gpu(stream_ptr);
            auto fence_t1 = std::chrono::high_resolution_clock::now();
            m_last_token_stats.groups[ng - 1].gpu_fence_ms = 
                std::chrono::duration<double, std::milli>(fence_t1 - fence_t0).count();
        }
    }
    
    // ====================================================================
    // Phase 3: Pinned TAIL layers — pure GPU, no IO needed
    // ====================================================================
    // These layers (total-pin_tail..total-1) stay in original USM memory.
    if (m_config.pin_tail_layers > 0) {
        auto tail_t0 = std::chrono::high_resolution_clock::now();
        
        uint32_t tail_first = m_config.total_decoder_layers - m_config.pin_tail_layers;
        bool ok = compute_fn(UINT32_MAX - 1,  // special group_idx: pinned tail
                              tail_first,
                              m_config.pin_tail_layers);
        if (!ok) {
            std::cerr << "[DenseStreaming] GPU compute failed for pinned tail layers\n";
            return false;
        }
        
        // Final GPU fence
        wait_for_gpu(stream_ptr);
        
        auto tail_t1 = std::chrono::high_resolution_clock::now();
        m_last_token_stats.pinned_tail_gpu_ms = 
            std::chrono::duration<double, std::milli>(tail_t1 - tail_t0).count();
        
        if (m_config.debug_logging) {
            std::cout << "[Pipeline] Pinned TAIL layers " << tail_first << "-"
                      << (m_config.total_decoder_layers - 1)
                      << ": " << m_last_token_stats.pinned_tail_gpu_ms << " ms\n";
        }
    }
    
    // ====================================================================
    // Compute aggregate stats
    // ====================================================================
    auto token_t1 = std::chrono::high_resolution_clock::now();
    m_last_token_stats.total_token_ms = 
        std::chrono::duration<double, std::milli>(token_t1 - token_t0).count();
    m_last_token_stats.aggregate();
    
    if (m_config.debug_logging) {
        m_last_token_stats.print();
    }
    
    return true;
}

void DenseWeightStreamingManager::wait_for_gpu(void* stream_ptr) {
#ifdef OPENVINO_GPU_RUNTIME_AVAILABLE
    if (stream_ptr) {
        // cldnn::stream::finish() blocks until all enqueued GPU work completes.
        // After this returns, it is safe to overwrite the weight buffer that
        // the GPU was reading from.
        auto* gpu_stream = static_cast<cldnn::stream*>(stream_ptr);
        gpu_stream->finish();
    }
#else
    // Standalone mode: no real GPU, nothing to wait for.
    // In testing, the compute_fn callback can simulate GPU latency with sleep().
    (void)stream_ptr;
#endif
}

bool DenseWeightStreamingManager::build_group_exec_order(void* network_ptr) {
#ifdef OPENVINO_GPU_RUNTIME_AVAILABLE
    // ====================================================================
    // Partition execution order into per-group primitive lists (H5+T5 aware)
    // ====================================================================
    //
    // IMPORTANT: _exec_order contains inter-layer primitives (rotary_emb,
    // Convert ops, etc.) without "layers.N" in their names, interleaved
    // among decoder layer primitives. We must preserve the EXACT order from
    // _exec_order by using a "current region" approach:
    //
    //   1. Find the range [first_decoder_pos, last_decoder_pos] in exec_order
    //   2. Everything before first_decoder_pos → pre-decoder
    //   3. Everything after last_decoder_pos → post-decoder
    //   4. Within the range: non-layer primitives inherit the current region
    //      (the region of the most recently seen decoder layer primitive)
    //
    // Layout (ng + 4 slots):
    //   [0..ng-1]  = streamed decoder groups (middle layers)
    //   [ng]       = pinned HEAD decoder layers (layers 0..pin_head-1)
    //   [ng+1]     = pinned TAIL decoder layers (layers total-pin_tail..total-1)
    //   [ng+2]     = pre-decoder (embeddings, vision encoder, etc.)
    //   [ng+3]     = post-decoder (final norm, lm_head, logits)
    
    auto* net = static_cast<cldnn::network*>(network_ptr);
    const uint32_t ng = m_header.num_groups;
    
    m_group_exec_order.clear();
    m_group_exec_order.resize(ng + 4);
    
    // Use exec_order for correct topological ordering
    const auto& exec_order = net->get_exec_order();
    
    // ---- Pass 1: Find the index range of decoder primitives ----
    size_t first_decoder_pos = SIZE_MAX;
    size_t last_decoder_pos = 0;
    size_t pos = 0;
    for (const auto& inst : exec_order) {
        int32_t layer_idx = parse_layer_index_from_name(inst->id());
        if (layer_idx >= 0 && layer_idx < static_cast<int32_t>(m_config.total_decoder_layers)) {
            if (first_decoder_pos == SIZE_MAX) first_decoder_pos = pos;
            last_decoder_pos = pos;
        }
        ++pos;
    }
    
    if (first_decoder_pos == SIZE_MAX) {
        // No decoder primitives found — everything is pre-decoder
        for (const auto& inst : exec_order) {
            m_group_exec_order[ng + 2].push_back(inst.get());
        }
        return true;
    }
    
    // ---- Pass 2: Assign primitives to groups preserving exec_order ----
    // "current_slot" tracks which group we're currently in.
    // Non-layer primitives inherit current_slot from the most recent decoder prim.
    //
    // IMPORTANT: _exec_order is a topological sort — the GPU compiler may
    // interleave constant operations from different layers. We must preserve
    // the exact _exec_order during execution (no reordering). The per-primitive
    // group assignment is used ONLY for detecting group transitions during
    // execution, NOT for reordering.
    int32_t current_layer = -1;  // -1 = pre-decoder
    pos = 0;
    
    // Build per-primitive group assignment (parallel to exec_order)
    m_exec_group_assignment.clear();
    m_exec_group_assignment.reserve(exec_order.size());
    
    for (const auto& inst : exec_order) {
        int32_t group_id;  // assignment for this primitive
        
        // Determine slot for this primitive
        if (pos < first_decoder_pos) {
            // Before any decoder primitive → always pre-decoder
            m_group_exec_order[ng + 2].push_back(inst.get());
            group_id = GROUP_PRE_DECODER;
        } else if (pos > last_decoder_pos) {
            // After all decoder primitives → always post-decoder
            m_group_exec_order[ng + 3].push_back(inst.get());
            group_id = static_cast<int32_t>(ng);
        } else {
            // Within decoder range: check if this has a layer index
            int32_t layer_idx = parse_layer_index_from_name(inst->id());
            if (layer_idx >= 0 && layer_idx < static_cast<int32_t>(m_config.total_decoder_layers)) {
                current_layer = layer_idx;
            }
            // else: non-layer primitive inherits current_layer
            
            if (current_layer < 0) {
                // Early non-layer prim before first decoder but within range
                m_group_exec_order[ng + 2].push_back(inst.get());
                group_id = GROUP_PRE_DECODER;
            } else if (m_config.is_pinned(static_cast<uint32_t>(current_layer))) {
                if (static_cast<uint32_t>(current_layer) < m_config.pin_head_layers) {
                    m_group_exec_order[ng].push_back(inst.get());
                    group_id = GROUP_PINNED_HEAD;
                } else {
                    m_group_exec_order[ng + 1].push_back(inst.get());
                    group_id = GROUP_PINNED_TAIL;
                }
            } else {
                uint32_t streamed_layer = static_cast<uint32_t>(current_layer) 
                                        - m_config.pin_head_layers;
                uint32_t group_idx = streamed_layer / m_header.group_size;
                if (group_idx < ng) {
                    m_group_exec_order[group_idx].push_back(inst.get());
                    group_id = static_cast<int32_t>(group_idx);
                } else {
                    m_group_exec_order[ng + 3].push_back(inst.get());
                    group_id = static_cast<int32_t>(ng);
                }
            }
        }
        
        m_exec_group_assignment.push_back(group_id);
        ++pos;
    }
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] build_group_exec_order (H5+T5):\n"
                  << "  Pre-decoder:  " << m_group_exec_order[ng + 2].size() << " primitives\n"
                  << "  Pinned HEAD:  " << m_group_exec_order[ng].size() 
                  << " primitives (layers 0-" << (m_config.pin_head_layers - 1) << ")\n";
        for (uint32_t g = 0; g < ng; ++g) {
            uint32_t first = m_group_table[g].first_layer;
            uint32_t last = first + m_group_table[g].num_layers - 1;
            std::cout << "  Group " << g << ":      " << m_group_exec_order[g].size() 
                      << " primitives (layers " << first << "-" << last << ")\n";
        }
        std::cout << "  Pinned TAIL:  " << m_group_exec_order[ng + 1].size()
                  << " primitives (layers " << (m_config.total_decoder_layers - m_config.pin_tail_layers)
                  << "-" << (m_config.total_decoder_layers - 1) << ")\n"
                  << "  Post-decoder: " << m_group_exec_order[ng + 3].size() << " primitives\n";
    }
    
    return true;
    
#else
    // ====================================================================
    // STANDALONE STUB
    // ====================================================================
    (void)network_ptr;
    
    const uint32_t ng = m_header.num_groups;
    m_group_exec_order.clear();
    m_group_exec_order.resize(ng + 4);
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] build_group_exec_order: standalone stub, "
                  << (ng + 4) << " groups created (empty)\n";
    }
    
    return true;
#endif
}

const std::vector<void*>& DenseWeightStreamingManager::get_group_exec_primitives(
    uint32_t group_idx) const {
    if (group_idx >= m_group_exec_order.size()) {
        return s_empty_exec_order;
    }
    return m_group_exec_order[group_idx];
}

// ============================================================================
// I/O Implementation: Async ReadFile (FILE_FLAG_OVERLAPPED)
// ============================================================================
//
// Architecture: True async I/O with Event-based completion
// =========================================================
// Instead of multi-threaded sync ReadFile (which required N threads per group
// transition), we now use native async I/O:
//
//   1. start_async_load() — issues N async ReadFile calls (one per handle),
//      each with its own OVERLAPPED + manual-reset Event. Returns IMMEDIATELY.
//      NVMe controller performs DMA in the background.
//
//   2. wait_async_load() — calls WaitForMultipleObjects to wait for all N
//      events to be signaled. Zero CPU overhead while waiting.
//
//   3. is_async_load_complete() — non-blocking check via WaitForMultipleObjects
//      with timeout=0.
//
// Benefits over multi-threaded sync ReadFile:
//   - No thread creation/destruction overhead (~200 μs per group transition saved)
//   - True kernel-level async: NVMe DMA without any CPU involvement
//   - GPU compute and NVMe DMA are truly independent on LPDDR5 bus
//   - LPDDR5 8533MHz bandwidth (>100 GB/s) >> NVMe (~12 GB/s) + iGPU (~50 GB/s)
//
// Data path (zero-copy):
//   NVMe SSD →(async DMA)→ LPDDR5 (USM host buffer) →(iGPU read)→ GPU compute
//   Both DMA paths run concurrently — no CPU involvement, no sync overhead.

bool DenseWeightStreamingManager::start_async_load(uint64_t file_offset,
                                                    uint64_t size,
                                                    void* dest_ptr) {
#ifdef _WIN32
    uint32_t num_handles = static_cast<uint32_t>(m_io_handles.size());
    if (num_handles == 0) return false;
    
    // Split data across file handles for NVMe queue depth
    // Each handle reads a contiguous, sector-aligned region.
    uint64_t bytes_per_handle = (size / num_handles / m_sector_size) * m_sector_size;
    
    OVERLAPPED* overlapped_arr = static_cast<OVERLAPPED*>(m_overlapped_storage);
    uint32_t op_idx = 0;
    
    for (uint32_t h = 0; h < num_handles; ++h) {
        uint64_t h_offset = file_offset + h * bytes_per_handle;
        uint64_t h_size = (h == num_handles - 1)
            ? (size - h * bytes_per_handle)   // Last handle gets remainder
            : bytes_per_handle;
        uint8_t* h_dest = static_cast<uint8_t*>(dest_ptr) + h * bytes_per_handle;
        HANDLE hFile = static_cast<HANDLE>(m_io_handles[h]);
        
        // Handle chunks > 256 MB (ReadFile DWORD limit)
        uint64_t remaining = h_size;
        uint64_t current_offset = h_offset;
        uint8_t* current_dest = h_dest;
        
        while (remaining > 0 && op_idx < MAX_ASYNC_OPS) {
            DWORD chunk = static_cast<DWORD>(
                std::min(remaining, static_cast<uint64_t>(256 * 1024 * 1024)));
            
            // Set up OVERLAPPED with event for this async ReadFile
            OVERLAPPED& ov = overlapped_arr[op_idx];
            memset(&ov, 0, sizeof(OVERLAPPED));
            ov.Offset = static_cast<DWORD>(current_offset & 0xFFFFFFFF);
            ov.OffsetHigh = static_cast<DWORD>(current_offset >> 32);
            ov.hEvent = static_cast<HANDLE>(m_io_events[op_idx]);
            ResetEvent(ov.hEvent);
            
            // Issue async ReadFile — returns immediately!
            // The NVMe controller performs DMA in the background.
            BOOL result = ReadFile(hFile, current_dest, chunk, nullptr, &ov);
            if (!result) {
                DWORD err = GetLastError();
                if (err != ERROR_IO_PENDING) {
                    std::cerr << "[DenseStreaming] Async ReadFile failed at offset "
                              << current_offset << ": error " << err << "\n";
                    m_async_num_ops = op_idx;
                    m_async_in_progress.store(false);
                    return false;
                }
                // ERROR_IO_PENDING = normal for async IO, operation is in progress
            }
            // If ReadFile returns TRUE, IO completed synchronously (cache hit)
            
            remaining -= chunk;
            current_offset += chunk;
            current_dest += chunk;
            ++op_idx;
        }
        
        if (remaining > 0) {
            std::cerr << "[DenseStreaming] Too many async ops needed (max "
                      << MAX_ASYNC_OPS << "), remaining " << remaining << " bytes\n";
            m_async_num_ops = op_idx;
            m_async_in_progress.store(false);
            return false;
        }
    }
    
    m_async_num_ops = op_idx;
    m_async_in_progress.store(true);
    
    if (m_config.debug_logging) {
        std::cout << "[DenseStreaming] start_async_load: " << op_idx
                  << " async ReadFile ops issued across " << num_handles
                  << " handles, " << (size / (1024.0*1024.0)) << " MB total\n";
    }
    
    return true;

#else
    return false;
#endif
}

bool DenseWeightStreamingManager::wait_async_load() {
#ifdef _WIN32
    if (m_async_num_ops == 0 || !m_async_in_progress.load()) return true;
    
    // WaitForMultipleObjects: wait for ALL async ReadFile ops to complete.
    // Zero CPU overhead — thread sleeps until all events are signaled.
    // Max MAXIMUM_WAIT_OBJECTS = 64, our MAX_ASYNC_OPS = 32 is safe.
    HANDLE events[MAX_ASYNC_OPS];
    for (uint32_t i = 0; i < m_async_num_ops; ++i) {
        events[i] = static_cast<HANDLE>(m_io_events[i]);
    }
    
    DWORD wait_result = WaitForMultipleObjects(
        m_async_num_ops, events, TRUE, INFINITE);
    
    if (wait_result == WAIT_FAILED) {
        std::cerr << "[DenseStreaming] WaitForMultipleObjects failed: error "
                  << GetLastError() << "\n";
        m_async_in_progress.store(false);
        return false;
    }
    
    // Verify each operation completed successfully
    OVERLAPPED* overlapped_arr = static_cast<OVERLAPPED*>(m_overlapped_storage);
    bool all_ok = true;
    for (uint32_t i = 0; i < m_async_num_ops; ++i) {
        DWORD bytes_read = 0;
        // Get each handle — ops are distributed round-robin across handles
        // We need the handle that was used for this op. Since we issue ops
        // sequentially per handle (handle 0's chunks, then handle 1's, etc.),
        // we track which handle was used for each op.
        // For simplicity, use GetOverlappedResult with any handle that
        // has the correct OVERLAPPED — actually we need the right handle.
        // Let's find it from the op index.
        uint32_t num_handles = static_cast<uint32_t>(m_io_handles.size());
        uint32_t handle_idx = 0;
        // Determine handle index: ops_per_handle varies by remainder, but
        // for verification we can just try each handle. Actually, with
        // FILE_FLAG_OVERLAPPED, GetOverlappedResult needs the file handle
        // that was used to start the operation.
        //
        // Since we issued ops sequentially per handle, we can calculate:
        // handle_idx = which handle's chunk range includes op_idx.
        // Simple approach: we stored overlapped[i].hEvent = m_io_events[i],
        // and the event is signaled by the kernel when IO completes.
        // Since WaitForMultipleObjects already confirmed all events are set,
        // the IO is done. We just need to check for errors.
        //
        // Alternative: use HasOverlappedIoCompleted() which checks the
        // OVERLAPPED's Internal field without needing the handle.
        if (!HasOverlappedIoCompleted(&overlapped_arr[i])) {
            std::cerr << "[DenseStreaming] Async op " << i
                      << " not completed despite event signal\n";
            all_ok = false;
        }
    }
    
    // Record timing and update buffer tracking
    if (all_ok && m_async_target_buffer >= 0) {
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(
            t1 - m_async_start_time).count();
        
        m_buffer_group[m_async_target_buffer] = static_cast<int32_t>(m_async_target_group);
        
        if (m_config.enable_timing) {
            m_stats.total_loads.fetch_add(1);
            if (m_async_target_group < m_stats.group_load_time_ms.size()) {
                m_stats.group_load_time_ms[m_async_target_group] = elapsed_ms;
            }
            
            // Estimate throughput from the group table
            if (m_async_target_group < m_group_table.size()) {
                uint64_t loaded_bytes = m_group_table[m_async_target_group].aligned_bytes;
                m_stats.total_bytes_loaded.fetch_add(loaded_bytes);
                atomic_add_double(m_stats.total_load_time_ms, elapsed_ms);
                
                double throughput_gbps = (loaded_bytes / (1024.0*1024.0*1024.0))
                                       / (elapsed_ms / 1000.0);
                double current_peak = m_stats.peak_throughput_gbps.load();
                while (throughput_gbps > current_peak) {
                    m_stats.peak_throughput_gbps.compare_exchange_weak(
                        current_peak, throughput_gbps);
                }
            }
        }
        
        if (m_config.debug_logging) {
            std::cout << "[DenseStreaming] wait_async_load: group "
                      << m_async_target_group << " loaded in "
                      << elapsed_ms << " ms (" << m_async_num_ops << " ops)\n";
        }
    }
    
    m_async_num_ops = 0;
    m_async_in_progress.store(false);
    return all_ok;

#else
    return true;
#endif
}

bool DenseWeightStreamingManager::is_async_load_complete() const {
#ifdef _WIN32
    if (m_async_num_ops == 0 || !m_async_in_progress.load()) return true;
    
    // Non-blocking check: WaitForMultipleObjects with timeout=0
    HANDLE events[MAX_ASYNC_OPS];
    for (uint32_t i = 0; i < m_async_num_ops; ++i) {
        events[i] = static_cast<HANDLE>(m_io_events[i]);
    }
    
    DWORD result = WaitForMultipleObjects(
        m_async_num_ops, events, TRUE, 0);
    
    return (result != WAIT_TIMEOUT);
#else
    return true;
#endif
}

bool DenseWeightStreamingManager::load_direct_io(uint64_t file_offset,
                                                  uint64_t size,
                                                  void* dest_ptr,
                                                  uint32_t buffer_idx) {
    // Synchronous wrapper: start async load then wait for completion.
    // Used by load_group() for cold-start synchronous reads.
#ifdef _WIN32
    m_async_start_time = std::chrono::high_resolution_clock::now();
    
    if (!start_async_load(file_offset, size, dest_ptr)) {
        return false;
    }
    return wait_async_load();

#else
    // Linux fallback: use pread (single-threaded)
    std::ifstream file(m_config.weights_file_path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(file_offset);
    file.read(static_cast<char*>(dest_ptr), size);
    return file.good();
#endif
}

// ============================================================================
// Debug
// ============================================================================

void DenseWeightStreamingManager::print_status() const {
    std::cout << "=== Dense Weight Streaming Manager Status ===\n"
              << "  Initialized: " << (m_initialized ? "YES" : "NO") << "\n"
              << "  IO method: Async ReadFile + OVERLAPPED (" << m_config.num_io_threads << " handles)\n"
              << "  Groups: " << m_header.num_groups << " x " << m_header.group_size << " layers\n"
              << "  Total weights: " << (m_header.total_weight_bytes / (1024.0*1024.0*1024.0)) << " GB\n"
              << "  Buffer sizes: " << (m_buffer_sizes[0] / (1024.0*1024.0)) << " MB x 2\n"
              << "  Active buffer: " << m_active_buffer << "\n"
              << "  Buffer[0] holds group: " << m_buffer_group[0] << "\n"
              << "  Buffer[1] holds group: " << m_buffer_group[1] << "\n"
              << "  Async IO active: " << (m_async_in_progress.load() ? "YES" : "NO") << "\n"
              << "  --- Hybrid Pinning ---\n"
              << "  Pin head layers: " << m_config.pin_head_layers
              << " (0-" << (m_config.pin_head_layers > 0 ? m_config.pin_head_layers - 1 : 0) << ")\n"
              << "  Pin tail layers: " << m_config.pin_tail_layers
              << " (" << (m_config.total_decoder_layers - m_config.pin_tail_layers) << "-"
              << (m_config.total_decoder_layers - 1) << ")\n"
              << "  Streamed layers: " << m_config.num_streamed_layers()
              << " (" << m_config.first_streamed_layer() << "-"
              << m_config.last_streamed_layer() << ")\n";
    
    if (m_config.enable_timing) {
        m_stats.print_summary();
    }
}

}  // namespace ov::intel_gpu
