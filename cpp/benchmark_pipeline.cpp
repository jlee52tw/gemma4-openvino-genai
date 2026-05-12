// Dense Weight Streaming — Pipeline Benchmark (Step 7)
// =====================================================
// End-to-end benchmark of the streamed decode pipeline:
//   NVMe IO → pointer swap → simulated GPU compute → GPU fence → prefetch
//
// Tests the full execute_streamed_decode() orchestrator with simulated GPU
// latency to validate pipeline timing, double-buffer correctness, and
// predict real-world performance.
//
// Also tests different IO thread counts (Step 8 tuning):
//   - 1, 2, 4, 8 IO threads
//   - With GPU latency = 0 (IO-only) and 7 ms (realistic)
//
// Build:
//   cl /EHsc /O2 /std:c++17 benchmark_pipeline.cpp dense_weight_streaming_manager.cpp
//      /link /out:benchmark_pipeline.exe
//
// Usage:
//   benchmark_pipeline.exe [path-to-dense_weights_streaming.bin] [iterations]

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")

#include "cldnn_stubs.hpp"
#include "dense_weight_streaming_manager.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace ov::intel_gpu;
using hrc = std::chrono::high_resolution_clock;

// ============================================================================
// Helpers
// ============================================================================

static double to_ms(hrc::time_point t0, hrc::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double median(std::vector<double>& v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 0) ? (v[n/2-1] + v[n/2]) / 2.0 : v[n/2];
}

static double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t idx = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[idx];
}

// ============================================================================
// Test 1: Sequential IO-only (no pipeline, no GPU)
// ============================================================================
// Measures raw IO throughput by loading all groups sequentially.
// This is the baseline — worst case with zero overlap.

struct IOOnlyResult {
    double total_ms;
    double throughput_gbps;
    std::vector<double> per_group_ms;
};

static IOOnlyResult benchmark_io_only(DenseWeightStreamingManager& mgr,
                                       int iterations) {
    uint32_t ng = mgr.num_groups();
    std::vector<double> total_times;
    std::vector<std::vector<double>> per_group_all(ng);
    
    for (int iter = 0; iter < iterations; ++iter) {
        auto t0 = hrc::now();
        
        for (uint32_t g = 0; g < ng; ++g) {
            auto g_t0 = hrc::now();
            mgr.load_group_sync(g);
            auto g_t1 = hrc::now();
            per_group_all[g].push_back(to_ms(g_t0, g_t1));
        }
        
        auto t1 = hrc::now();
        total_times.push_back(to_ms(t0, t1));
    }
    
    IOOnlyResult result;
    result.total_ms = median(total_times);
    result.per_group_ms.resize(ng);
    for (uint32_t g = 0; g < ng; ++g) {
        result.per_group_ms[g] = median(per_group_all[g]);
    }
    
    double total_bytes = static_cast<double>(mgr.total_weight_bytes());
    result.throughput_gbps = (total_bytes / (1024.0*1024.0*1024.0)) / (result.total_ms / 1000.0);
    
    return result;
}

// ============================================================================
// Test 2: Pipeline with simulated GPU
// ============================================================================
// Uses execute_streamed_decode() with a callback that simulates GPU compute
// by sleeping for a specified duration.

struct PipelineResult {
    double total_ms;
    double estimated_tps;
    TokenPipelineStats stats;
    std::vector<double> all_token_ms;
};

static PipelineResult benchmark_pipeline(DenseWeightStreamingManager& mgr,
                                          double gpu_ms_per_group,
                                          int iterations) {
    // GPU compute simulation callback
    // Uses num_layers to compute accurate GPU time for both pinned and streamed phases
    constexpr double GPU_MS_PER_LAYER_SIM = 41.7 / 42.0;  // ~0.99 ms
    auto compute_fn = [gpu_ms_per_group, GPU_MS_PER_LAYER_SIM](uint32_t group_idx,
                                          uint32_t first_layer,
                                          uint32_t num_layers) -> bool {
        // For pinned phases (UINT32_MAX, UINT32_MAX-1), use per-layer time × num_layers
        // For streamed groups, use gpu_ms_per_group (= per-layer × group_size)
        double target_ms = (group_idx >= 0xFFFFFFFE) 
            ? (num_layers * GPU_MS_PER_LAYER_SIM) 
            : gpu_ms_per_group;
        if (target_ms > 0) {
            // Simulate GPU work with pure spin-wait for sub-ms precision.
            // timeBeginPeriod(1) is set in main(), so Sleep(1) ≈ 1ms.
            auto t0 = hrc::now();
            double target_us = target_ms * 1000.0;
            while (true) {
                auto now = hrc::now();
                double elapsed_us = std::chrono::duration<double, std::micro>(now - t0).count();
                if (elapsed_us >= target_us) break;
                // Use Sleep only for long waits (> 3ms remaining)
                double remaining_us = target_us - elapsed_us;
                if (remaining_us > 3000) {
                    Sleep(1);
                }
            }
        }
        return true;
    };
    
    std::vector<double> all_token_ms;
    TokenPipelineStats last_stats;
    
    for (int iter = 0; iter < iterations; ++iter) {
        mgr.execute_streamed_decode(compute_fn, nullptr, nullptr);
        const auto& stats = mgr.get_last_token_stats();
        all_token_ms.push_back(stats.total_token_ms);
        last_stats = stats;  // Keep last iteration's detailed stats
    }
    
    PipelineResult result;
    result.total_ms = median(all_token_ms);
    result.estimated_tps = 1000.0 / result.total_ms;
    result.stats = last_stats;
    result.all_token_ms = std::move(all_token_ms);
    
    return result;
}

// ============================================================================
// Test 3: Prefetch overlap measurement
// ============================================================================
// Measures how much IO can be hidden behind simulated GPU compute.
// Runs the pipeline with varying GPU latencies and measures the delta.

struct OverlapResult {
    double io_only_ms;       // Pipeline with 0ms GPU (pure IO cost)
    double with_gpu_ms;      // Pipeline with real GPU latency
    double overlap_saved_ms; // Time saved by overlapping
    double overlap_ratio;    // Fraction of IO hidden by overlap
};

static OverlapResult benchmark_overlap(DenseWeightStreamingManager& mgr,
                                        double gpu_ms_per_group,
                                        int iterations) {
    // IO-only pipeline (gpu_ms = 0)
    auto io_result = benchmark_pipeline(mgr, 0.0, iterations);
    
    // Full pipeline with GPU
    auto full_result = benchmark_pipeline(mgr, gpu_ms_per_group, iterations);
    
    OverlapResult result;
    result.io_only_ms = io_result.total_ms;
    result.with_gpu_ms = full_result.total_ms;
    
    // Expected serial time = IO_time + GPU_time
    double expected_serial_ms = io_result.total_ms + 
        gpu_ms_per_group * mgr.num_groups();
    result.overlap_saved_ms = expected_serial_ms - full_result.total_ms;
    result.overlap_ratio = result.overlap_saved_ms / 
        (gpu_ms_per_group * mgr.num_groups());
    if (result.overlap_ratio < 0) result.overlap_ratio = 0;
    if (result.overlap_ratio > 1) result.overlap_ratio = 1;
    
    return result;
}

// ============================================================================
// Test 4: Data integrity check
// ============================================================================
// Verifies that double-buffer swapping doesn't corrupt data by loading
// each group twice and comparing the content.

static bool verify_data_integrity(DenseWeightStreamingManager& mgr) {
    uint32_t ng = mgr.num_groups();
    bool all_ok = true;
    
    for (uint32_t g = 0; g < ng; ++g) {
        // First load
        mgr.load_group_sync(g);
        void* ptr1 = mgr.get_group_buffer_ptr(g);
        uint64_t size = mgr.group_bytes(g);
        
        // Save first 4 KB of data
        std::vector<uint8_t> snapshot(std::min<uint64_t>(size, 4096));
        memcpy(snapshot.data(), ptr1, snapshot.size());
        
        // Load another group to force buffer swap
        uint32_t other_g = (g + 1) % ng;
        mgr.load_group_sync(other_g);
        
        // Load original group again (into different buffer slot)
        mgr.load_group_sync(g);
        void* ptr2 = mgr.get_group_buffer_ptr(g);
        
        // Compare first 4 KB
        if (memcmp(snapshot.data(), ptr2, snapshot.size()) != 0) {
            std::cerr << "  Data integrity FAILED for group " << g << "!\n";
            all_ok = false;
        }
    }
    
    return all_ok;
}

// ============================================================================
// Print helpers
// ============================================================================

static void print_separator(const char* title) {
    std::cout << "\n" << std::string(70, '=') << "\n"
              << "  " << title << "\n"
              << std::string(70, '=') << "\n";
}

static void print_group_table(const DenseWeightStreamingManager& mgr) {
    uint32_t ng = mgr.num_groups();
    uint32_t first_streamed = mgr.pin_head_layers();  // H5+T5: streamed layers start after pinned head
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "  " << std::setw(6) << "Group" 
              << std::setw(14) << "Size (MB)"
              << std::setw(18) << "Layers"
              << "\n";
    std::cout << "  " << std::string(38, '-') << "\n";
    
    for (uint32_t g = 0; g < ng; ++g) {
        double mb = mgr.group_bytes(g) / (1024.0 * 1024.0);
        // Map group index to actual decoder layer range (offset by pinned head)
        uint32_t first = first_streamed + g * mgr.group_size();
        uint32_t last = first + mgr.group_size() - 1;
        std::cout << "  " << std::setw(6) << g
                  << std::setw(12) << mb << " MB"
                  << std::setw(8) << first << "-" << last
                  << "\n";
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Set Windows timer resolution to 1 ms for accurate Sleep(1) calls
    timeBeginPeriod(1);
    
    // Parse arguments
    std::string bin_path = "C:\\working\\gemma4-openvino\\gemma-4-E4B-it-ov\\dense_weights_streaming.bin";
    int iterations = 5;
    
    if (argc >= 2) bin_path = argv[1];
    if (argc >= 3) iterations = std::atoi(argv[2]);
    
    std::cout << "Dense Weight Streaming — Pipeline Benchmark (H5+T5 Hybrid)\n"
              << "================================================================\n"
              << "  Binary file: " << bin_path << "\n"
              << "  Iterations:  " << iterations << "\n";
    
    // Baseline parameters from actual Gemma4 measurements
    constexpr double BASELINE_TPOT_MS = 41.7;  // Output TPOT at 24 tps
    constexpr uint32_t NUM_DECODER_LAYERS = 42;
    constexpr double GPU_MS_PER_LAYER = BASELINE_TPOT_MS / NUM_DECODER_LAYERS;  // ~0.99 ms
    
    // ====================================================================
    // Phase 1: Initialize & print model info
    // ====================================================================
    
    // Standalone stub engine (defined in dense_weight_streaming_manager.cpp
    // when OPENVINO_GPU_RUNTIME_AVAILABLE is not defined)
    cldnn::engine engine;
    DenseStreamingConfig config;
    config.weights_file_path = bin_path;
    config.num_io_threads = 4;
    config.enable_timing = true;
    config.debug_logging = false;
    
    DenseWeightStreamingManager mgr(engine, config);
    if (!mgr.initialize()) {
        std::cerr << "Failed to initialize streaming manager!\n";
        return 1;
    }
    
    std::cout << "\n  Model info:\n"
              << "    Streamed weight bytes: " << std::fixed << std::setprecision(2)
              << (mgr.total_weight_bytes() / (1024.0*1024.0*1024.0)) << " GB\n"
              << "    Streamed groups: " << mgr.num_groups() << " x " << mgr.group_size() << " layers\n"
              << "    Pinned head: " << mgr.pin_head_layers() << " layers (0-"
              << (mgr.pin_head_layers() > 0 ? mgr.pin_head_layers() - 1 : 0) << ")\n"
              << "    Pinned tail: " << mgr.pin_tail_layers() << " layers ("
              << (NUM_DECODER_LAYERS - mgr.pin_tail_layers()) << "-"
              << (NUM_DECODER_LAYERS - 1) << ")\n"
              << "    GPU per-layer: " << std::setprecision(2) << GPU_MS_PER_LAYER << " ms\n"
              << "    GPU per-group: " << std::setprecision(1) 
              << (GPU_MS_PER_LAYER * mgr.group_size()) << " ms (estimated)\n"
              << "    Pinned GPU:    " << std::setprecision(1)
              << (GPU_MS_PER_LAYER * (mgr.pin_head_layers() + mgr.pin_tail_layers()))
              << " ms\n";
    
    print_group_table(mgr);
    
    // ====================================================================
    // Phase 2: Data integrity check
    // ====================================================================
    
    print_separator("Phase 2: Data Integrity Check");
    std::cout << "  Verifying double-buffer correctness...\n";
    bool integrity_ok = verify_data_integrity(mgr);
    std::cout << "  Result: " << (integrity_ok ? "PASS" : "FAIL") << "\n";
    if (!integrity_ok) {
        std::cerr << "  ABORTING: Data integrity check failed!\n";
        return 1;
    }
    
    // ====================================================================
    // Phase 3: IO-only baseline (sequential, no pipeline)
    // ====================================================================
    
    print_separator("Phase 3: IO-Only Baseline (Sequential)");
    std::cout << "  Loading all groups sequentially (no prefetch, no GPU)...\n";
    
    auto io_result = benchmark_io_only(mgr, iterations);
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Results (" << iterations << " iterations, median):\n"
              << "    Total IO time:  " << io_result.total_ms << " ms\n"
              << "    IO throughput:  " << io_result.throughput_gbps << " GB/s\n"
              << "    Per-group:\n";
    for (uint32_t g = 0; g < mgr.num_groups(); ++g) {
        double mb = mgr.group_bytes(g) / (1024.0*1024.0);
        double gbps = (mb / 1024.0) / (io_result.per_group_ms[g] / 1000.0);
        std::cout << "      Group " << g << ": " 
                  << std::setprecision(1) << io_result.per_group_ms[g] << " ms  ("
                  << std::setprecision(2) << mb << " MB, " 
                  << std::setprecision(1) << gbps << " GB/s)\n";
    }
    
    // ====================================================================
    // Phase 4: Pipeline with simulated GPU (realistic)
    // ====================================================================
    
    double gpu_per_group = GPU_MS_PER_LAYER * mgr.group_size();
    
    print_separator("Phase 4: Pipeline with Simulated GPU");
    std::cout << "  Simulated GPU compute: " << std::setprecision(1) 
              << gpu_per_group << " ms per group\n"
              << "  Pipeline: load → swap → GPU + async prefetch → fence → repeat\n";
    
    auto pipe_result = benchmark_pipeline(mgr, gpu_per_group, iterations);
    
    std::cout << "\n  Results (" << iterations << " iterations, median):\n"
              << "    Total token time:  " << std::setprecision(1) 
              << pipe_result.total_ms << " ms\n"
              << "    Estimated TPS:     " << std::setprecision(2) 
              << pipe_result.estimated_tps << "\n"
              << "    vs baseline 24 tps: " << std::setprecision(1)
              << (pipe_result.estimated_tps / 24.0 * 100.0) << "% throughput\n";
    
    // Print last iteration's detailed stats
    const auto& stats = pipe_result.stats;
    std::cout << "\n  Pipeline breakdown (last iteration):\n"
              << "    Pinned head GPU: " << std::setprecision(1)
              << stats.pinned_head_gpu_ms << " ms\n"
              << "    First load (cold): " << std::setprecision(1) 
              << stats.first_load_ms << " ms\n"
              << "    Total IO wait:     " << stats.total_io_wait_ms << " ms\n"
              << "    Total GPU compute: " << stats.total_gpu_ms << " ms\n"
              << "    Total swap:        " << std::setprecision(2) 
              << stats.total_swap_ms << " ms\n"
              << "    Total GPU fence:   " << std::setprecision(2) 
              << stats.total_gpu_fence_ms << " ms\n"
              << "    Pinned tail GPU:   " << std::setprecision(1)
              << stats.pinned_tail_gpu_ms << " ms\n"
              << "    Pipeline overlap:  " << std::setprecision(1) 
              << (stats.pipeline_efficiency * 100.0) << "%\n";
    
    std::cout << "\n  Per-group timing (last iteration, streamed only):\n"
              << "    " << std::setw(6) << "Group"
              << std::setw(12) << "IO wait"
              << std::setw(10) << "Swap"
              << std::setw(10) << "GPU"
              << std::setw(12) << "GPU fence"
              << std::setw(12) << "Total"
              << "\n    " << std::string(62, '-') << "\n";
    
    for (size_t g = 0; g < stats.groups.size(); ++g) {
        const auto& gt = stats.groups[g];
        double group_total = gt.io_wait_ms + gt.swap_ms + gt.gpu_ms + gt.gpu_fence_ms;
        std::cout << "    " << std::setw(6) << g
                  << std::setw(10) << std::setprecision(1) << gt.io_wait_ms << " ms"
                  << std::setw(8) << std::setprecision(2) << gt.swap_ms << " ms"
                  << std::setw(8) << std::setprecision(1) << gt.gpu_ms << " ms"
                  << std::setw(10) << std::setprecision(2) << gt.gpu_fence_ms << " ms"
                  << std::setw(10) << std::setprecision(1) << group_total << " ms"
                  << "\n";
    }
    
    // ====================================================================
    // Phase 5: IO-only pipeline (measure pure streaming overhead)
    // ====================================================================
    
    print_separator("Phase 5: Pipeline IO-Only (GPU = 0ms)");
    std::cout << "  Pipeline with zero GPU time — measures pure IO pipelining overhead\n";
    
    auto io_pipe = benchmark_pipeline(mgr, 0.0, iterations);
    
    std::cout << "  Results:\n"
              << "    Pipeline IO time: " << std::setprecision(1) 
              << io_pipe.total_ms << " ms\n"
              << "    Sequential IO:    " << io_result.total_ms << " ms\n"
              << "    Pipeline overhead: " << std::setprecision(1)
              << (io_pipe.total_ms - io_result.total_ms) << " ms  ("
              << std::setprecision(1)
              << ((io_pipe.total_ms / io_result.total_ms - 1.0) * 100.0) << "% overhead)\n";
    
    // ====================================================================
    // Phase 6: IO thread scaling (Step 8 tuning)
    // ====================================================================
    
    print_separator("Phase 6: IO Thread Scaling");
    std::cout << "  Testing 1, 2, 4, 8 IO threads...\n\n";
    std::cout << "  " << std::setw(10) << "Threads"
              << std::setw(14) << "IO Only (ms)"
              << std::setw(14) << "IO GB/s"
              << std::setw(16) << "Pipeline (ms)"
              << std::setw(10) << "Est. TPS"
              << "\n  " << std::string(64, '-') << "\n";
    
    for (uint32_t threads : {1u, 2u, 4u, 8u}) {
        // Reinitialize with different thread count
        DenseStreamingConfig cfg2 = config;
        cfg2.num_io_threads = threads;
        
        DenseWeightStreamingManager mgr2(engine, cfg2);
        if (!mgr2.initialize()) {
            std::cerr << "  Failed to initialize with " << threads << " threads\n";
            continue;
        }
        
        // IO-only
        auto io_r = benchmark_io_only(mgr2, iterations);
        // Pipeline with GPU
        auto pipe_r = benchmark_pipeline(mgr2, gpu_per_group, iterations);
        
        std::cout << "  " << std::setw(10) << threads
                  << std::setw(12) << std::setprecision(1) << io_r.total_ms << " ms"
                  << std::setw(12) << std::setprecision(2) << io_r.throughput_gbps << " GB/s"
                  << std::setw(14) << std::setprecision(1) << pipe_r.total_ms << " ms"
                  << std::setw(10) << std::setprecision(2) << pipe_r.estimated_tps
                  << "\n";
    }
    
    // ====================================================================
    // Phase 7: Overlap analysis
    // ====================================================================
    
    print_separator("Phase 7: IO/GPU Overlap Analysis");
    std::cout << "  Measuring how much IO is hidden behind GPU compute...\n";
    
    auto overlap = benchmark_overlap(mgr, gpu_per_group, iterations);
    
    double serial_ms = overlap.io_only_ms + gpu_per_group * mgr.num_groups();
    std::cout << "  IO-only pipeline:   " << std::setprecision(1) 
              << overlap.io_only_ms << " ms\n"
              << "  GPU total:          " << (gpu_per_group * mgr.num_groups()) << " ms\n"
              << "  Expected serial:    " << serial_ms << " ms\n"
              << "  Actual with GPU:    " << overlap.with_gpu_ms << " ms\n"
              << "  Time saved:         " << overlap.overlap_saved_ms << " ms\n"
              << "  Overlap ratio:      " << std::setprecision(0) 
              << (overlap.overlap_ratio * 100.0) << "%\n";
    
    // ====================================================================
    // Summary
    // ====================================================================
    
    print_separator("Summary");
    
    double mem_per_buffer_mb = 0;
    for (uint32_t g = 0; g < mgr.num_groups(); ++g) {
        double mb = mgr.group_bytes(g) / (1024.0*1024.0);
        if (mb > mem_per_buffer_mb) mem_per_buffer_mb = mb;
    }
    double total_streaming_mem_mb = mem_per_buffer_mb * 2;  // double buffer
    
    std::cout << std::fixed;
    std::cout << "  Model total weights:     " << std::setprecision(2) 
              << (mgr.total_weight_bytes() / (1024.0*1024.0*1024.0)) << " GB\n"
              << "  Streaming buffer (2x):   " << std::setprecision(0) 
              << total_streaming_mem_mb << " MB ("
              << std::setprecision(2) << (total_streaming_mem_mb / 1024.0) << " GB)\n"
              << "  Memory saved:            " << std::setprecision(2)
              << ((mgr.total_weight_bytes() / (1024.0*1024.0) - total_streaming_mem_mb) / 1024.0)
              << " GB\n"
              << "\n"
              << "  IO throughput:           " << std::setprecision(1)
              << io_result.throughput_gbps << " GB/s\n"
              << "  Baseline (all in RAM):   " << BASELINE_TPOT_MS << " ms/token  (24 tps)\n"
              << "  Streamed pipeline:       " << std::setprecision(1) 
              << pipe_result.total_ms << " ms/token  ("
              << std::setprecision(2) << pipe_result.estimated_tps << " tps)\n"
              << "  Streaming penalty:       " << std::setprecision(1)
              << (pipe_result.total_ms / BASELINE_TPOT_MS) << "x slowdown\n"
              << "\n"
              << "  Bottleneck:              " 
              << (stats.total_io_wait_ms > stats.total_gpu_ms ? "IO-bound" : "GPU-bound")
              << "\n";
    
    // Prediction for 8 GB system
    // With H5+T5 hybrid:
    //   - Non-decoder weights (embed + vision + lm_head): ~1480 MB
    //   - Pinned head+tail decoder: ~477 MB
    //   - Streaming buffers: 2 x max_group_size
    //   - KV cache + runtime: ~1.2 GB
    double non_decoder_gb = 1.48;   // embeddings, vision encoder, lm_head
    double pinned_decoder_gb = (mgr.pin_head_layers() + mgr.pin_tail_layers()) * 48.0 / 1024.0;
    double kv_runtime_gb = 1.2;
    double streaming_gb = total_streaming_mem_mb / 1024.0;
    double predicted_rss_gb = non_decoder_gb + pinned_decoder_gb + streaming_gb + kv_runtime_gb;
    
    std::cout << "\n  8 GB system prediction (H" << mgr.pin_head_layers() 
              << "+T" << mgr.pin_tail_layers() << " hybrid):\n"
              << "    Non-decoder weights:   " << std::setprecision(2) << non_decoder_gb << " GB\n"
              << "    Pinned decoder:        " << pinned_decoder_gb << " GB  ("
              << (mgr.pin_head_layers() + mgr.pin_tail_layers()) << " layers)\n"
              << "    Streaming buffers:     " << streaming_gb << " GB\n"
              << "    KV cache + runtime:    " << kv_runtime_gb << " GB\n"
              << "    Predicted RSS:         " << predicted_rss_gb << " GB\n"
              << "    Fits in 8 GB:          " 
              << (predicted_rss_gb <= 8.0 ? "YES" : "NO (need fewer pinned layers)")
              << "\n";
    
    timeEndPeriod(1);
    return 0;
}
