// Dense Weight Streaming — IO Benchmark v3
// ==========================================================
// Focus: Direct I/O vs DirectStorage for sequential dense weight loading
// mmap removed (creates memory pressure on 8 GB systems)
//
// Methods:
//   1. ReadFile (Buffered)     — FILE_FLAG_SEQUENTIAL_SCAN, single thread
//   2. ReadFile (Direct I/O)   — FILE_FLAG_NO_BUFFERING, single thread
//   3. ReadFile (Direct I/O)   — multi-threaded (N threads per group)
//   4. DirectStorage           — single queue, batched 64 MB chunks
//   5. DirectStorage           — multi-queue parallel (N queues per group)
//
// All methods:
//   - Same per-group loop with per-group timing
//   - Read exactly g.raw_bytes per group into same VirtualAlloc'd buffer
//   - 3 warmup + 10 measured iterations
//   - Unified IO_CHUNK_SIZE = 64 MB
//
// Build (without DirectStorage):
//   cl /EHsc /O2 /std:c++17 benchmark_directstorage_io.cpp /link /out:benchmark_ds.exe
//
// Build (with DirectStorage):
//   cl /EHsc /O2 /std:c++17 /DOPENVINO_USE_DIRECTSTORAGE
//      /I<ds-sdk>/include benchmark_directstorage_io.cpp
//      /link dstorage.lib /LIBPATH:<ds-sdk>/lib/x64 /out:benchmark_ds.exe

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>
#include <thread>
#include <atomic>
#include <immintrin.h>

#ifdef OPENVINO_USE_DIRECTSTORAGE
#include <dstorage.h>
#pragma comment(lib, "dstorage.lib")
#endif

// ============================================================================
// File format (matches pack_dense_weights.py output)
// ============================================================================
#pragma pack(push, 1)
struct DenseWeightsFileHeader {
    char magic[4];         // "DNSW"
    uint32_t version;
    uint32_t num_layers;
    uint32_t num_groups;
    uint32_t group_size;
    uint32_t reserved_0;
    uint64_t total_weight_bytes;
    uint64_t total_file_size;
    uint32_t sector_size;
};

struct GroupTableEntry {
    uint64_t file_offset;
    uint64_t raw_bytes;
    uint64_t aligned_bytes;
    uint32_t first_layer;
    uint32_t num_layers;
};
#pragma pack(pop)

static constexpr size_t SECTOR_SIZE = 4096;
static constexpr uint64_t IO_CHUNK_SIZE = 64ULL * 1024 * 1024;  // 64 MB

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Statistics
// ============================================================================
struct Stats {
    double min_ms, avg_ms, median_ms, max_ms;
    double throughput_gbps;
};

Stats compute_stats(const std::vector<double>& times_ms, uint64_t bytes) {
    if (times_ms.empty()) return {0, 0, 0, 0, 0};
    std::vector<double> sorted = times_ms;
    std::sort(sorted.begin(), sorted.end());
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    double avg = sum / sorted.size();
    double median = sorted[sorted.size() / 2];
    double gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (avg / 1000.0);
    return {sorted.front(), avg, median, sorted.back(), gbps};
}

// ============================================================================
// Unified benchmark runner
// ============================================================================
using ReadGroupFn = std::function<bool(const GroupTableEntry& group, void* dest)>;

struct MethodResult {
    std::string name;
    double total_ms;
    double throughput_gbps;
    uint64_t total_bytes;
    std::vector<Stats> per_group_stats;
    Stats overall_stats;
};

MethodResult run_benchmark(const std::string& name,
                           const std::vector<GroupTableEntry>& groups,
                           void* buffer,
                           int warmup_iters,
                           int measure_iters,
                           ReadGroupFn read_fn) {
    MethodResult result;
    result.name = name;
    result.total_bytes = 0;

    uint32_t num_groups = static_cast<uint32_t>(groups.size());
    std::vector<std::vector<double>> per_group_times(num_groups);
    std::vector<double> iteration_times;

    // Warmup
    for (int iter = 0; iter < warmup_iters; ++iter) {
        for (uint32_t gi = 0; gi < num_groups; ++gi) {
            if (!read_fn(groups[gi], buffer)) {
                printf("  [%s] WARMUP ERROR at group %u\n", name.c_str(), gi);
                return result;
            }
        }
    }

    // Measured iterations
    for (int iter = 0; iter < measure_iters; ++iter) {
        auto t_iter_start = Clock::now();

        for (uint32_t gi = 0; gi < num_groups; ++gi) {
            auto t0 = Clock::now();
            if (!read_fn(groups[gi], buffer)) {
                printf("  [%s] ERROR at iter %d group %u\n", name.c_str(), iter, gi);
                return result;
            }
            auto t1 = Clock::now();
            per_group_times[gi].push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        auto t_iter_end = Clock::now();
        iteration_times.push_back(
            std::chrono::duration<double, std::milli>(t_iter_end - t_iter_start).count());

        uint64_t bytes_per_iter = 0;
        for (const auto& g : groups) bytes_per_iter += g.raw_bytes;
        result.total_bytes += bytes_per_iter;
    }

    result.total_ms = std::accumulate(iteration_times.begin(), iteration_times.end(), 0.0);
    result.throughput_gbps = (result.total_bytes / (1024.0*1024.0*1024.0)) / (result.total_ms / 1000.0);

    result.per_group_stats.resize(num_groups);
    for (uint32_t gi = 0; gi < num_groups; ++gi)
        result.per_group_stats[gi] = compute_stats(per_group_times[gi], groups[gi].raw_bytes);

    uint64_t bpi = 0;
    for (const auto& g : groups) bpi += g.raw_bytes;
    result.overall_stats = compute_stats(iteration_times, bpi);

    return result;
}

// ============================================================================
// Print result
// ============================================================================
void print_result(const MethodResult& r, const std::vector<GroupTableEntry>& groups,
                  double gpu_compute_ms) {
    uint64_t bpi = 0;
    for (const auto& g : groups) bpi += g.raw_bytes;
    int niters = (int)(r.total_bytes / bpi);

    printf("  Aggregate: %.1f ms over %d iters | %.2f GB | %.2f GB/s\n",
           r.total_ms, niters, r.total_bytes / (1024.0*1024.0*1024.0), r.throughput_gbps);
    printf("  Per-iteration: min=%.1f  avg=%.1f  median=%.1f  max=%.1f ms  | %.2f GB/s\n",
           r.overall_stats.min_ms, r.overall_stats.avg_ms,
           r.overall_stats.median_ms, r.overall_stats.max_ms,
           r.overall_stats.throughput_gbps);

    double io_ms = r.overall_stats.median_ms;
    double tpot_serial = io_ms + gpu_compute_ms;
    double tpot_overlap = std::max(io_ms, gpu_compute_ms);
    printf("  TPOT: IO=%.1fms + GPU=%.1fms -> serial=%.1fms (%.1f tps) | overlap=%.1fms (%.1f tps)\n",
           io_ms, gpu_compute_ms, tpot_serial, 1000.0 / tpot_serial,
           tpot_overlap, 1000.0 / tpot_overlap);

    printf("\n  Per-group breakdown (median):\n");
    printf("    %5s  %6s  %8s  %8s  %8s  %8s  %8s\n",
           "Group", "Size", "Min", "Avg", "Median", "Max", "GB/s");
    for (uint32_t gi = 0; gi < groups.size(); ++gi) {
        const auto& s = r.per_group_stats[gi];
        printf("    %5u  %5.1fMB  %6.1fms  %6.1fms  %6.1fms  %6.1fms  %6.2f\n",
               gi, groups[gi].raw_bytes / (1024.0*1024.0),
               s.min_ms, s.avg_ms, s.median_ms, s.max_ms, s.throughput_gbps);
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char* argv[]) {
    std::string file_path = R"(C:\working\gemma4-openvino\gemma-4-E4B-it-ov\dense_weights_streaming.bin)";
    int num_threads = 4;
    if (argc > 1) file_path = argv[1];
    if (argc > 2) num_threads = std::atoi(argv[2]);

    printf("================================================================\n");
    printf("  Dense Weight Streaming - IO Benchmark v3\n");
    printf("  Focus: Direct I/O vs DirectStorage\n");
    printf("================================================================\n");
    printf("  File: %s\n", file_path.c_str());
    printf("  Threads: %d\n", num_threads);

    // --- Read header & group table ---
    HANDLE hFile = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        printf("ERROR: Cannot open file (error %lu)\n", GetLastError());
        return 1;
    }

    DenseWeightsFileHeader header = {};
    DWORD bytes_read = 0;
    ReadFile(hFile, &header, sizeof(header), &bytes_read, nullptr);
    if (memcmp(header.magic, "DNSW", 4) != 0) {
        printf("ERROR: Invalid file magic.\n");
        CloseHandle(hFile);
        return 1;
    }

    SetFilePointer(hFile, SECTOR_SIZE, nullptr, FILE_BEGIN);
    std::vector<GroupTableEntry> groups(header.num_groups);
    ReadFile(hFile, groups.data(), header.num_groups * sizeof(GroupTableEntry), &bytes_read, nullptr);
    CloseHandle(hFile);

    uint64_t total_weight_bytes = 0;
    uint64_t max_group_aligned = 0;
    for (const auto& g : groups) {
        total_weight_bytes += g.raw_bytes;
        max_group_aligned = std::max(max_group_aligned, g.aligned_bytes);
    }

    printf("  Groups: %u x %u layers | Total: %.3f GB\n",
           header.num_groups, header.group_size,
           total_weight_bytes / (1024.0*1024.0*1024.0));
    printf("  Group sizes: ");
    for (uint32_t i = 0; i < header.num_groups; ++i)
        printf("%.1fMB ", groups[i].raw_bytes / (1024.0*1024.0));
    printf("\n");

    // Allocate page-aligned buffer
    void* buffer = VirtualAlloc(nullptr, max_group_aligned,
                                MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buffer) {
        printf("ERROR: VirtualAlloc failed\n");
        return 1;
    }
    memset(buffer, 0, max_group_aligned);
    printf("  Buffer: %.1f MB (VirtualAlloc, page-aligned)\n", max_group_aligned / (1024.0*1024.0));
    printf("  IO chunk: %llu MB\n", IO_CHUNK_SIZE / (1024*1024));

    int warmup = 3, iters = 10;
    double gpu_ms = 42.0;
    printf("  Warmup: %d | Measured: %d iterations\n", warmup, iters);
    printf("  GPU compute (baseline): %.1f ms\n\n", gpu_ms);

    // ====================================================================
    // [1] ReadFile (Buffered)
    // ====================================================================
    printf("----------------------------------------------------------------\n");
    printf("  [1] ReadFile (Buffered, SEQUENTIAL_SCAN)\n");
    printf("----------------------------------------------------------------\n");
    {
        HANDLE h = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            printf("  ERROR: open failed\n");
        } else {
            auto read_fn = [&](const GroupTableEntry& g, void* dest) -> bool {
                LARGE_INTEGER off; off.QuadPart = (LONGLONG)g.file_offset;
                SetFilePointerEx(h, off, nullptr, FILE_BEGIN);
                uint64_t rem = g.raw_bytes;
                uint8_t* d = (uint8_t*)dest;
                while (rem > 0) {
                    DWORD chunk = (DWORD)std::min(rem, IO_CHUNK_SIZE);
                    DWORD br = 0;
                    if (!ReadFile(h, d, chunk, &br, nullptr) || br == 0) return false;
                    d += br; rem -= br;
                }
                return true;
            };
            auto r = run_benchmark("Buffered", groups, buffer, warmup, iters, read_fn);
            print_result(r, groups, gpu_ms);
            CloseHandle(h);
        }
    }

    // ====================================================================
    // [2] ReadFile (Direct I/O, single thread)
    // ====================================================================
    printf("\n----------------------------------------------------------------\n");
    printf("  [2] ReadFile (Direct I/O, single thread)\n");
    printf("----------------------------------------------------------------\n");
    {
        HANDLE h = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            printf("  ERROR: open failed\n");
        } else {
            auto read_fn = [&](const GroupTableEntry& g, void* dest) -> bool {
                uint64_t aligned_offset = (g.file_offset / SECTOR_SIZE) * SECTOR_SIZE;
                uint64_t padding = g.file_offset - aligned_offset;
                uint64_t read_size = ((g.raw_bytes + padding + SECTOR_SIZE - 1) / SECTOR_SIZE) * SECTOR_SIZE;
                uint64_t rem = read_size;
                uint8_t* d = (uint8_t*)dest;
                uint64_t cur = aligned_offset;
                while (rem > 0) {
                    DWORD chunk = (DWORD)std::min(rem, IO_CHUNK_SIZE);
                    OVERLAPPED ov = {};
                    ov.Offset = (DWORD)(cur & 0xFFFFFFFF);
                    ov.OffsetHigh = (DWORD)(cur >> 32);
                    DWORD br = 0;
                    if (!ReadFile(h, d, chunk, &br, &ov) || br == 0) return false;
                    d += br; rem -= br; cur += br;
                }
                return true;
            };
            auto r = run_benchmark("DirectIO-1T", groups, buffer, warmup, iters, read_fn);
            print_result(r, groups, gpu_ms);
            CloseHandle(h);
        }
    }

    // ====================================================================
    // [3] ReadFile (Direct I/O, multi-threaded per group)
    //     Split each group's 338MB into N thread chunks, read in parallel
    // ====================================================================
    printf("\n----------------------------------------------------------------\n");
    printf("  [3] ReadFile (Direct I/O, %d threads per group)\n", num_threads);
    printf("----------------------------------------------------------------\n");
    {
        HANDLE h = CreateFileA(file_path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            printf("  ERROR: open failed\n");
        } else {
            auto read_fn = [&](const GroupTableEntry& g, void* dest) -> bool {
                uint64_t aligned_offset = (g.file_offset / SECTOR_SIZE) * SECTOR_SIZE;
                uint64_t padding = g.file_offset - aligned_offset;
                uint64_t total_read = ((g.raw_bytes + padding + SECTOR_SIZE - 1) / SECTOR_SIZE) * SECTOR_SIZE;

                // Split into thread_count chunks
                uint64_t per_thread = ((total_read / num_threads + SECTOR_SIZE - 1) / SECTOR_SIZE) * SECTOR_SIZE;

                std::atomic<bool> all_ok{true};
                std::vector<std::thread> threads;

                for (int t = 0; t < num_threads; ++t) {
                    uint64_t t_offset = aligned_offset + t * per_thread;
                    uint64_t t_size = std::min(per_thread, total_read - t * per_thread);
                    if (t * per_thread >= total_read) break;

                    threads.emplace_back([&, t_offset, t_size, t]() {
                        uint8_t* d = (uint8_t*)dest + (t_offset - aligned_offset);
                        uint64_t rem = t_size;
                        uint64_t cur = t_offset;
                        while (rem > 0 && all_ok.load()) {
                            DWORD chunk = (DWORD)std::min(rem, IO_CHUNK_SIZE);
                            OVERLAPPED ov = {};
                            ov.Offset = (DWORD)(cur & 0xFFFFFFFF);
                            ov.OffsetHigh = (DWORD)(cur >> 32);
                            ov.hEvent = CreateEvent(nullptr, TRUE, FALSE, nullptr);
                            DWORD br = 0;
                            BOOL ok = ReadFile(h, d, chunk, &br, &ov);
                            if (!ok && GetLastError() == ERROR_IO_PENDING) {
                                GetOverlappedResult(h, &ov, &br, TRUE);
                            } else if (!ok) {
                                all_ok = false;
                            }
                            CloseHandle(ov.hEvent);
                            if (br == 0) { all_ok = false; break; }
                            d += br; rem -= br; cur += br;
                        }
                    });
                }
                for (auto& t : threads) t.join();
                return all_ok.load();
            };
            auto r = run_benchmark("DirectIO-MT", groups, buffer, warmup, iters, read_fn);
            print_result(r, groups, gpu_ms);
            CloseHandle(h);
        }
    }

    // ====================================================================
    // [4] DirectStorage (single queue, batched chunks per group)
    // ====================================================================
#ifdef OPENVINO_USE_DIRECTSTORAGE
    printf("\n----------------------------------------------------------------\n");
    printf("  [4] DirectStorage (single queue, batched)\n");
    printf("----------------------------------------------------------------\n");
    {
        IDStorageFactory* factory = nullptr;
        HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            printf("  ERROR: DStorageGetFactory failed (0x%08X)\n", hr);
        } else {
            factory->SetStagingBufferSize(static_cast<uint32_t>(IO_CHUNK_SIZE));

            DSTORAGE_QUEUE_DESC qd = {};
            qd.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
            qd.Priority = DSTORAGE_PRIORITY_HIGH;
            qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            qd.Device = nullptr;

            IDStorageQueue* queue = nullptr;
            IDStorageStatusArray* sa = nullptr;
            factory->CreateQueue(&qd, IID_PPV_ARGS(&queue));
            factory->CreateStatusArray(1, nullptr, IID_PPV_ARGS(&sa));

            std::wstring wp(file_path.begin(), file_path.end());
            IDStorageFile* dsFile = nullptr;
            factory->OpenFile(wp.c_str(), IID_PPV_ARGS(&dsFile));

            if (!queue || !sa || !dsFile) {
                printf("  ERROR: DS init failed\n");
            } else {
                auto read_fn = [&](const GroupTableEntry& g, void* dest) -> bool {
                    uint64_t rem = g.raw_bytes;
                    uint64_t off = g.file_offset;
                    uint8_t* d = (uint8_t*)dest;
                    while (rem > 0) {
                        uint32_t sz = (uint32_t)std::min(rem, IO_CHUNK_SIZE);
                        DSTORAGE_REQUEST req = {};
                        req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                        req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                        req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                        req.Source.File.Source = dsFile;
                        req.Source.File.Offset = off;
                        req.Source.File.Size = sz;
                        req.Destination.Memory.Buffer = d;
                        req.Destination.Memory.Size = sz;
                        req.UncompressedSize = sz;
                        queue->EnqueueRequest(&req);
                        rem -= sz; off += sz; d += sz;
                    }
                    queue->EnqueueStatus(sa, 0);
                    queue->Submit();
                    while (!sa->IsComplete(0)) { _mm_pause(); }
                    return SUCCEEDED(sa->GetHResult(0));
                };
                auto r = run_benchmark("DS-Batched-1Q", groups, buffer, warmup, iters, read_fn);
                print_result(r, groups, gpu_ms);
            }
            if (dsFile) dsFile->Release();
            if (sa) sa->Release();
            if (queue) queue->Release();
            factory->Release();
        }
    }

    // ====================================================================
    // [5] DirectStorage (multi-queue parallel per group)
    //     Split each group into N parts, one queue per part, all submit
    // ====================================================================
    printf("\n----------------------------------------------------------------\n");
    printf("  [5] DirectStorage (%d queues, parallel per group)\n", num_threads);
    printf("----------------------------------------------------------------\n");
    {
        IDStorageFactory* factory = nullptr;
        HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            printf("  ERROR: DStorageGetFactory failed (0x%08X)\n", hr);
        } else {
            factory->SetStagingBufferSize(static_cast<uint32_t>(IO_CHUNK_SIZE));

            // Create N queues + status arrays
            struct DSQ {
                IDStorageQueue* queue = nullptr;
                IDStorageStatusArray* sa = nullptr;
            };
            std::vector<DSQ> dsqs(num_threads);
            bool init_ok = true;
            for (int q = 0; q < num_threads; ++q) {
                DSTORAGE_QUEUE_DESC qd = {};
                qd.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
                qd.Priority = DSTORAGE_PRIORITY_HIGH;
                qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                qd.Device = nullptr;
                if (FAILED(factory->CreateQueue(&qd, IID_PPV_ARGS(&dsqs[q].queue)))) init_ok = false;
                if (FAILED(factory->CreateStatusArray(1, nullptr, IID_PPV_ARGS(&dsqs[q].sa)))) init_ok = false;
            }

            std::wstring wp(file_path.begin(), file_path.end());
            IDStorageFile* dsFile = nullptr;
            factory->OpenFile(wp.c_str(), IID_PPV_ARGS(&dsFile));

            if (!init_ok || !dsFile) {
                printf("  ERROR: DS multi-queue init failed\n");
            } else {
                auto read_fn = [&](const GroupTableEntry& g, void* dest) -> bool {
                    // Split group into N parts
                    uint64_t per_q = ((g.raw_bytes / num_threads + SECTOR_SIZE - 1) / SECTOR_SIZE) * SECTOR_SIZE;

                    std::vector<std::thread> threads;
                    std::atomic<bool> all_ok{true};

                    for (int q = 0; q < num_threads; ++q) {
                        uint64_t q_start = q * per_q;
                        if (q_start >= g.raw_bytes) break;
                        uint64_t q_bytes = std::min(per_q, g.raw_bytes - q_start);

                        threads.emplace_back([&, q, q_start, q_bytes]() {
                            uint64_t rem = q_bytes;
                            uint64_t off = g.file_offset + q_start;
                            uint8_t* d = (uint8_t*)dest + q_start;

                            while (rem > 0) {
                                uint32_t sz = (uint32_t)std::min(rem, IO_CHUNK_SIZE);
                                DSTORAGE_REQUEST req = {};
                                req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                                req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                                req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                                req.Source.File.Source = dsFile;
                                req.Source.File.Offset = off;
                                req.Source.File.Size = sz;
                                req.Destination.Memory.Buffer = d;
                                req.Destination.Memory.Size = sz;
                                req.UncompressedSize = sz;
                                dsqs[q].queue->EnqueueRequest(&req);
                                rem -= sz; off += sz; d += sz;
                            }

                            dsqs[q].queue->EnqueueStatus(dsqs[q].sa, 0);
                            dsqs[q].queue->Submit();
                            while (!dsqs[q].sa->IsComplete(0)) { _mm_pause(); }
                            if (FAILED(dsqs[q].sa->GetHResult(0))) all_ok = false;
                        });
                    }
                    for (auto& t : threads) t.join();
                    return all_ok.load();
                };
                auto r = run_benchmark("DS-Parallel-NQ", groups, buffer, warmup, iters, read_fn);
                print_result(r, groups, gpu_ms);
            }

            for (auto& dq : dsqs) {
                if (dq.queue) dq.queue->Release();
                if (dq.sa) dq.sa->Release();
            }
            if (dsFile) dsFile->Release();
            factory->Release();
        }
    }

    // ====================================================================
    // [6] DirectStorage (single queue, NO chunking — full group as one request)
    //     Test: does removing chunk overhead help?
    // ====================================================================
    printf("\n----------------------------------------------------------------\n");
    printf("  [6] DirectStorage (single queue, full-group request, larger staging)\n");
    printf("----------------------------------------------------------------\n");
    {
        IDStorageFactory* factory = nullptr;
        HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
        if (FAILED(hr)) {
            printf("  ERROR: DStorageGetFactory failed (0x%08X)\n", hr);
        } else {
            // Set staging buffer large enough for full group (~342 MB)
            uint32_t staging_sz = (uint32_t)std::min((uint64_t)512*1024*1024, max_group_aligned);
            factory->SetStagingBufferSize(staging_sz);
            printf("  Staging buffer: %u MB\n", staging_sz / (1024*1024));

            DSTORAGE_QUEUE_DESC qd = {};
            qd.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
            qd.Priority = DSTORAGE_PRIORITY_HIGH;
            qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
            qd.Device = nullptr;

            IDStorageQueue* queue = nullptr;
            IDStorageStatusArray* sa = nullptr;
            factory->CreateQueue(&qd, IID_PPV_ARGS(&queue));
            factory->CreateStatusArray(1, nullptr, IID_PPV_ARGS(&sa));

            std::wstring wp(file_path.begin(), file_path.end());
            IDStorageFile* dsFile = nullptr;
            factory->OpenFile(wp.c_str(), IID_PPV_ARGS(&dsFile));

            if (!queue || !sa || !dsFile) {
                printf("  ERROR: DS init failed\n");
            } else {
                auto read_fn = [&](const GroupTableEntry& g, void* dest) -> bool {
                    // Single request for entire group — no chunking
                    // DS 1.3 max request size check: if >32MB, may need chunking
                    // Try full group first; if it fails, fall back to chunks
                    uint64_t rem = g.raw_bytes;
                    uint64_t off = g.file_offset;
                    uint8_t* d = (uint8_t*)dest;

                    // Use larger chunks (staging buffer sized)
                    uint64_t chunk_size = (uint64_t)staging_sz;
                    while (rem > 0) {
                        uint32_t sz = (uint32_t)std::min(rem, chunk_size);
                        DSTORAGE_REQUEST req = {};
                        req.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
                        req.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
                        req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
                        req.Source.File.Source = dsFile;
                        req.Source.File.Offset = off;
                        req.Source.File.Size = sz;
                        req.Destination.Memory.Buffer = d;
                        req.Destination.Memory.Size = sz;
                        req.UncompressedSize = sz;
                        queue->EnqueueRequest(&req);
                        rem -= sz; off += sz; d += sz;
                    }
                    queue->EnqueueStatus(sa, 0);
                    queue->Submit();
                    while (!sa->IsComplete(0)) { _mm_pause(); }

                    HRESULT result = sa->GetHResult(0);
                    if (FAILED(result)) {
                        printf("    DS error: 0x%08X (staging=%u MB, group=%.1f MB)\n",
                               result, staging_sz/(1024*1024), g.raw_bytes/(1024.0*1024.0));
                        return false;
                    }
                    return true;
                };
                auto r = run_benchmark("DS-LargeStaging", groups, buffer, warmup, iters, read_fn);
                print_result(r, groups, gpu_ms);
            }
            if (dsFile) dsFile->Release();
            if (sa) sa->Release();
            if (queue) queue->Release();
            factory->Release();
        }
    }
#else
    printf("\n----------------------------------------------------------------\n");
    printf("  [4-6] DirectStorage — SKIPPED (build with /DOPENVINO_USE_DIRECTSTORAGE)\n");
    printf("----------------------------------------------------------------\n");
#endif

    // ====================================================================
    // Summary
    // ====================================================================
    printf("\n================================================================\n");
    printf("  Target Throughput Requirements\n");
    printf("================================================================\n");
    double wb = total_weight_bytes / (1024.0*1024.0*1024.0);
    printf("  Weights per token: %.3f GB | GPU compute: %.1f ms\n\n", wb, gpu_ms);
    printf("  %8s  %8s  %10s  %10s\n", "Target", "TPOT", "IO budget", "Need GB/s");
    double targets[] = {3, 5, 8, 10, 12};
    for (double tps : targets) {
        double tpot = 1000.0 / tps;
        double io_budget = tpot - gpu_ms;
        if (io_budget <= 0)
            printf("  %6.0f tps  %6.1fms  impossible\n", tps, tpot);
        else
            printf("  %6.0f tps  %6.1fms  %7.1f ms  %7.1f\n", tps, tpot, io_budget, wb / (io_budget / 1000.0));
    }

    VirtualFree(buffer, 0, MEM_RELEASE);
    printf("\nDone.\n");
    return 0;
}
