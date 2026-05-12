// ==========================================================================
// DirectStorage vs ReadFile vs mmap Performance Benchmark
// ==========================================================================
// Standalone benchmark for MoE expert weight loading from .bin file.
//
// Tests I/O paths:
//   1. DirectStorage (BypassIO) - Windows 11 22H2+ with NVMe
//      - Batched / Parallel variants at PRIORITY_NORMAL and PRIORITY_HIGH
//   2. ReadFile (buffered) - standard Win32 file I/O
//   3. ReadFile (Direct I/O) - FILE_FLAG_NO_BUFFERING, bypass page cache
//   4. Memory-mapped I/O (mmap) - MapViewOfFile + memcpy
//
// Usage:
//   ds_perf_test.exe --file <path_to_moe_weights.bin>
//                    [--layers <N>]         (default: all from header)
//                    [--topk <K>]           (default: 4)
//                    [--iterations <N>]     (default: 10)
//                    [--method ds|readfile|directio|mmap|all]
//                    [--warmup <N>]         (default: 2)
//                    [--threads <N>]        (default: 4, for DS queues)
//                    [--seed <N>]           (default: 42)
//                    [--purge-cache]        (drop file from page cache between runs)
// ==========================================================================

#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <queue>

#ifdef HAS_DIRECTSTORAGE
#include <dstorage.h>
#pragma comment(lib, "dstorage.lib")
#endif

// ==========================================================================
// File header (must match moe_expert_weight_manager.hpp)
// ==========================================================================
#pragma pack(push, 1)
struct MoEWeightsFileHeader {
    char     magic[4];                // "MOEW"
    uint32_t version;
    uint32_t num_layers;
    uint32_t num_experts_per_layer;
    uint64_t expert_up_weight_size;
    uint64_t expert_down_weight_size;
    uint64_t expert_up_scale_size;
    uint64_t expert_down_scale_size;
    uint64_t expert_up_bias_size;
    uint64_t expert_down_bias_size;
    uint64_t data_offset;
    uint64_t reserved[7];
};
#pragma pack(pop)

// ==========================================================================
// High-resolution timer
// ==========================================================================
struct HRTimer {
    LARGE_INTEGER freq, start, stop;
    HRTimer() { QueryPerformanceFrequency(&freq); }
    void begin() { QueryPerformanceCounter(&start); }
    void end()   { QueryPerformanceCounter(&stop);  }
    double elapsed_ms()  const { return (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart * 1000.0; }
    double elapsed_us()  const { return (double)(stop.QuadPart - start.QuadPart) / freq.QuadPart * 1e6; }
    double elapsed_sec() const { return (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart; }
};

// ==========================================================================
// Expert location: which file offset, which size
// ==========================================================================
struct ExpertRead {
    uint32_t layer_idx;
    uint32_t expert_idx;
    uint64_t up_offset;
    uint64_t up_size;
    uint64_t down_offset;
    uint64_t down_size;
    uint64_t total_size() const { return up_size + down_size; }
};

// ==========================================================================
// Simple thread pool for parallel reads
// ==========================================================================
class SimpleThreadPool {
public:
    SimpleThreadPool(size_t num_threads) : m_stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            m_workers.emplace_back([this, i] { worker_loop(i); });
        }
    }

    ~SimpleThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_cv.notify_all();
        for (auto& w : m_workers) w.join();
    }

    // Submit tasks with thread index
    void submit(std::function<void(size_t)> task) {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_tasks.push(std::move(task));
        }
        m_cv.notify_one();
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_done_cv.wait(lock, [this] { return m_tasks.empty() && m_active == 0; });
    }

    size_t thread_count() const { return m_workers.size(); }

private:
    void worker_loop(size_t thread_idx) {
        while (true) {
            std::function<void(size_t)> task;
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                m_cv.wait(lock, [this] { return m_stop || !m_tasks.empty(); });
                if (m_stop && m_tasks.empty()) return;
                task = std::move(m_tasks.front());
                m_tasks.pop();
                ++m_active;
            }
            task(thread_idx);
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                --m_active;
            }
            m_done_cv.notify_all();
        }
    }

    std::vector<std::thread> m_workers;
    std::queue<std::function<void(size_t)>> m_tasks;
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::condition_variable m_done_cv;
    bool m_stop;
    int m_active = 0;
};

// ==========================================================================
// Statistics helper
// ==========================================================================
struct Stats {
    double min_ms, max_ms, avg_ms, median_ms, total_bytes;
    double throughput_gbps;
    std::vector<double> samples;

    static Stats compute(const std::vector<double>& times_ms, double bytes) {
        Stats s;
        s.samples = times_ms;
        s.total_bytes = bytes;
        std::sort(s.samples.begin(), s.samples.end());
        s.min_ms = s.samples.front();
        s.max_ms = s.samples.back();
        s.avg_ms = std::accumulate(s.samples.begin(), s.samples.end(), 0.0) / s.samples.size();
        s.median_ms = s.samples[s.samples.size() / 2];
        s.throughput_gbps = (bytes / (1024.0 * 1024.0 * 1024.0)) / (s.median_ms / 1000.0);
        return s;
    }

    void print(const char* label) const {
        printf("  %-14s  min=%8.2f ms  avg=%8.2f ms  median=%8.2f ms  max=%8.2f ms  |  %.2f GB/s  (%.1f MB in median)\n",
               label, min_ms, avg_ms, median_ms, max_ms, throughput_gbps,
               total_bytes / (1024.0 * 1024.0));
    }
};

// ==========================================================================
// Command-line arguments
// ==========================================================================
struct Config {
    std::string file_path;
    int layers       = -1;     // -1 = use all from header
    int topk         = 4;
    int iterations   = 10;
    int warmup       = 2;
    int threads      = 4;
    int seed         = 42;
    bool purge_cache = false;
    std::string method = "all";  // ds, readfile, directio, mmap, all
};

Config parse_args(int argc, char* argv[]) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--file"     && i+1 < argc) cfg.file_path   = argv[++i];
        else if (arg == "--layers"    && i+1 < argc) cfg.layers      = std::stoi(argv[++i]);
        else if (arg == "--topk"      && i+1 < argc) cfg.topk        = std::stoi(argv[++i]);
        else if (arg == "--iterations"&& i+1 < argc) cfg.iterations  = std::stoi(argv[++i]);
        else if (arg == "--warmup"    && i+1 < argc) cfg.warmup      = std::stoi(argv[++i]);
        else if (arg == "--threads"   && i+1 < argc) cfg.threads     = std::stoi(argv[++i]);
        else if (arg == "--seed"      && i+1 < argc) cfg.seed        = std::stoi(argv[++i]);
        else if (arg == "--method"    && i+1 < argc) cfg.method      = argv[++i];
        else if (arg == "--purge-cache") cfg.purge_cache = true;
        else if (arg == "--help" || arg == "-h") {
            printf("Usage: ds_perf_test.exe --file <moe_weights.bin> [options]\n\n");
            printf("Options:\n");
            printf("  --file <path>       Path to MoE weights .bin file (required)\n");
            printf("  --layers <N>        Number of layers to test (default: all from header)\n");
            printf("  --topk <K>          Random experts per layer to load (default: 4)\n");
            printf("  --iterations <N>    Benchmark iterations (default: 10)\n");
            printf("  --warmup <N>        Warmup iterations (default: 2)\n");
            printf("  --threads <N>       Thread/queue count for parallel reads (default: 4)\n");
            printf("  --seed <N>          Random seed for expert selection (default: 42)\n");
            printf("  --method <m>        Test method: ds, ds_batched, ds_parallel,\n");
            printf("                                     ds_high, ds_batched_high, ds_parallel_high,\n");
            printf("                                     readfile, directio, mmap, all (default: all)\n");
            printf("  --purge-cache       Purge page cache between iterations (requires admin)\n");
            printf("  -h, --help          Show this help\n");
            exit(0);
        }
    }
    if (cfg.file_path.empty()) {
        fprintf(stderr, "Error: --file <path> is required. Use --help for usage.\n");
        exit(1);
    }
    return cfg;
}

// ==========================================================================
// Read and validate the file header
// ==========================================================================
MoEWeightsFileHeader read_header(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Error: Cannot open file: %s\n", path.c_str());
        exit(1);
    }
    MoEWeightsFileHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "Error: Cannot read header from: %s\n", path.c_str());
        fclose(f);
        exit(1);
    }
    fclose(f);

    if (memcmp(hdr.magic, "MOEW", 4) != 0) {
        fprintf(stderr, "Error: Invalid magic number in header. Expected 'MOEW'.\n");
        exit(1);
    }
    return hdr;
}

// ==========================================================================
// Generate random expert selections for all layers
// ==========================================================================
std::vector<ExpertRead> generate_expert_reads(
    const MoEWeightsFileHeader& hdr, int num_layers, int topk, int seed)
{
    std::mt19937 rng(seed);
    std::vector<ExpertRead> reads;

    uint64_t per_expert_size = hdr.expert_up_weight_size + hdr.expert_down_weight_size +
                               hdr.expert_up_scale_size + hdr.expert_down_scale_size +
                               hdr.expert_up_bias_size + hdr.expert_down_bias_size;
    uint64_t per_layer_size = hdr.num_experts_per_layer * per_expert_size;

    for (int layer = 0; layer < num_layers; ++layer) {
        // Generate random topk expert indices for this layer
        std::vector<int> indices(hdr.num_experts_per_layer);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        int k = std::min(topk, (int)hdr.num_experts_per_layer);
        for (int i = 0; i < k; ++i) {
            ExpertRead r;
            r.layer_idx  = layer;
            r.expert_idx = indices[i];

            uint64_t expert_base = hdr.data_offset + layer * per_layer_size +
                                   r.expert_idx * per_expert_size;
            r.up_offset   = expert_base;
            r.up_size     = hdr.expert_up_weight_size;
            r.down_offset = expert_base + hdr.expert_up_weight_size +
                            hdr.expert_up_scale_size + hdr.expert_up_bias_size;
            r.down_size   = hdr.expert_down_weight_size;
            reads.push_back(r);
        }
    }
    return reads;
}

// ==========================================================================
// Try to purge file from page cache (requires admin)
// ==========================================================================
void try_purge_cache(const std::string& path) {
    // Open with no buffering to force cache purge on close
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                           nullptr, OPEN_EXISTING,
                           FILE_FLAG_NO_BUFFERING | FILE_FLAG_WRITE_THROUGH, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        // Reading a byte with NO_BUFFERING flag helps invalidate cache
        CloseHandle(h);
    }
    // Also try FlushFileBuffers via normal handle
    h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h != INVALID_HANDLE_VALUE) {
        FlushFileBuffers(h);
        CloseHandle(h);
    }
}

// ==========================================================================
// Get sector size for the drive containing the file
// ==========================================================================
DWORD get_sector_size(const std::string& path) {
    std::string root = path.substr(0, 3);  // "C:\"
    DWORD spc, bps, nfc, tc;
    if (GetDiskFreeSpaceA(root.c_str(), &spc, &bps, &nfc, &tc)) {
        return bps;
    }
    return 4096;  // fallback
}

// ==========================================================================
// Check BypassIO support
// ==========================================================================
void check_bypassio(const std::string& path) {
    printf("\n--- BypassIO Check ---\n");
    std::string drive = path.substr(0, 2);  // "C:"

    // Get volume info
    char vol_name[MAX_PATH], fs_name[MAX_PATH];
    DWORD serial, max_comp_len, flags;
    std::string root = path.substr(0, 3);
    if (GetVolumeInformationA(root.c_str(), vol_name, MAX_PATH, &serial,
                              &max_comp_len, &flags, fs_name, MAX_PATH)) {
        printf("  Volume: %s  FileSystem: %s\n", vol_name, fs_name);
    }

    DWORD sector_size = get_sector_size(path);
    printf("  Sector size: %u bytes\n", sector_size);
    printf("  Drive: %s\n", root.c_str());
    printf("  Tip: Run 'fsutil bypassIo state %s' in admin PowerShell to check BypassIO\n", drive.c_str());
}

// ==========================================================================
// BENCHMARK: ReadFile (buffered, standard)
// ==========================================================================
double bench_readfile_buffered(const std::string& path,
                               const std::vector<ExpertRead>& reads,
                               uint8_t* buffer, size_t buffer_size,
                               int threads_count)
{
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "  Error: Cannot open file for ReadFile\n");
        return -1;
    }

    HRTimer timer;
    timer.begin();

    // Parallel reads using thread pool
    SimpleThreadPool pool(threads_count);
    std::atomic<uint64_t> total_read{0};
    size_t per_expert_buf = buffer_size / reads.size();

    for (size_t i = 0; i < reads.size(); ++i) {
        pool.submit([&, i](size_t /*tid*/) {
            const auto& r = reads[i];
            uint8_t* dst = buffer + i * per_expert_buf;

            // Read up weight
            OVERLAPPED ov_up = {};
            ov_up.Offset     = (DWORD)(r.up_offset & 0xFFFFFFFF);
            ov_up.OffsetHigh = (DWORD)(r.up_offset >> 32);
            DWORD bytes_read = 0;
            ReadFile(hFile, dst, (DWORD)r.up_size, &bytes_read, &ov_up);
            total_read += bytes_read;

            // Read down weight
            OVERLAPPED ov_dn = {};
            ov_dn.Offset     = (DWORD)(r.down_offset & 0xFFFFFFFF);
            ov_dn.OffsetHigh = (DWORD)(r.down_offset >> 32);
            ReadFile(hFile, dst + r.up_size, (DWORD)r.down_size, &bytes_read, &ov_dn);
            total_read += bytes_read;
        });
    }
    pool.wait_all();

    timer.end();
    CloseHandle(hFile);
    return timer.elapsed_ms();
}

// ==========================================================================
// BENCHMARK: ReadFile (Direct I/O, FILE_FLAG_NO_BUFFERING)
// ==========================================================================
double bench_readfile_directio(const std::string& path,
                                const std::vector<ExpertRead>& reads,
                                uint8_t* buffer, size_t buffer_size,
                                int threads_count)
{
    DWORD sector_size = get_sector_size(path);

    // Need per-thread aligned temp buffer
    size_t max_read = 0;
    for (auto& r : reads) {
        max_read = std::max(max_read, (size_t)std::max(r.up_size, r.down_size));
    }
    size_t aligned_buf_size = ((max_read + sector_size * 2) / sector_size) * sector_size;

    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_RANDOM_ACCESS,
                               nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "  Error: Cannot open file for Direct I/O (err=%lu)\n", GetLastError());
        return -1;
    }

    // Allocate per-thread aligned temp buffers
    std::vector<uint8_t*> thread_bufs(threads_count);
    for (int t = 0; t < threads_count; ++t) {
        thread_bufs[t] = (uint8_t*)_aligned_malloc(aligned_buf_size, sector_size);
        if (!thread_bufs[t]) {
            fprintf(stderr, "  Error: Failed to allocate aligned buffer for thread %d\n", t);
            CloseHandle(hFile);
            return -1;
        }
    }

    HRTimer timer;
    timer.begin();

    SimpleThreadPool pool(threads_count);
    size_t per_expert_buf = buffer_size / reads.size();

    for (size_t i = 0; i < reads.size(); ++i) {
        pool.submit([&, i](size_t tid) {
            const auto& r = reads[i];
            uint8_t* dst = buffer + i * per_expert_buf;
            uint8_t* tmp = thread_bufs[tid];

            auto read_aligned = [&](uint64_t offset, uint64_t size, uint8_t* dest) {
                uint64_t aligned_off = (offset / sector_size) * sector_size;
                uint64_t off_adj = offset - aligned_off;
                uint64_t read_sz = ((off_adj + size + sector_size - 1) / sector_size) * sector_size;

                OVERLAPPED ov = {};
                ov.Offset     = (DWORD)(aligned_off & 0xFFFFFFFF);
                ov.OffsetHigh = (DWORD)(aligned_off >> 32);
                DWORD bytes_read = 0;
                ReadFile(hFile, tmp, (DWORD)read_sz, &bytes_read, &ov);
                memcpy(dest, tmp + off_adj, (size_t)size);
            };

            read_aligned(r.up_offset, r.up_size, dst);
            read_aligned(r.down_offset, r.down_size, dst + r.up_size);
        });
    }
    pool.wait_all();

    timer.end();
    CloseHandle(hFile);
    for (auto p : thread_bufs) _aligned_free(p);
    return timer.elapsed_ms();
}

// ==========================================================================
// BENCHMARK: Memory-mapped I/O (MapViewOfFile + memcpy)
// ==========================================================================
double bench_mmap(const std::string& path,
                  const std::vector<ExpertRead>& reads,
                  uint8_t* buffer, size_t buffer_size,
                  int threads_count)
{
    HANDLE hFile = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "  Error: Cannot open file for mmap\n");
        return -1;
    }

    LARGE_INTEGER file_size;
    GetFileSizeEx(hFile, &file_size);

    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) {
        fprintf(stderr, "  Error: CreateFileMapping failed (err=%lu)\n", GetLastError());
        CloseHandle(hFile);
        return -1;
    }

    // Map entire file
    void* view = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    if (!view) {
        fprintf(stderr, "  Error: MapViewOfFile failed (err=%lu)\n", GetLastError());
        CloseHandle(hMap);
        CloseHandle(hFile);
        return -1;
    }

    const uint8_t* file_data = (const uint8_t*)view;

    HRTimer timer;
    timer.begin();

    SimpleThreadPool pool(threads_count);
    size_t per_expert_buf = buffer_size / reads.size();

    for (size_t i = 0; i < reads.size(); ++i) {
        pool.submit([&, i](size_t /*tid*/) {
            const auto& r = reads[i];
            uint8_t* dst = buffer + i * per_expert_buf;
            memcpy(dst, file_data + r.up_offset, (size_t)r.up_size);
            memcpy(dst + r.up_size, file_data + r.down_offset, (size_t)r.down_size);
        });
    }
    pool.wait_all();

    timer.end();

    UnmapViewOfFile(view);
    CloseHandle(hMap);
    CloseHandle(hFile);
    return timer.elapsed_ms();
}

// ==========================================================================
// BENCHMARK: DirectStorage (BypassIO)
// ==========================================================================
#ifdef HAS_DIRECTSTORAGE
double bench_directstorage_batched(const std::string& path,
                                    const std::vector<ExpertRead>& reads,
                                    uint8_t* buffer, size_t buffer_size,
                                    int num_queues,
                                    DSTORAGE_PRIORITY priority = DSTORAGE_PRIORITY_NORMAL)
{
    // --- Init DS ---
    IDStorageFactory* factory = nullptr;
    HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        fprintf(stderr, "  Error: DStorageGetFactory failed (hr=0x%08X)\n", (unsigned)hr);
        if (hr == 0x80004001) {
            fprintf(stderr, "    => E_NOTIMPL: DirectStorage runtime not available.\n");
            fprintf(stderr, "    => Check: Windows 11 22H2+, NVMe with BypassIO, stornvme driver.\n");
            fprintf(stderr, "    => Run: fsutil bypassIo state %c:\n", path[0]);
        }
        return -1;
    }

    // Use default staging buffer (32MB) — setting to 0 can cause
    // E_DSTORAGE_REQUEST_TOO_LARGE for large requests on some systems.
    // factory->SetStagingBufferSize(DSTORAGE_STAGING_BUFFER_SIZE_32MB);

    // Enable debug errors if available
    factory->SetDebugFlags(DSTORAGE_DEBUG_SHOW_ERRORS);

    // Open file via DS
    std::wstring wpath(path.begin(), path.end());
    IDStorageFile* ds_file = nullptr;
    hr = factory->OpenFile(wpath.c_str(), IID_PPV_ARGS(&ds_file));
    if (FAILED(hr)) {
        fprintf(stderr, "  Error: DS OpenFile failed (hr=0x%08X)\n", (unsigned)hr);
        factory->Release();
        return -1;
    }

    // Create queues + status arrays
    struct DSQueue {
        IDStorageQueue* queue = nullptr;
        IDStorageStatusArray* status = nullptr;
        HANDLE event = nullptr;
    };
    std::vector<DSQueue> queues(num_queues);

    for (int q = 0; q < num_queues; ++q) {
        DSTORAGE_QUEUE_DESC qd = {};
        qd.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
        qd.Priority   = priority;
        qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        qd.Device     = nullptr;  // CPU destination

        hr = factory->CreateQueue(&qd, IID_PPV_ARGS(&queues[q].queue));
        if (FAILED(hr)) {
            fprintf(stderr, "  Error: CreateQueue[%d] failed (hr=0x%08X)\n", q, (unsigned)hr);
            ds_file->Release();
            factory->Release();
            return -1;
        }
        factory->CreateStatusArray(1, nullptr, IID_PPV_ARGS(&queues[q].status));
        queues[q].event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    }

    size_t per_expert_buf = buffer_size / reads.size();

    HRTimer timer;
    timer.begin();

    // Enqueue all reads to queue 0 as a single batch (like batched DS in moe code)
    {
        auto& dq = queues[0];
        for (size_t i = 0; i < reads.size(); ++i) {
            const auto& r = reads[i];
            uint8_t* dst = buffer + i * per_expert_buf;

            // Up weight
            DSTORAGE_REQUEST req = {};
            req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
            req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_MEMORY;
            req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            req.Source.File.Source         = ds_file;
            req.Source.File.Offset        = r.up_offset;
            req.Source.File.Size          = (UINT32)r.up_size;
            req.Destination.Memory.Buffer = dst;
            req.Destination.Memory.Size   = (UINT32)r.up_size;
            req.UncompressedSize          = (UINT32)r.up_size;
            dq.queue->EnqueueRequest(&req);

            // Down weight
            req.Source.File.Offset        = r.down_offset;
            req.Source.File.Size          = (UINT32)r.down_size;
            req.Destination.Memory.Buffer = dst + r.up_size;
            req.Destination.Memory.Size   = (UINT32)r.down_size;
            req.UncompressedSize          = (UINT32)r.down_size;
            dq.queue->EnqueueRequest(&req);
        }

        dq.queue->EnqueueStatus(dq.status, 0);
        dq.queue->Submit();

        while (!dq.status->IsComplete(0)) {
            std::this_thread::yield();
        }

        hr = dq.status->GetHResult(0);
        if (FAILED(hr)) {
            fprintf(stderr, "  Error: DS batch read failed (hr=0x%08X)\n", (unsigned)hr);
            // Retrieve detailed error info
            DSTORAGE_ERROR_RECORD err = {};
            dq.queue->RetrieveErrorRecord(&err);
            if (err.FailureCount > 0) {
                fprintf(stderr, "    DS Error: FailureCount=%u, CommandType=%u, hr=0x%08X\n",
                        err.FailureCount,
                        (unsigned)err.FirstFailure.CommandType,
                        (unsigned)err.FirstFailure.HResult);
                if (err.FirstFailure.CommandType == DSTORAGE_COMMAND_TYPE_REQUEST) {
                    auto& req_err = err.FirstFailure.Request.Request;
                    fprintf(stderr, "    Request: FileOffset=%llu, FileSize=%u, DestSize=%u\n",
                            (unsigned long long)req_err.Source.File.Offset,
                            req_err.Source.File.Size,
                            req_err.Destination.Memory.Size);
                    fprintf(stderr, "    File: %ls\n", err.FirstFailure.Request.Filename);
                }
            }
        }
    }

    timer.end();

    // Cleanup
    for (auto& dq : queues) {
        if (dq.queue)  dq.queue->Release();
        if (dq.status) dq.status->Release();
        if (dq.event)  CloseHandle(dq.event);
    }
    ds_file->Release();
    factory->Release();

    return timer.elapsed_ms();
}

double bench_directstorage_parallel(const std::string& path,
                                     const std::vector<ExpertRead>& reads,
                                     uint8_t* buffer, size_t buffer_size,
                                     int num_queues,
                                     DSTORAGE_PRIORITY priority = DSTORAGE_PRIORITY_NORMAL)
{
    // --- Init DS ---
    IDStorageFactory* factory = nullptr;
    HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        fprintf(stderr, "  Error: DStorageGetFactory failed (hr=0x%08X)\n", (unsigned)hr);
        return -1;
    }

    // Use default 32MB staging buffer
    factory->SetDebugFlags(DSTORAGE_DEBUG_SHOW_ERRORS);

    std::wstring wpath(path.begin(), path.end());
    IDStorageFile* ds_file = nullptr;
    hr = factory->OpenFile(wpath.c_str(), IID_PPV_ARGS(&ds_file));
    if (FAILED(hr)) {
        fprintf(stderr, "  Error: DS OpenFile failed (hr=0x%08X)\n", (unsigned)hr);
        factory->Release();
        return -1;
    }

    // Per-thread queues
    struct DSQueue {
        IDStorageQueue* queue = nullptr;
        IDStorageStatusArray* status = nullptr;
    };
    std::vector<DSQueue> queues(num_queues);

    for (int q = 0; q < num_queues; ++q) {
        DSTORAGE_QUEUE_DESC qd = {};
        qd.Capacity   = DSTORAGE_MAX_QUEUE_CAPACITY;
        qd.Priority   = priority;
        qd.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        qd.Device     = nullptr;

        factory->CreateQueue(&qd, IID_PPV_ARGS(&queues[q].queue));
        factory->CreateStatusArray(1, nullptr, IID_PPV_ARGS(&queues[q].status));
    }

    size_t per_expert_buf = buffer_size / reads.size();

    HRTimer timer;
    timer.begin();

    // Parallel: each thread uses its own queue
    SimpleThreadPool pool(num_queues);

    for (size_t i = 0; i < reads.size(); ++i) {
        pool.submit([&, i](size_t tid) {
            const auto& r = reads[i];
            uint8_t* dst = buffer + i * per_expert_buf;
            auto& dq = queues[tid];

            // Up weight
            DSTORAGE_REQUEST req = {};
            req.Options.SourceType        = DSTORAGE_REQUEST_SOURCE_FILE;
            req.Options.DestinationType   = DSTORAGE_REQUEST_DESTINATION_MEMORY;
            req.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
            req.Source.File.Source         = ds_file;
            req.Source.File.Offset        = r.up_offset;
            req.Source.File.Size          = (UINT32)r.up_size;
            req.Destination.Memory.Buffer = dst;
            req.Destination.Memory.Size   = (UINT32)r.up_size;
            req.UncompressedSize          = (UINT32)r.up_size;
            dq.queue->EnqueueRequest(&req);

            // Down weight
            req.Source.File.Offset        = r.down_offset;
            req.Source.File.Size          = (UINT32)r.down_size;
            req.Destination.Memory.Buffer = dst + r.up_size;
            req.Destination.Memory.Size   = (UINT32)r.down_size;
            req.UncompressedSize          = (UINT32)r.down_size;
            dq.queue->EnqueueRequest(&req);

            dq.queue->EnqueueStatus(dq.status, 0);
            dq.queue->Submit();

            while (!dq.status->IsComplete(0)) {
                std::this_thread::yield();
            }

            HRESULT rhr = dq.status->GetHResult(0);
            if (FAILED(rhr)) {
                fprintf(stderr, "  Error: DS parallel read failed (tid=%zu, hr=0x%08X)\n",
                        tid, (unsigned)rhr);
            }
        });
    }
    pool.wait_all();

    timer.end();

    for (auto& dq : queues) {
        if (dq.queue)  dq.queue->Release();
        if (dq.status) dq.status->Release();
    }
    ds_file->Release();
    factory->Release();

    return timer.elapsed_ms();
}
#endif // HAS_DIRECTSTORAGE

// ==========================================================================
// Per-layer breakdown benchmark
// ==========================================================================
struct LayerResult {
    int layer_idx;
    double time_ms;
    double bytes;
    int experts_loaded;
};

template<typename BenchFn>
std::vector<LayerResult> bench_per_layer(
    const MoEWeightsFileHeader& /*hdr*/,
    const std::vector<ExpertRead>& all_reads,
    int num_layers, int /*topk*/,
    uint8_t* buffer, size_t buffer_size,
    BenchFn bench_fn)
{
    std::vector<LayerResult> results;

    for (int layer = 0; layer < num_layers; ++layer) {
        // Collect reads for this layer
        std::vector<ExpertRead> layer_reads;
        for (auto& r : all_reads) {
            if ((int)r.layer_idx == layer) {
                layer_reads.push_back(r);
            }
        }
        if (layer_reads.empty()) continue;

        double total_bytes = 0;
        for (auto& r : layer_reads) total_bytes += r.total_size();

        double ms = bench_fn(layer_reads, buffer, buffer_size);

        LayerResult lr;
        lr.layer_idx = layer;
        lr.time_ms = ms;
        lr.bytes = total_bytes;
        lr.experts_loaded = (int)layer_reads.size();
        results.push_back(lr);
    }
    return results;
}

// ==========================================================================
// Main
// ==========================================================================
int main(int argc, char* argv[])
{
    Config cfg = parse_args(argc, argv);

    printf("============================================================\n");
    printf("  DirectStorage vs ReadFile vs mmap  --  MoE Weight Benchmark\n");
    printf("============================================================\n\n");

    // Read header
    MoEWeightsFileHeader hdr = read_header(cfg.file_path);
    uint64_t per_expert = hdr.expert_up_weight_size + hdr.expert_down_weight_size +
                          hdr.expert_up_scale_size + hdr.expert_down_scale_size +
                          hdr.expert_up_bias_size + hdr.expert_down_bias_size;
    uint64_t per_layer  = hdr.num_experts_per_layer * per_expert;
    uint64_t file_data_size = hdr.num_layers * per_layer;

    int num_layers = (cfg.layers > 0) ? std::min(cfg.layers, (int)hdr.num_layers) : (int)hdr.num_layers;

    printf("File: %s\n", cfg.file_path.c_str());
    printf("Header:\n");
    printf("  Magic:           %.4s\n", hdr.magic);
    printf("  Version:         %u\n", hdr.version);
    printf("  Layers:          %u (testing: %d)\n", hdr.num_layers, num_layers);
    printf("  Experts/Layer:   %u\n", hdr.num_experts_per_layer);
    printf("  Up weight:       %llu bytes (%.2f MB)\n", hdr.expert_up_weight_size,
           hdr.expert_up_weight_size / (1024.0 * 1024.0));
    printf("  Down weight:     %llu bytes (%.2f MB)\n", hdr.expert_down_weight_size,
           hdr.expert_down_weight_size / (1024.0 * 1024.0));
    printf("  Up scale:        %llu bytes\n", hdr.expert_up_scale_size);
    printf("  Down scale:      %llu bytes\n", hdr.expert_down_scale_size);
    printf("  Up bias:         %llu bytes\n", hdr.expert_up_bias_size);
    printf("  Down bias:       %llu bytes\n", hdr.expert_down_bias_size);
    printf("  Per-expert total: %llu bytes (%.2f MB)\n", per_expert, per_expert / (1024.0 * 1024.0));
    printf("  Per-layer total:  %llu bytes (%.2f MB)\n", per_layer, per_layer / (1024.0 * 1024.0));
    printf("  Data offset:     %llu bytes\n", hdr.data_offset);
    printf("  Data region:     %.2f GB\n", file_data_size / (1024.0 * 1024.0 * 1024.0));
    printf("\n");

    printf("Benchmark config:\n");
    printf("  Top-K:           %d experts/layer\n", cfg.topk);
    printf("  Layers:          %d\n", num_layers);
    printf("  Total reads:     %d experts (%d reads x 2 components)\n",
           num_layers * cfg.topk, num_layers * cfg.topk);
    printf("  Iterations:      %d (+ %d warmup)\n", cfg.iterations, cfg.warmup);
    printf("  Threads/Queues:  %d\n", cfg.threads);
    printf("  Seed:            %d\n", cfg.seed);
    printf("  Method:          %s\n", cfg.method.c_str());
    printf("  Purge cache:     %s\n", cfg.purge_cache ? "YES" : "no");

    check_bypassio(cfg.file_path);

    // Generate random expert reads
    auto reads = generate_expert_reads(hdr, num_layers, cfg.topk, cfg.seed);
    uint64_t total_bytes = 0;
    for (auto& r : reads) total_bytes += r.total_size();
    printf("\n  Total data per iteration: %.2f MB (%.2f GB)\n",
           total_bytes / (1024.0 * 1024.0), total_bytes / (1024.0 * 1024.0 * 1024.0));

    // Allocate destination buffer (simulating USM host buffer)
    size_t buf_size = (size_t)(per_expert * reads.size());
    uint8_t* buffer = (uint8_t*)VirtualAlloc(nullptr, buf_size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!buffer) {
        fprintf(stderr, "Error: Cannot allocate %.2f MB buffer\n", buf_size / (1024.0 * 1024.0));
        return 1;
    }
    printf("  Buffer allocated: %.2f MB (VirtualAlloc)\n\n", buf_size / (1024.0 * 1024.0));

    // Helper to decide which methods to run
    auto should_run = [&](const char* name) {
        return cfg.method == "all" || cfg.method == name;
    };

    printf("============================================================\n");
    printf("  FULL-PASS RESULTS (all %d layers x top-%d = %zu experts)\n",
           num_layers, cfg.topk, reads.size());
    printf("============================================================\n\n");

    // ---- ReadFile (buffered) ----
    if (should_run("readfile")) {
        printf("--- ReadFile (Buffered, %d threads) ---\n", cfg.threads);
        std::vector<double> times;
        for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
            if (cfg.purge_cache) try_purge_cache(cfg.file_path);
            double ms = bench_readfile_buffered(cfg.file_path, reads, buffer, buf_size, cfg.threads);
            if (ms < 0) break;
            if (i >= cfg.warmup) times.push_back(ms);
            printf("    [%s %2d] %.2f ms  (%.2f GB/s)\n",
                   i < cfg.warmup ? "warmup" : "iter  ", i,
                   ms, (total_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0));
        }
        if (!times.empty()) {
            auto s = Stats::compute(times, (double)total_bytes);
            s.print("ReadFile");
        }
        printf("\n");
    }

    // ---- ReadFile (Direct I/O) ----
    if (should_run("directio")) {
        printf("--- ReadFile (Direct I/O, FILE_FLAG_NO_BUFFERING, %d threads) ---\n", cfg.threads);
        std::vector<double> times;
        for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
            if (cfg.purge_cache) try_purge_cache(cfg.file_path);
            double ms = bench_readfile_directio(cfg.file_path, reads, buffer, buf_size, cfg.threads);
            if (ms < 0) break;
            if (i >= cfg.warmup) times.push_back(ms);
            printf("    [%s %2d] %.2f ms  (%.2f GB/s)\n",
                   i < cfg.warmup ? "warmup" : "iter  ", i,
                   ms, (total_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0));
        }
        if (!times.empty()) {
            auto s = Stats::compute(times, (double)total_bytes);
            s.print("DirectIO");
        }
        printf("\n");
    }

    // ---- mmap ----
    if (should_run("mmap")) {
        printf("--- Memory-Mapped I/O (MapViewOfFile + memcpy, %d threads) ---\n", cfg.threads);
        std::vector<double> times;
        for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
            if (cfg.purge_cache) try_purge_cache(cfg.file_path);
            double ms = bench_mmap(cfg.file_path, reads, buffer, buf_size, cfg.threads);
            if (ms < 0) break;
            if (i >= cfg.warmup) times.push_back(ms);
            printf("    [%s %2d] %.2f ms  (%.2f GB/s)\n",
                   i < cfg.warmup ? "warmup" : "iter  ", i,
                   ms, (total_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0));
        }
        if (!times.empty()) {
            auto s = Stats::compute(times, (double)total_bytes);
            s.print("mmap");
        }
        printf("\n");
    }

    // ---- DirectStorage (batched single queue) ----
#ifdef HAS_DIRECTSTORAGE
    if (should_run("ds") || should_run("ds_batched")) {
        printf("--- DirectStorage BATCHED (single queue, all enqueued then submit) ---\n");
        std::vector<double> times;
        for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
            if (cfg.purge_cache) try_purge_cache(cfg.file_path);
            double ms = bench_directstorage_batched(cfg.file_path, reads, buffer, buf_size, cfg.threads);
            if (ms < 0) break;
            if (i >= cfg.warmup) times.push_back(ms);
            printf("    [%s %2d] %.2f ms  (%.2f GB/s)\n",
                   i < cfg.warmup ? "warmup" : "iter  ", i,
                   ms, (total_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0));
        }
        if (!times.empty()) {
            auto s = Stats::compute(times, (double)total_bytes);
            s.print("DS-Batched");
        }
        printf("\n");
    }

    // ---- DirectStorage (parallel per-thread queues) ----
    if (should_run("ds") || should_run("ds_parallel")) {
        printf("--- DirectStorage PARALLEL (%d queues, per-thread submit) ---\n", cfg.threads);
        std::vector<double> times;
        for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
            if (cfg.purge_cache) try_purge_cache(cfg.file_path);
            double ms = bench_directstorage_parallel(cfg.file_path, reads, buffer, buf_size, cfg.threads);
            if (ms < 0) break;
            if (i >= cfg.warmup) times.push_back(ms);
            printf("    [%s %2d] %.2f ms  (%.2f GB/s)\n",
                   i < cfg.warmup ? "warmup" : "iter  ", i,
                   ms, (total_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0));
        }
        if (!times.empty()) {
            auto s = Stats::compute(times, (double)total_bytes);
            s.print("DS-Parallel");
        }
        printf("\n");
    }

    // ---- DirectStorage BATCHED with PRIORITY_HIGH ----
    if (should_run("ds") || should_run("ds_high") || should_run("ds_batched_high")) {
        printf("--- DirectStorage BATCHED HIGH (single queue, PRIORITY_HIGH) ---\n");
        std::vector<double> times;
        for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
            if (cfg.purge_cache) try_purge_cache(cfg.file_path);
            double ms = bench_directstorage_batched(cfg.file_path, reads, buffer, buf_size, cfg.threads,
                                                     DSTORAGE_PRIORITY_HIGH);
            if (ms < 0) break;
            if (i >= cfg.warmup) times.push_back(ms);
            printf("    [%s %2d] %.2f ms  (%.2f GB/s)\n",
                   i < cfg.warmup ? "warmup" : "iter  ", i,
                   ms, (total_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0));
        }
        if (!times.empty()) {
            auto s = Stats::compute(times, (double)total_bytes);
            s.print("DS-Batch-High");
        }
        printf("\n");
    }

    // ---- DirectStorage PARALLEL with PRIORITY_HIGH ----
    if (should_run("ds") || should_run("ds_high") || should_run("ds_parallel_high")) {
        printf("--- DirectStorage PARALLEL HIGH (%d queues, PRIORITY_HIGH) ---\n", cfg.threads);
        std::vector<double> times;
        for (int i = 0; i < cfg.warmup + cfg.iterations; ++i) {
            if (cfg.purge_cache) try_purge_cache(cfg.file_path);
            double ms = bench_directstorage_parallel(cfg.file_path, reads, buffer, buf_size, cfg.threads,
                                                      DSTORAGE_PRIORITY_HIGH);
            if (ms < 0) break;
            if (i >= cfg.warmup) times.push_back(ms);
            printf("    [%s %2d] %.2f ms  (%.2f GB/s)\n",
                   i < cfg.warmup ? "warmup" : "iter  ", i,
                   ms, (total_bytes / (1024.0*1024.0*1024.0)) / (ms / 1000.0));
        }
        if (!times.empty()) {
            auto s = Stats::compute(times, (double)total_bytes);
            s.print("DS-Para-High");
        }
        printf("\n");
    }
#else
    if (should_run("ds") || should_run("ds_batched") || should_run("ds_parallel") ||
        should_run("ds_high") || should_run("ds_batched_high") || should_run("ds_parallel_high")) {
        printf("--- DirectStorage: NOT AVAILABLE (built without HAS_DIRECTSTORAGE) ---\n");
        printf("    Rebuild with -DDIRECTSTORAGE_SDK_PATH=<path> to enable.\n\n");
    }
#endif

    // ====================================================================
    //  PER-LAYER BREAKDOWN
    // ====================================================================
    printf("============================================================\n");
    printf("  PER-LAYER BREAKDOWN  (top-%d experts per layer)\n", cfg.topk);
    printf("============================================================\n\n");

    auto print_layer_table = [](const char* method, const std::vector<LayerResult>& results) {
        printf("  %-20s | Layer | Experts | Data (MB) | Time (ms) | GB/s\n", method);
        printf("  %-20s |-------|---------|-----------|-----------|-------\n", "");
        double tot_ms = 0, tot_bytes = 0;
        for (auto& r : results) {
            double gbps = (r.bytes / (1024.0*1024.0*1024.0)) / (r.time_ms / 1000.0);
            printf("  %-20s |  %3d  |   %3d   | %8.2f  | %8.2f  | %.2f\n",
                   "", r.layer_idx, r.experts_loaded,
                   r.bytes / (1024.0*1024.0), r.time_ms, gbps);
            tot_ms += r.time_ms;
            tot_bytes += r.bytes;
        }
        double tot_gbps = (tot_bytes / (1024.0*1024.0*1024.0)) / (tot_ms / 1000.0);
        printf("  %-20s |-------|---------|-----------|-----------|-------\n", "");
        printf("  %-20s | TOTAL | %5zu   | %8.2f  | %8.2f  | %.2f\n",
               "", results.size() * (results.empty() ? 0 : results[0].experts_loaded),
               tot_bytes / (1024.0*1024.0), tot_ms, tot_gbps);
        printf("\n");
    };

    if (should_run("readfile")) {
        auto lr = bench_per_layer(hdr, reads, num_layers, cfg.topk, buffer, buf_size,
            [&](const std::vector<ExpertRead>& layer_reads, uint8_t* buf, size_t bsz) {
                return bench_readfile_buffered(cfg.file_path, layer_reads, buf, bsz, cfg.threads);
            });
        print_layer_table("ReadFile", lr);
    }

    if (should_run("directio")) {
        auto lr = bench_per_layer(hdr, reads, num_layers, cfg.topk, buffer, buf_size,
            [&](const std::vector<ExpertRead>& layer_reads, uint8_t* buf, size_t bsz) {
                return bench_readfile_directio(cfg.file_path, layer_reads, buf, bsz, cfg.threads);
            });
        print_layer_table("DirectIO", lr);
    }

    if (should_run("mmap")) {
        auto lr = bench_per_layer(hdr, reads, num_layers, cfg.topk, buffer, buf_size,
            [&](const std::vector<ExpertRead>& layer_reads, uint8_t* buf, size_t bsz) {
                return bench_mmap(cfg.file_path, layer_reads, buf, bsz, cfg.threads);
            });
        print_layer_table("mmap", lr);
    }

#ifdef HAS_DIRECTSTORAGE
    if (should_run("ds") || should_run("ds_batched")) {
        auto lr = bench_per_layer(hdr, reads, num_layers, cfg.topk, buffer, buf_size,
            [&](const std::vector<ExpertRead>& layer_reads, uint8_t* buf, size_t bsz) {
                return bench_directstorage_batched(cfg.file_path, layer_reads, buf, bsz, cfg.threads);
            });
        print_layer_table("DS-Batched", lr);
    }

    if (should_run("ds") || should_run("ds_parallel")) {
        auto lr = bench_per_layer(hdr, reads, num_layers, cfg.topk, buffer, buf_size,
            [&](const std::vector<ExpertRead>& layer_reads, uint8_t* buf, size_t bsz) {
                return bench_directstorage_parallel(cfg.file_path, layer_reads, buf, bsz, cfg.threads);
            });
        print_layer_table("DS-Parallel", lr);
    }

    if (should_run("ds") || should_run("ds_high") || should_run("ds_batched_high")) {
        auto lr = bench_per_layer(hdr, reads, num_layers, cfg.topk, buffer, buf_size,
            [&](const std::vector<ExpertRead>& layer_reads, uint8_t* buf, size_t bsz) {
                return bench_directstorage_batched(cfg.file_path, layer_reads, buf, bsz, cfg.threads,
                                                   DSTORAGE_PRIORITY_HIGH);
            });
        print_layer_table("DS-Batch-High", lr);
    }

    if (should_run("ds") || should_run("ds_high") || should_run("ds_parallel_high")) {
        auto lr = bench_per_layer(hdr, reads, num_layers, cfg.topk, buffer, buf_size,
            [&](const std::vector<ExpertRead>& layer_reads, uint8_t* buf, size_t bsz) {
                return bench_directstorage_parallel(cfg.file_path, layer_reads, buf, bsz, cfg.threads,
                                                    DSTORAGE_PRIORITY_HIGH);
            });
        print_layer_table("DS-Para-High", lr);
    }
#endif

    // ====================================================================
    //  SUMMARY TABLE
    // ====================================================================
    printf("============================================================\n");
    printf("  SUMMARY\n");
    printf("============================================================\n\n");
    printf("  File:       %s\n", cfg.file_path.c_str());
    printf("  Layers:     %d  Top-K: %d  Experts/iter: %zu\n",
           num_layers, cfg.topk, reads.size());
    printf("  Data/iter:  %.2f MB (%.2f GB)\n",
           total_bytes / (1024.0*1024.0), total_bytes / (1024.0*1024.0*1024.0));
    printf("  Iterations: %d  Warmup: %d  Threads: %d  Seed: %d\n",
           cfg.iterations, cfg.warmup, cfg.threads, cfg.seed);

    // Cleanup
    VirtualFree(buffer, 0, MEM_RELEASE);

    printf("\nDone.\n");
    return 0;
}
