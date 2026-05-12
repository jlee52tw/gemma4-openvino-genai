# MoE Expert Weight Manager — Key Patterns for Dense Weight Streaming

## Source Location
`jlee52tw/openvino` branch `moe-otd-pr-squash`  
Path: `src/plugins/intel_gpu/src/graph/impls/ocl_v2/moe/moe_expert_weight_manager.cpp`

---

## 1. Initialization Flow (initialize())

```
1. Open weights file (ifstream for header reading)
2. Read & validate header (magic "MOEW", version 1)
3. Initialize layer infos (offset calculations per expert)
4. HYBRID I/O ARCHITECTURE setup:
   a. Calculate dedicated_layers from resident_experts config
   b. Open Direct I/O handle (FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)
   c. Allocate sector-aligned read buffer
   d. Set up mmap for dynamic layers (MapViewOfFile at offset)
5. allocate_buffers() — USM host memory for GPU
6. initialize_directstorage() — DS factory, queues, status arrays, file
7. Create thread pool (ExpertLoadingThreadPool)
8. preload_hot_experts() — pin frequently-used experts
```

---

## 2. DirectStorage Initialization (initialize_directstorage())

```cpp
// Key pattern: Per-thread queues, no mutex needed
#ifdef OPENVINO_USE_DIRECTSTORAGE

IDStorageFactory* factory = nullptr;
HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&factory));

// Queue config
DSTORAGE_QUEUE_DESC queue_desc = {};
queue_desc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;  // 8192 requests
queue_desc.Priority = DSTORAGE_PRIORITY_HIGH;
queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
queue_desc.Device = nullptr;  // CPU memory destination (USM host buffer)

// Create NUM_DS_QUEUES (4) independent queues + status arrays
for (size_t i = 0; i < NUM_DS_QUEUES; ++i) {
    IDStorageQueue* queue = nullptr;
    hr = factory->CreateQueue(&queue_desc, IID_PPV_ARGS(&queue));
    m_ds_queues[i] = queue;
    
    IDStorageStatusArray* status_array = nullptr;
    hr = factory->CreateStatusArray(1, nullptr, IID_PPV_ARGS(&status_array));
    m_ds_status_arrays[i] = status_array;
    
    HANDLE fence_event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    m_ds_fence_events[i] = fence_event;
}

// Open file (shared across queues - thread-safe for reads)
std::wstring wide_path = /* UTF-8 to wide conversion */;
hr = factory->OpenFile(wide_path.c_str(), IID_PPV_ARGS(&file));
m_ds_file = file;

#endif
```

---

## 3. DirectStorage Read (load_expert_via_directstorage())

```cpp
// Per-thread queue design: each worker uses its own queue (no mutex)
auto* queue = static_cast<IDStorageQueue*>(m_ds_queues[queue_idx]);
auto* file = static_cast<IDStorageFile*>(m_ds_file);
auto* status_array = static_cast<IDStorageStatusArray*>(m_ds_status_arrays[queue_idx]);

// Build read request
DSTORAGE_REQUEST request = {};
request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MEMORY;
request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_NONE;
request.Source.File.Source = file;
request.Source.File.Offset = file_offset;
request.Source.File.Size = static_cast<UINT32>(size);
request.Destination.Memory.Buffer = dest_ptr;  // USM host buffer
request.Destination.Memory.Size = static_cast<UINT32>(size);
request.UncompressedSize = static_cast<UINT32>(size);

// Submit & wait
queue->EnqueueRequest(&request);
queue->EnqueueStatus(status_array, 0);
queue->Submit();

// Spin-wait (fast with BypassIO)
while (!status_array->IsComplete(0)) {
    std::this_thread::yield();
}

// Check errors
HRESULT status = status_array->GetHResult(0);
if (FAILED(status)) {
    DSTORAGE_ERROR_RECORD error_record = {};
    queue->RetrieveErrorRecord(&error_record);
    // Handle error...
}
```

---

## 4. Buffer Allocation (allocate_buffers())

```cpp
// Memory budget: 60% of GPU memory or OV_MOE_MAX_BUFFER_GB
uint64_t gpu_mem_total = m_engine.get_device_info().max_global_mem_size;
uint64_t max_buffer_bytes = static_cast<uint64_t>(gpu_mem_total * 0.60);

// Allocate USM host buffers (CPU and GPU accessible on iGPU)
cldnn::layout up_weight_layout({static_cast<int64_t>(up_weight_buffer_size)}, 
                                cldnn::data_types::u8, cldnn::format::bfyx);
m_up_weight_buffer = m_engine.allocate_memory(up_weight_layout, 
                                              cldnn::allocation_type::usm_host, false);

// Optional: Lock in physical memory (VirtualLock)
if (moe_lock_memory_enabled()) {
    void* ptr = m_up_weight_buffer->buffer_ptr();
    lock_memory_pages(ptr, up_weight_buffer_size, "up_weight_buffer");
}
```

---

## 5. Expert Loading 3-Phase Pattern (load_experts())

### Phase 1: DECISION (under lock)
```cpp
std::lock_guard<std::mutex> lock(m_mutex);
// For each expert: check cache (expert_to_slot map)
// Cache hit → reuse existing slot
// Cache miss → find_available_slot() via tiered LRU
// Update slot metadata IMMEDIATELY (prevent same slot for multiple misses)
```

### Phase 2: DATA COPY (parallel, no lock)
```cpp
// Batched DirectStorage for ≤4 experts:
//   Single queue, multiple EnqueueRequest, one Submit, one wait
// Parallel thread pool for >4:
//   Each thread uses its own DS queue (thread_idx % NUM_DS_QUEUES)
// Each expert writes to DIFFERENT GPU memory region (no conflicts)
```

### Phase 3: BOOKKEEPING (update mappings)
```cpp
// Update slot_to_expert, expert_to_slot maps
// Tiered access time: PINNED (UINT64_MAX-1), HOT (UINT64_MAX/2+counter), COLD (counter)
```

---

## 6. I/O Priority Chain (load_expert_direct_to_gpu())

```
1. DirectStorage (BypassIO) — works for ALL layers, absolute file offset
2. Direct I/O (FILE_FLAG_NO_BUFFERING) — static layers only (0 to dedicated_layers-1)
3. mmap (MapViewOfFile) — dynamic layers, relative offset from mmap start
4. Fall back to staging buffer (ifstream)
```

---

## 7. Tiered LRU Eviction (find_available_slot())

```cpp
// Priority tiers (evict from lowest priority first):
// Tier 1 (evict first):  cold experts (not in hot expert set)
// Tier 2 (evict second): hot experts (in hot set but not pinned)
// Tier 3 (evict last):   pinned experts (highest-frequency)

// Access time encoding:
// - Pinned: UINT64_MAX - 1 (never evicted normally)
// - Hot:    UINT64_MAX/2 + counter (resist eviction)
// - Cold:   counter (first to be evicted)
```

---

## 8. Direct I/O Implementation (load_expert_via_direct_io())

```cpp
// FILE_FLAG_NO_BUFFERING requires sector-aligned reads:
// 1. File offset must be aligned to sector boundary
// 2. Read size must be aligned to sector boundary
// 3. Buffer must be aligned to sector boundary

size_t aligned_offset = (weight_desc.offset / sector_size) * sector_size;
size_t offset_within_sector = weight_desc.offset - aligned_offset;
size_t aligned_read_size = ((bytes_to_read + sector_size - 1) / sector_size) * sector_size;

// Get sector-aligned buffer address
uintptr_t buffer_addr = reinterpret_cast<uintptr_t>(m_aligned_read_buffer.data());
uintptr_t aligned_buffer_addr = ((buffer_addr + sector_size - 1) / sector_size) * sector_size;
uint8_t* aligned_buffer = reinterpret_cast<uint8_t*>(aligned_buffer_addr);

// Use OVERLAPPED for position (synchronous mode)
OVERLAPPED overlapped = {};
overlapped.Offset = static_cast<DWORD>(aligned_offset & 0xFFFFFFFF);
overlapped.OffsetHigh = static_cast<DWORD>(aligned_offset >> 32);

DWORD bytes_read = 0;
ReadFile(m_direct_io_handle, aligned_buffer, aligned_read_size, &bytes_read, &overlapped);

// Copy from aligned buffer to GPU USM buffer (skip sector alignment padding)
std::memcpy(gpu_ptr + dst_offset, aligned_buffer + offset_within_sector, weight_desc.size);
```

---

## 9. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| OV_MOE_OTD_DEBUG | 0 | Enable debug logging |
| OV_MOE_USE_DIRECTSTORAGE | 1 (if SDK) | Enable/disable DirectStorage |
| OV_MOE_LOCK_MEMORY | 0 | VirtualLock USM buffers |
| OV_MOE_MAX_BUFFER_GB | 60% GPU | Memory budget cap |
| OV_MOE_LRU_SLOTS | 64 | Cold expert LRU pool size |
| OV_MOE_PARALLEL_LOAD | 1 | Enable thread pool |
| OV_MOE_PIN_HOT_EXPERTS | 1 | Enable hot expert pinning |
| OV_MOE_PROFILE_EXPERTS | 0 | Enable runtime profiling |
| OV_MOE_HOT_EXPERTS_FILE | - | JSON file with hot expert IDs |
| OV_MOE_120B_HOT_LAYERS | 10 | Layers to pin for 120B model |
| OV_MOE_120B_HOT_PIN_MODE | FULL_LAYERS | Pin mode for 120B |

---

## 10. Key Adaption Points for Dense Weight Streaming

### What to reuse:
1. **DirectStorage initialization pattern** — factory, per-thread queues, status arrays
2. **USM host allocation** — `m_engine.allocate_memory(layout, usm_host, false)`
3. **Per-thread queue I/O** — zero-mutex parallel reads
4. **VirtualLock for pinned buffers** — prevent swap-out
5. **Batched DS submission** — single submit for multiple EnqueueRequest

### What to change for dense model:
1. **No LRU/slot management** — dense model loads ALL layers sequentially
2. **No expert granularity** — load entire layer weights at once
3. **Double-buffering** — while GPU computes layer N, prefetch layer N+1 from NVMe
4. **get_arguments() pattern** — swap weight USM pointers between layers (from MoE kernel dispatch)
5. **Larger I/O sizes** — each dense layer ~50 MB vs MoE expert ~6 MB

### Critical insight:
The MoE OTD code uses `buffer_ptr()` to get CPU-accessible USM host pointer, 
then does `memcpy`/DirectStorage directly into it. For dense streaming, we need 
to pre-allocate TWO layer-sized USM buffers and swap the pointer that the 
compiled model's kernel uses — this is the `get_arguments()` approach.
