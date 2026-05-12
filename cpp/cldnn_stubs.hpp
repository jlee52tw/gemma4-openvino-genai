// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Standalone stubs for cldnn types when building outside OpenVINO GPU plugin.
// Used by DenseWeightStreamingManager and benchmark_pipeline in standalone mode.
//
// When OPENVINO_GPU_RUNTIME_AVAILABLE is defined, the real headers are used instead.

#pragma once

#ifndef OPENVINO_GPU_RUNTIME_AVAILABLE

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <cstdlib>
#endif

#include <cstddef>
#include <memory>

namespace cldnn {

class engine {
public:
    // Stub: allocate USM host memory (page-aligned, satisfies NO_BUFFERING)
    void* allocate_usm_host(size_t size) {
#ifdef _WIN32
        return VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
#else
        return aligned_alloc(4096, size);
#endif
    }
    void free_usm_host(void* ptr) {
#ifdef _WIN32
        VirtualFree(ptr, 0, MEM_RELEASE);
#else
        free(ptr);
#endif
    }
};

class memory {
public:
    using ptr = std::shared_ptr<memory>;
    void* buffer_ptr() { return m_ptr; }
    size_t size() const { return m_size; }
    void* m_ptr = nullptr;
    size_t m_size = 0;
};

using memory_ptr = memory::ptr;

}  // namespace cldnn

#endif  // !OPENVINO_GPU_RUNTIME_AVAILABLE
