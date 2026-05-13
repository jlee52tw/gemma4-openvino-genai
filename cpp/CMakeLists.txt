# Gemma 4 — C++ inference with OpenVINO GenAI VLMPipeline
# =======================================================

cmake_minimum_required(VERSION 3.23)
project(run_gemma4_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dynamic CRT on MSVC (match openvino pip package)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# ── Find OpenVINO GenAI ────────────────────────────────────────────────────
# If openvino-genai was installed into the venv via `pip install .`, the
# CMake config is at:  .venv/Lib/site-packages/openvino_genai/
# Pass it with:  -DOpenVINOGenAI_DIR=<path>
find_package(OpenVINOGenAI REQUIRED)

# ── Download stb_image.h (single-header image loader) ─────────────────────
file(DOWNLOAD
    https://raw.githubusercontent.com/nothings/stb/f75e8d1cad7d90d72ef7a4661f1b994ef78b4e31/stb_image.h
    ${CMAKE_BINARY_DIR}/stb_image.h
    EXPECTED_HASH MD5=27932e6fb3a2f26aee2fc33f2cb4e696)

# ── Build executable: run_gemma4 ───────────────────────────────────────────
add_executable(run_gemma4 run_gemma4.cpp load_image.cpp)
target_include_directories(run_gemma4 PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(run_gemma4 PRIVATE openvino::genai)

# ── Build executable: benchmark_kpi ────────────────────────────────────────
add_executable(benchmark_kpi benchmark_kpi.cpp)
target_link_libraries(benchmark_kpi PRIVATE openvino::genai)
if(WIN32)
    target_link_libraries(benchmark_kpi PRIVATE psapi)
endif()

install(TARGETS run_gemma4 benchmark_kpi RUNTIME DESTINATION bin)

# ── Build library: dense_weight_streaming ──────────────────────────────────
# This is the DirectStorage-based weight streaming manager for dense models.
# Requires Windows SDK and optionally DirectStorage 1.3.0+ NuGet package.
option(ENABLE_DIRECTSTORAGE "Enable DirectStorage for weight streaming" OFF)

add_library(dense_weight_streaming STATIC
    dense_weight_streaming_manager.cpp
    dense_weight_streaming_manager.hpp)
target_include_directories(dense_weight_streaming PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

if(WIN32)
    target_link_libraries(dense_weight_streaming PRIVATE kernel32)
endif()

if(ENABLE_DIRECTSTORAGE)
    target_compile_definitions(dense_weight_streaming PRIVATE OPENVINO_USE_DIRECTSTORAGE)
    # DirectStorage SDK (install via NuGet: Microsoft.Direct3D.DirectStorage 1.3.0)
    # Set DSTORAGE_DIR to the NuGet package path:
    #   -DDSTORAGE_DIR=<path>/Microsoft.Direct3D.DirectStorage.1.3.0
    if(DSTORAGE_DIR)
        target_include_directories(dense_weight_streaming PRIVATE
            "${DSTORAGE_DIR}/include")
        target_link_directories(dense_weight_streaming PRIVATE
            "${DSTORAGE_DIR}/lib/x64")
        target_link_libraries(dense_weight_streaming PRIVATE dstorage)
    else()
        message(WARNING "ENABLE_DIRECTSTORAGE=ON but DSTORAGE_DIR not set. "
                        "Set -DDSTORAGE_DIR=<NuGet-package-path>")
    endif()
endif()
