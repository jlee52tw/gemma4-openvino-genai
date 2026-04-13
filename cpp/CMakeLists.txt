# Gemma 4 — C++ inference with OpenVINO GenAI VLMPipeline
# =======================================================

cmake_minimum_required(VERSION 3.23)
project(run_gemma4_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static CRT on MSVC (match openvino.genai build)
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
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

# ── Build executable ───────────────────────────────────────────────────────
add_executable(run_gemma4 run_gemma4.cpp load_image.cpp)
target_include_directories(run_gemma4 PRIVATE "${CMAKE_BINARY_DIR}")
target_link_libraries(run_gemma4 PRIVATE openvino::genai)

install(TARGETS run_gemma4 RUNTIME DESTINATION bin)
