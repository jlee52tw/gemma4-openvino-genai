// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Gemma 4 — Simple inference with OpenVINO GenAI VLMPipeline (C++)
 * =================================================================
 * Demonstrates text-only and image+text inference using the
 * ov::genai::VLMPipeline C++ API with Gemma 4 models exported
 * to OpenVINO IR format.  After generation the built-in
 * PerfMetrics are printed (TTFT, TPOT, throughput, etc.).
 *
 * Usage:
 *   # Text-only (GPU)
 *   run_gemma4.exe --model-dir gemma-4-E2B-it-ov --prompt "Explain quantum computing."
 *
 *   # Image + text
 *   run_gemma4.exe --model-dir gemma-4-E2B-it-ov --prompt "Describe this image." --image photo.jpg
 *
 *   # CPU
 *   run_gemma4.exe --model-dir gemma-4-E2B-it-ov --device CPU --prompt "Hello!"
 *
 *   # Disable mmap + show memory
 *   run_gemma4.exe --model-dir gemma-4-E2B-it-ov --no-mmap --show-memory
 *
 *   # Prompt from file
 *   run_gemma4.exe --model-dir gemma-4-E2B-it-ov --prompt-file prompt.txt
 */

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include "openvino/genai/visual_language/pipeline.hpp"
#include "load_image.hpp"

namespace fs = std::filesystem;

// ── Helpers ────────────────────────────────────────────────────────────────

struct Args {
    std::string model_dir;
    std::string device    = "GPU";
    std::string prompt    = "Explain quantum computing in simple terms.";
    std::string prompt_file;
    std::string image_path;
    size_t      max_new_tokens = 256;
    bool        no_mmap     = false;
    bool        show_memory = false;
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Options:\n"
        << "  --model-dir <path>        Path to OpenVINO IR model directory (required)\n"
        << "  --device <CPU|GPU>        Inference device (default: GPU)\n"
        << "  --prompt <text>           Text prompt (default: \"Explain quantum computing...\")\n"
        << "  --prompt-file <path>      Read prompt from a text file (overrides --prompt)\n"
        << "  --image <path>            Optional image file for multimodal inference\n"
        << "  --max-new-tokens <N>      Maximum tokens to generate (default: 256)\n"
        << "  --no-mmap                 Disable memory-mapped model loading\n"
        << "  --show-memory             Print process memory (RSS / peak) at key stages\n"
        << "  --help                    Show this help message\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--model-dir") && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if ((a == "--device") && i + 1 < argc) {
            args.device = argv[++i];
        } else if ((a == "--prompt") && i + 1 < argc) {
            args.prompt = argv[++i];
        } else if ((a == "--prompt-file") && i + 1 < argc) {
            args.prompt_file = argv[++i];
        } else if ((a == "--image") && i + 1 < argc) {
            args.image_path = argv[++i];
        } else if ((a == "--max-new-tokens") && i + 1 < argc) {
            args.max_new_tokens = std::stoull(argv[++i]);
        } else if (a == "--no-mmap") {
            args.no_mmap = true;
        } else if (a == "--show-memory") {
            args.show_memory = true;
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown option: " << a << "\n";
            print_usage(argv[0]);
            std::exit(1);
        }
    }
    if (args.model_dir.empty()) {
        std::cerr << "Error: --model-dir is required\n";
        print_usage(argv[0]);
        std::exit(1);
    }
    return args;
}

static std::string read_text_file(const fs::path& path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    // Trim trailing whitespace
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
    }
    return text;
}

static std::string read_file_raw(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Cannot open file: " + path.string());
    }
    auto size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::string data(static_cast<size_t>(size), '\0');
    f.read(data.data(), size);
    return data;
}

static ov::Tensor read_bin_to_tensor(const fs::path& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Cannot open binary file: " + path.string());
    }
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    ov::Tensor tensor(ov::element::u8, {size});
    f.read(reinterpret_cast<char*>(tensor.data()), size);
    return tensor;
}

// ── Memory measurement (Windows) ───────────────────────────────────────────

struct MemInfo {
    double rss_mb  = 0.0;
    double peak_mb = 0.0;
};

static MemInfo get_memory() {
    MemInfo info;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        info.rss_mb  = static_cast<double>(pmc.WorkingSetSize)     / (1024.0 * 1024.0);
        info.peak_mb = static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
#endif
    return info;
}

static void print_memory(const char* label) {
    auto mem = get_memory();
    if (mem.rss_mb > 0) {
        std::cout << "  [" << label << "]  RSS: " << std::fixed << std::setprecision(0)
                  << mem.rss_mb << " MB  |  Peak: " << mem.peak_mb << " MB\n";
    }
}

// ── Print performance metrics ──────────────────────────────────────────────

static void print_perf_metrics(ov::genai::VLMPerfMetrics& metrics) {
    const std::string sep(60, '-');
    std::cout << std::fixed << std::setprecision(2);
    std::cout << sep << "\n";
    std::cout << "  OpenVINO GenAI - Performance Metrics\n";
    std::cout << sep << "\n";

    // Token counts
    std::cout << "  Input tokens          : " << metrics.get_num_input_tokens()     << "\n";
    std::cout << "  Generated tokens      : " << metrics.get_num_generated_tokens() << "\n";
    std::cout << "\n";

    // Latency
    std::cout << "  Load time             : " << metrics.get_load_time() << " ms\n";
    std::cout << "  TTFT                  : " << metrics.get_ttft().mean  << " ms\n";
    std::cout << "  TPOT                  : " << metrics.get_tpot().mean  << " ms\n";
    std::cout << "\n";

    // Throughput
    std::cout << "  Throughput            : " << metrics.get_throughput().mean << " tok/s\n";
    std::cout << "\n";

    // Duration breakdown
    std::cout << "  Generate duration     : " << metrics.get_generate_duration().mean     << " ms\n";
    std::cout << "  Inference duration    : " << metrics.get_inference_duration().mean     << " ms\n";
    std::cout << "  Tokenization duration : " << metrics.get_tokenization_duration().mean  << " ms\n";
    std::cout << "  Detokenization dur.   : " << metrics.get_detokenization_duration().mean << " ms\n";
    std::cout << "  Prepare embeddings    : " << metrics.get_prepare_embeddings_duration().mean << " ms\n";
    std::cout << sep << "\n";
}

// ── Streamer callback ──────────────────────────────────────────────────────

static ov::genai::StreamingStatus print_subword(std::string&& subword) {
    std::cout << subword << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
}

// ── Main ───────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    try {
        Args args = parse_args(argc, argv);

        // Resolve prompt
        std::string prompt = args.prompt;
        if (!args.prompt_file.empty()) {
            fs::path pf(args.prompt_file);
            if (!fs::exists(pf)) {
                std::cerr << "Error: prompt file not found: " << pf << "\n";
                return 1;
            }
            prompt = read_text_file(pf);
        }

        // Validate model directory
        fs::path model_dir(args.model_dir);
        if (!fs::exists(model_dir)) {
            std::cerr << "Error: model directory not found: " << model_dir << "\n";
            return 1;
        }

        // Load model
        std::cout << "Loading VLMPipeline from " << model_dir << " on " << args.device << "...\n";
        if (args.no_mmap) {
            std::cout << "  (mmap disabled -- weights will be copied into RAM)\n";
        }
        if (args.show_memory) {
            std::cout << "\n";
            print_memory("Before loading");
        }

        auto t0 = std::chrono::steady_clock::now();
        ov::AnyMap properties;

        // Build pipeline — with or without mmap
        std::unique_ptr<ov::genai::VLMPipeline> pipe_ptr;

        if (args.no_mmap) {
            // Load model components manually into heap memory (no mmap).
            // VLMPipeline second constructor: models map + tokenizer + config_dir.
            const std::vector<std::string> model_names = {
                "language", "text_embeddings", "text_embeddings_per_layer",
                "vision_embeddings",
            };
            std::map<std::string, std::pair<std::string, ov::Tensor>> models;
            for (const auto& name : model_names) {
                auto xml_path = model_dir / ("openvino_" + name + "_model.xml");
                auto bin_path = model_dir / ("openvino_" + name + "_model.bin");
                if (fs::exists(xml_path) && fs::exists(bin_path)) {
                    models[name] = {read_file_raw(xml_path), read_bin_to_tensor(bin_path)};
                }
            }

            // Load tokenizer + detokenizer
            auto tok_xml   = read_file_raw(model_dir / "openvino_tokenizer.xml");
            auto tok_bin   = read_bin_to_tensor(model_dir / "openvino_tokenizer.bin");
            auto detok_xml = read_file_raw(model_dir / "openvino_detokenizer.xml");
            auto detok_bin = read_bin_to_tensor(model_dir / "openvino_detokenizer.bin");
            ov::genai::Tokenizer tokenizer(tok_xml, tok_bin, detok_xml, detok_bin);

            pipe_ptr = std::make_unique<ov::genai::VLMPipeline>(
                models, tokenizer, model_dir, args.device, properties
            );
        } else {
            pipe_ptr = std::make_unique<ov::genai::VLMPipeline>(
                model_dir, args.device, properties
            );
        }
        auto& pipe = *pipe_ptr;

        auto t1 = std::chrono::steady_clock::now();
        double load_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Model loaded in " << std::fixed << std::setprecision(1) << load_s << "s\n";

        if (args.show_memory) {
            print_memory("After loading (peak)");
        }

        // Generation config
        ov::genai::GenerationConfig config;
        config.max_new_tokens = args.max_new_tokens;

        // Load image (if provided)
        std::vector<ov::Tensor> images;
        if (!args.image_path.empty()) {
            images = utils::load_images(args.image_path);
            std::cout << "Image loaded: " << args.image_path << "\n";
        }

        // Generate
        std::cout << "\nPrompt: " << prompt << "\n\n";
        std::cout << "Response: " << std::flush;

        ov::genai::VLMDecodedResults result;
        if (!images.empty()) {
            result = pipe.generate(
                prompt,
                ov::genai::images(images),
                ov::genai::generation_config(config),
                ov::genai::streamer(print_subword)
            );
        } else {
            result = pipe.generate(
                prompt,
                ov::genai::generation_config(config),
                ov::genai::streamer(print_subword)
            );
        }

        std::cout << "\n\n";

        // Memory after generation (stabilized)
        if (args.show_memory) {
            print_memory("After generation (stabilized)");
            std::cout << "\n";
        }

        // Print performance metrics
        print_perf_metrics(result.perf_metrics);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
