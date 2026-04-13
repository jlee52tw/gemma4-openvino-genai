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

// ── Print performance metrics ──────────────────────────────────────────────

static std::string fmt_ms(ov::genai::MeanStdPair p) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(2) << p.mean << " +/- " << p.std << " ms";
    return s.str();
}

static std::string fmt_tok_s(ov::genai::MeanStdPair p) {
    std::ostringstream s;
    s << std::fixed << std::setprecision(2) << p.mean << " +/- " << p.std << " tok/s";
    return s.str();
}

static void print_perf_metrics(ov::genai::VLMPerfMetrics& metrics) {
    const std::string sep(60, '-');
    std::cout << sep << "\n";
    std::cout << "  OpenVINO GenAI - Performance Metrics\n";
    std::cout << sep << "\n";

    // Token counts
    std::cout << "  Input tokens          : " << metrics.get_num_input_tokens()     << "\n";
    std::cout << "  Generated tokens      : " << metrics.get_num_generated_tokens() << "\n";
    std::cout << "\n";

    // Latency
    std::cout << "  Load time             : " << std::fixed << std::setprecision(2)
              << metrics.get_load_time() << " ms\n";
    std::cout << "  TTFT                  : " << fmt_ms(metrics.get_ttft())  << "\n";
    std::cout << "  TPOT                  : " << fmt_ms(metrics.get_tpot())  << "\n";
    std::cout << "  iPOT                  : " << fmt_ms(metrics.get_ipot())  << "\n";
    std::cout << "\n";

    // Throughput
    std::cout << "  Throughput            : " << fmt_tok_s(metrics.get_throughput()) << "\n";
    std::cout << "\n";

    // Duration breakdown
    std::cout << "  Generate duration     : " << fmt_ms(metrics.get_generate_duration())     << "\n";
    std::cout << "  Inference duration    : " << fmt_ms(metrics.get_inference_duration())     << "\n";
    std::cout << "  Tokenization duration : " << fmt_ms(metrics.get_tokenization_duration())  << "\n";
    std::cout << "  Detokenization dur.   : " << fmt_ms(metrics.get_detokenization_duration())<< "\n";
    std::cout << "  Prepare embeddings    : " << fmt_ms(metrics.get_prepare_embeddings_duration()) << "\n";
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
        auto t0 = std::chrono::steady_clock::now();
        ov::genai::VLMPipeline pipe(model_dir, args.device);
        auto t1 = std::chrono::steady_clock::now();
        double load_s = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Model loaded in " << std::fixed << std::setprecision(1) << load_s << "s\n";

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

        // Print performance metrics
        print_perf_metrics(result.perf_metrics);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
