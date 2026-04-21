// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

/**
 * Gemma 4 E4B — KPI Benchmark (C++)
 * ==================================
 * C++ port of benchmark_kpi.py.  Measures key performance indicators
 * for Gemma-4-e4b-it (INT4) via VLMPipeline.
 *
 * KPIs measured:
 *   - Model size on disk (GB)
 *   - Model load time (s)
 *   - Per-scenario (varying input token counts):
 *       * Input tokens, Output tokens
 *       * Prefill Speed (tokens/s) = input_tokens / TTFT
 *       * Output TPS (tokens/s)    = 1000 / TPOT
 *       * Total peak memory (GB)   = process peak working set
 *       * Private memory (GB)      = process private bytes
 *   - mmap on / off comparison
 *   - CACHE_DIR support
 *
 * Usage:
 *   benchmark_kpi.exe --model-dir gemma-4-E4B-it-ov --device GPU --prompt-file prompt.txt
 *   benchmark_kpi.exe --model-dir gemma-4-E4B-it-ov --device GPU --no-mmap --prompt-file p1.txt p2.txt
 *   benchmark_kpi.exe --model-dir gemma-4-E4B-it-ov --device GPU --prompt-file prompt.txt --max-new-tokens 300
 */

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "openvino/genai/visual_language/pipeline.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

namespace fs = std::filesystem;

// ═══════════════════════════════════════════════════════════════════════════════
// Argument parsing
// ═══════════════════════════════════════════════════════════════════════════════

struct Args {
    std::string model_dir;
    std::string device       = "GPU";
    std::string cache_dir;
    std::string output_json;
    std::vector<std::string> prompt_files;
    int         max_new_tokens = 0;   // 0 = no limit (generate until EOS)
    int         warmup         = 1;
    bool        no_mmap        = false;
    bool        skip_cache_measurement = false;
};

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n\n"
        << "Options:\n"
        << "  --model-dir <path>         Path to OpenVINO IR model directory (required)\n"
        << "  --device <CPU|GPU>         Inference device (default: GPU)\n"
        << "  --prompt-file <f> [f] ...  Prompt text file(s), one per scenario (required)\n"
        << "  --max-new-tokens <N>       Max output tokens (default: 0 = no limit)\n"
        << "  --no-mmap                  Disable memory-mapped model loading\n"
        << "  --cache-dir <path>         Set CACHE_DIR for compiled model cache\n"
        << "  --skip-cache-measurement   Skip cache creation measurement\n"
        << "  --warmup <N>               Warmup runs (default: 1)\n"
        << "  --output-json <path>       Save results to JSON file\n"
        << "  --help                     Show this help message\n";
}

static Args parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model-dir" && i + 1 < argc) {
            args.model_dir = argv[++i];
        } else if (a == "--device" && i + 1 < argc) {
            args.device = argv[++i];
        } else if (a == "--cache-dir" && i + 1 < argc) {
            args.cache_dir = argv[++i];
        } else if (a == "--output-json" && i + 1 < argc) {
            args.output_json = argv[++i];
        } else if (a == "--max-new-tokens" && i + 1 < argc) {
            args.max_new_tokens = std::stoi(argv[++i]);
        } else if (a == "--warmup" && i + 1 < argc) {
            args.warmup = std::stoi(argv[++i]);
        } else if (a == "--prompt-file") {
            // Collect all following non-flag arguments as prompt files
            while (i + 1 < argc && argv[i + 1][0] != '-') {
                args.prompt_files.push_back(argv[++i]);
            }
        } else if (a == "--no-mmap") {
            args.no_mmap = true;
        } else if (a == "--skip-cache-measurement") {
            args.skip_cache_measurement = true;
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
    if (args.prompt_files.empty()) {
        std::cerr << "Error: --prompt-file is required (one or more prompt text files)\n";
        print_usage(argv[0]);
        std::exit(1);
    }
    return args;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Memory measurement (Windows)
// ═══════════════════════════════════════════════════════════════════════════════

static double get_peak_working_set_gb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

static double get_private_memory_gb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                             sizeof(pmc))) {
        return static_cast<double>(pmc.PrivateUsage) / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

static double get_rss_gb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

static void reset_peak_working_set() {
#ifdef _WIN32
    // EmptyWorkingSet resets the peak working set counter
    EmptyWorkingSet(GetCurrentProcess());
#endif
}

static double get_system_memory_gb() {
#ifdef _WIN32
    MEMORYSTATUSEX statex;
    statex.dwLength = sizeof(statex);
    if (GlobalMemoryStatusEx(&statex)) {
        return static_cast<double>(statex.ullTotalPhys) / (1024.0 * 1024.0 * 1024.0);
    }
#endif
    return 0.0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// File size helpers
// ═══════════════════════════════════════════════════════════════════════════════

static double get_model_size_gb(const fs::path& model_dir) {
    uint64_t total = 0;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            total += entry.file_size();
        }
    }
    return static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0);
}

static double get_cache_size_gb(const fs::path& model_dir) {
    uint64_t total_size = 0;
    bool found_cache_dir = false;

    std::vector<fs::path> cache_dirs = {
        model_dir / "cache",
        model_dir / "model_cache",
    };

    for (const auto& cd : cache_dirs) {
        if (fs::exists(cd) && fs::is_directory(cd)) {
            found_cache_dir = true;
            for (const auto& entry : fs::recursive_directory_iterator(cd)) {
                if (entry.is_regular_file()) {
                    total_size += entry.file_size();
                }
            }
        }
    }

    // Only count top-level .blob files if no cache directory was found
    if (!found_cache_dir) {
        for (const auto& entry : fs::directory_iterator(model_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".blob") {
                total_size += entry.file_size();
            }
        }
    }

    return static_cast<double>(total_size) / (1024.0 * 1024.0 * 1024.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Prompt loader
// ═══════════════════════════════════════════════════════════════════════════════

static std::string load_prompt_file(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        std::cerr << "Error: prompt file not found: " << filepath << "\n";
        std::exit(1);
    }
    std::ifstream f(filepath);
    if (!f) {
        std::cerr << "Error: cannot read prompt file: " << filepath << "\n";
        std::exit(1);
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();
    // Trim trailing whitespace
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
        text.pop_back();
    if (text.empty()) {
        std::cerr << "Error: prompt file is empty: " << filepath << "\n";
        std::exit(1);
    }
    return text;
}

static int tokenize_prompt(ov::genai::Tokenizer& tokenizer, const std::string& prompt) {
    auto encoded = tokenizer.encode(prompt);
    return static_cast<int>(encoded.input_ids.get_shape()[1]);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Scenario result
// ═══════════════════════════════════════════════════════════════════════════════

struct ScenarioResult {
    std::string scenario;
    int    input_tokens        = 0;
    int    output_tokens       = 0;
    double prefill_speed_tps   = 0.0;
    double output_tps          = 0.0;
    double ttft_ms             = 0.0;
    double tpot_ms             = 0.0;
    double total_peak_memory_gb = 0.0;
    double private_memory_gb   = 0.0;
    double total_time_s        = 0.0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// Run one inference scenario
// ═══════════════════════════════════════════════════════════════════════════════

static ScenarioResult run_scenario(
    ov::genai::VLMPipeline& pipe,
    const std::string& prompt,
    const std::string& scenario_name,
    int input_tokens,
    int max_new_tokens)
{
    std::cout << "\n  --- Scenario: " << scenario_name
              << " (input ~" << input_tokens << " tokens) ---\n";

    ov::genai::GenerationConfig config;
    if (max_new_tokens > 0) {
        config.max_new_tokens = max_new_tokens;
    }

    // Let memory settle, then reset peak
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    reset_peak_working_set();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    // Generate with streaming output
    bool first_token_printed = false;
    auto streamer = [&first_token_printed](std::string subword) -> ov::genai::StreamingStatus {
        if (!first_token_printed) {
            std::cout << "\n    [Output] " << std::flush;
            first_token_printed = true;
        }
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    auto gen_start = std::chrono::steady_clock::now();
    auto result = pipe.generate(prompt,
        ov::genai::generation_config(config),
        ov::genai::streamer(streamer));
    auto gen_end = std::chrono::steady_clock::now();
    std::cout << "\n\n";  // newline after streamed output

    double total_time = std::chrono::duration<double>(gen_end - gen_start).count();

    // Memory after generation
    double total_peak = get_peak_working_set_gb();
    double private_mem = get_private_memory_gb();

    // Get PerfMetrics
    auto& perf = result.perf_metrics;
    double pm_ttft = perf.get_ttft().mean;        // ms
    double pm_tpot = perf.get_tpot().mean;        // ms
    double pm_throughput = perf.get_throughput().mean;  // tok/s
    int pm_in_tok = static_cast<int>(perf.get_num_input_tokens());
    int pm_out_tok = static_cast<int>(perf.get_num_generated_tokens());

    std::cout << "    PerfMetrics: TTFT=" << std::fixed << std::setprecision(1)
              << pm_ttft << "ms, TPOT=" << pm_tpot << "ms, Throughput="
              << pm_throughput << " tok/s\n";
    std::cout << "    PerfMetrics: InTok=" << pm_in_tok
              << ", OutTok=" << pm_out_tok << "\n";

    // Use PerfMetrics values (most accurate)
    double ttft_ms = pm_ttft;
    double ttft_s = pm_ttft / 1000.0;
    double prefill_speed = (ttft_s > 0) ? pm_in_tok / ttft_s : 0.0;
    double output_tps = (pm_tpot > 0) ? 1000.0 / pm_tpot : 0.0;
    double tpot_ms = pm_tpot;

    ScenarioResult sr;
    sr.scenario = scenario_name;
    sr.input_tokens = pm_in_tok;
    sr.output_tokens = pm_out_tok;
    sr.prefill_speed_tps = prefill_speed;
    sr.output_tps = output_tps;
    sr.ttft_ms = ttft_ms;
    sr.tpot_ms = tpot_ms;
    sr.total_peak_memory_gb = total_peak;
    sr.private_memory_gb = private_mem;
    sr.total_time_s = total_time;

    std::cout << "    Input tokens     : " << sr.input_tokens << "\n";
    std::cout << "    Output tokens    : " << sr.output_tokens << "\n";
    std::cout << "    TTFT             : " << std::fixed << std::setprecision(1)
              << sr.ttft_ms << " ms\n";
    std::cout << "    Prefill Speed    : " << sr.prefill_speed_tps << " tokens/s\n";
    std::cout << "    Output TPS       : " << sr.output_tps << " tokens/s\n";
    std::cout << "    TPOT             : " << sr.tpot_ms << " ms\n";
    std::cout << "    Total peak mem   : " << std::setprecision(2)
              << sr.total_peak_memory_gb << " GB\n";
    std::cout << "    Private mem      : " << sr.private_memory_gb << " GB\n";
    std::cout << "    Total time       : " << sr.total_time_s << " s\n";

    return sr;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Report printer
// ═══════════════════════════════════════════════════════════════════════════════

static void print_report(
    const Args& args,
    double model_size_gb,
    double sys_mem_gb,
    double model_load_time_s,
    double cache_size_gb,
    const std::vector<ScenarioResult>& results)
{
    const std::string sep(72, '=');
    std::cout << "\n" << sep << "\n";
    std::cout << "  KPI BENCHMARK REPORT\n";
    std::cout << sep << "\n";
    std::cout << "  Model          : " << fs::path(args.model_dir).filename().string() << "\n";
    std::cout << "  Model dir      : " << args.model_dir << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "  Model size     : " << model_size_gb << " GB\n";
    std::cout << "  Device         : " << args.device << "\n";
    std::cout << "  mmap           : " << (args.no_mmap ? "OFF" : "ON") << "\n";
    if (!args.cache_dir.empty()) {
        std::cout << "  CACHE_DIR      : " << args.cache_dir << "\n";
    }
    std::cout << std::setprecision(1);
    std::cout << "  System memory  : " << sys_mem_gb << " GB\n";
    std::cout << "\n";
    std::cout << std::setprecision(1);
    std::cout << "  Model load time: " << model_load_time_s << " s\n";
    std::cout << std::setprecision(2);
    std::cout << "  Cache size     : " << cache_size_gb << " GB\n";
    std::cout << "\n";

    // Table 1: Main KPIs
    std::cout << "  " << std::setw(13) << std::right << "Input tokens"
              << std::setw(14) << "Output tokens"
              << std::setw(14) << "Prefill (t/s)"
              << std::setw(11) << "Output TPS"
              << std::setw(16) << "Total peak (GB)"
              << std::setw(13) << "Private (GB)" << "\n";
    std::cout << "  " << std::string(81, '-') << "\n";

    for (const auto& s : results) {
        std::cout << "  " << std::setw(13) << s.input_tokens
                  << std::setw(14) << s.output_tokens
                  << std::setw(14) << std::fixed << std::setprecision(1) << s.prefill_speed_tps
                  << std::setw(11) << s.output_tps
                  << std::setw(16) << s.total_peak_memory_gb
                  << std::setw(13) << s.private_memory_gb << "\n";
    }

    // Table 2: TTFT / TPOT
    // Compute dynamic scenario name column width
    int scn_w = 8;  // minimum width for "Scenario" header
    for (const auto& s : results) {
        if (static_cast<int>(s.scenario.size()) > scn_w)
            scn_w = static_cast<int>(s.scenario.size());
    }
    int tbl2_width = scn_w + 10 + 10 + 14 + 11 + 4;

    std::cout << "\n";
    std::cout << "  " << std::setw(scn_w) << std::right << "Scenario"
              << std::setw(10) << "TTFT (ms)"
              << std::setw(10) << "TPOT (ms)"
              << std::setw(14) << "Prefill (t/s)"
              << std::setw(11) << "Output TPS" << "\n";
    std::cout << "  " << std::string(tbl2_width, '-') << "\n";

    for (const auto& s : results) {
        std::cout << "  " << std::setw(scn_w) << s.scenario
                  << std::setw(10) << std::fixed << std::setprecision(1) << s.ttft_ms
                  << std::setw(10) << s.tpot_ms
                  << std::setw(14) << s.prefill_speed_tps
                  << std::setw(11) << s.output_tps << "\n";
    }

    std::cout << sep << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// JSON report saver
// ═══════════════════════════════════════════════════════════════════════════════

static void save_report_json(
    const Args& args,
    double model_size_gb,
    double sys_mem_gb,
    double model_load_time_s,
    double cache_size_gb,
    const std::vector<ScenarioResult>& results,
    const std::string& path)
{
    std::ofstream f(path);
    if (!f) {
        std::cerr << "  Error: cannot write to " << path << "\n";
        return;
    }

    f << "{\n";
    f << "  \"model_name\": \"" << fs::path(args.model_dir).filename().string() << "\",\n";
    f << "  \"model_dir\": \"" << args.model_dir << "\",\n";
    f << "  \"model_size_gb\": " << std::fixed << std::setprecision(6) << model_size_gb << ",\n";
    f << "  \"device\": \"" << args.device << "\",\n";
    f << "  \"mmap_mode\": \"" << (args.no_mmap ? "OFF" : "ON") << "\",\n";
    f << "  \"cache_dir\": \"" << args.cache_dir << "\",\n";
    f << "  \"system_memory_gb\": " << sys_mem_gb << ",\n";
    f << "  \"model_load_time_s\": " << std::setprecision(2) << model_load_time_s << ",\n";
    f << "  \"cache_info\": {\n";
    f << "    \"cache_size_gb\": " << std::setprecision(6) << cache_size_gb << "\n";
    f << "  },\n";
    f << "  \"scenarios\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        const auto& s = results[i];
        f << "    {\n";
        f << "      \"scenario\": \"" << s.scenario << "\",\n";
        f << "      \"input_tokens\": " << s.input_tokens << ",\n";
        f << "      \"output_tokens\": " << s.output_tokens << ",\n";
        f << "      \"prefill_speed_tps\": " << std::setprecision(6) << s.prefill_speed_tps << ",\n";
        f << "      \"output_tps\": " << s.output_tps << ",\n";
        f << "      \"ttft_ms\": " << s.ttft_ms << ",\n";
        f << "      \"tpot_ms\": " << s.tpot_ms << ",\n";
        f << "      \"total_peak_memory_gb\": " << s.total_peak_memory_gb << ",\n";
        f << "      \"private_memory_gb\": " << s.private_memory_gb << ",\n";
        f << "      \"total_time_s\": " << s.total_time_s << "\n";
        f << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
    }

    f << "  ]\n";
    f << "}\n";
    f.close();

    std::cout << "  Report saved to: " << path << "\n";
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    try {
        Args args = parse_args(argc, argv);

        fs::path model_dir(args.model_dir);
        if (!fs::exists(model_dir)) {
            std::cerr << "Error: model directory not found: " << model_dir << "\n";
            return 1;
        }

        // ── System info ────────────────────────────────────────────────────
        double sys_mem_gb = get_system_memory_gb();
        double model_size_gb = get_model_size_gb(model_dir);
        double cache_size_gb = get_cache_size_gb(model_dir);

        const std::string sep(72, '=');
        std::cout << sep << "\n";
        std::cout << "  Gemma 4 E4B — KPI Benchmark (C++)\n";
        std::cout << sep << "\n";

        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        std::cout << "  Date           : " << std::put_time(std::localtime(&time_t_now), "%Y-%m-%d %H:%M:%S") << "\n";
        std::cout << std::fixed << std::setprecision(1);
        std::cout << "  System memory  : " << sys_mem_gb << " GB\n";
        std::cout << "  Model dir      : " << model_dir.string() << "\n";
        std::cout << "  Device         : " << args.device << "\n";
        std::cout << "  mmap           : " << (args.no_mmap ? "OFF (--no-mmap)" : "ON (default)") << "\n";
        std::cout << "  CACHE_DIR      : " << (args.cache_dir.empty() ? "not set" : args.cache_dir) << "\n";
        std::cout << "  Prompt files   : [";
        for (size_t i = 0; i < args.prompt_files.size(); ++i) {
            std::cout << args.prompt_files[i] << (i + 1 < args.prompt_files.size() ? ", " : "");
        }
        std::cout << "]\n";
        std::cout << "  Max new tokens : " << (args.max_new_tokens > 0 ? std::to_string(args.max_new_tokens) : "no limit") << "\n";
        std::cout << "  Warmup runs    : " << args.warmup << "\n";
        std::cout << std::setprecision(2);
        std::cout << "  Model size     : " << model_size_gb << " GB\n";

        if (args.skip_cache_measurement) {
            std::cout << "\n  Skipping cache creation measurement.\n";
        }

        // ── Load model ─────────────────────────────────────────────────────
        std::cout << "\n--- Loading model for scenario runs ---\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));

        auto t0 = std::chrono::steady_clock::now();

        ov::AnyMap properties;
        if (args.no_mmap) {
            properties["ENABLE_MMAP"] = false;
        }
        if (!args.cache_dir.empty()) {
            properties["CACHE_DIR"] = args.cache_dir;
        }

        auto pipe = ov::genai::VLMPipeline(model_dir, args.device, properties);

        auto t1 = std::chrono::steady_clock::now();
        double load_time = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "  Model loaded in " << std::fixed << std::setprecision(1)
                  << load_time << "s, RSS=" << std::setprecision(2)
                  << get_rss_gb() << " GB\n";

        // ── Load prompts from files ───────────────────────────────────────────
        std::cout << "\n--- Loading prompts from files ---\n";
        auto tokenizer = ov::genai::Tokenizer(model_dir);

        struct ScenarioInput {
            std::string name;
            std::string prompt;
            int actual_tokens;
        };
        std::vector<ScenarioInput> scenario_prompts;

        for (const auto& pf : args.prompt_files) {
            std::string prompt = load_prompt_file(pf);
            int actual = tokenize_prompt(tokenizer, prompt);
            std::string name = fs::path(pf).stem().string();
            scenario_prompts.push_back({name, prompt, actual});
            std::cout << "  " << name << ": " << actual << " input tokens  (" << pf << ")\n";
        }

        // ── Warmup ─────────────────────────────────────────────────────────
        if (args.warmup > 0) {
            std::cout << "\n--- Warmup (" << args.warmup << " runs with shortest prompt) ---\n";
            ov::genai::GenerationConfig warmup_config;
            warmup_config.max_new_tokens = 16;
            for (int i = 0; i < args.warmup; ++i) {
                std::cout << "  Warmup " << (i + 1) << "/" << args.warmup << "...\n";
                pipe.generate(scenario_prompts[0].prompt, ov::genai::generation_config(warmup_config));
            }
            std::cout << "  Warmup complete.\n";
        }

        // ── Run scenarios ──────────────────────────────────────────────────
        std::cout << "\n--- Running " << scenario_prompts.size() << " scenarios ---\n";

        std::vector<ScenarioResult> results;
        for (const auto& sp : scenario_prompts) {
            auto sr = run_scenario(pipe, sp.prompt, sp.name, sp.actual_tokens, args.max_new_tokens);
            results.push_back(sr);
        }

        // ── Print report ───────────────────────────────────────────────────
        print_report(args, model_size_gb, sys_mem_gb, load_time, cache_size_gb, results);

        // ── Save JSON ──────────────────────────────────────────────────────
        if (!args.output_json.empty()) {
            save_report_json(args, model_size_gb, sys_mem_gb, load_time,
                            cache_size_gb, results, args.output_json);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
