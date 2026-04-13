// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <set>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "load_image.hpp"

namespace fs = std::filesystem;

std::vector<ov::Tensor> utils::load_images(const std::filesystem::path& input_path) {
    if (input_path.empty() || !fs::exists(input_path)) {
        throw std::runtime_error{"Path to images is empty or does not exist."};
    }
    if (fs::is_directory(input_path)) {
        std::set<fs::path> sorted_images{fs::directory_iterator(input_path),
                                         fs::directory_iterator()};
        std::vector<ov::Tensor> images;
        for (const fs::path& entry : sorted_images) {
            images.push_back(utils::load_image(entry));
        }
        return images;
    }
    return {utils::load_image(input_path)};
}

ov::Tensor utils::load_image(const std::filesystem::path& image_path) {
    int x = 0, y = 0, channels_in_file = 0;
    constexpr int desired_channels = 3;
    unsigned char* image = stbi_load(
        image_path.string().c_str(),
        &x, &y, &channels_in_file, desired_channels);
    if (!image) {
        std::stringstream ss;
        ss << "Failed to load the image '" << image_path << "'";
        throw std::runtime_error{ss.str()};
    }
    struct SharedImageAllocator {
        unsigned char* image;
        int channels, height, width;
        void* allocate(size_t bytes, size_t) const {
            if (image && static_cast<size_t>(channels) * height * width == bytes) {
                return image;
            }
            throw std::runtime_error{"Unexpected number of bytes was requested to allocate."};
        }
        void deallocate(void*, size_t, size_t) noexcept {
            stbi_image_free(image);
            image = nullptr;
        }
        bool is_equal(const SharedImageAllocator& other) const noexcept {
            return this == &other;
        }
    };
    return ov::Tensor(
        ov::element::u8,
        ov::Shape{1, static_cast<size_t>(y), static_cast<size_t>(x),
                  static_cast<size_t>(desired_channels)},
        SharedImageAllocator{image, desired_channels, y, x});
}
