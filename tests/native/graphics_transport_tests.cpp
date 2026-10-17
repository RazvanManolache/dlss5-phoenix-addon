// CPU layout and routing checks; live API tests separately verify GPU transfers.
#include "../../src/integration/compatibility/pixel_transport.h"
#include "../../src/native/support/embedded_runtime.h"
#include <array>
#include <cstdlib>
#include <iostream>
int main() {
    using namespace phoenix_compat;
    using reshade::api::device_api;
    unsigned checks = 0;
    auto check = [&](bool value) {
        ++checks;
        if (!value)
            throw std::runtime_error("Graphics transport contract failed");
    };
    for (auto api : {device_api::d3d9, device_api::d3d10, device_api::d3d11, device_api::d3d12,
                     device_api::opengl, device_api::vulkan}) {
        check(uses_helper(api, 32));
        check(uses_helper(api, 64) == (api != device_api::d3d12));
        for (unsigned width : {1, 63, 64, 65, 257, 1920, 8192}) {
            const auto pitch = transfer_row_pitch(api, width);
            check(pitch >= width * 4);
            check(api != device_api::d3d12 || pitch % 256 == 0);
        }
    }
    for (unsigned width : {1, 65, 257}) {
        constexpr unsigned height = 3;
        std::vector<uint8_t> rgba(size_t(width) * height * 4), roundtrip(rgba.size());
        for (size_t index = 0; index < rgba.size(); ++index)
            rgba[index] = uint8_t(index * 31 + index / 17);
        const auto pitch = transfer_row_pitch(device_api::d3d12, width);
        for (bool flip : {false, true})
            for (bool bgra : {false, true}) {
                std::vector<uint8_t> native(size_t(pitch) * height, 0xCD);
                copy_pixel_rows(native.data(), pitch, rgba.data(), width * 4, width, height, flip,
                                bgra);
                copy_pixel_rows(roundtrip.data(), width * 4, native.data(), pitch, width, height,
                                flip, bgra);
                check(roundtrip == rgba);
                for (unsigned row = 0; row < height; ++row)
                    for (unsigned byte = width * 4; byte < pitch; ++byte)
                        check(native[size_t(row) * pitch + byte] == 0xCD);
                // Independently check corner identity; a double upside-down or
                // double-swizzled roundtrip alone would not detect a wrong boundary.
                check(native[0] ==
                      rgba[size_t(flip ? height - 1 : 0) * width * 4 + (bgra ? 2 : 0)]);
            }
    }
    for (unsigned width : {1, 65, 257}) {
        constexpr unsigned height = 3;
        const auto pitch = transfer_row_pitch(device_api::d3d12, width);
        std::vector<uint8_t> rgba(size_t(width) * height * 4), packed(size_t(pitch) * height, 0xCD), decoded(rgba.size());
        for (size_t pixel = 0; pixel < rgba.size() / 4; ++pixel) {
            rgba[pixel * 4] = uint8_t(pixel);
            rgba[pixel * 4 + 1] = uint8_t(pixel * 31);
            rgba[pixel * 4 + 2] = uint8_t(pixel * 17);
            rgba[pixel * 4 + 3] = uint8_t((pixel % 4) * 85);
        }
        copy_rgb10a2_rows(packed.data(), pitch, rgba.data(), width * 4, width, height, false);
        copy_rgb10a2_rows(decoded.data(), width * 4, packed.data(), pitch, width, height, true);
        check(decoded == rgba);
        for (unsigned row = 0; row < height; ++row)
            for (unsigned byte = width * 4; byte < pitch; ++byte)
                check(packed[size_t(row) * pitch + byte] == 0xCD);
    }
    for (uint32_t value = 0; value <= 1023; ++value) {
        const uint32_t packed = value | (value << 10) | (value << 20) | (3u << 30);
        std::array<uint8_t, 4> rgba{}, native{};
        copy_rgb10a2_rows(rgba.data(), 4, reinterpret_cast<const uint8_t*>(&packed), 4, 1, 1, true);
        check(rgba[0] == uint8_t((value * 255u + 511u) / 1023u));
        check(rgba[0] == rgba[1] && rgba[1] == rgba[2] && rgba[3] == 255);
        copy_rgb10a2_rows(native.data(), 4, rgba.data(), 4, 1, 1, false);
        uint32_t restored = 0; std::memcpy(&restored, native.data(), 4);
        check(std::abs(int(restored & 1023u) - int(value)) <= 2);
    }
    bool rejected = false;
    try { uint8_t pixel[4]{}; copy_rgb10a2_rows(pixel, 3, pixel, 4, 1, 1, true); }
    catch (const std::exception&) { rejected = true; }
    check(rejected);
    const auto temporary = std::filesystem::temp_directory_path() /
                           ("phoenix-hash-" + std::to_string(GetCurrentProcessId()) + ".bin");
    std::vector<uint8_t> bytes(65536 * 2 + 31, 0xA7);
    {
        std::ofstream stream(temporary, std::ios::binary);
        stream.write(reinterpret_cast<char*>(bytes.data()), bytes.size());
    }
    check(phoenix_embedded::file_digest(temporary) ==
          phoenix_model::weight_digest(bytes.data(), bytes.size()));
    std::filesystem::remove(temporary);
    std::cout << "{\"passed\":true,\"checks\":" << checks << ",\"bits\":" << sizeof(void*) * 8
              << "}\n";
}
