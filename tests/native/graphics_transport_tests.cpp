// CPU layout and routing checks; live API tests separately verify GPU transfers.
#include "../../src/integration/compatibility/pixel_transport.h"
#include "../../src/native/support/embedded_runtime.h"
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
