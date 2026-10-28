// CPU-only format refusal and PNG normalization contracts. No graphics device
// or model is opened; live screenshot/frame identity is tested by the API fixture.
#include "../../src/integration/compatibility/screenshot.h"
#include <iostream>
int main() {
    using namespace phoenix_compat;
    using namespace reshade::api;
    unsigned checks = 0;
    const auto check = [&](bool valid) {
        ++checks;
        require(valid, "Screenshot CPU contract failed");
    };
    const auto refused = [&](auto callback) {
        bool failed = false;
        try {
            callback();
        } catch (const std::runtime_error&) {
            failed = true;
        }
        check(failed);
    };
    resource_desc canvas(resource_type::texture_2d, 257, 193, 1, 1, format::r8g8b8a8_unorm, 1,
                         memory_heap::gpu_only, resource_usage::render_target);
    for (auto api : {device_api::d3d9, device_api::d3d10, device_api::d3d11, device_api::d3d12,
                     device_api::vulkan, device_api::opengl}) {
        for (auto pixel_format :
             {format::r8g8b8a8_unorm, format::b8g8r8a8_unorm, format::b8g8r8x8_unorm,
              format::b8g8r8a8_typeless, format::r8g8b8a8_unorm_srgb}) {
            canvas.texture.format = pixel_format;
            const auto layout = screenshot_layout(api, canvas);
            const auto staging = screenshot_readback_description(api, canvas);
            check(staging.heap == memory_heap::readback &&
                  staging.usage == resource_usage::copy_dest);
            check(staging.type == (layout.buffers ? resource_type::buffer :
                                  api == device_api::d3d9 ? resource_type::surface :
                                  resource_type::texture_2d));
            check(layout.width == 257 && layout.height == 193);
            check(layout.row_pitch == (api == device_api::d3d12 ? 1280u : 1028u));
            check(layout.flip_rows == (api == device_api::opengl));
            check(layout.bgra == (pixel_format == format::b8g8r8a8_unorm ||
                                  pixel_format == format::b8g8r8x8_unorm ||
                                  pixel_format == format::b8g8r8a8_typeless));
        }
        for (auto packed : {format::r10g10b10a2_unorm, format::r10g10b10a2_typeless}) {
            canvas.texture.format = packed;
            if (supports_rgb10_readback(api)) check(screenshot_layout(api, canvas).rgb10);
            else refused([&] { screenshot_layout(api, canvas); });
        }
        for (auto unsupported : {format::r16g16b16a16_float,
                                 format::r8_unorm, format::unknown}) {
            canvas.texture.format = unsupported;
            refused([&] { screenshot_layout(api, canvas); });
        }
    }
    canvas.texture.format = format::r8g8b8a8_unorm;
    canvas.texture.samples = 4;
    refused([&] { screenshot_layout(device_api::d3d11, canvas); });
    for (auto samples : {2, 4, 8}) {
        canvas.texture.samples = uint16_t(samples);
        const auto layout = screenshot_layout(device_api::d3d9, canvas);
        check(layout.width == 257 && layout.height == 193 && !layout.buffers);
        const auto staging = screenshot_readback_description(device_api::d3d9, canvas);
        check(staging.type == resource_type::surface && staging.texture.samples == 1 &&
              staging.texture.width == 257 && staging.texture.height == 193);
    }
    canvas.texture.samples = 1;
    canvas.texture.width = 0;
    refused([&] { screenshot_layout(device_api::d3d12, canvas); });
    canvas.texture.width = 8193;
    refused([&] { screenshot_layout(device_api::vulkan, canvas); });
    canvas.texture.width = 257;
    canvas.type = resource_type::buffer;
    refused([&] { screenshot_layout(device_api::d3d12, canvas); });
    const fs::path image = fs::current_path() / "screenshot-encoder-test.png";
    refused([&] { save_rgba8_png({}, 0, 1, image); });
    refused([&] { save_rgba8_png({0, 0, 0}, 1, 1, image); });
    refused([&] { save_rgba8_png({}, UINT32_MAX, UINT32_MAX, image); });
    constexpr unsigned width = 17, height = 9;
    std::vector<uint8_t> expected(width * height * 4);
    for (unsigned row = 0; row < height; ++row)
        for (unsigned column = 0; column < width; ++column) {
            const auto offset = (row * width + column) * 4;
            expected[offset] = uint8_t(column * 13);
            expected[offset + 1] = uint8_t(row * 23);
            expected[offset + 2] = uint8_t((row ^ column) * 11);
            expected[offset + 3] = uint8_t(33 + row * 7 + column * 3);
        }
    save_rgba8_png(expected, width, height, image);
    hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
    {
        Ptr<IWICImagingFactory> factory;
        hr(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                            IID_PPV_ARGS(&factory)));
        Ptr<IWICBitmapDecoder> decoder;
        hr(factory->CreateDecoderFromFilename(image.c_str(), nullptr, GENERIC_READ,
                                              WICDecodeMetadataCacheOnLoad, &decoder));
        Ptr<IWICBitmapFrameDecode> frame;
        hr(decoder->GetFrame(0, &frame));
        UINT decoded_width = 0, decoded_height = 0;
        hr(frame->GetSize(&decoded_width, &decoded_height));
        check(decoded_width == width && decoded_height == height);
        Ptr<IWICFormatConverter> converter;
        hr(factory->CreateFormatConverter(&converter));
        hr(converter->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone,
                                 nullptr, 0, WICBitmapPaletteTypeCustom));
        std::vector<uint8_t> actual(expected.size());
        hr(converter->CopyPixels(nullptr, width * 4, UINT(actual.size()), actual.data()));
        check(actual == expected);
    }
    CoUninitialize();
    std::cout << Json({{"passed", true},
                       {"checks", checks},
                       {"bits", sizeof(void*) * 8},
                       {"native_hdr_refused_before_allocation", true},
                       {"png_rgba_pixels_identical", true}})
                     .dump()
              << '\n';
}
