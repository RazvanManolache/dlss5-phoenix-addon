#pragma once
// One-shot exports of an owned, completed frame pair. No capture storage is
// allocated by the idle path, and encoding never re-runs the neural network.
#include <windows.h>
#include <d3d12.h>
#include <dxgiformat.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "../../third_party/nlohmann/json.hpp"

namespace phoenix_pipeline::screenshots {
using Json = nlohmann::json;
namespace fs = std::filesystem;
template <class T> using Com = Microsoft::WRL::ComPtr<T>;
inline void demand(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
inline void checked(HRESULT value) {
    if (FAILED(value))
        throw std::runtime_error("Screenshot graphics/PNG operation failed");
}
inline void png(const std::vector<uint8_t>& rgba, unsigned width, unsigned height,
                const fs::path& path) {
    demand(width && height && uint64_t(width) * height <= UINT_MAX / 4 &&
               rgba.size() == uint64_t(width) * height * 4,
           "Screenshot dimensions or pixels are invalid");
    const HRESULT apartment = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    demand(SUCCEEDED(apartment) || apartment == RPC_E_CHANGED_MODE,
           "Screenshot COM initialization failed");
    struct Apartment {
        bool owned;
        ~Apartment() {
            if (owned)
                CoUninitialize();
        }
    } cleanup{SUCCEEDED(apartment)};
    Com<IWICImagingFactory> factory;
    checked(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                             IID_PPV_ARGS(&factory)));
    Com<IWICStream> stream;
    checked(factory->CreateStream(&stream));
    checked(stream->InitializeFromFilename(path.c_str(), GENERIC_WRITE));
    Com<IWICBitmapEncoder> encoder;
    checked(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, &encoder));
    checked(encoder->Initialize(stream.Get(), WICBitmapEncoderNoCache));
    Com<IWICBitmapFrameEncode> frame;
    Com<IPropertyBag2> properties;
    checked(encoder->CreateNewFrame(&frame, &properties));
    checked(frame->Initialize(properties.Get()));
    checked(frame->SetSize(width, height));
    auto format = GUID_WICPixelFormat32bppBGRA;
    checked(frame->SetPixelFormat(&format));
    demand(IsEqualGUID(format, GUID_WICPixelFormat32bppBGRA),
           "PNG encoder changed the pixel layout");
    auto pixels = rgba;
    // Engine alpha often carries an auxiliary value or zero. These are viewable
    // screenshots, so keep RGB intact and make the PNG opaque on both sides.
    for (size_t i = 0; i < pixels.size(); i += 4) {
        std::swap(pixels[i], pixels[i + 2]);
        pixels[i + 3] = 255;
    }
    checked(frame->WritePixels(height, width * 4, UINT(pixels.size()), pixels.data()));
    checked(frame->Commit());
    checked(encoder->Commit());
}
inline float unsigned_float(unsigned word, unsigned fraction_bits) {
    const unsigned exponent = word >> fraction_bits, fraction = word & ((1u << fraction_bits) - 1);
    if (exponent == 31)
        return fraction ? 0.f : INFINITY;
    return std::ldexp(float(exponent ? fraction + (1u << fraction_bits) : fraction),
                      int(exponent ? exponent : 1) - 15 - int(fraction_bits));
}
inline float half(uint16_t word) {
    return (word & 0x8000 ? -1.f : 1.f) * unsigned_float(word & 0x7fff, 10);
}
inline unsigned pixel_bytes(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
        return 8;
    case DXGI_FORMAT_R11G11B10_FLOAT:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
        return 4;
    default:
        throw std::runtime_error("Paired PNG capture does not support this pixel format");
    }
}
inline std::vector<uint8_t> rgba8(const uint8_t* source, size_t count, unsigned width,
                                  unsigned height, size_t pitch, DXGI_FORMAT format, bool linear) {
    const auto bpp = pixel_bytes(format);
    demand(width && height && width <= 8192 && height <= 8192 && pitch >= size_t(width) * bpp &&
               count >= pitch * (height - 1) + size_t(width) * bpp,
           "Screenshot readback extent is invalid");
    std::vector<uint8_t> result(size_t(width) * height * 4);
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x) {
            const auto* p = source + y * pitch + x * bpp;
            float channels[4]{0, 0, 0, 1};
            if (format == DXGI_FORMAT_R16G16B16A16_FLOAT) {
                uint16_t words[4];
                std::memcpy(words, p, 8);
                for (unsigned c = 0; c < 4; ++c)
                    channels[c] = half(words[c]);
            } else if (format == DXGI_FORMAT_R11G11B10_FLOAT) {
                uint32_t word;
                std::memcpy(&word, p, 4);
                channels[0] = unsigned_float(word & 2047, 6);
                channels[1] = unsigned_float((word >> 11) & 2047, 6);
                channels[2] = unsigned_float(word >> 22, 5);
            } else if (format == DXGI_FORMAT_R10G10B10A2_UNORM) {
                uint32_t word;
                std::memcpy(&word, p, 4);
                for (unsigned c = 0; c < 3; ++c)
                    channels[c] = float((word >> (10 * c)) & 1023) / 1023;
                channels[3] = float(word >> 30) / 3;
            } else {
                const bool bgra = format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                                  format == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
                for (unsigned c = 0; c < 4; ++c)
                    channels[c] = float(p[bgra && c < 3 ? 2 - c : c]) / 255;
            }
            for (unsigned c = 0; c < 4; ++c) {
                float value = std::isnan(channels[c]) ? 0.f : std::clamp(channels[c], 0.f, 1.f);
                if (linear && c < 3)
                    value = value <= .0031308f ? 12.92f * value
                                               : 1.055f * std::pow(value, 1.f / 2.4f) - .055f;
                result[(size_t(y) * width + x) * 4 + c] = uint8_t(std::lround(value * 255));
            }
        }
    return result;
}
struct PairExport {
    fs::path destination;
    Json result = Json::object();
    bool pending = false;
    bool request(const fs::path& folder) {
        if (pending || folder.empty())
            return false;
        destination = folder;
        pending = true;
        result = {{"state", "pending"}};
        return true;
    }
    void fail(const std::string& error) {
        pending = false;
        result = {{"state", "error"}, {"error", error}};
    }
    void save(const std::vector<uint8_t>& original, const std::vector<uint8_t>& modified,
              unsigned width, unsigned height, uint64_t frame,
              Json facts = Json::object()) noexcept {
        if (!pending)
            return;
        fs::path folder;
        bool owns_folder = false;
        try {
            demand(original.size() == modified.size() &&
                       original.size() == uint64_t(width) * height * 4,
                   "Paired screenshot pixels have different extents");
            fs::create_directories(destination);
            FILETIME clock{};
            GetSystemTimeAsFileTime(&clock);
            static std::atomic<uint64_t> sequence{0};
            const uint64_t stamp = (uint64_t(clock.dwHighDateTime) << 32) | clock.dwLowDateTime;
            folder =
                destination / ("pair-" + std::to_string(stamp) + "-" + std::to_string(++sequence));
            owns_folder = fs::create_directory(folder);
            demand(owns_folder, "Cannot reserve paired screenshot folder");
            png(original, width, height, folder / "original.png.tmp");
            png(modified, width, height, folder / "modified.png.tmp");
            fs::rename(folder / "original.png.tmp", folder / "original.png");
            fs::rename(folder / "modified.png.tmp", folder / "modified.png");
            facts["schema"] = "phoenix-screenshot-pair/1";
            facts["frame"] = frame;
            facts["same_frame"] = true;
            facts["width"] = width;
            facts["height"] = height;
            facts["alpha"] = "opaque screenshot; source RGB preserved";
            facts["original"] = "original.png";
            facts["modified"] = "modified.png";
            facts["scope"] =
                "Matching processing input and current modified output; excludes later game/ReShade effects and overlay";
            std::ofstream receipt(folder / "pair.json.tmp", std::ios::binary);
            receipt << facts.dump(2) << '\n';
            receipt.close();
            demand(bool(receipt), "Cannot finish paired screenshot receipt");
            fs::rename(folder / "pair.json.tmp", folder / "pair.json");
            const auto utf8 = folder.u8string();
            result = {{"state", "saved"},
                      {"directory", std::string(utf8.begin(), utf8.end())},
                      {"frame", frame}};
            pending = false;
        } catch (const std::exception& error) {
            if (owns_folder) {
                std::error_code ignored;
                for (const auto* name : {"original.png.tmp", "modified.png.tmp", "original.png",
                                         "modified.png", "pair.json.tmp"})
                    fs::remove(folder / name, ignored);
                fs::remove(folder, ignored);
            }
            fail(error.what());
        }
    }
    std::string message() const {
        const auto state = result.value("state", std::string());
        if (state == "pending")
            return "Paired screenshot pending: waiting for a completed processed frame.";
        if (state == "error")
            return "Paired screenshot failed: " + result.value("error", std::string());
        if (state == "saved")
            return "Saved original + modified: " + result.value("directory", std::string());
        return {};
    }
};

// Readback helpers may finish an older request after the user presses Capture.
// Retain only the next accepted input; ignore results with any other token.
struct CompletedPair {
    PairExport output;
    std::vector<uint8_t> original;
    uint64_t frame = 0;
    void remember(uint64_t token, const std::vector<uint8_t>& pixels) {
        if (output.pending && !frame && token) {
            original = pixels;
            frame = token;
        }
    }
    void rejected(uint64_t token) {
        if (frame != token)
            return;
        frame = 0;
        std::vector<uint8_t>().swap(original);
    }
    void complete(uint64_t token, const std::vector<uint8_t>& pixels, unsigned width,
                  unsigned height) {
        if (!output.pending || !frame || frame != token)
            return;
        output.save(original, pixels, width, height, frame, {{"route", "fenced helper/readback"}});
        rejected(token);
    }
};

class D3D12Pair {
    std::array<Com<ID3D12Resource>, 2> readbacks_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};
    uint64_t bytes_ = 0, retired_at_ = 0, frame_ = 0;
    unsigned width_ = 0, height_ = 0;
    DXGI_FORMAT format_ = DXGI_FORMAT_UNKNOWN;
    bool linear_ = false, started_ = false;
    ID3D12GraphicsCommandList* recording_ = nullptr;
    bool submitted_ = false;
    Json facts_;
    void copy(ID3D12GraphicsCommandList* command, ID3D12Resource* image,
              D3D12_RESOURCE_STATES state, unsigned index) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition = {image, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, state,
                              D3D12_RESOURCE_STATE_COPY_SOURCE};
        if (state != D3D12_RESOURCE_STATE_COPY_SOURCE)
            command->ResourceBarrier(1, &barrier);
        D3D12_TEXTURE_COPY_LOCATION source{}, destination{};
        source.pResource = image;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.pResource = readbacks_[index].Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint = footprint_;
        command->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        std::swap(barrier.Transition.StateBefore, barrier.Transition.StateAfter);
        if (state != D3D12_RESOURCE_STATE_COPY_SOURCE)
            command->ResourceBarrier(1, &barrier);
    }

  public:
    PairExport output;
    bool start(ID3D12GraphicsCommandList* command, ID3D12Resource* original,
               D3D12_RESOURCE_STATES state, uint64_t frame, bool linear,
               Json facts = Json::object()) noexcept {
        if (!output.pending || started_)
            return false;
        try {
            auto desc = original->GetDesc();
            pixel_bytes(desc.Format);
            demand(desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D && desc.Width &&
                       desc.Width <= 8192 && desc.Height && desc.Height <= 8192 &&
                       desc.DepthOrArraySize == 1 && desc.MipLevels == 1 &&
                       desc.SampleDesc.Count == 1,
                   "Paired screenshot texture layout is unsupported");
            Com<ID3D12Device> device;
            checked(original->GetDevice(IID_PPV_ARGS(&device)));
            device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint_, nullptr, nullptr, &bytes_);
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_READBACK;
            D3D12_RESOURCE_DESC buffer{};
            buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            buffer.Width = bytes_;
            buffer.Height = 1;
            buffer.DepthOrArraySize = buffer.MipLevels = 1;
            buffer.SampleDesc.Count = 1;
            buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            for (auto& resource : readbacks_)
                checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                        IID_PPV_ARGS(&resource)));
            width_ = unsigned(desc.Width);
            height_ = desc.Height;
            format_ = desc.Format;
            linear_ = linear;
            frame_ = frame;
            facts_ = std::move(facts);
            copy(command, original, state, 0);
            recording_ = command;
            started_ = true;
            return true;
        } catch (const std::exception& error) {
            output.fail(error.what());
            for (auto& resource : readbacks_)
                resource.Reset();
            bytes_ = 0;
            return false;
        }
    }
    void finish(ID3D12GraphicsCommandList* command, ID3D12Resource* modified,
                D3D12_RESOURCE_STATES state, uint64_t fence) {
        if (!started_ || retired_at_)
            return;
        demand(command == recording_, "Paired screenshot command list changed");
        copy(command, modified, state, 1);
        retired_at_ = fence;
    }
    void submitted(ID3D12CommandList* command) {
        if (command == recording_)
            submitted_ = true;
    }
    void abandon(ID3D12GraphicsCommandList* command) {
        if (command != recording_ || submitted_)
            return;
        output.fail("Capture canceled because its command list was discarded.");
        for (auto& resource : readbacks_)
            resource.Reset();
        bytes_ = retired_at_ = 0;
        started_ = false;
        recording_ = nullptr;
    }
    void poll(uint64_t completed) noexcept {
        if (!submitted_ || !retired_at_ || completed == UINT64_MAX || completed < retired_at_)
            return;
        try {
            std::array<std::vector<uint8_t>, 2> images;
            for (unsigned i = 0; i < 2; ++i) {
                void* mapped = nullptr;
                D3D12_RANGE range{0, size_t(bytes_)};
                checked(readbacks_[i]->Map(0, &range, &mapped));
                try {
                    images[i] = rgba8(static_cast<const uint8_t*>(mapped) + footprint_.Offset,
                                      size_t(bytes_ - footprint_.Offset), width_, height_,
                                      footprint_.Footprint.RowPitch, format_, linear_);
                } catch (...) {
                    readbacks_[i]->Unmap(0, nullptr);
                    throw;
                }
                D3D12_RANGE written{0, 0};
                readbacks_[i]->Unmap(0, &written);
            }
            facts_["pixel_conversion"] = linear_ ? "linear RGB clipped to SDR then sRGB encoded"
                                                 : "SDR values clipped/quantized to RGBA8";
            output.save(images[0], images[1], width_, height_, frame_, facts_);
        } catch (const std::exception& error) {
            output.fail(error.what());
        }
        for (auto& resource : readbacks_)
            resource.Reset();
        bytes_ = retired_at_ = 0;
        started_ = submitted_ = false;
        recording_ = nullptr;
    }
    uint64_t readback_bytes() const {
        return bytes_ * 2;
    }
};
} // namespace phoenix_pipeline::screenshots
