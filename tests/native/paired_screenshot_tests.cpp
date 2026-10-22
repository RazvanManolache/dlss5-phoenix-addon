// WARP only: snapshot a texture before and after a mutation on one command
// list, then decode both PNGs. No model, game or hardware GPU is used.
#include "../../src/pipeline/paired_screenshot.h"
#include <dxgi1_6.h>
#include <iostream>
using namespace phoenix_pipeline::screenshots;
static unsigned checks = 0;
static void check(bool value) {
    ++checks;
    demand(value, "Paired screenshot regression failed");
}
static std::vector<uint8_t> decode(const fs::path& file, unsigned width, unsigned height) {
    Com<IWICImagingFactory> factory;
    checked(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                             IID_PPV_ARGS(&factory)));
    Com<IWICBitmapDecoder> decoder;
    checked(factory->CreateDecoderFromFilename(file.c_str(), nullptr, GENERIC_READ,
                                               WICDecodeMetadataCacheOnLoad, &decoder));
    Com<IWICBitmapFrameDecode> frame;
    checked(decoder->GetFrame(0, &frame));
    UINT w = 0, h = 0;
    checked(frame->GetSize(&w, &h));
    check(w == width && h == height);
    Com<IWICFormatConverter> convert;
    checked(factory->CreateFormatConverter(&convert));
    checked(convert->Initialize(frame.Get(), GUID_WICPixelFormat32bppRGBA, WICBitmapDitherTypeNone,
                                nullptr, 0, WICBitmapPaletteTypeCustom));
    std::vector<uint8_t> pixels(size_t(width) * height * 4);
    checked(convert->CopyPixels(nullptr, width * 4, UINT(pixels.size()), pixels.data()));
    return pixels;
}
int main() {
    try {
        checked(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
        const fs::path folder = fs::current_path() / "paired-screenshot-test";
        demand(fs::create_directory(folder), "Test folder already exists");
        struct Cleanup {
            fs::path folder;
            ~Cleanup() {
                std::error_code error;
                fs::remove_all(folder, error);
                CoUninitialize();
            }
        } cleanup{folder};
        const uint8_t bgra[]{3, 7, 11, 255};
        check(rgba8(bgra, 4, 1, 1, 4, DXGI_FORMAT_B8G8R8A8_UNORM, false) ==
              std::vector<uint8_t>({11, 7, 3, 255}));
        const uint16_t floating[]{0x3c00, 0x3800, 0xbc00, 0x3c00};
        check(rgba8(reinterpret_cast<const uint8_t*>(floating), 8, 1, 1, 8,
                    DXGI_FORMAT_R16G16B16A16_FLOAT,
                    false) == std::vector<uint8_t>({255, 128, 0, 255}));
        check(half(0x0001) > 0 && half(0x0001) < .001f && half(0x7c00) == INFINITY);
        bool rejected = false;
        try {
            rgba8(bgra, 3, 1, 1, 4, DXGI_FORMAT_R8G8B8A8_UNORM, false);
        } catch (...) {
            rejected = true;
        }
        check(rejected);
        PairExport malformed;
        check(malformed.request(folder));
        malformed.save({}, {}, 0, 0, 0);
        check(malformed.result["state"] == "error");
        png({11,17,33,0},1,1,folder/"opaque.png");
        check(decode(folder/"opaque.png",1,1)==std::vector<uint8_t>({11,17,33,255}));
        CompletedPair latch;
        check(latch.output.request(folder));
        const std::vector<uint8_t> red{255, 0, 0, 255}, blue{0, 0, 255, 255};
        latch.remember(5, red);
        latch.complete(4, blue, 1, 1);
        check(latch.output.pending && latch.frame == 5);
        latch.rejected(5);
        check(latch.frame == 0 && latch.original.empty());
        latch.remember(6, red);
        latch.remember(7, blue);
        check(latch.frame == 6 && latch.original == red);
        latch.complete(6, blue, 1, 1);
        check(latch.output.result["state"] == "saved" && latch.original.empty());
        const auto paired = fs::u8path(latch.output.result["directory"].get<std::string>());
        check(decode(paired / "original.png", 1, 1) == red &&
              decode(paired / "modified.png", 1, 1) == blue);
        Com<IDXGIFactory4> factory;
        checked(CreateDXGIFactory1(IID_PPV_ARGS(&factory)));
        Com<IDXGIAdapter> adapter;
        checked(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)));
        Com<ID3D12Device> device;
        checked(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)));
        Com<ID3D12CommandQueue> queue;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        checked(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)));
        Com<ID3D12Fence> fence;
        checked(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));
        D3D12Pair pair;
        check(pair.readback_bytes() == 0);
        fs::path previous;
        for (unsigned pass = 1; pass <= 3; ++pass) {
            Com<ID3D12CommandAllocator> allocator;
            checked(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                   IID_PPV_ARGS(&allocator)));
            Com<ID3D12GraphicsCommandList> command;
            checked(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(),
                                              nullptr, IID_PPV_ARGS(&command)));
            constexpr unsigned width = 17, height = 9;
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            desc.Width = width;
            desc.Height = height;
            desc.DepthOrArraySize = desc.MipLevels = 1;
            desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            desc.SampleDesc.Count = 1;
            D3D12_HEAP_PROPERTIES heap{};
            heap.Type = D3D12_HEAP_TYPE_DEFAULT;
            Com<ID3D12Resource> texture;
            checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&texture)));
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
            uint64_t bytes = 0;
            device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &bytes);
            std::array<Com<ID3D12Resource>, 2> uploads;
            const std::array<std::array<uint8_t, 4>, 2> colors{
                {{255, 0, 128, 255}, {0, 128, 255, 255}}};
            for (unsigned i = 0; i < 2; ++i) {
                auto buffer = desc;
                buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
                buffer.Width = bytes;
                buffer.Height = 1;
                buffer.Format = DXGI_FORMAT_UNKNOWN;
                buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
                heap.Type = D3D12_HEAP_TYPE_UPLOAD;
                checked(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                        IID_PPV_ARGS(&uploads[i])));
                void* data = nullptr;
                checked(uploads[i]->Map(0, nullptr, &data));
                for (unsigned y = 0; y < height; ++y)
                    for (unsigned x = 0; x < width; ++x)
                        std::memcpy(static_cast<uint8_t*>(data) + y * footprint.Footprint.RowPitch +
                                        x * 4,
                                    colors[i].data(), 4);
                uploads[i]->Unmap(0, nullptr);
            }
            auto upload = [&](unsigned i) {
                D3D12_TEXTURE_COPY_LOCATION source{}, target{};
                source.pResource = uploads[i].Get();
                source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
                source.PlacedFootprint = footprint;
                target.pResource = texture.Get();
                target.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
                command->CopyTextureRegion(&target, 0, 0, 0, &source, nullptr);
            };
            check(pair.output.request(folder));
            check(!pair.output.request(folder));
            upload(0);
            check(pair.start(command.Get(), texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                             100 + pass, false));
            check(pair.readback_bytes() > 0);
            upload(1);
            pair.finish(command.Get(), texture.Get(), D3D12_RESOURCE_STATE_COPY_DEST, pass);
            if (pass == 3) {
                checked(command->Close());
                pair.abandon(command.Get());
                check(pair.output.result["state"] == "error" && pair.readback_bytes() == 0);
                continue;
            }
            pair.poll(pass);
            check(pair.output.pending);
            pair.poll(pass - 1);
            check(pair.output.pending);
            checked(command->Close());
            ID3D12CommandList* lists[]{command.Get()};
            queue->ExecuteCommandLists(1, lists);
            pair.submitted(command.Get());
            checked(queue->Signal(fence.Get(), pass));
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            demand(event != nullptr, "No test fence event");
            checked(fence->SetEventOnCompletion(pass, event));
            const auto waited = WaitForSingleObject(event, 10000);
            CloseHandle(event);
            check(waited == WAIT_OBJECT_0);
            pair.poll(fence->GetCompletedValue());
            if (pair.output.result["state"] != "saved")
                throw std::runtime_error(pair.output.result.dump());
            check(!pair.output.pending && pair.output.result["state"] == "saved");
            check(pair.readback_bytes() == 0);
            const auto saved = fs::u8path(pair.output.result["directory"].get<std::string>());
            check(saved != previous);
            previous = saved;
            for (unsigned i = 0; i < 2; ++i) {
                const auto pixels =
                    decode(saved / (i ? "modified.png" : "original.png"), width, height);
                for (size_t offset = 0; offset < pixels.size(); offset += 4)
                    check(std::equal(colors[i].begin(), colors[i].end(), pixels.begin() + offset));
            }
            std::ifstream receipt(saved / "pair.json");
            Json facts;
            receipt >> facts;
            check(facts["same_frame"] == true && facts["frame"] == 100 + pass);
        }
        std::cout << Json({{"passed", true},
                           {"checks", checks},
                           {"software_d3d12", true},
                           {"gpu_inference", false},
                           {"games_launched", false},
                           {"same_texture_before_after_verified", true},
                           {"readbacks_released", true}})
                         .dump()
                  << '\n';
    } catch (const std::exception& error) {
        std::cerr << "check " << checks << ": " << error.what() << '\n';
        return 1;
    }
}
