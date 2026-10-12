#pragma once
#include <windows.h>
#include <dxgi1_6.h>
#include <d3d12sdklayers.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdio>
#include <wrl/client.h>
#include <string>
#include <stdexcept>
#include <cstring>
#include <d3d12.h>
using Microsoft::WRL::ComPtr;
static void check(HRESULT result) {
    if (FAILED(result))
        throw std::runtime_error("D3D12 probe failure " + std::to_string(unsigned(result)));
}
struct Texture {
    unsigned width = 0, height = 0, bits = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    ComPtr<ID3D12Resource> resource, transfer;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 bytes = 0;
};
static std::vector<unsigned char> read(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Missing HDR fixture " + path.string());
    auto size = file.tellg();
    if (size <= 0 || size > 256 * 1024 * 1024)
        throw std::runtime_error("Invalid fixture size");
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
        throw std::runtime_error("Short fixture read");
    return bytes;
}
static void barrier(ID3D12GraphicsCommandList* command_list, ID3D12Resource* resource,
                    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER transition{};
    transition.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    transition.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    command_list->ResourceBarrier(1, &transition);
}
static Texture texture(ID3D12Device* device, unsigned width, unsigned height, unsigned bits,
                       bool output, unsigned channels = 4) {
    if (!width || !height || width > 16384 || height > 16384 || (bits != 16 && bits != 32) ||
        (channels != 4 && !(channels == 2 && bits == 16)))
        throw std::invalid_argument("Fixture geometry or format");
    Texture value;
    value.width = width;
    value.height = height;
    value.bits = bits;
    // Motion fixtures use RG16F, matching the NVIDIA comparison harness.
    value.format = channels == 2 ? DXGI_FORMAT_R16G16_FLOAT
                   : bits == 16 ? DXGI_FORMAT_R16G16B16A16_FLOAT
                                : DXGI_FORMAT_R32G32B32A32_FLOAT;
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    description.Width = width;
    description.Height = height;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = value.format;
    description.SampleDesc.Count = 1;
    if (output)
        description.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
                                          output ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                                                 : D3D12_RESOURCE_STATE_COPY_DEST,
                                          nullptr, IID_PPV_ARGS(&value.resource)));
    device->GetCopyableFootprints(&description, 0, 1, 0, &value.footprint, nullptr, nullptr,
                                  &value.bytes);
    description = {};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Width = value.bytes;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.SampleDesc.Count = 1;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    heap.Type = output ? D3D12_HEAP_TYPE_READBACK : D3D12_HEAP_TYPE_UPLOAD;
    check(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &description,
                                          output ? D3D12_RESOURCE_STATE_COPY_DEST
                                                 : D3D12_RESOURCE_STATE_GENERIC_READ,
                                          nullptr, IID_PPV_ARGS(&value.transfer)));
    return value;
}
// Copy tightly packed fixture rows into the D3D12-aligned upload footprint,
// then transition the GPU texture for compute sampling. RowPitch includes padding.
static Texture upload(ID3D12Device* device, ID3D12GraphicsCommandList* command_list,
                      const std::filesystem::path& path) {
    auto bytes = read(path);
    if (bytes.size() < 12)
        throw std::invalid_argument("Texture header missing");
    unsigned header[3];
    memcpy(header, bytes.data(), 12);
    auto value = texture(device, header[0], header[1], header[2], false);
    const size_t row = size_t(value.width) * 4 * value.bits / 8;
    if (bytes.size() != 12 + row * value.height)
        throw std::invalid_argument("Texture payload mismatch");
    void* mapped = nullptr;
    D3D12_RANGE empty{};
    check(value.transfer->Map(0, &empty, &mapped));
    for (unsigned row_index = 0; row_index < value.height; ++row_index)
        memcpy(static_cast<unsigned char*>(mapped) + value.footprint.Offset +
                   row_index * value.footprint.Footprint.RowPitch,
               bytes.data() + 12 + row_index * row, row);
    value.transfer->Unmap(0, nullptr);
    D3D12_TEXTURE_COPY_LOCATION from{}, to{};
    from.pResource = value.transfer.Get();
    from.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    from.PlacedFootprint = value.footprint;
    to.pResource = value.resource.Get();
    to.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    command_list->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
    barrier(command_list, value.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return value;
}
// Strip readback row padding and preserve the fixture header (width, height,
// component bits); only the four stored color components are written per pixel.
static void save(const Texture& value, const std::filesystem::path& path) {
    void* mapped = nullptr;
    D3D12_RANGE range{0, SIZE_T(value.bytes)}, empty{};
    check(value.transfer->Map(0, &range, &mapped));
    std::ofstream file(path, std::ios::binary);
    unsigned header[] = {value.width, value.height, value.bits};
    file.write(reinterpret_cast<const char*>(header), 12);
    for (unsigned row_index = 0; row_index < value.height; ++row_index)
        file.write(static_cast<const char*>(mapped) + value.footprint.Offset +
                       row_index * value.footprint.Footprint.RowPitch,
                   size_t(value.width) * 4 * value.bits / 8);
    value.transfer->Unmap(0, &empty);
    if (!file)
        throw std::runtime_error("Cannot write HDR output");
}
