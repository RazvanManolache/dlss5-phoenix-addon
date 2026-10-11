// Stages game D3D12 resources at the portable-runner boundary.
// Transport waits and copies belong to end-to-end timing, not to neural operator timing alone.

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <memory>
using Microsoft::WRL::ComPtr;

// Transport foundation for a later GPU-resident frame ABI. This is deliberately
// not advertised as a backend capability until each vendor can import it.
// The v1 host-plane runtime does not call these exports.
struct SharedBuffer {
    ComPtr<ID3D12Resource> buffer;
    ComPtr<ID3D12Fence> fence;
    HANDLE memory = nullptr, sync = nullptr;
    ~SharedBuffer() {
        if (memory)
            CloseHandle(memory);
        if (sync)
            CloseHandle(sync);
    }
};
extern "C" __declspec(dllexport) HRESULT phoenix_shared_buffer_create(ID3D12Device* device,
                                                                      uint64_t bytes,
                                                                      BOOL cross_adapter,
                                                                      void** result) {
    if (!result)
        return E_POINTER;
    *result = nullptr;
    if (!device || !bytes || bytes > uint64_t(16) * 1024 * 1024 * 1024)
        return E_INVALIDARG;
    auto object = std::make_unique<SharedBuffer>();
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = heap.VisibleNodeMask = 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    auto heap_flags = D3D12_HEAP_FLAG_SHARED;
    auto fence_flags = D3D12_FENCE_FLAG_SHARED;
    if (cross_adapter) {
        heap_flags |= D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER;
        desc.Flags |= D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
        fence_flags |= D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER;
    }
    HRESULT status =
        device->CreateCommittedResource(&heap, heap_flags, &desc, D3D12_RESOURCE_STATE_COMMON,
                                        nullptr, IID_PPV_ARGS(&object->buffer));
    if (FAILED(status))
        return status;
    if (FAILED(status = device->CreateFence(0, fence_flags, IID_PPV_ARGS(&object->fence))))
        return status;
    if (FAILED(status = device->CreateSharedHandle(object->buffer.Get(), nullptr, GENERIC_ALL,
                                                   nullptr, &object->memory)))
        return status;
    if (FAILED(status = device->CreateSharedHandle(object->fence.Get(), nullptr, GENERIC_ALL,
                                                   nullptr, &object->sync)))
        return status;
    *result = object.release();
    return S_OK;
}
extern "C" __declspec(dllexport) HRESULT phoenix_shared_buffer_duplicate(void* opaque,
                                                                         HANDLE target,
                                                                         HANDLE* memory,
                                                                         HANDLE* fence) {
    if (!opaque || !target || !memory || !fence)
        return E_INVALIDARG;
    *memory = *fence = nullptr;
    auto& object = *static_cast<SharedBuffer*>(opaque);
    if (!DuplicateHandle(GetCurrentProcess(), object.memory, target, memory, 0, FALSE,
                         DUPLICATE_SAME_ACCESS))
        return HRESULT_FROM_WIN32(GetLastError());
    if (!DuplicateHandle(GetCurrentProcess(), object.sync, target, fence, 0, FALSE,
                         DUPLICATE_SAME_ACCESS)) {
        DWORD error = GetLastError();
        HANDLE local = nullptr;
        if (DuplicateHandle(target, *memory, GetCurrentProcess(), &local, 0, FALSE,
                            DUPLICATE_CLOSE_SOURCE))
            CloseHandle(local);
        *memory = nullptr;
        return HRESULT_FROM_WIN32(error);
    }
    return S_OK;
}
extern "C" __declspec(dllexport) HRESULT phoenix_shared_buffer_signal(void* opaque,
                                                                      ID3D12CommandQueue* queue,
                                                                      uint64_t value) {
    if (!opaque || !queue || !value)
        return E_INVALIDARG;
    return queue->Signal(static_cast<SharedBuffer*>(opaque)->fence.Get(), value);
}
extern "C" __declspec(dllexport) HRESULT phoenix_shared_buffer_wait(void* opaque,
                                                                    ID3D12CommandQueue* queue,
                                                                    uint64_t value) {
    if (!opaque || !queue || !value)
        return E_INVALIDARG;
    return queue->Wait(static_cast<SharedBuffer*>(opaque)->fence.Get(), value);
}
extern "C" __declspec(dllexport) ID3D12Resource* phoenix_shared_buffer_resource(void* opaque) {
    if (!opaque)
        return nullptr;
    auto resource = static_cast<SharedBuffer*>(opaque)->buffer.Get();
    resource->AddRef();
    return resource;
}
extern "C" __declspec(dllexport) void phoenix_shared_buffer_destroy(void* opaque) {
    delete static_cast<SharedBuffer*>(opaque);
}
