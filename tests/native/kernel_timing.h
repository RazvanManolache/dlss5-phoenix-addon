// Optional offline attribution. Timestamped single launches disturb overlap;
// use this only to locate expensive kernels, never for end-to-end speed claims.
#pragma once

class KernelTiming {
    static constexpr unsigned launch_count = 154;
    ComPtr<ID3D12QueryHeap> queries_;
    ComPtr<ID3D12Resource> readback_;
    uint64_t frequency_ = 0;
    unsigned recorded_ = 0;

  public:
    KernelTiming(ID3D12Device* device, ID3D12CommandQueue* queue) {
        phoenix_native::hr(queue->GetTimestampFrequency(&frequency_));
        D3D12_QUERY_HEAP_DESC description{};
        description.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        description.Count = launch_count * 2;
        phoenix_native::hr(device->CreateQueryHeap(&description, IID_PPV_ARGS(&queries_)));
        readback_ =
            phoenix_native::buffer(device, description.Count * sizeof(uint64_t),
                                   D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    }
    void begin_frame() {
        recorded_ = 0;
    }
    NvAPI_Status dispatch(ID3D12GraphicsCommandList* commands,
                          const NVAPI_CU_KERNEL_LAUNCH_PARAMS* launches, NvU32 count,
                          decltype(phoenix_native::Api::dispatch) forward) {
        for (unsigned index = 0; index < count; ++index) {
            check(recorded_ < launch_count, "Kernel timing capacity");
            const unsigned query = recorded_++ * 2;
            commands->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query);
            const auto status = forward(commands, launches + index, 1);
            commands->EndQuery(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, query + 1);
            if (status != NVAPI_OK)
                return status;
        }
        return NVAPI_OK;
    }
    void resolve(ID3D12GraphicsCommandList* commands) {
        check(recorded_ == launch_count, "Kernel timing launch count");
        commands->ResolveQueryData(queries_.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, launch_count * 2,
                                   readback_.Get(), 0);
    }
    // The caller has waited for the submission's completion fence.
    Json read() const {
        uint64_t* timestamps = nullptr;
        D3D12_RANGE range{0, launch_count * 2 * sizeof(uint64_t)}, empty{};
        phoenix_native::hr(readback_->Map(0, &range, reinterpret_cast<void**>(&timestamps)));
        Json timings = Json::array();
        for (unsigned index = 0; index < launch_count; ++index)
            timings.push_back(double(timestamps[index * 2 + 1] - timestamps[index * 2]) * 1000.0 /
                              double(frequency_));
        readback_->Unmap(0, &empty);
        return timings;
    }
};

static KernelTiming* kernel_timer = nullptr;
static decltype(phoenix_native::Api::dispatch) untimed_dispatch = nullptr;
static NvAPI_Status __cdecl timed_kernel_dispatch(ID3D12GraphicsCommandList* commands,
                                                  const NVAPI_CU_KERNEL_LAUNCH_PARAMS* launches,
                                                  NvU32 count) {
    return kernel_timer->dispatch(commands, launches, count, untimed_dispatch);
}
