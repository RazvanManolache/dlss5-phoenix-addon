#pragma once

// Runs only inside the disposable x86 probe's native worker. Exercise the same
// TLS/UMD adaptations as production, without loading a model or launching kernels.
static void check_installed_driver(HMODULE cuda,bridge::Channel& channel) {
    auto& proof=*reinterpret_cast<bridge::CompatibilityEvidence*>(channel.output);
    require(proof.abi==1&&proof.bytes==sizeof(proof),"Invalid driver compatibility evidence buffer");
    int version=0,count=0;
    require(function<int(*)(int*)>(cuda,"cuDriverGetVersion")(&version)==0&&version>0,"CUDA driver version query failed");
    require(function<int(*)(int*)>(cuda,"cuDeviceGetCount")(&count)==0&&count>0,"No CUDA-capable device available");
    proof.cuda_initialized=1;proof.cuda_version=uint32_t(version);proof.device_count=uint32_t(count);
    Driver driver;hr(CreateDXGIFactory1(IID_PPV_ARGS(&driver.factory)));
    for(unsigned index=0;;++index) {
        hr(driver.factory->EnumAdapters1(index,&driver.adapter));DXGI_ADAPTER_DESC1 desc{};hr(driver.adapter->GetDesc1(&desc));
        if(desc.VendorId==0x10de&&!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)) {
            const auto get=function<int(*)(int*,int)>(cuda,"cuDeviceGet");
            const auto luid=function<int(*)(char*,unsigned*,int)>(cuda,"cuDeviceGetLuid");
            for(int ordinal=0;ordinal<count;++ordinal) {
                int device=0;char identity[8]{};unsigned mask=0;
                if(get(&device,ordinal)==0&&luid(identity,&mask,device)==0&&mask&&memcmp(identity,&desc.AdapterLuid,8)==0)proof.adapter_matched=1;
            }
            if(proof.adapter_matched)break;
        }
        driver.adapter.Reset();
    }
    hr(D3D12CreateDevice(driver.adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&driver.device)));
    D3D12_COMMAND_QUEUE_DESC desc{};hr(driver.device->CreateCommandQueue(&desc,IID_PPV_ARGS(&driver.queue)));
    hr(driver.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&driver.allocator)));
    hr(driver.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,driver.allocator.Get(),nullptr,IID_PPV_ARGS(&driver.list)));
    hr(driver.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&driver.fence)));
    auto nvapi=LoadLibraryExW(L"nvapi64.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);require(nvapi!=nullptr,"Native NVAPI unavailable");
    wow64_umd_probe::patch(nvapi);
    const auto query=function<void*(__cdecl*)(unsigned)>(nvapi,"nvapi_QueryInterface");
    const auto initialize=reinterpret_cast<NvAPI_Status(__cdecl*)()>(query(0x0150e828));require(initialize!=nullptr,"Native NVAPI init unavailable");
    wow64_umd_probe::patch(GetModuleHandleW(L"nvapi64_impl.dll"));nv(initialize());
    for(unsigned id:{0xad1a677du,0xe2436e22u,0x41c65285u,0xdf295ea6u,0x24973538u,0x329fe6e0u,0x48f5b2eeu}) {
        require(query(id)!=nullptr,"Required D3D12 CUDA interface unavailable");++proof.nvapi_entries;
    }
    auto input=driver.buffer(16,D3D12_HEAP_TYPE_UPLOAD,D3D12_RESOURCE_STATE_GENERIC_READ);
    auto output=driver.buffer(16,D3D12_HEAP_TYPE_READBACK,D3D12_RESOURCE_STATE_COPY_DEST);
    const uint32_t expected[]{0x50485835,0x11223344,0x55667788,0xaabbccdd};void* mapped=nullptr;D3D12_RANGE empty{};
    hr(input->Map(0,&empty,&mapped));memcpy(mapped,expected,sizeof(expected));input->Unmap(0,nullptr);
    driver.list->CopyBufferRegion(output.Get(),0,input.Get(),0,sizeof(expected));driver.submit();
    D3D12_RANGE range{0,sizeof(expected)};hr(output->Map(0,&range,&mapped));
    const bool equal=memcmp(mapped,expected,sizeof(expected))==0;output->Unmap(0,&empty);
    require(equal,"Native D3D12 queue/fence/readback verification failed");proof.d3d12_copy_verified=1;
}
