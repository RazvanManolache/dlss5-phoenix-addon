// Compare real GPU enlargement filters with an independent CPU reference.
// Exercise SDK NIS, original-alpha preservation and identity residuals on WARP.
#include "d3d12_test_context.h"
#include "../../src/integration/compatibility/processing_pixels.h"
#include <iostream>
using namespace phoenix_addon;
int main()try {
    phoenix_test::Gpu gpu;constexpr unsigned width=32,height=32;
    std::vector<float> original;
    for(unsigned y=0;y<height;++y)for(unsigned x=0;x<width;++x)for(unsigned c=0;c<3;++c)
        original.push_back(DirectX::PackedVector::XMConvertHalfToFloat(
            DirectX::PackedVector::XMConvertFloatToHalf(float((x*37+y*11+c*51)%256)/255)));
    std::vector<std::vector<float>> explicit_enlargements;
    unsigned cases=0,nis_cases=0;float maximum_reference_error=0;
    for(float scale:{50.f,400.f})for(int method:{1,2,3,4,5}) {
        phoenix::Processing p;p.enabled=true;p.scale=scale;p.resize_method=method;p.encoding=2;p.domain=2;
        const auto geometry=phoenix::processing_resolution(width,height,p,true);
        auto source=gpu.texture(width,height);
        auto working=phoenix_native::image(gpu.device.Get(),geometry.width,geometry.height,
            DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        auto neural=phoenix_native::image(gpu.device.Get(),geometry.width,geometry.height,
            DXGI_FORMAT_R16G16B16A16_FLOAT,D3D12_RESOURCE_STATE_COPY_DEST);
        phoenix::direct::ResolutionBridge bridge(gpu.device.Get(),geometry,DXGI_FORMAT_R16G16B16A16_FLOAT,{},p,true);
        bridge.begin_frame(gpu.serial,gpu.serial+1);
        bridge.reduce(gpu.cmd.Get(),source.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,working.Get(),false);
        const auto pre=gpu.readback(working.Get());
        phoenix_native::transition(gpu.cmd.Get(),working.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COPY_SOURCE);
        gpu.cmd->CopyResource(neural.Get(),working.Get());
        phoenix_native::transition(gpu.cmd.Get(),working.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        phoenix_native::transition(gpu.cmd.Get(),neural.Get(),D3D12_RESOURCE_STATE_COPY_DEST,D3D12_RESOURCE_STATE_COPY_SOURCE);
        bridge.present(gpu.cmd.Get(),neural.Get(),source.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,working.Get());
        const auto post=gpu.readback(source.Get());gpu.submit();
        auto pixels=[&](const auto& capture,unsigned w,unsigned h) {
            std::vector<float> result;void* mapped=nullptr;
            phoenix_native::hr(capture.first->Map(0,nullptr,&mapped));
            for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x) {
                const auto* pixel=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(mapped)+
                    capture.second.Offset+y*capture.second.Footprint.RowPitch+x*8);
                for(unsigned c=0;c<3;++c){const float value=DirectX::PackedVector::XMConvertHalfToFloat(pixel[c]);
                    check(std::isfinite(value),"Resize generated a nonfinite pixel");result.push_back(value);}
                check(pixel[3]==0x3c00,"Resize lost original opaque alpha");
            }
            capture.first->Unmap(0,nullptr);return result;
        };
        const auto encoded=pixels(pre,geometry.width,geometry.height);
        const auto restored=pixels(post,width,height);
        for(size_t i=0;i<original.size();++i)check(std::abs(restored[i]-original[i])<.002f,
            "An identity neural result changed the full-resolution residual reference");
        if(method>=3) {
            phoenix_compat::pixels::Frame reference(original,width,height,p,true);
            check(reference.encoded.size()==encoded.size(),"CPU/GPU resize extent mismatch");
            for(size_t i=0;i<encoded.size();++i)maximum_reference_error=std::max(maximum_reference_error,
                std::abs(encoded[i]-reference.encoded[i]));
            check(maximum_reference_error<.006f,"GPU enlargement did not match the selected CPU filter");
            if(scale==400)explicit_enlargements.push_back(encoded);
        } else {
            const auto status=bridge.status();
            check(status.at("nis_dispatches").get<unsigned>()>0,"Auto/NIS used no NVIDIA shader dispatches");
            check(status.at("last_nis_passes")==unsigned(scale==400?2:1),"NIS pass count does not match the resize ratio");
            ++nis_cases;
        }
        ++cases;gpu.next();
    }
    check(explicit_enlargements.size()==3&&explicit_enlargements[0]!=explicit_enlargements[1]&&
        explicit_enlargements[1]!=explicit_enlargements[2]&&explicit_enlargements[0]!=explicit_enlargements[2],
        "Bilinear, Bicubic and Nearest produced indistinguishable enlargement fixtures");
    // Three outstanding recordings share image geometry but require distinct
    // NIS constants. A 0/1/0 sharpness sequence catches upload-buffer reuse.
    phoenix::Processing asynchronous;asynchronous.enabled=true;asynchronous.encoding=2;
    asynchronous.resize_method=2;asynchronous.scale=400;
    const auto g=phoenix::processing_resolution(width,height,asynchronous,true);
    auto source=gpu.texture(width,height);
    auto working=phoenix_native::image(gpu.device.Get(),g.width,g.height,DXGI_FORMAT_R16G16B16A16_FLOAT,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    phoenix::direct::ResolutionBridge pending(gpu.device.Get(),g,DXGI_FORMAT_R16G16B16A16_FLOAT,{},asynchronous,true);
    const auto retired=gpu.serial;
    std::vector<std::pair<Microsoft::WRL::ComPtr<ID3D12Resource>,D3D12_PLACED_SUBRESOURCE_FOOTPRINT>> captures;
    for(float sharpness:{0.f,1.f,0.f}) {
        asynchronous.nis_sharpness=sharpness;pending.update_processing(asynchronous);
        pending.begin_frame(retired,gpu.serial+1);
        pending.reduce(gpu.cmd.Get(),source.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,working.Get(),false);
        captures.push_back(gpu.readback(working.Get()));gpu.submit(false);gpu.next();
    }
    gpu.finish();
    std::vector<std::vector<uint16_t>> values;
    for(const auto& capture:captures) {
        void* mapped=nullptr;phoenix_native::hr(capture.first->Map(0,nullptr,&mapped));
        std::vector<uint16_t> pixels;
        for(unsigned y=0;y<g.height;++y) {
            const auto* row=reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(mapped)+
                capture.second.Offset+y*capture.second.Footprint.RowPitch);
            pixels.insert(pixels.end(),row,row+g.width*4);
        }
        capture.first->Unmap(0,nullptr);values.push_back(std::move(pixels));
    }
    check(values[0]==values[2]&&values[0]!=values[1],"Outstanding NIS recordings overwrote each other's sharpness constants");
    gpu.validate();
    std::cout<<Json{{"passed",true},{"software_d3d12","WARP"},{"cases",cases},{"real_nis_cases",nis_cases},
        {"maximum_cpu_gpu_filter_error",maximum_reference_error},{"identity_residual_and_alpha_preserved",true},
        {"explicit_filters_distinct",true},{"outstanding_nis_recordings",3},{"debug_errors",0}}.dump()<<'\n';
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
