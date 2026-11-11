#pragma once
#include "../../pipeline/nis.h"
#include "memory.h"

namespace phoenix_addon::phoenix::direct {
// Each recording owns its descriptors, constants and temporary images until
// the parent bridge's fence retires. SDK NIS supports at most 2x per axis;
// larger ratios use a sequence of real NIS dispatches, never a renamed filter.
class NisResize {
    struct Step {
        Microsoft::WRL::ComPtr<ID3D12Resource> image;
        std::unique_ptr<phoenix_pipeline::NisScaler> scaler;
    };
    std::vector<Step> steps_;
public:
    NisResize(ID3D12Device* device,unsigned sw,unsigned sh,unsigned dw,unsigned dh,DXGI_FORMAT format,
        const phoenix_pipeline::NisScaler& prototype,
        const std::shared_ptr<phoenix_native::memory::Generation>& accounting) {
        check(sw&&sh&&dw>=sw&&dh>=sh&&(dw!=sw||dh!=sh),"NIS resize requires enlargement");
        while(sw!=dw||sh!=dh) {
            sw=std::min(dw,sw*2);sh=std::min(dh,sh*2);
            Step step;
            step.scaler=std::make_unique<phoenix_pipeline::NisScaler>(prototype);
            step.image=phoenix_native::image(device,sw,sh,format,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            phoenix_native::memory::resource(device,step.image.Get(),accounting,phoenix_native::memory::Kind::bridge);
            steps_.push_back(std::move(step));
        }
    }
    unsigned passes()const{return unsigned(steps_.size());}
    ID3D12Resource* record(ID3D12GraphicsCommandList* command,ID3D12Resource* source,
        D3D12_RESOURCE_STATES state,float sharpness) {
        for(auto& step:steps_) {
            check(step.scaler->record(command,source,state,step.image.Get(),
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,sharpness),"NIS resize rejected its staged geometry");
            source=step.image.Get();state=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        }
        return source;
    }
};
}
