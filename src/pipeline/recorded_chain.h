#pragma once
#include "recipe.h"
#include "image_ops.h"
#include "nis.h"
#include "tiny.h"
#include "../native/support/compute_pass.h"
#include "../native/support/dispatch.h"
#include <map>
#include <functional>

namespace phoenix_pipeline {
// Records into the host's command list. It never submits or waits on a game
// queue. The owner commits histories only after the observed submission, and
// retires this entire object before changing geometry or architecture.
class RecordedChain {
    using Resource=Microsoft::WRL::ComPtr<ID3D12Resource>;
    using Pass=phoenix_addon::phoenix::direct::Pass;
    using Json=nlohmann::json;
public:
    using Plan=std::function<Json(unsigned,unsigned)>;
    struct NeuralStage {
        virtual ~NeuralStage()=default;
        virtual ID3D12Resource* record(ID3D12GraphicsCommandList*,ID3D12Resource*,ID3D12Resource*,
            phoenix_addon::Controls,unsigned,ID3D12CommandQueue*)=0;
        virtual void commit(ID3D12GraphicsCommandList*)=0;
        virtual void invalidate()=0;
        virtual uint64_t launches()const=0;
    };
    using Factory=std::function<std::unique_ptr<NeuralStage>(unsigned,unsigned)>;
private:
    Microsoft::WRL::ComPtr<ID3D12Device> device_;
    std::shared_ptr<phoenix_native::memory::Generation> accounting_;
    Options options_;
    unsigned width_,height_,output_width_,output_height_,passes_;
    uint64_t serial_=0;
    std::map<std::string,Resource> images_;
    std::map<std::string,std::unique_ptr<Pass>> operations_;
    std::map<std::string,std::unique_ptr<NeuralStage>> networks_;
    std::vector<ChainStep> schedule_;
    std::array<std::unique_ptr<NisScaler>,8> nis_;
    std::unique_ptr<TinyUpscaler> tiny_;
    Json executed_=Json::array();
    ID3D12Resource* image(const std::string& key,unsigned w,unsigned h,DXGI_FORMAT format=DXGI_FORMAT_R16G16B16A16_FLOAT){
        auto& result=images_[key];
        if(!result){result=phoenix_native::image(device_.Get(),w,h,format,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,true);
            phoenix_native::memory::resource(device_.Get(),result.Get(),accounting_,phoenix_native::memory::Kind::bridge);}
        return result.Get();
    }
    ID3D12Resource* run(ID3D12GraphicsCommandList* cmd,const std::string& key,ID3D12Resource* source,
        unsigned w,unsigned h,unsigned mode=0,ID3D12Resource* other=nullptr,float a=0,float b=1,float c=1,float d=1){
        auto* target=image(key,w,h,mode==1?DXGI_FORMAT_R16G16_FLOAT:DXGI_FORMAT_R16G16B16A16_FLOAT);
        auto& pass=operations_[key];
        if(!pass)pass=std::make_unique<Pass>(device_.Get(),image_ops_shader,accounting_,8);
        phoenix_native::transition(cmd,target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        pass->record(cmd,{source,other,nullptr},target,ImageConstants{w,h,mode,0,a,b,c,d,0},false,unsigned(serial_%8));
        phoenix_native::transition(cmd,target,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        return target;
    }
    void prepare(const Factory& factory,const std::string& key,unsigned w,unsigned h){
        for(unsigned pass=0;pass<passes_;++pass)networks_.emplace(key+"/"+std::to_string(pass),factory(w,h));
    }
    ID3D12Resource* neural(ID3D12GraphicsCommandList* cmd,const std::string& key,ID3D12Resource* source,
        ID3D12Resource* motion,phoenix_addon::Controls controls,unsigned blocks,ID3D12CommandQueue* queue,
        unsigned w,unsigned h,float scale){
        const unsigned nw=std::max(1u,unsigned(std::lround(w*scale))),nh=std::max(1u,unsigned(std::lround(h*scale)));
        auto* current=run(cmd,key+"/input",source,nw,nh);
        auto* mv=run(cmd,key+"/motion",motion,nw,nh,1);
        // Input motion is already in the network grid's pixel units. Changing
        // the grid scales displacement, rather than inventing geometry/depth.
        for(unsigned pass=0;pass<passes_;++pass)
            current=networks_.at(key+"/"+std::to_string(pass))->record(cmd,current,mv,controls,blocks,queue);
        return (nw==w&&nh==h)?current:run(cmd,key+"/restore",current,w,h);
    }
public:
    RecordedChain(ID3D12Device* device,unsigned width,unsigned height,unsigned output_width,unsigned output_height,
        const Options& options,unsigned passes,const Factory& factory,const std::filesystem::path& tiny_path={},
        std::shared_ptr<phoenix_native::memory::Generation> accounting={})
        :device_(device),accounting_(std::move(accounting)),options_(options),width_(width),height_(height),
         output_width_(output_width),output_height_(output_height),passes_(passes),schedule_(plan(options,width,height,output_width,output_height)){
        if(passes<1||passes>10)throw std::invalid_argument("Chain network passes outside 1..10");
        unsigned w=width,h=height;
        for(const auto& step:schedule_){
            if(step.stage==Stage::Neural)prepare(factory,"neural",w,h);
            if(step.stage==Stage::Fusion){
                if(!fusion_reuses_neural(options,width,height,output_width,output_height))prepare(factory,"fusion0",w,h);
                for(int layer=1;layer<options.fusion_layers;++layer){
                const float scale=layer==1?options.fusion_second_scale:options.fusion_third_scale;
                prepare(factory,"fusion"+std::to_string(layer),std::max(1u,unsigned(std::lround(w*scale))),std::max(1u,unsigned(std::lround(h*scale))));
                }
            }
            if(step.stage==Stage::Scale){
                if(options.scaler==3){
                    if(step.width!=w*2||step.height!=h*2)throw std::invalid_argument("Tiny reconstruction requires exactly 2x output geometry");
                    tiny_=std::make_unique<TinyUpscaler>(device,w,h,tiny_path);
                }else if(options.scaler==1){
                    if(step.width<w||step.height<h||step.width>w*2||step.height>h*2)throw std::invalid_argument("NIS supports 1x to 2x per axis");
                    nis_[0]=std::make_unique<NisScaler>(device);
                    for(unsigned slot=1;slot<8;++slot)nis_[slot]=std::make_unique<NisScaler>(*nis_[0]);
                }else if(options.scaler==2)throw std::invalid_argument("VSR runtime is unavailable");
                w=step.width;h=step.height;
            }
        }
    }
    ID3D12Resource* resample(ID3D12GraphicsCommandList* cmd,const std::string& key,ID3D12Resource* source,unsigned w,unsigned h){return run(cmd,key,source,w,h);}
    ID3D12Resource* record(ID3D12GraphicsCommandList* cmd,ID3D12Resource* source,ID3D12Resource* motion,
        phoenix_addon::Controls controls,unsigned blocks,ID3D12CommandQueue* queue,uint64_t serial){
        serial_=serial;auto* current=source;unsigned w=width_,h=height_;executed_=Json::array();
        ID3D12Resource* neural_source=nullptr;
        for(const auto& step:schedule_){
            executed_.push_back({{"stage",unsigned(step.stage)},{"width",step.width},{"height",step.height}});
            if(step.stage==Stage::Neural){neural_source=current;current=neural(cmd,"neural",current,motion,controls,blocks,queue,w,h,1);}
            else if(step.stage==Stage::Fusion){
                const bool reuse=fusion_reuses_neural(options_,width_,height_,output_width_,output_height_);
                auto* base=reuse?neural_source:current;
                if(!reuse)current=neural(cmd,"fusion0",base,motion,controls,blocks,queue,w,h,1);
                for(int layer=1;layer<options_.fusion_layers;++layer){
                    auto* branch=neural(cmd,"fusion"+std::to_string(layer),base,motion,controls,blocks,queue,w,h,
                        layer==1?options_.fusion_second_scale:options_.fusion_third_scale);
                    current=run(cmd,"blend"+std::to_string(layer),current,w,h,3,branch,1.f/(layer+1));
                }
            }else if(step.stage==Stage::Color){
                if(options_.color_brightness!=0||options_.color_contrast!=1||options_.color_saturation!=1||options_.color_gamma!=1)
                    current=run(cmd,"color",current,w,h,2,nullptr,options_.color_brightness,options_.color_contrast,options_.color_saturation,options_.color_gamma);
            }else{
                if(tiny_||nis_[0]){
                    current=run(cmd,"scale/input",current,w,h);
                    auto* target=image("scale",step.width,step.height);
                    if(tiny_)tiny_->record(cmd,current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    else if(!nis_[serial_%8]->record(cmd,current,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,target,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,options_.sharpness))
                        throw std::runtime_error("NIS chain execution rejected geometry");
                    current=target;
                }else current=run(cmd,"scale",current,step.width,step.height);
                w=step.width;h=step.height;
            }
        }
        return current;
    }
    void commit(ID3D12GraphicsCommandList* cmd){for(auto& [name,stage]:networks_)stage->commit(cmd);}
    void invalidate(){for(auto& [name,stage]:networks_)stage->invalidate();}
    uint64_t launches()const{uint64_t count=0;for(const auto& [name,stage]:networks_)count+=stage->launches();return count;}
    Json status()const{return {{"executed_chain",executed_},{"independent_temporal_stages",networks_.size()},{"submission","host command list"}};}
};

class RecordedNativeStage final:public RecordedChain::NeuralStage {
    unsigned width_,height_;
    Microsoft::WRL::ComPtr<ID3D12Resource> input_,motion_;
    phoenix_addon::FrameDispatch dispatch_;
    std::unique_ptr<phoenix_native::Network> network_;
    static void copy(ID3D12GraphicsCommandList* cmd,ID3D12Resource* source,ID3D12Resource* target){
        const auto read=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        phoenix_native::transition(cmd,source,read,D3D12_RESOURCE_STATE_COPY_SOURCE);
        phoenix_native::transition(cmd,target,read,D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyResource(target,source);
        phoenix_native::transition(cmd,target,D3D12_RESOURCE_STATE_COPY_DEST,read);
        phoenix_native::transition(cmd,source,D3D12_RESOURCE_STATE_COPY_SOURCE,read);
    }
public:
    RecordedNativeStage(ID3D12Device* device,unsigned w,unsigned h,const nlohmann::json& plan,
        const phoenix_model::WeightView& weights,const std::vector<phoenix_kernel::Binary>& kernels,
        const std::array<int,8>& repeats,std::shared_ptr<phoenix_native::memory::Generation> accounting={})
        :width_(w),height_(h){
        const auto read=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        input_=phoenix_native::image(device,w,h,DXGI_FORMAT_R16G16B16A16_FLOAT,read);
        motion_=phoenix_native::image(device,w,h,DXGI_FORMAT_R16G16_FLOAT,read);
        for(auto* texture:{input_.Get(),motion_.Get()})phoenix_native::memory::resource(device,texture,accounting,phoenix_native::memory::Kind::bridge);
        dispatch_.configure_repeats(device,plan,repeats);
        network_=std::make_unique<phoenix_native::Network>(device,dispatch_.api(),plan,weights,kernels,input_.Get(),motion_.Get(),accounting);
    }
    ID3D12Resource* record(ID3D12GraphicsCommandList* cmd,ID3D12Resource* source,ID3D12Resource* motion,
        phoenix_addon::Controls controls,unsigned blocks,ID3D12CommandQueue* queue)override{
        copy(cmd,source,input_.Get());copy(cmd,motion,motion_.Get());
        controls.color={0,0,int(width_),int(height_)};controls.motion=controls.color;controls.output=controls.color;
        const auto style=controls.style;controls.style=0;controls.intensity=1;controls.ui_correction=0;
        auto resolved=phoenix_addon::resolve_controls(controls,width_,height_,true,network_->next_sampling_index()>0,network_->next_sampling_index());
        resolved.style=float(style)/128.f;
        dispatch_.prepare_repeats(cmd);
        phoenix_addon::FrameDispatch::Scope scope(dispatch_,cmd,resolved,blocks);
        auto* result=network_->record(cmd,!resolved.recurrent,0,queue,true);scope.finish();return result;
    }
    void commit(ID3D12GraphicsCommandList* cmd)override{network_->commit_any_thread(cmd);}
    void invalidate()override{network_->invalidate();}
    uint64_t launches()const override{return dispatch_.launches();}
};
} // namespace phoenix_pipeline
