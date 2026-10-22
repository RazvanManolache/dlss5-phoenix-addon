#pragma once
#include "recipe.h"
#include <functional>
#include <span>

namespace phoenix_pipeline {
// Native C++ fallback for portable backends that already exchange host planes.
// This adds no transfers to the default executor. Each named neural callback
// must own its recurrence; a branch never consumes another branch's prediction.
struct HostImage {
    unsigned width=0,height=0,channels=0;
    std::vector<float> pixels;
    HostImage()=default;
    HostImage(unsigned w,unsigned h,unsigned c):width(w),height(h),channels(c){
        if(!w||!h||!c||w>8192||h>8192||c>4)throw std::invalid_argument("Invalid host image extent");
        pixels.resize(size_t(w)*h*c);
    }
    HostImage(unsigned w,unsigned h,unsigned c,const float* source):HostImage(w,h,c){
        if(!source)throw std::invalid_argument("Missing host image");
        std::copy_n(source,pixels.size(),pixels.begin());
    }
};
enum class HostSampling {Image,Nearest,Motion,Validity};
inline HostImage host_resample(const HostImage& source,unsigned w,unsigned h,HostSampling mode=HostSampling::Image){
    if(source.width==w&&source.height==h)return source;
    HostImage result(w,h,source.channels);
    const float rx=float(source.width)/w,ry=float(source.height)/h;
    auto at=[&](int x,int y,unsigned c){return source.pixels[(size_t(std::clamp(y,0,int(source.height)-1))*source.width+
        std::clamp(x,0,int(source.width)-1))*source.channels+c];};
    auto sample=[&](float x,float y,unsigned c){
        const int left=int(std::floor(x)),top=int(std::floor(y));const float fx=x-left,fy=y-top;
        return std::lerp(std::lerp(at(left,top,c),at(left+1,top,c),fx),
                         std::lerp(at(left,top+1,c),at(left+1,top+1,c),fx),fy);
    };
    for(unsigned y=0;y<h;++y)for(unsigned x=0;x<w;++x){
        const float px=(x+.5f)*rx-.5f,py=(y+.5f)*ry-.5f;
        for(unsigned c=0;c<source.channels;++c){
            float value;
            if(mode==HostSampling::Validity){
                value=1;
                // Neither reduction nor enlargement may hide an invalid
                // source pixel inside the represented footprint.
                const unsigned x0=std::min(source.width-1,unsigned(std::floor(x*rx))),
                    y0=std::min(source.height-1,unsigned(std::floor(y*ry))),
                    x1=std::min(source.width,unsigned(std::ceil((x+1)*rx))),
                    y1=std::min(source.height,unsigned(std::ceil((y+1)*ry)));
                for(unsigned sy=y0;sy<y1;++sy)for(unsigned sx=x0;sx<x1;++sx)value=std::min(value,at(int(sx),int(sy),c));
            }else if(mode==HostSampling::Nearest||mode==HostSampling::Motion){
                value=at(int(std::floor(px+.5f)),int(std::floor(py+.5f)),c);
                if(mode==HostSampling::Motion&&c<2)value/=c==0?rx:ry;
            }else{
                const float dx=std::max(rx-1,0.f)*.25f,dy=std::max(ry-1,0.f)*.25f;
                value=.25f*(sample(px+dx,py+dy,c)+sample(px-dx,py-dy,c)+sample(px+dx,py-dy,c)+sample(px-dx,py+dy,c));
            }
            result.pixels[(size_t(y)*w+x)*source.channels+c]=value;
        }
    }return result;
}
inline void host_color(HostImage& image,const Options& p){
    if(image.channels!=3)throw std::invalid_argument("Host color stage requires RGB");
    if(p.color_brightness==0&&p.color_contrast==1&&p.color_saturation==1&&p.color_gamma==1)return;
    for(size_t i=0;i<image.pixels.size();i+=3){
        const float l=image.pixels[i]*.2126f+image.pixels[i+1]*.7152f+image.pixels[i+2]*.0722f;
        for(unsigned c=0;c<3;++c){
            const float v=std::lerp(l,image.pixels[i+c],p.color_saturation);
            image.pixels[i+c]=std::pow(std::clamp((v-.5f)*p.color_contrast+.5f+p.color_brightness,0.f,1.f),1.f/p.color_gamma);
        }
    }
}
using HostNeural=std::function<HostImage(const std::string&,const HostImage&)>;
// Geometry-changing scale stages are supported by the shared host planner;
// the legacy portable frame ABI admits equal input/output extents only.
inline HostImage host_chain(const Options& p,const HostImage& source,unsigned ow,unsigned oh,
                            const HostNeural& neural,nlohmann::json* executed=nullptr){
    if(source.channels!=3)throw std::invalid_argument("Host chain requires RGB");
    const auto steps=plan(p,source.width,source.height,ow,oh);
    if(p.chain_enabled&&p.enabled[unsigned(Stage::Scale)]&&p.scaler!=0)
        throw std::invalid_argument("This portable host chain requires the existing spatial scaler");
    HostImage current=source,neural_source;
    const bool reuse=fusion_reuses_neural(p,source.width,source.height,ow,oh);
    auto evaluate=[&](const std::string& key,const HostImage& input){
        if(!neural)throw std::invalid_argument("Missing host neural stage");
        auto result=neural(key,input);
        if(result.width!=input.width||result.height!=input.height||result.channels!=3||
           result.pixels.size()!=input.pixels.size()||
           !std::all_of(result.pixels.begin(),result.pixels.end(),[](float v){return std::isfinite(v);}))
            throw std::runtime_error("Host neural stage returned invalid pixels");
        return result;
    };
    for(const auto& step:steps){
        if(step.stage==Stage::Neural){neural_source=current;current=evaluate("neural",current);}
        else if(step.stage==Stage::Color)host_color(current,p);
        else if(step.stage==Stage::Scale)current=host_resample(current,step.width,step.height);
        else{
            const auto& input=reuse?neural_source:current;
            auto mixed=reuse?current:evaluate("fusion0",input);
            for(int layer=1;layer<p.fusion_layers;++layer){
                const float scale=layer==1?p.fusion_second_scale:p.fusion_third_scale;
                const unsigned w=std::max(1u,unsigned(std::lround(input.width*scale))),
                    h=std::max(1u,unsigned(std::lround(input.height*scale)));
                auto prediction=evaluate("fusion"+std::to_string(layer),host_resample(input,w,h));
                prediction=host_resample(prediction,input.width,input.height);
                for(size_t i=0;i<mixed.pixels.size();++i)mixed.pixels[i]=std::lerp(mixed.pixels[i],prediction.pixels[i],1.f/float(layer+1));
            }
            current=std::move(mixed);
        }
        if(executed)executed->push_back({{"stage",int(step.stage)},{"width",current.width},{"height",current.height}});
    }
    return current;
}
} // namespace phoenix_pipeline
