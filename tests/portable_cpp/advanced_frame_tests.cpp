#include "../../native_runtime/frame.h"
#include "../../native_runtime/advanced_frame.h"
#include <iostream>
using namespace phoenix_cpp;
int main() {
    unsigned checks=0;
    auto check=[&](bool ok,const char* message) { ++checks; if(!ok) throw std::runtime_error(message); };
    auto close_enough=[](float a,float b) { return std::fabs(a-b)<1e-6f; };
    try {
        RGB red{1,0,0}, gray{.5f,.5f,.5f};
        check(style_grade(red,0,1)==red,"style zero");
        check(style_grade(gray,2,1)==gray,"cinematic gray neutrality");
        const auto cinema=style_grade(red,2,1);
        check(close_enough(cinema[0],.925f)&&close_enough(cinema[1],.075f)&&close_enough(cinema[2],.075f),"cinematic saturation");
        check(style_grade(red,1,0)==red,"tone zero disables grade");
        float map[]={.5f,1,1,1};
        const RGB original{.25f,.25f,.25f}, neural{.75f,.75f,.75f};
        check(close_enough(compose_display(original,neural,0,.5f,0,1,DisplayInputs{map})[0],.375f),"mask times display intensity");
        check(compose_display(original,neural,0,2,0,1,{})==neural,"above-one pass activation");
        map[0]=1;
        check(compose_display(original,neural,0,2,0,1,DisplayInputs{map})[0]==1,"postprocess extrapolation clamp");
        float ui[]={.1f,.2f,.3f,1};
        DisplayInputs hud; hud.ui=ui;
        check(compose_display(original,neural,0,1,0,1,hud)==RGB{.1f,.2f,.3f},"opaque UI protection");
        ui[0]=ui[1]=ui[2]=.1f; ui[3]=.5f;
        check(close_enough(compose_display(original,neural,0,1,0,1,hud)[0],.475f),"premultiplied fractional UI");
        std::vector<float> image={0,0,0, .25f,.25f,.25f, .5f,.5f,.5f, 1,1,1};
        std::vector<float> field(16,0), vectors(8,0);
        check(warp_plane(image.data(),field.data(),4,1,3,2)==image,"identity distortion");
        for(unsigned p=0;p<4;++p) { field[p*4]=-.25f; field[p*4+2]=.25f; }
        CanonicalFrame canonical;
        canonical.prepare(image.data(),vectors.data(),nullptr,nullptr,nullptr,field.data(),field.data(),4,1);
        check(canonical.color[0]==.25f,"canonical input direction");
        check(canonical.motion[0]==0&&canonical.motion[2]==0,"static distortion motion alignment");
        check(canonical.validity[3]==0,"distortion outside-image history rejection");
        const auto restored=warp_plane(canonical.color.data(),field.data(),4,1,3,0);
        check(restored[3]==image[3]&&restored[6]==image[6],"inverse field interior");
        const std::vector<float> previous_identity(16,0);
        canonical.prepare(image.data(),vectors.data(),nullptr,nullptr,nullptr,field.data(),previous_identity.data(),4,1);
        check(canonical.motion[2]==1,"dynamic distortion uses the previous accepted field");
        FrameState state(2,1,4,2,1);
        Controls controls; controls.tone=2; controls.structure=.5f;
        std::vector<float> color(6,.25f),prior(6,.5f);
        std::vector<uint8_t> valid(2,1);
        float controls_map[]={0,.5f,.25f,1,1,1,0,1};
        const auto expected=state.features(color.data(),controls,prior,valid,controls_map);
        std::vector<uint32_t> joined(9,0);
        std::memcpy(joined.data(),valid.data(),2);
        std::memcpy(joined.data()+1,controls_map,sizeof(controls_map));
        std::vector<uint16_t> actual(expected.size());
        phx::Operation op{}; op.family=phx::FrameFeatures;
        op.left_input=color.data(); op.right_input=prior.data(); op.auxiliary=state.noise.data();
        op.secondary_output=joined.data(); op.output=actual.data(); op.row_count=1; op.column_count=2;
        op.reduction_count=4; op.batch_count=2; op.flags=3;
        op.right_batch_stride=std::bit_cast<uint32_t>(controls.tone);
        op.output_batch_stride=std::bit_cast<uint32_t>(controls.structure);
        op.left_row_stride=std::bit_cast<uint32_t>(controls.skin);
        phx::Kernel<HostMath> kernel{op};
        for(size_t p=0;p<8;++p) kernel(p);
        check(actual==expected,"shared device-operation map packing differs from host");
        std::cout<<checks<<" advanced appearance/UI/distortion/device-map contracts passed\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

