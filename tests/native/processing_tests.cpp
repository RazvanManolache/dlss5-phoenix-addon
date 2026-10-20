// Focused CPU contracts and shader compilation. Creates no graphics device.
#include "../../src/native/support/direct.h"
#include "../../src/native/support/present_guides.h"
#include "../../src/integration/compatibility/processing_pixels.h"
#include "../../src/integration/compatibility/session_policy.h"
#include <iostream>
#include <limits>
int main() {
    using namespace phoenix_addon;
    using namespace phoenix_addon::phoenix;
    using namespace phoenix_compat;
    unsigned checks = 0;
    auto expect = [&](bool ok, const char* message) {
        ++checks;
        check(ok, message);
    };
    try {
        Settings s;
        expect(s.valid(), "Existing defaults remain valid");
        expect(!s.processing.enabled, "Legacy profiles retain the existing bridge");
        s.refinement_passes = 10;
        expect(s.valid(), "Ten passes accepted");
        s.refinement_passes = 11;
        expect(!s.valid(), "Eleven passes refused");
        s.refinement_passes = 10;
        expect(nvidia_preset(-1, "original") == 1 && nvidia_preset(-1, "small-fp8") == 2,
               "Matched NVIDIA architectures");
        expect(nvidia_preset(0, "small-fp8") == 0 && nvidia_preset(1, "small-fp8") == 1,
               "Explicit vendor preset overrides matching");
        expect(model_profile("small-fp8").native_launches == 78 &&
                   model_profile("original").learned_blocks == 71,
               "Model profiles");
        auto& p = s.processing;
        p.enabled = true;
        p.scale = 0;
        auto g = processing_resolution(1920, 1080, p, true);
        expect(g.width == 16 && g.height == 16, "Zero scale has a real minimum canvas");
        p.scale = 200;
        g = processing_resolution(1920, 1080, p, true);
        expect(g.width == 3840 && g.height == 2160, "200 percent scales both axes");
        p.scale = std::numeric_limits<float>::quiet_NaN();
        expect(!s.valid(), "Nonfinite scale refused");
        p.scale = 63;
        p.exposure = 100;
        expect(p.effective_exposure() == 10, "Runtime exposure clamp");
        p.exposure_mode = 2;
        expect(std::abs(p.effective_exposure() - .1f) < 1e-6, "Inverse exposure");
        p.exposure = 0;
        expect(p.effective_exposure() == 0, "Zero inversion finite");
        p.exposure_mode = 0;
        expect(p.effective_exposure() == 1, "Exposure Off");
        p.white_nits = 250;
        expect(p.diffuse_white() == 100, "Automatic white ignores saved override");
        p.white_override = true;
        expect(p.diffuse_white() == 250, "Explicit white applied");
        p.ui_policy = 0;
        expect(p.correction(true) && !p.correction(false), "Auto correction host policy");
        p.ui_policy = 1;
        expect(!p.correction(true), "Correction Off");
        p.ui_policy = 2;
        expect(p.correction(false), "Correction On");
        auto json = settings_json(s);
        Settings loaded;
        visit_settings(loaded, [&](const char* key, auto& value) {
            value = json.at(key).get<std::decay_t<decltype(value)>>();
        });
        expect(settings_json(loaded) == json && loaded.valid(),
               "Full persistence and helper transport roundtrip");
        expect(helper_settings_changed(Settings{}, s), "Pipeline edits rebuild resource owners");
        expect(!live_tuning_json(s).contains("Encoding"),
               "Resource domain changes never race live tuning");
        for (int encoding = 1; encoding <= 5; ++encoding)
            for (float white : {48.f, 100.f, 500.f}) {
                pixels::RGB v{.002f, .1f, .8f};
                auto encoded = pixels::transfer(v, encoding, white, false);
                auto back = pixels::transfer(encoded, encoding, white, true);
                for (unsigned c = 0; c < 3; ++c)
                    expect(std::abs(back[c] - v[c]) < .001f, "Input transfer roundtrip");
            }
        for (float value : {-2.f, -.1f, 0.f, .001f, .5f, 2.f}) {
            expect(std::abs(pixels::srgb(pixels::srgb(value, false), true) - value) < 1e-5,
                   "Signed sRGB roundtrip");
            expect(std::abs(pixels::pq(pixels::pq(value / 100, false), true) * 100 - value) < .001,
                   "Signed PQ roundtrip");
        }
        expect(pixels::pq(0,false)==0 && pixels::pq(0,true)==0, "Signed PQ black agrees with the GPU codec");
        std::vector<float> checker(4 * 4 * 3);
        for (unsigned y = 0; y < 4; ++y)
            for (unsigned x = 0; x < 4; ++x)
                for (unsigned c = 0; c < 3; ++c)
                    checker[(y * 4 + x) * 3 + c] = float((x + y) % 2);
        auto area = pixels::resample(checker, 4, 4, 1, 1, 0, true);
        auto nearest = pixels::resample(checker, 4, 4, 1, 1, 2, true);
        expect(area[0] == .5f && nearest[0] == 0,
               "Area integrates source coverage; nearest preserves its selected texel");
        std::vector<float> image(35 * 23 * 3);
        for (size_t i = 0; i < image.size(); ++i)
            image[i] = .01f + float((i * 37) % 101) / 105.f;
        for (int domain = 0; domain < 3; ++domain)
            for (int signal = 0; signal < 3; ++signal)
                for (int up = 0; up < 3; ++up) {
                    Processing options;
                    options.enabled = true;
                    options.scale = 51;
                    options.domain = domain;
                    options.signal = signal;
                    options.upsample = up;
                    pixels::Frame frame(image, 35, 23, options);
                    const auto result = frame.finish(frame.encoded);
                    for (size_t i = 0; i < image.size(); ++i)
                        expect(
                            std::abs(image[i] - result[i]) < .002,
                            "Identity NR preserves full-resolution detail through residual reconstruction");
                }
        direct::Pass::compile(direct::PresentGuides::shader_source());
        expect(true,"Present guide resampling shader compiled without a graphics device");
        direct::Pass::compile(direct::processing_shader);
        expect(true, "Explicit processing HLSL compiled without creating a device");
        direct::Pass::compile((std::string(direct::bridge_common) + direct::output_shader).c_str());
        expect(true, "Legacy and explicit-domain comparison shader compiled");
        std::cout << Json{{"passed", true},
                          {"checks", checks},
                          {"gpu_device_created", false},
                          {"inference", false}}
                         .dump()
                  << '\n';
    } catch (const std::exception& e) {
        std::cout << Json{{"passed", false}, {"checks", checks}, {"error", e.what()}}.dump()
                  << '\n';
        return 1;
    }
}
