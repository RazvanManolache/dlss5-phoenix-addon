// Deterministic shared frame preparation tests. The runner compiles this once
// with the pinned control header and once with the current maintained header.
#include "../../native_runtime/frame.h"
#include "../../src/native/model/weight_source.h"
#include <chrono>
#include <iostream>
using namespace phoenix_cpp;
int main() {
    Json report = {{"passed", true}, {"cases", Json::array()}};
    for (const auto extent : {std::pair{37u, 29u}, std::pair{1920u, 1080u}, std::pair{2560u, 1080u}}) {
        const auto [width, height] = extent;
        FrameState frame(width, height, (width + 31) / 32 * 32, (height + 31) / 32 * 32);
        std::vector<float> color(size_t(width) * height * 3), prior(color.size()),
            mask(size_t(width) * height * 4);
        std::vector<uint8_t> valid(size_t(width) * height);
        for (size_t i = 0; i < color.size(); ++i) {
            color[i] = float(int(i * 19 % 193) - 64) / 128;
            prior[i] = float(int(i * 13 % 97) - 32) / 64;
        }
        for (size_t i = 0; i < mask.size(); ++i) mask[i] = float(i * 7 % 23) / 22;
        for (size_t i = 0; i < valid.size(); ++i) valid[i] = uint8_t(i % 3 != 0);
        for (bool spatial : {false, true})
            for (bool automatic : {false, true}) {
                Controls controls;
                controls.style = 2; controls.tone = 1.73f; controls.structure = .37f;
                controls.skin = -.0f; controls.auto_mask = automatic;
                std::vector<double> samples;
                std::vector<uint16_t> features;
                for (unsigned repeat = 0; repeat < 11; ++repeat) {
                    auto start = std::chrono::steady_clock::now();
                    features = frame.features(color.data(), controls, prior, valid,
                                              spatial ? mask.data() : nullptr);
                    if (repeat >= 2) samples.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count());
                }
                std::sort(samples.begin(), samples.end());
                report["cases"].push_back({{"extent", {width, height}}, {"spatial_mask", spatial},
                    {"auto_mask", automatic}, {"median_ms", samples[samples.size() / 2]},
                    {"sha256", phoenix_model::weight_digest(
                        reinterpret_cast<const uint8_t*>(features.data()), features.size() * 2)}});
            }
    }
    std::cout << report.dump() << '\n';
}
