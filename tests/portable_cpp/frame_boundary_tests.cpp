// Synthetic frame-boundary contracts: no model, driver, device or game needed.
#include "../../native_runtime/frame.h"
#include "../../native_runtime/frame_planes.h"
#include <iostream>
#include <limits>
using namespace phoenix_cpp;
static unsigned checks = 0;
static void check(bool ok, const char* message) {
    ++checks;
    if (!ok) throw std::runtime_error(message);
}
template<class F> static void rejects(F operation, const char* message) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
int main() {
    try {
        const float gain = checked_model_history_gain(.73974609375f);
        check(resolve_gate(0, true, gain) == .369873046875f, "model-derived gate");
        check(resolve_gate(0, true, 1) == .5f, "explicit legacy unit gain");
        check(resolve_gate(NAN, false, gain) == 0, "invalid history must not consume gate");
        for (float bad : {-1.f, 2.f, NAN, INFINITY})
            rejects([&] { checked_model_history_gain(bad); }, "invalid model gain accepted");
        SamplingPlan fixed{17, 9, false}, video{17, 9, true}, wrap{0, UINT32_MAX, true};
        check(fixed.seed(0) == 17 && fixed.seed(3) == 17 && fixed.next(4) == 9, "fixed replay");
        check(video.seed(0) == 26 && video.seed(3) == 29 && video.next(4) == 13, "per-pass sequence");
        check(wrap.seed(1) == 0 && wrap.next(1) == 0, "defined sampling wrap");
        check(video.seed(0) == 26, "uncommitted sampling plan changed state");
        rejects([] { Controls::from(Json{{"sampling_mode", "guess"}}); }, "unknown sampling accepted");
        check(Controls::from(Json::object()).advancing_noise == false, "legacy fixed default");

        FrameState state(2, 1, 4, 2, 1);
        std::vector<float> color(6, .25f), prior(6, .75f);
        std::vector<uint8_t> valid(2, 1), invalid(2, 0);
        std::vector<float> head(4 * 2 * 4, 0.f), next, masked_next;
        Controls controls;
        controls.legacy_output = true;
        controls.history_strength = gain;
        auto display = state.resolve(color.data(), head, controls, prior, valid, next);
        check(std::fabs(display[0] - (.25f + .5f * .369873046875f)) < 1e-7f, "history resolve algebra");
        auto no_history = state.resolve(color.data(), head, controls, prior, invalid, masked_next);
        check(no_history == color, "no-history baseline changed");
        float mask[] = {0, .5f, .25f, 1, 1, 1, 0, 1};
        validate_control_mask(mask, 2);
        auto masked = state.resolve(color.data(), head, controls, prior, valid, masked_next, mask);
        check(masked[0] == color[0] && masked[3] == display[3], "R display composition");
        check(masked_next == next, "display mask leaked into recurrent history");
        controls.intensity = 2;
        masked = state.resolve(color.data(), head, controls, prior, valid, masked_next, mask);
        check(masked[0] == color[0], "zero R must protect original at intensity above one");
        controls.tone = 2;
        controls.structure = .5f;
        auto scalar = state.features(color.data(), controls, prior, valid);
        auto packed = state.features(color.data(), controls, prior, valid, mask);
        auto half = [&](size_t i) { return state.math.half_value(packed[i]); };
        check(half(11) == 1 && half(12) == .125f, "G/B global multiplication");
        check(half(13) == 0 && half(14) == 0, "map automatic skin precedence");
        check(half(16 + 11) == 2 && half(16 + 12) == 0, "per-pixel channels");
        check(packed[2 * 16 + 11] == packed[11], "map mirrored padding");
        for (size_t pixel = 0; pixel < 8; ++pixel)
            for (size_t channel = 0; channel < 16; ++channel)
                if (channel < 11 || channel > 14)
                    check(packed[pixel * 16 + channel] == scalar[pixel * 16 + channel],
                          "map altered non-control features");
        mask[3] = 0;
        rejects([&] { validate_control_mask(mask, 2); }, "unknown alpha semantics accepted");
        mask[3] = 1; mask[0] = NAN;
        rejects([&] { validate_control_mask(mask, 2); }, "nonfinite mask accepted");
        mask[0] = 1.1f;
        rejects([&] { validate_control_mask(mask, 2); }, "out-of-range mask accepted");

        float rows[] = {1, 2, 3, 99, 4, 5, 6, 99};
        phoenix_plane_f32 plane{rows, sizeof(rows) - sizeof(float), 4 * sizeof(float), 1, 2, 3, 0};
        check(copy_plane(plane, 1, 2, 3, true) == std::vector<float>({1,2,3,4,5,6}), "padded row copy");
        auto bad = plane; bad.capacity_bytes--;
        rejects([&] { copy_plane(bad, 1, 2, 3, true); }, "short capacity accepted");
        bad = plane; bad.row_stride_bytes = UINT64_MAX;
        rejects([&] { copy_plane(bad, 1, 2, 3, true); }, "stride overflow accepted");
        bad = plane; bad.row_stride_bytes = 4;
        rejects([&] { copy_plane(bad, 1, 2, 3, true); }, "overlapping rows accepted");
        bad = plane; bad.channels = 4;
        rejects([&] { copy_plane(bad, 1, 2, 3, true); }, "wrong channels accepted");
        bad = plane; bad.reserved = 1;
        rejects([&] { copy_plane(bad, 1, 2, 3, true); }, "reserved fields accepted");
        bad = plane; bad.data = reinterpret_cast<const float*>(UINTPTR_MAX - 3);
        rejects([&] { copy_plane(bad, 1, 2, 3, true); }, "address wrap accepted");
        bad = {}; bad.width = 1;
        rejects([&] { copy_plane(bad, 1, 2, 3, false); }, "malformed absent plane accepted");
        check(copy_plane({}, 1, 2, 3, false).empty(), "omitted optional plane");
        rejects([&] { copy_plane({}, 1, 2, 3, true); }, "missing required plane accepted");
        std::cout << checks << " frame boundary checks passed; no model/GPU execution\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
