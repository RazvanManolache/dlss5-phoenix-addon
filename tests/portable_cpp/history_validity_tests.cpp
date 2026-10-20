// Synthetic history reprojection: no model, device, driver or inference.
#include "../../native_runtime/frame.h"
#include <iostream>
#include <stdexcept>
using namespace phoenix_cpp;

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
struct Warped {
    std::vector<float> conditioning, resolve;
    std::vector<uint8_t> conditioning_valid, resolve_valid;
    bool operator==(const Warped&) const = default;
};
static Warped warp(FrameState& state, const std::vector<float>& motion,
                   const float* depth, const float* validity, bool reversed) {
    Warped result;
    state.warp(motion.data(), depth, validity, reversed, result.conditioning, result.resolve,
               result.conditioning_valid, result.resolve_valid);
    return result;
}
int main() {
    try {
        FrameState state(3, 3, 3, 3, 1);
        state.history.resize(27);
        for (unsigned i = 0; i < state.history.size(); ++i)
            state.history[i] = float(i + 1) / 32.f;
        std::vector<float> all_valid(9, 1.f), motion(18, 0.f), depth(9, .5f);
        unsigned cases = 0;
        for (bool reversed : {false, true}) {
            for (bool use_depth : {false, true}) {
                for (float displacement : {-100.f, -2.f, -1.f, 0.f, 1.f, 2.f, 100.f}) {
                    for (unsigned axis = 0; axis < 2; ++axis) {
                        std::fill(motion.begin(), motion.end(), 0.f);
                        motion[4 * 2 + axis] = displacement;
                        const auto implicit = warp(state, motion, use_depth ? depth.data() : nullptr, nullptr, reversed);
                        const auto explicit_mask = warp(state, motion, use_depth ? depth.data() : nullptr, all_valid.data(), reversed);
                        require(implicit == explicit_mask, "Omitted mask differs from an all-valid mask");
                        const bool inside = displacement >= -1.f && displacement <= 1.f;
                        require(bool(implicit.conditioning_valid[4]) == inside, "Conditioning bounds rejection failed");
                        require(bool(implicit.resolve_valid[4]) == inside, "Resolve bounds rejection failed");
                        if (!inside)
                            for (unsigned c = 0; c < 3; ++c) {
                                require(implicit.conditioning[12 + c] == 0.f, "Invalid conditioning history leaked");
                                require(implicit.resolve[12 + c] == 0.f, "Invalid resolve history leaked");
                            }
                        ++cases;
                    }
                }
            }
        }
        // Depth-dilated conditioning may select a different motion vector from
        // resolve. Reject only the out-of-frame history view, with either mask form.
        for (bool reversed : {false, true}) {
            std::fill(motion.begin(), motion.end(), 0.f);
            std::fill(depth.begin(), depth.end(), .5f);
            depth[0] = reversed ? .9f : .1f;
            motion[0] = 100.f;
            const auto implicit = warp(state, motion, depth.data(), nullptr, reversed);
            const auto explicit_mask = warp(state, motion, depth.data(), all_valid.data(), reversed);
            require(implicit == explicit_mask, "Depth-selected motion depends on mask presence");
            require(!implicit.conditioning_valid[4] && implicit.resolve_valid[4],
                    "Conditioning and resolve bounds must stay independent");
            all_valid[0] = 0.f;
            const auto masked_neighbor = warp(state, motion, depth.data(), all_valid.data(), reversed);
            require(masked_neighbor.conditioning_valid[4] && masked_neighbor.resolve_valid[4],
                    "Invalid depth neighbor must not supply conditioning motion");
            all_valid[0] = 1.f;
        }
        std::fill(motion.begin(), motion.end(), 0.f);
        all_valid[4] = 0.f;
        const auto rejected = warp(state, motion, nullptr, all_valid.data(), false);
        require(!rejected.conditioning_valid[4] && !rejected.resolve_valid[4], "Explicit invalid mask ignored");
        require(rejected.resolve_valid[3], "Invalid mask affected a neighboring valid pixel");
        std::cout << "{\"passed\":true,\"mask_equivalence_cases\":" << cases
                  << ",\"gpu_used\":false}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
