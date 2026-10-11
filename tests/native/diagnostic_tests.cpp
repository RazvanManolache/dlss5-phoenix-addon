#include "../../src/native/support/diagnostic.h"
#include <iostream>
using namespace phoenix_addon;
using namespace phoenix_addon::diagnostic;

struct Parameters final : NVSDK_NGX_Parameter {
    unsigned writes = 0;
    void* color = reinterpret_cast<void*>(0x1000);
    void* output = reinterpret_cast<void*>(0x2000);
#define PHOENIX_TEST_PARAM(T, V)                                                                   \
    void Set(const char*, T) override {                                                            \
        ++writes;                                                                                  \
    }                                                                                              \
    NVSDK_NGX_Result Get(const char*, T* value) const override {                                   \
        *value = V;                                                                                \
        return NVSDK_NGX_Result_Success;                                                           \
    }
    PHOENIX_TEST_PARAM(unsigned long long, 0x100000007ull)
    PHOENIX_TEST_PARAM(float, 0.375f)
    PHOENIX_TEST_PARAM(double, 0.25)
    PHOENIX_TEST_PARAM(unsigned, 3u)
    PHOENIX_TEST_PARAM(int, -4)
    PHOENIX_TEST_PARAM(ID3D11Resource*, nullptr)
#undef PHOENIX_TEST_PARAM
    void Set(const char*, ID3D12Resource*) override {
        ++writes;
    }
    void Set(const char*, void*) override {
        ++writes;
    }
    NVSDK_NGX_Result Get(const char*, ID3D12Resource**) const override {
        return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    NVSDK_NGX_Result Get(const char* key, void** value) const override {
        if (!std::strcmp(key, "DLSSNR.Color"))
            *value = color;
        else if (!std::strcmp(key, "DLSSNR.Output"))
            *value = output;
        else {
            *value = nullptr;
            return NVSDK_NGX_Result_FAIL_InvalidParameter;
        }
        return NVSDK_NGX_Result_Success;
    }
    void Reset() override {
        ++writes;
    }
};
static unsigned cases = 0;
static void test(bool condition) {
    check(condition, "phoenix diagnostic CPU assertion");
    ++cases;
}
template <class F> void rejected(F action) {
    bool caught = false;
    try {
        action();
    } catch (const std::exception&) {
        caught = true;
    }
    test(caught);
}
int main() {
    try {
        std::vector<uint8_t> padded(80, 0xcd);
        for (unsigned row = 0; row < 2; ++row)
            for (unsigned column = 0; column < 16; ++column)
                padded[8 + row * 32 + column] = uint8_t(row * 16 + column);
        auto packed = pack_rows(padded.data(), padded.size(), 8, 32, 16, 2);
        test(packed.size() == 32 && packed.front() == 0 && packed.back() == 31);
        auto padded2 = padded;
        padded2[30] = 0;
        test(compare_rgba16f(packed, pack_rows(padded2.data(), padded2.size(), 8, 32, 16, 2))
                 .exact());
        test(compare_rgba16f(packed, packed).first_byte == std::numeric_limits<size_t>::max());
        auto different = packed;
        different[9] ^= 1;
        different[10] ^= 2;
        different[24] ^= 4;
        const auto delta = compare_rgba16f(packed, different);
        test(delta.bytes == 3 && delta.pixels == 2 && delta.first_byte == 9);
        std::vector<uint8_t> zero(8), negative_zero(8);
        negative_zero[1] = 0x80;
        test(!compare_rgba16f(zero, negative_zero).exact());
        std::vector<uint8_t> nan(8);
        nan[0] = 1;
        nan[1] = 0x7e;
        test(compare_rgba16f(nan, nan).exact());
        // Bounded HDR error: identical frames measure zero; a 2% network-
        // scale difference passes the interim ceiling; a missing
        // denormalize (candidate at paper 1.0 against native at white 78x,
        // rel ~0.99) fails it; invented or clipped non-finites fail
        // structurally while matched overflow is counted, not measured.
        {
            auto pixel = [](float red, float green, float blue, float alpha) {
                const auto to_half = [](float value) {
                    uint32_t bits = 0;
                    std::memcpy(&bits, &value, 4);
                    const int exp = int((bits >> 23) & 0xff) - 127;
                    const uint32_t mant = bits & 0x7fffff;
                    if ((bits & 0x7fffffff) == 0)
                        return uint16_t((bits >> 16) & 0x8000);
                    if (exp > 15)
                        return uint16_t(((bits >> 16) & 0x8000) | 0x7c00);
                    if (exp < -24)
                        return uint16_t((bits >> 16) & 0x8000);
                    if (exp < -14) {
                        const uint32_t shifted = (mant | 0x800000u) >> unsigned(-14 - exp);
                        return uint16_t(((bits >> 16) & 0x8000) |
                                        ((shifted + 0xfffu + ((shifted >> 13) & 1u)) >> 13));
                    }
                    return uint16_t(((bits >> 16) & 0x8000) | (unsigned(exp + 15) << 10) |
                                    ((mant + 0xfffu + ((mant >> 13) & 1u)) >> 13));
                };
                std::vector<uint8_t> out(8);
                const uint16_t channels[4]{to_half(red), to_half(green), to_half(blue),
                                           to_half(alpha)};
                for (unsigned channel_index = 0; channel_index < 4; ++channel_index) {
                    out[channel_index * 2] = uint8_t(channels[channel_index] & 0xff);
                    out[channel_index * 2 + 1] = uint8_t(channels[channel_index] >> 8);
                }
                return out;
            };
            test(decode_half_float(0x3c00) == 1.0f);
            test(decode_half_float(0x3800) == 0.5f);
            test(decode_half_float(0x0000) == 0.0f);
            test(decode_half_float(0x7c00) == std::numeric_limits<float>::infinity());
            test(decode_half_float(0x7e00) != decode_half_float(0x7e00));
            const auto flat = pixel(1.0f, 0.5f, 0.25f, 1.0f);
            const auto same = bounded_difference(flat, flat);
            test(same.finite && same.max_abs == 0.0 && same.max_rel == 0.0 &&
                 same.within(hdr_sanity_ceiling));
            const auto native2 = pixel(1.0f, 0.5f, 0.25f, 1.0f);
            const auto cand2 = pixel(1.02f, 0.49f, 0.255f, 1.0f);
            const auto narrow = bounded_difference(cand2, native2);
            test(narrow.finite && narrow.max_rel < 0.05 && narrow.within(hdr_sanity_ceiling));
            const auto native78 = pixel(78.0f, 39.0f, 19.5f, 1.0f);
            const auto nodesc = pixel(1.0f, 0.5f, 0.25f, 1.0f);
            const auto missing = bounded_difference(nodesc, native78);
            test(missing.finite && missing.max_rel > 0.95 && !missing.within(hdr_sanity_ceiling));
            auto inf_cand = pixel(1.0f, 0.5f, 0.25f, 1.0f);
            inf_cand[0] = 0x00;
            inf_cand[1] = 0x7c;
            const auto invented = bounded_difference(inf_cand, flat);
            test(!invented.finite && invented.nonfinite_mismatches == 1 &&
                 !invented.within(hdr_sanity_ceiling));
            auto nan_a = flat;
            nan_a[0] = 0x01;
            nan_a[1] = 0x7e;
            auto nan_b = flat;
            nan_b[0] = 0x02;
            nan_b[1] = 0x7e;
            const auto matched = bounded_difference(nan_a, nan_b);
            test(matched.finite && matched.matched_nonfinite == 1 && matched.max_rel == 0.0 &&
                 matched.within(hdr_sanity_ceiling));
            auto inf_nat = flat;
            inf_nat[2] = 0x00;
            inf_nat[3] = 0x7c;
            const auto clipped = bounded_difference(flat, inf_nat);
            test(!clipped.finite && clipped.nonfinite_mismatches == 1);
            auto alpha_only = flat;
            alpha_only[6] = 0x00;
            alpha_only[7] = 0x38;
            const auto alpha = bounded_difference(alpha_only, flat);
            test(alpha.finite && alpha.max_rel == 0.0 && alpha.alpha_max_abs > 0.0 &&
                 alpha.within(hdr_sanity_ceiling));
            const auto black = pixel(0.0f, 0.0f, 0.0f, 1.0f);
            auto near_black = pixel(0.005f, 0.0f, 0.0f, 1.0f);
            const auto floor = bounded_difference(near_black, black);
            test(floor.finite && floor.max_rel <= 0.51 && floor.max_abs <= 0.006);
            rejected([&] { bounded_difference({}, {}); });
            rejected([&] { bounded_difference(flat, std::vector<uint8_t>(16, 0)); });
            const auto stats = candidate_statistics(flat);
            test(stats.finite && stats.channels == 4 && stats.max_abs == 1.0 &&
                 stats.min_value == 0.25 && stats.max_value == 1.0 &&
                 stats.mean_abs == (1.0 + 0.5 + 0.25 + 1.0) / 4.0);
            auto inf_stats = flat;
            inf_stats[0] = 0x00;
            inf_stats[1] = 0x7c;
            const auto clipped_stats = candidate_statistics(inf_stats);
            test(!clipped_stats.finite && clipped_stats.channels == 3);
            const auto empty_stats = candidate_statistics(std::vector<uint8_t>{});
            test(empty_stats.finite && empty_stats.channels == 0 && empty_stats.mean_abs == 0.0);
            rejected([&] { candidate_statistics(std::vector<uint8_t>(7)); });
            rejected([&] { bounded_difference(std::vector<uint8_t>(7), std::vector<uint8_t>(7)); });
        }
        rejected([&] { pack_rows(nullptr, 80, 0, 32, 16, 2); });
        rejected([&] { pack_rows(padded.data(), 80, 81, 32, 16, 2); });
        rejected([&] { pack_rows(padded.data(), 80, 8, 15, 16, 2); });
        rejected([&] { pack_rows(padded.data(), 55, 8, 32, 16, 2); });
        rejected([&] { pack_rows(padded.data(), 80, 8, 32, 16, 0); });
        rejected([&] { pack_rows(padded.data(), 80, 8, SIZE_MAX, 16, SIZE_MAX); });
        rejected([&] { compare_rgba16f({}, {}); });
        rejected([&] { compare_rgba16f(zero, packed); });
        rejected([&] { compare_rgba16f(std::vector<uint8_t>(7), std::vector<uint8_t>(7)); });
        Parameters original;
        auto* separate = reinterpret_cast<ID3D12Resource*>(0x3000);
        CandidateParameters proxy(original, separate);
        test(parameter_resource(&proxy, "DLSSNR.Output") == separate);
        test(parameter_resource(&original, "DLSSNR.Output") == original.output);
        test(parameter_resource(&proxy, "DLSSNR.Color") == original.color);
        test(parameter_resource(&proxy, "DLSSNR.MVec") == nullptr);
        ID3D12Resource* typed = nullptr;
        test(NVSDK_NGX_SUCCEED(proxy.Get("DLSSNR.Output", &typed)) && typed == separate);
        test(!NVSDK_NGX_SUCCEED(proxy.Get("DLSSNR.Color", &typed)));
        using NativeGet = NVSDK_NGX_Result (*)(const NVSDK_NGX_Parameter*, const char*, void**);
        const auto slot8 = reinterpret_cast<NativeGet>((*reinterpret_cast<void***>(&proxy))[8]);
        void* raw = nullptr;
        test(NVSDK_NGX_SUCCEED(slot8(&proxy, "DLSSNR.Output", &raw)) && raw == separate);
        float float_value = 0;
        int iteration_index = 0;
        unsigned unsigned_value = 0;
        unsigned long long wide_value = 0;
        double double_value = 0;
        proxy.Get("tone", &float_value);
        proxy.Get("reset", &iteration_index);
        proxy.Get("style", &unsigned_value);
        proxy.Get("qword", &wide_value);
        proxy.Get("double", &double_value);
        test(float_value == 0.375f && iteration_index == -4 && unsigned_value == 3 &&
             wide_value == 0x100000007ull && double_value == 0.25);
        rejected([&] { proxy.Set("DLSSNR.Output", separate); });
        rejected([&] { proxy.Set("tone", 1.0f); });
        rejected([&] { proxy.Reset(); });
        test(original.writes == 0 && original.output == reinterpret_cast<void*>(0x2000));
        rejected([&] { CandidateParameters invalid(original, nullptr); });
        // The private dispatch patch must override the parent's new defaults.
        std::array<uint8_t, 264> stem{};
        std::array<uint8_t, 184> head{};
        put(stem.data(), 8, uint64_t(0x1111));
        put(stem.data(), 16, uint64_t(0x2222));
        put(head.data(), 88, uint64_t(0x1111));
        put(head.data(), 96, uint64_t(0x2222));
        const auto defaults = resolve_controls(Controls{}, 640, 384, true, true, 1);
        patch_stem(stem.data(), stem.size(), defaults);
        patch_head(head.data(), head.size(), defaults);
        Controls live;
        live.tone = 0.25f;
        live.structure = 0.75f;
        live.motion_x = -640;
        live.motion_y = 192;
        const auto resolved = resolve_controls(live, 640, 384, true, true, 0x100000007ull);
        patch_stem(stem.data(), stem.size(), resolved);
        patch_head(head.data(), head.size(), resolved);
        uint64_t sampling = 0, descriptor = 0;
        float tone = 0, mx = 0;
        std::memcpy(&sampling, stem.data() + 200, 8);
        std::memcpy(&tone, stem.data() + 172, 4);
        std::memcpy(&descriptor, stem.data() + 16, 8);
        std::memcpy(&mx, head.data() + 164, 4);
        test(sampling == 0x100000007ull && tone == 0.25f && descriptor == 0x2222 && mx == -1);
        dispatch_audit::Counts counts;
        {
            dispatch_audit::Call outside(99);
            outside.accepted();
        }
        test(counts.own_attempted == 0 && counts.other_attempted == 0);
        {
            dispatch_audit::Scope scope(counts);
            {
                dispatch_audit::OwnCall own;
                dispatch_audit::Call batch(154);
                batch.accepted();
            }
            {
                dispatch_audit::Call native(155);
                native.accepted();
            }
            {
                dispatch_audit::Call rejected_call(1);
            }
            rejected([&] { dispatch_audit::Scope nested(counts); });
        }
        test(counts.own_attempted == 154 && counts.own_accepted == 154 &&
             counts.other_attempted == 156 && counts.other_accepted == 155);
        test(dispatch_audit::active == nullptr && !dispatch_audit::own_call);
        test(completed_submission_count(0, 32, 32) == 0);
        test(completed_submission_count(31, 32, 32) == 31);
        test(completed_submission_count(32, 32, 32) == 32);
        test(completed_submission_count(100, 32, 32) == 32);
        test(completed_submission_count(32, 32, 31) == 31);
        test(completed_submission_count(UINT64_MAX, 32, 32) == 0);
        // Output-state refusal contract: resolve_engine_output refuses via
        // check(), i.e. std::runtime_error carrying the refusal. The live
        // path degrades to candidate-only by catching exactly that form, so
        // this pins it: any other throw type would turn a routine RE9
        // refusal (persistent-state output, untracked list) terminal again.
        {
            bool runtime_form = false;
            std::string refusal;
            try {
                check(false, "phoenix diagnostic engine output state unestablished");
            } catch (const std::runtime_error& exception) {
                runtime_form = true;
                refusal = exception.what();
            } catch (...) {
            }
            test(runtime_form && refusal.find("output state unestablished") != std::string::npos);
        }
        test(!GetModuleHandleW(L"nvapi64.dll") && !GetModuleHandleW(L"nvngx_dlssnr.dll"));
        std::cout << Json({{"cpu_only", true},
                           {"passed", true},
                           {"cases", cases},
                           {"nvapi_loaded", false},
                           {"native_nr_loaded", false}})
                         .dump()
                  << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
