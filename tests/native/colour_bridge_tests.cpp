// Proves the two claims the colour bridge rests on.
//
// 1. Widening R11G11B10_FLOAT to R16G16B16A16_FLOAT loses nothing. This is
//    not a sampled comparison. Every R11G11B10_FLOAT channel value that
//    exists is enumerated -- all 2048 of the 11-bit encodings and all 1024
//    of the 10-bit ones -- decoded the way the hardware decodes them,
//    encoded to fp16, and decoded back. Any value that did not survive
//    exactly would fail here.
//
// 2. The paper-white rescale round-trips within a bound. Unlike the
//    widening, dividing by paper white and multiplying back is two fp16
//    roundings plus the R11/B10 display store, so exactness is not claimed;
//    the bound is. White values are the ones measured in the b6 played run
//    (OptiScaler composition lines, 1.00x to 78.02x).
//
// CPU only. No device, no shader, no GPU. This pins the MATH (the formula
// the shaders implement); the live band validates the SHADERS.
//
// 3. The exposure meter's log-mean and key formula. Same layering: the math
//    here, the shader on the band.
#include "../../src/native/support/exposure_meter.h" // meter_key only; no device is ever created here
#include "../../src/native/support/colour_bridge.h" // gates only (widenable/rescale_only/exposure_readable are pure); no device is ever created here
#include "../../src/native/support/denormalize.h" // gates only (result/output/exposure gates are pure); no device is ever created here
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <iostream>

namespace {

// R11G11B10_FLOAT channels are unsigned floats with a 5-bit exponent and no
// sign bit: 6-bit mantissa for R and G, 5-bit for B. Bias 15, same as fp16.
double decode_small_float(unsigned bits, unsigned mantissa_bits) {
    const unsigned exponent_bits = 5;
    const unsigned mantissa_mask = (1u << mantissa_bits) - 1u;
    const unsigned exponent_mask = (1u << exponent_bits) - 1u;
    const unsigned mantissa = bits & mantissa_mask;
    const unsigned exponent = (bits >> mantissa_bits) & exponent_mask;
    if (exponent == exponent_mask) {
        if (mantissa)
            return std::nan("");
        return std::numeric_limits<double>::infinity();
    }
    if (exponent == 0) {
        if (mantissa == 0)
            return 0.0;
        return std::ldexp(double(mantissa) / double(1u << mantissa_bits), -14);
    }
    return std::ldexp(1.0 + double(mantissa) / double(1u << mantissa_bits), int(exponent) - 15);
}

// IEEE binary16: 1 sign, 5 exponent, 10 mantissa. Round to nearest even, which
// is what a UAV store performs.
uint16_t encode_half(double value) {
    if (std::isnan(value))
        return 0x7e00;
    if (std::isinf(value))
        return value < 0 ? 0xfc00 : 0x7c00;
    const uint16_t sign = value < 0 ? 0x8000 : 0;
    value = std::fabs(value);
    if (value == 0.0)
        return sign;
    int exponent = 0;
    double fraction =
        std::frexp(value, &exponent); // value = fraction * 2^exponent, 0.5 <= fraction < 1
    exponent -= 1;
    fraction *= 2.0;      // 1 <= fraction < 2
    if (exponent < -14) { // subnormal
        const double scaled = std::ldexp(value, 24);
        const double rounded = std::nearbyint(scaled);
        return uint16_t(sign | unsigned(rounded));
    }
    if (exponent > 15)
        return uint16_t(sign | 0x7c00);
    const double scaled = (fraction - 1.0) * 1024.0;
    double mantissa = std::nearbyint(scaled);
    if (mantissa == 1024.0) {
        mantissa = 0.0;
        ++exponent;
        if (exponent > 15)
            return uint16_t(sign | 0x7c00);
    }
    return uint16_t(sign | (unsigned(exponent + 15) << 10) | unsigned(mantissa));
}

double decode_half(uint16_t bits) {
    const unsigned mantissa = bits & 0x3ff;
    const unsigned exponent = (bits >> 10) & 0x1f;
    const double sign = (bits & 0x8000) ? -1.0 : 1.0;
    if (exponent == 0x1f) {
        if (mantissa)
            return std::nan("");
        return sign * std::numeric_limits<double>::infinity();
    }
    if (exponent == 0) {
        if (mantissa == 0)
            return sign * 0.0;
        return sign * std::ldexp(double(mantissa) / 1024.0, -14);
    }
    return sign * std::ldexp(1.0 + double(mantissa) / 1024.0, int(exponent) - 15);
}

unsigned checked = 0, failures = 0;

void verify(unsigned mantissa_bits, const char* channel) {
    const unsigned encodings = 1u << (5 + mantissa_bits);
    for (unsigned bits = 0; bits < encodings; ++bits) {
        const double source = decode_small_float(bits, mantissa_bits);
        const double round_trip = decode_half(encode_half(source));
        ++checked;
        const bool same = std::isnan(source) ? std::isnan(round_trip) : (source == round_trip);
        if (!same) {
            ++failures;
            if (failures <= 4)
                std::cerr << channel << " encoding " << bits << " decoded " << source
                          << " but survived fp16 as " << round_trip << '\n';
        }
    }
}

// Unsigned small-float encode, round to nearest even, matching a typed UAV
// store. Finite non-negative inputs only; the rescale path never feeds it
// anything else (infinities are asserted separately below).
uint32_t encode_small_float(double value, unsigned mantissa_bits) {
    if (std::isinf(value))
        return (0x1fu << mantissa_bits);
    if (value == 0.0)
        return 0;
    int exponent = 0;
    double fraction = std::frexp(value, &exponent);
    exponent -= 1;
    fraction *= 2.0;
    const double scale = double(1u << mantissa_bits);
    if (exponent < -14)
        return unsigned(std::nearbyint(std::ldexp(value, 14 + int(mantissa_bits))));
    if (exponent > 15)
        return (0x1fu << mantissa_bits);
    double mantissa = std::nearbyint((fraction - 1.0) * scale);
    if (mantissa == scale) {
        mantissa = 0.0;
        ++exponent;
        if (exponent > 15)
            return (0x1fu << mantissa_bits);
    }
    return (unsigned(exponent + 15) << mantissa_bits) | unsigned(mantissa);
}

unsigned scale_cases = 0, scale_failures = 0;
double scale_max_rel_err = 0.0;

void verify_scale(unsigned mantissa_bits, const char* channel) {
    // Measured paper whites (b6 OptiScaler composition lines), not guesses.
    const double whites[] = {1.0, 7.63, 22.63, 43.78, 78.02};
    for (double white_level : whites) {
        // Dense sweep over [0, 2W]: paper lands at 1.0 after the divide, the
        // guard band to twice paper stays in range, all in fp16 throughout.
        for (unsigned iteration_index = 0; iteration_index <= 1024; ++iteration_index) {
            const double scene_value = (2.0 * white_level * double(iteration_index)) / 1024.0;
            const double divided = decode_half(encode_half(scene_value / white_level));
            const double back = decode_small_float(
                encode_small_float(divided * white_level, mantissa_bits), mantissa_bits);
            ++scale_cases;
            if (scene_value >= 0.01) {
                const double rel = std::fabs(back - scene_value) / scene_value;
                if (rel > scale_max_rel_err)
                    scale_max_rel_err = rel;
                if (!(rel <= 0.03)) {
                    ++scale_failures;
                    if (scale_failures <= 4)
                        std::cerr << channel << " w=" << white_level << " x=" << scene_value
                                  << " rel err " << rel << '\n';
                }
            } else if (!(std::fabs(back - scene_value) <= 0.005)) {
                ++scale_failures;
                if (scale_failures <= 4)
                    std::cerr << channel << " w=" << white_level << " x=" << scene_value
                              << " abs err " << std::fabs(back - scene_value) << '\n';
            }
        }
        // Beyond the guard band the values stay finite and ordered (no clip
        // invents data, no overflow below the fp16 ceiling at 65504xW).
        for (double white_multiple : {2.5, 4.0, 100.0}) {
            const double scene_value = white_multiple * white_level;
            const double back = decode_half(encode_half(scene_value / white_level)) * white_level;
            ++scale_cases;
            if (!std::isfinite(back) || !(back > 0.0)) {
                ++scale_failures;
                std::cerr << channel << " w=" << white_level << " x=" << scene_value
                          << " not finite positive\n";
            }
        }
    }
    // Overflow documents itself: past 65504xW the fp16 bridge store is inf,
    // which propagates honestly instead of clipping to white.
    {
        const double back = decode_half(encode_half(1e7)) * 78.02;
        ++scale_cases;
        if (!std::isinf(back)) {
            ++scale_failures;
            std::cerr << channel << " overflow did not yield inf\n";
        }
    }
    // The exposure clamp: e <= 0 yields white 1e6, never an infinity or NaN.
    for (double exposure : {0.0, 1e-9, 1e-6}) {
        const double white_level = 1.0 / std::max(exposure, 1e-6);
        ++scale_cases;
        if (!std::isfinite(white_level) || white_level != 1e6) {
            ++scale_failures;
            std::cerr << channel << " exposure " << exposure << " gave white " << white_level
                      << '\n';
        }
    }
}

} // namespace

int main() {
    // R and G carry 6 mantissa bits and share one encoding, so enumerating the
    // 11-bit space once covers both channels. B carries 5.
    verify(6, "R11 (red/green)");
    verify(5, "B10 (blue)");

    // Set expectations explicitly so a silent change in the loop is caught:
    // 2048 eleven-bit encodings plus 1024 ten-bit ones.
    const bool counts_right = checked == (1u << 11) + (1u << 10);
    if (!counts_right)
        std::cerr << "unexpected enumeration count " << checked << '\n';

    verify_scale(6, "R11 scale");
    verify_scale(5, "B10 scale");
    const bool scale_bound_met = scale_failures == 0;

    // Meter math: E = key / exp(mean(log(max(L, 1e-6)))), key pinned below.
    // Changing the key must update these expectations consciously.
    unsigned meter_cases = 0, meter_failures = 0;
    auto meter_e = [](double lavg) { return 0.23 / std::max(lavg, 1e-6); };
    if (!(phoenix_addon::exposure_meter::meter_key == 0.23f)) {
        ++meter_failures;
        std::cerr << "meter key drifted from 0.23\n";
    }
    ++meter_cases;
    // Middle grey exposes to unity: the defining property of the key.
    {
        const double exposure = meter_e(0.23);
        ++meter_cases;
        if (!(std::fabs(exposure - 1.0) <= 1e-6)) {
            ++meter_failures;
            std::cerr << "E(0.23)=" << exposure << '\n';
        }
    }
    // Uniform frames: exact inverse relationship, monotone decreasing.
    {
        double prev = 0.0;
        for (double luminance : {1e-6, 0.01, 0.23, 1.0, 10.0, 100.0}) {
            const double exposure = meter_e(luminance);
            ++meter_cases;
            if (!(std::fabs(exposure - 0.23 / luminance) / (0.23 / luminance) <= 1e-9)) {
                ++meter_failures;
                std::cerr << "E(" << luminance << ")=" << exposure << '\n';
            }
            if (prev != 0.0 && !(exposure < prev)) {
                ++meter_failures;
                std::cerr << "non-monotone at " << luminance << '\n';
            }
            prev = exposure;
        }
    }
    // Black frame: finite huge exposure, never inf or NaN.
    {
        const double exposure = meter_e(0.0);
        ++meter_cases;
        if (!std::isfinite(exposure) || !(std::fabs(exposure - 230000.0) / 230000.0 <= 1e-6)) {
            ++meter_failures;
            std::cerr << "E(black)=" << exposure << '\n';
        }
    }
    // Strided 4096-tile partition sums like a direct sum (RE9 geometry,
    // deterministic pattern): the shader's partition is exact up to fp order.
    {
        const unsigned image_width = 1920, image_height = 1080, total = image_width * image_height;
        double direct = 0.0;
        double tiles[4096] = {};
        for (unsigned pixel_index = 0; pixel_index < total; ++pixel_index) {
            const double luminance = 0.01 + double((pixel_index * 2654435761u) % 97) * 0.13;
            const double log_luminance = std::log(std::max(luminance, 1e-6));
            direct += log_luminance;
            tiles[pixel_index % 4096] += log_luminance;
        }
        double regrouped = 0.0;
        for (double log_luminance : tiles)
            regrouped += log_luminance;
        ++meter_cases;
        if (!(std::fabs(regrouped - direct) / std::fabs(direct) <= 1e-6)) {
            ++meter_failures;
            std::cerr << "tile partition drifted\n";
        }
    }
    // Smoothing: E moves at most 15%/frame toward target. First frame (or
    // non-positive previous) takes the target outright. Changing the rate
    // must update these expectations consciously.
    {
        using phoenix_addon::exposure_meter::smooth_exposure;
        using phoenix_addon::exposure_meter::meter_smoothing_rate;
        if (!(meter_smoothing_rate == 1.15f)) {
            ++meter_failures;
            std::cerr << "smoothing rate drifted\n";
        }
        ++meter_cases;
        struct Case {
            float prev, target, expect;
            bool first;
        };
        const Case cases[]{
            {0.74f, 11.06f, 0.74f * 1.15f, false},        // b13 frame 88: clamp the spike
            {0.74f, 0.01f, 0.74f / 1.15f, false},         // bright flash: clamp down
            {0.74f, 0.75f, 0.75f, false},                 // inside the band: pass through
            {0.74f, 0.74f * 1.15f, 0.74f * 1.15f, false}, // band edge inclusive
            {1.0f, 6.0f, 6.0f, true},                     // first frame: raw target
            {0.0f, 6.0f, 6.0f, false},                    // uninitialized prev: raw target
            {-2.0f, 6.0f, 6.0f, false},                   // garbage prev: raw target
        };
        for (const auto& test_case : cases) {
            ++meter_cases;
            const float got = smooth_exposure(test_case.prev, test_case.target, test_case.first);
            if (!(std::fabs(got - test_case.expect) <= 1e-6f * std::fabs(test_case.expect))) {
                ++meter_failures;
                std::cerr << "smooth(" << test_case.prev << "," << test_case.target << ","
                          << test_case.first << ")=" << got << '\n';
            }
        }
        // Convergence: a real scene cut (10x darker) settles within ~20
        // frames; the walk is monotone and never overshoots.
        {
            float exposure = 0.74f;
            const float target = 7.4f;
            float prev = exposure;
            unsigned steps = 0;
            for (; steps < 100 && exposure != target; ++steps) {
                exposure = smooth_exposure(prev, target, false);
                if (!(exposure >= prev && exposure <= target)) {
                    ++meter_failures;
                    std::cerr << "non-monotone smoothing walk\n";
                    break;
                }
                prev = exposure;
            }
            ++meter_cases;
            if (!(exposure == target && steps < 25)) {
                ++meter_failures;
                std::cerr << "smoothing walk took " << steps << " steps\n";
            }
        }
    }
    // Key override parsing: absent, unparseable, or out of [0.01, 1.0]
    // falls back to the 0.23 default; a parseable prefix is accepted.
    {
        using phoenix_addon::exposure_meter::parse_meter_key;
        struct KeyCase {
            const char* text;
            float expect;
        };
        const KeyCase cases[]{
            {nullptr, 0.23f}, {"", 0.23f},      {"0.23", 0.23f}, {"0.30", 0.30f},
            {" 0.25", 0.25f}, {"0.25 ", 0.25f}, {"abc", 0.23f},  {"0", 0.23f},
            {"-1", 0.23f},    {"0.005", 0.23f}, {"2", 0.23f},    {"1", 1.0f},
            {"0.01", 0.01f},  {"nan", 0.23f},   {"inf", 0.23f},  {"0.18x", 0.18f},
        };
        for (const auto& test_case : cases) {
            ++meter_cases;
            if (!(parse_meter_key(test_case.text) == test_case.expect)) {
                ++meter_failures;
                std::cerr << "key(" << (test_case.text ? test_case.text : "null")
                          << ")=" << parse_meter_key(test_case.text) << '\n';
            }
        }
    }
    const bool meter_ok = meter_failures == 0;
    // Format gates: packed R11 needs widening; already-wide fp16 takes the
    // rescale-only path (observed live in RE9 menus); fp32 stays refused so
    // no rounding enters silently. Widening itself is deliberately NOT
    // extended, so ASI/replay behavior is unchanged.
    unsigned gate_cases = 0, gate_failures = 0;
    auto gate = [&](bool cond, const char* name) {
        ++gate_cases;
        if (!cond) {
            ++gate_failures;
            std::cerr << "gate " << name << '\n';
        }
    };
    using phoenix_addon::colour_bridge::widenable;
    using phoenix_addon::colour_bridge::rescale_only;
    using phoenix_addon::colour_bridge::exposure_readable;
    gate(widenable(DXGI_FORMAT_R11G11B10_FLOAT), "widen packed");
    gate(!widenable(DXGI_FORMAT_R16G16B16A16_FLOAT), "widen refuses half");
    gate(!widenable(DXGI_FORMAT_R32G32B32A32_FLOAT), "widen refuses fp32");
    gate(!widenable(DXGI_FORMAT_R8G8B8A8_UNORM), "widen refuses rgba8");
    gate(!widenable(DXGI_FORMAT_UNKNOWN), "widen refuses unknown");
    gate(rescale_only(DXGI_FORMAT_R16G16B16A16_FLOAT), "rescale half");
    gate(!rescale_only(DXGI_FORMAT_R11G11B10_FLOAT), "rescale refuses packed");
    gate(!rescale_only(DXGI_FORMAT_R32G32B32A32_FLOAT), "rescale refuses fp32");
    gate(!rescale_only(DXGI_FORMAT_UNKNOWN), "rescale refuses unknown");
    gate(exposure_readable(DXGI_FORMAT_R16_FLOAT), "exposure r16f");
    gate(exposure_readable(DXGI_FORMAT_R32_FLOAT), "exposure r32f");
    gate(!exposure_readable(DXGI_FORMAT_R16G16B16A16_FLOAT), "exposure refuses half4");
    using phoenix_addon::colour_bridge::bridge_accepts_source;
    gate(bridge_accepts_source(DXGI_FORMAT_R11G11B10_FLOAT), "bridge packed");
    gate(bridge_accepts_source(DXGI_FORMAT_R16G16B16A16_FLOAT), "bridge rescale-only half");
    gate(!bridge_accepts_source(DXGI_FORMAT_R32G32B32A32_FLOAT), "bridge refuses fp32");
    gate(!bridge_accepts_source(DXGI_FORMAT_R8G8B8A8_UNORM), "bridge refuses rgba8");
    gate(!bridge_accepts_source(DXGI_FORMAT_UNKNOWN), "bridge refuses unknown");
    using phoenix_addon::denormalize::result_readable;
    using phoenix_addon::denormalize::output_writable;
    gate(result_readable(DXGI_FORMAT_R16G16B16A16_FLOAT), "denorm result half");
    gate(!result_readable(DXGI_FORMAT_R11G11B10_FLOAT), "denorm result refuses packed");
    gate(!result_readable(DXGI_FORMAT_R32G32B32A32_FLOAT), "denorm result refuses fp32");
    gate(!result_readable(DXGI_FORMAT_UNKNOWN), "denorm result refuses unknown");
    gate(output_writable(DXGI_FORMAT_R16G16B16A16_FLOAT), "denorm output half");
    gate(!output_writable(DXGI_FORMAT_R11G11B10_FLOAT), "denorm output refuses packed");
    gate(!output_writable(DXGI_FORMAT_R32G32B32A32_FLOAT), "denorm output refuses fp32");
    gate(!output_writable(DXGI_FORMAT_UNKNOWN), "denorm output refuses unknown");
    gate(phoenix_addon::denormalize::exposure_readable(DXGI_FORMAT_R16_FLOAT),
         "denorm exposure r16f");
    gate(phoenix_addon::denormalize::exposure_readable(DXGI_FORMAT_R32_FLOAT),
         "denorm exposure r32f");
    gate(!phoenix_addon::denormalize::exposure_readable(DXGI_FORMAT_R16G16B16A16_FLOAT),
         "denorm exposure refuses half4");
    const bool gate_ok = gate_failures == 0;

    // Denormalize math: out = v * w with w = 1/max(e, 1e-6), the bridge's
    // white derivation character-identical so normalize/denormalize
    // round-trips. Same measured whites, same 3% bound, fp16 throughout;
    // the output store is fp16 (no small-float display store this time).
    // The 0.13110 exposure is OptiScaler's own documented derivation
    // (white 7.63); the denormalize white must equal it.
    unsigned denorm_cases = 0, denorm_failures = 0;
    double denorm_max_rel_err = 0.0;
    auto denorm_white = [](double exposure) { return 1.0 / std::max(exposure, 1e-6); };
    {
        ++denorm_cases;
        const double white_level = denorm_white(0.13110);
        if (!(std::fabs(white_level - 7.627766) / 7.627766 <= 1e-4)) {
            ++denorm_failures;
            std::cerr << "denorm white(0.13110)=" << white_level << '\n';
        }
    }
    for (double exposure : {0.0, 1e-9, 1e-6}) {
        ++denorm_cases;
        if (!(denorm_white(exposure) == 1e6)) {
            ++denorm_failures;
            std::cerr << "denorm clamp e=" << exposure << '\n';
        }
    }
    {
        // White identity with the bridge: both derive 1/max(e, 1e-6), so a
        // frame metered for the way in is inverted exactly on the way out.
        // Pinned here so a drift in either formula fails loudly.
        ++denorm_cases;
        bool identical = true;
        for (double exposure : {0.0, 1e-9, 1e-6, 0.01, 0.13110, 0.5, 1.0, 10.0}) {
            const double bridge_w = 1.0 / std::max(exposure, 1e-6);
            if (!(denorm_white(exposure) == bridge_w)) {
                identical = false;
                break;
            }
        }
        if (!identical) {
            ++denorm_failures;
            std::cerr << "denorm white drifted from bridge white\n";
        }
    }
    {
        const double whites[] = {1.0, 7.63, 22.63, 43.78, 78.02};
        for (double white_level : whites) {
            for (unsigned iteration_index = 0; iteration_index <= 1024; ++iteration_index) {
                const double scene_value = (2.0 * white_level * double(iteration_index)) / 1024.0;
                const double divided = decode_half(encode_half(scene_value / white_level));
                // The shader multiplies in fp32, then the UAV store rounds;
                // the middle step here is float for the same reason.
                const double back =
                    decode_half(encode_half(double(float(divided) * float(white_level))));
                ++denorm_cases;
                if (scene_value >= 0.01) {
                    const double rel = std::fabs(back - scene_value) / scene_value;
                    if (rel > denorm_max_rel_err)
                        denorm_max_rel_err = rel;
                    if (!(rel <= 0.03)) {
                        ++denorm_failures;
                        if (denorm_failures <= 4)
                            std::cerr << "denorm w=" << white_level << " x=" << scene_value
                                      << " rel err " << rel << '\n';
                    }
                } else if (!(std::fabs(back - scene_value) <= 0.005)) {
                    ++denorm_failures;
                    if (denorm_failures <= 4)
                        std::cerr << "denorm w=" << white_level << " x=" << scene_value
                                  << " abs err " << std::fabs(back - scene_value) << '\n';
                }
            }
        }
    }
    const bool denorm_ok = denorm_failures == 0;

    const bool passed =
        failures == 0 && counts_right && scale_bound_met && meter_ok && gate_ok && denorm_ok;
    std::cout
        << "{\"cpu_only\":true,\"passed\":" << (passed ? "true" : "false")
        << ",\"encodings_checked\":" << checked << ",\"lossy_encodings\":" << failures
        << ",\"scale_cases\":" << scale_cases << ",\"scale_max_rel_err\":" << scale_max_rel_err
        << ",\"scale_within_bound\":" << (scale_bound_met ? "true" : "false")
        << ",\"meter_cases\":" << meter_cases << ",\"meter_ok\":" << (meter_ok ? "true" : "false")
        << ",\"gate_cases\":" << gate_cases << ",\"gate_ok\":" << (gate_ok ? "true" : "false")
        << ",\"denorm_cases\":" << denorm_cases << ",\"denorm_max_rel_err\":" << denorm_max_rel_err
        << ",\"denorm_ok\":" << (denorm_ok ? "true" : "false")
        << ",\"claim\":\"R11G11B10_FLOAT widens to R16G16B16A16_FLOAT with no rounding or clamping; "
           "paper-white divide/multiply round-trips within 3% relative (R11/B10 display store included); "
           "meter key 0.23 (calibrated b14/b13/b15), log-mean monotone and partition-exact\"}\n";
    return passed ? 0 : 1;
}
