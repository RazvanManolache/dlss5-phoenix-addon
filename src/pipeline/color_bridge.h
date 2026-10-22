#pragma once
namespace phoenix_pipeline {
inline constexpr const char* hdr_bridge_shader = R"(
// Reversible BT.709 gamut and magnitude compression. Recompute the
// inverse scale from the same original pixel to preserve scene exposure.

float SanitizeFinite(float value, float fallback) {
    return ((asuint(value) & 0x7F800000u) == 0x7F800000u) ? fallback : value;
}
float3 SanitizeFinite3(float3 value, float3 fallback) {
    return float3(SanitizeFinite(value.x, fallback.x), SanitizeFinite(value.y, fallback.y),
                  SanitizeFinite(value.z, fallback.z));
}
float SafeDivide(float numerator, float denominator, float fallback) {
    return abs(denominator) > 1e-8 ? numerator / denominator : fallback;
}
float3 BT709ToLMS(float3 color) {
    const float3x3 transform_matrix = {0.30569589, 0.62271286, 0.04528636, 0.15776262, 0.76968599,
                                       0.08807030, 0.01933082, 0.11919478, 0.95053215};
    return mul(transform_matrix, color);
}
float3 LMSToBT709(float3 color) {
    const float3x3 transform_matrix = {5.62059812,  -4.57145756, 0.15577924,
                                       -1.15555585, 2.25800438,  -0.15415806,
                                       0.03059913,  -0.19018011, 1.06820532};
    return mul(transform_matrix, color);
}
float LuminanceBT709(float3 color) {
    return dot(color, float3(.21263901, .71516868, .07219232));
}
float3 D65NeutralBT709(float3 lms, float luminance) {
    float3 d65 = LMSToBT709(max(lms, 1e-8));
    return d65 * (luminance / max(LuminanceBT709(d65), 1e-8));
}
float GamutScaleBT709(float3 source_rgb, float3 lms) {
    float3 color = SanitizeFinite3(source_rgb, 0.0);
    float luminance = LuminanceBT709(color);
    float gamut_scale = 1.0;
    if (luminance > 1e-8) {
        float3 neutral_rgb = D65NeutralBT709(lms, luminance);
        if (color.r < 0.0 && neutral_rgb.r > color.r)
            gamut_scale = min(gamut_scale, SafeDivide(neutral_rgb.r, neutral_rgb.r - color.r, 1.0));
        if (color.g < 0.0 && neutral_rgb.g > color.g)
            gamut_scale = min(gamut_scale, SafeDivide(neutral_rgb.g, neutral_rgb.g - color.g, 1.0));
        if (color.b < 0.0 && neutral_rgb.b > color.b)
            gamut_scale = min(gamut_scale, SafeDivide(neutral_rgb.b, neutral_rgb.b - color.b, 1.0));
        gamut_scale = saturate(SanitizeFinite(gamut_scale, 1.0));
    }
    return gamut_scale;
}
float3 GamutCompressBT709(float3 source_rgb, float3 lms, float gamut_scale) {
    float3 color = SanitizeFinite3(source_rgb, 0.0);
    float luminance = LuminanceBT709(color);
    float3 outc = color;
    if (luminance > 1e-8) {
        float3 neutral_rgb = D65NeutralBT709(lms, luminance);
        outc = SanitizeFinite3(neutral_rgb + (color - neutral_rgb) * saturate(gamut_scale),
                               max(neutral_rgb, 0.0));
    }
    return outc;
}
float3 GamutDecompressBT709(float3 source_rgb, float3 lms, float gamut_scale) {
    float3 color = SanitizeFinite3(source_rgb, 0.0);
    float luminance = LuminanceBT709(color);
    float3 outc = color;
    if (luminance > 1e-8) {
        float3 neutral_rgb = D65NeutralBT709(lms, luminance);
        float safe_scale = max(gamut_scale, 1e-6);
        outc = SanitizeFinite3(neutral_rgb + (color - neutral_rgb) / safe_scale, color);
    }
    return outc;
}
float Neutwo(float magnitude) {
    return magnitude * rsqrt(magnitude * magnitude + 1.0);
}
float NeutwoScale(float3 color) {
    float maximum_channel = max(max(abs(color.r), abs(color.g)), abs(color.b));
    float outv = 1.0;
    if (maximum_channel > 1e-8)
        outv = SanitizeFinite(Neutwo(maximum_channel) / maximum_channel, 1.0);
    return outv;
}
float SrgbEncodeSafe(float value) {
    value = saturate(SanitizeFinite(value, 0.0));
    return value <= 0.0031308 ? value * 12.92 : 1.055 * pow(value, 1.0 / 2.4) - 0.055;
}
float3 SrgbEncodeSafe3(float3 color) {
    return float3(SrgbEncodeSafe(color.r), SrgbEncodeSafe(color.g), SrgbEncodeSafe(color.b));
}
float SrgbDecodeSafe(float value) {
    value = saturate(SanitizeFinite(value, 0.0));
    return value <= 0.04045 ? value / 12.92 : pow((value + 0.055) / 1.055, 2.4);
}
float3 SrgbDecodeSafe3(float3 color) {
    return float3(SrgbDecodeSafe(color.r), SrgbDecodeSafe(color.g), SrgbDecodeSafe(color.b));
}
float3 ReferenceEncode(float3 original) {
    float3 src = SanitizeFinite3(original, 0.0);
    float3 lms = BT709ToLMS(float3(.18, .18, .18));
    float gamut_scale = GamutScaleBT709(src, lms);
    float3 comp = GamutCompressBT709(src, lms, gamut_scale);
    return SrgbEncodeSafe3(max(comp * NeutwoScale(comp), 0.0));
}
float3 ReferenceDecode(float3 original, float3 neural) {
    float3 lms = BT709ToLMS(float3(.18, .18, .18));
    float gamut_scale = GamutScaleBT709(original, lms);
    float magnitude_scale = NeutwoScale(GamutCompressBT709(original, lms, gamut_scale));
    float3 graded = SrgbDecodeSafe3(neural);
    return GamutDecompressBT709((abs(magnitude_scale) > 1e-8) ? graded / magnitude_scale : graded,
                                lms, gamut_scale);
}
)";
} // namespace phoenix_pipeline
