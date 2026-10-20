#pragma once
#include "../../native/support/processing.h"
#include <array>
#include <vector>
#include <stdexcept>
namespace phoenix_compat::pixels {
using phoenix_addon::phoenix::Processing;
using RGB = std::array<float, 3>;
inline float srgb(float v, bool inverse) {
    const float a = std::abs(v);
    return std::copysign(
        inverse ? (a <= .04045f ? a / 12.92f : std::pow((a + .055f) / 1.055f, 2.4f))
                : (a <= .0031308f ? a * 12.92f : 1.055f * std::pow(a, 1 / 2.4f) - .055f),
        v);
}
inline float pq(float v, bool inverse) {
    if (v == 0)
        return 0; // Signed PQ has exact black, matching the GPU codec.
    const float a = std::pow(std::min(std::abs(v), 1.f), inverse ? 1 / 78.84375f : .1593017578125f);
    return std::copysign(
        inverse
            ? std::pow(std::max(a - .8359375f, 0.f) / std::max(18.8515625f - 18.6875f * a, 1e-6f),
                       1 / .1593017578125f)
            : std::pow((.8359375f + 18.8515625f * a) / (1 + 18.6875f * a), 78.84375f),
        v);
}
inline RGB gamut(RGB v, bool to709) {
    constexpr float forward[9] = {1.660491f, -.587641f, -.072850f, -.124550f, 1.132900f,
                                  -.008349f, -.018151f, -.100579f, 1.118730f};
    constexpr float reverse[9] = {.627404f, .329283f, .043313f, .069097f, .919540f,
                                  .011362f, .016391f, .088013f, .895595f};
    const auto* m = to709 ? forward : reverse;
    return {m[0] * v[0] + m[1] * v[1] + m[2] * v[2], m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2]};
}
inline RGB transfer(RGB v, int encoding, float white, bool inverse) {
    if (!inverse && encoding == 3)
        v = gamut(v, false);
    for (auto& c : v) {
        if (encoding == 2)
            c = srgb(c, inverse);
        if (encoding == 3)
            c = inverse ? pq(c, true) * 10000 / white : pq(c * white / 10000, false);
        if (encoding == 4)
            c *= inverse ? 80 / white : white / 80;
        if (encoding == 5)
            c = inverse ? srgb(c, true) * 80 / white : srgb(c * white / 80, false);
    }
    return inverse && encoding == 3 ? gamut(v, true) : v;
}
inline RGB domain(RGB v, const Processing& p, bool inverse) {
    for (auto& c : v) {
        const float white = p.diffuse_white();
        if (p.domain == 0)
            c = inverse ? srgb(c, true) * 80 / white : srgb(c * white / 80, false);
        if (p.domain == 1)
            c = inverse ? pq(c, true) * 10000 / white : pq(c * white / 10000, false);
    }
    return v;
}
inline RGB signal(RGB v, const Processing& p, bool inverse) {
    for (auto& c : v) {
        if (p.signal == 0)
            c = srgb(c, inverse);
        if (p.signal == 1)
            c = inverse ? pq(c, true) * 10000 / p.diffuse_white()
                        : pq(c * p.diffuse_white() / 10000, false);
    }
    return v;
}
inline RGB load(const std::vector<float>& image, unsigned width, unsigned height, int x, int y) {
    const auto i =
        (size_t(std::clamp(y, 0, int(height) - 1)) * width + std::clamp(x, 0, int(width) - 1)) * 3;
    return {image[i], image[i + 1], image[i + 2]};
}
inline float cubic(float x) {
    x = std::abs(x);
    return x < 1   ? 1.5f * x * x * x - 2.5f * x * x + 1
           : x < 2 ? -.5f * x * x * x + 2.5f * x * x - 4 * x + 2
                   : 0;
}
inline std::vector<float> resample(const std::vector<float>& source, unsigned sw, unsigned sh,
                                   unsigned dw, unsigned dh, int filter, bool down) {
    if (!sw || !sh || !dw || !dh || source.size() != size_t(sw) * sh * 3)
        throw std::runtime_error("Invalid processing image");
    std::vector<float> result(size_t(dw) * dh * 3);
    for (unsigned y = 0; y < dh; ++y)
        for (unsigned x = 0; x < dw; ++x) {
            const float fx = float(sw) / dw, fy = float(sh) / dh;
            const float px = (x + .5f) * fx - .5f, py = (y + .5f) * fy - .5f;
            RGB sum{};
            float total = 0;
            auto tap = [&](int tx, int ty, float weight) {
                const auto v = load(source, sw, sh, tx, ty);
                for (unsigned c = 0; c < 3; ++c)
                    sum[c] += v[c] * weight;
                total += weight;
            };
            if (filter == 2)
                tap(int(std::floor(px + .5f)), int(std::floor(py + .5f)), 1);
            else if (down && filter == 0 && (fx > 1 || fy > 1)) {
                float l = px + .5f - fx * .5f, r = l + fx, t = py + .5f - fy * .5f, b = t + fy;
                for (int ty = int(std::floor(t)); ty < int(std::ceil(b)); ++ty)
                    for (int tx = int(std::floor(l)); tx < int(std::ceil(r)); ++tx)
                        tap(tx, ty,
                            std::max(0.f, std::min(float(tx + 1), r) - std::max(float(tx), l)) *
                                std::max(0.f, std::min(float(ty + 1), b) - std::max(float(ty), t)));
            } else if (!down && filter == 1) {
                const int bx = int(std::floor(px)), by = int(std::floor(py));
                for (int ty = -1; ty <= 2; ++ty)
                    for (int tx = -1; tx <= 2; ++tx)
                        tap(bx + tx, by + ty, cubic(px - bx - tx) * cubic(py - by - ty));
            } else {
                const int bx = int(std::floor(px)), by = int(std::floor(py));
                const float dx = px - bx, dy = py - by;
                tap(bx, by, (1 - dx) * (1 - dy));
                tap(bx + 1, by, dx * (1 - dy));
                tap(bx, by + 1, (1 - dx) * dy);
                tap(bx + 1, by + 1, dx * dy);
            }
            for (unsigned c = 0; c < 3; ++c)
                result[(size_t(y) * dw + x) * 3 + c] = sum[c] / std::max(total, 1e-8f);
        }
    return result;
}
template <class F> inline void map(std::vector<float>& image, F f) {
    for (size_t i = 0; i < image.size(); i += 3) {
        const auto v = f(RGB{image[i], image[i + 1], image[i + 2]});
        for (unsigned c = 0; c < 3; ++c)
            image[i + c] = v[c];
    }
}
struct Frame {
    Processing options;
    unsigned width, height, working_width, working_height;
    std::vector<float> original, baseline, encoded;
    Frame(const std::vector<float>& source, unsigned w, unsigned h, Processing p,
          bool native = false)
        : options(p), width(w), height(h), original(source) {
        const auto g = phoenix_addon::phoenix::processing_resolution(w, h, p, native);
        // The portable canvas contains the same aspect ratio; use exact scaled
        // image dimensions and no opaque letterbox pixels in its inference.
        working_width = g.image_width;
        working_height = g.image_height;
        auto linear = source;
        map(linear, [&](RGB v) {
            return domain(transfer(v, p.input_encoding(true), p.diffuse_white(), true), p, false);
        });
        baseline = resample(linear, w, h, working_width, working_height, p.downsample, true);
        map(baseline, [&](RGB v) { return domain(v, p, true); });
        encoded = baseline;
        map(encoded, [&](RGB v) {
            for (auto& c : v)
                c *= p.effective_exposure();
            return signal(v, p, false);
        });
    }
    std::vector<float> finish(std::vector<float> result) const {
        if (result.size() != baseline.size())
            throw std::runtime_error("Processing output extent mismatch");
        const auto& p = options;
        map(result, [&](RGB v) {
            v = signal(v, p, true);
            for (auto& c : v)
                c /= std::max(p.effective_exposure(), 1e-8f);
            return v;
        });
        if (p.effective_exposure() == 0)
            return original;
        if (p.reconstruction == 0)
            for (size_t i = 0; i < result.size(); ++i)
                result[i] -= baseline[i];
        else
            map(result, [&](RGB v) { return domain(v, p, false); });
        result = resample(result, working_width, working_height, width, height, p.upsample, false);
        if (p.reconstruction == 0) {
            auto decoded = original;
            map(decoded, [&](RGB v) {
                return transfer(v, p.input_encoding(true), p.diffuse_white(), true);
            });
            for (size_t i = 0; i < result.size(); ++i)
                result[i] += decoded[i];
        } else
            map(result, [&](RGB v) { return domain(v, p, true); });
        map(result,
            [&](RGB v) { return transfer(v, p.input_encoding(true), p.diffuse_white(), false); });
        return result;
    }
};
} // namespace phoenix_compat::pixels
