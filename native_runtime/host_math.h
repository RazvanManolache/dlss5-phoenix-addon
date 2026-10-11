#pragma once
#include <bit>
#include <cmath>
struct HostMath {
    static uint32_t bits(float value_or_bits) {
        return std::bit_cast<uint32_t>(value_or_bits);
    }
    static float value(uint32_t value_or_bits) {
        return std::bit_cast<float>(value_or_bits);
    }
    static float fma(float left, float right, float addend) {
        return std::fma(left, right, addend);
    }
    static float sqrt(float value_or_bits) {
        return std::sqrt(value_or_bits);
    }
    static float abs(float value_or_bits) {
        return std::fabs(value_or_bits);
    }
    static float min(float left, float right) {
        return std::fmin(left, right);
    }
    static float max(float left, float right) {
        return std::fmax(left, right);
    }
    static float even(float value_or_bits) {
        return std::nearbyint(value_or_bits);
    }
    static bool inf(float value_or_bits) {
        return std::isinf(value_or_bits);
    }
};
