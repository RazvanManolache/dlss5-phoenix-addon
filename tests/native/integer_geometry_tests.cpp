#include "../../src/native/kernels/production/common/integer_geometry.h"
#include <iostream>
#include <stdexcept>

static unsigned checks = 0;
static void check(bool condition) {
    ++checks;
    if (!condition) throw std::runtime_error("Integer geometry differs from independent division");
}
static void division(uint32_t numerator, uint32_t divisor) {
    const auto prepared = make_word_divisor(divisor);
    const uint32_t expected_unsigned = divisor ? uint32_t(uint64_t(numerator) / divisor) : UINT32_MAX;
    const uint32_t expected_signed = divisor
        ? uint32_t(int64_t(int32_t(numerator)) / int64_t(int32_t(divisor))) : UINT32_MAX;
    check(exact_word_quotient(numerator, prepared) == expected_unsigned);
    check(exact_signed_word_quotient(numerator, divisor) == expected_signed);
}
template <bool Signed> static void tail(uint32_t next, uint32_t end, uint32_t stride) {
    const bool has_next = Signed ? int64_t(int32_t(next)) < int64_t(int32_t(end))
                                : uint64_t(next) < uint64_t(end);
    // Derive a ceiling quotient in 64 bits, rather than repeating production's
    // reciprocal or remaining-minus-one calculation. Differences may wrap in
    // the signed-index ABI; lift that modulo-32-bit distance before dividing.
    const uint64_t distance = has_next ? uint32_t(end - next) : 0u;
    const uint32_t expected = stride
        ? uint32_t((distance + uint64_t(stride) - 1u) / stride)
        : uint32_t(has_next) + UINT32_MAX;
    check(geometry_tail_steps<Signed>(next, end, stride) == expected);
}
int main() {
    try {
        constexpr uint32_t edges[] = {0, 1, 2, 3, 7, 31, 32, 127, 128, 129,
            65535, 65536, 0x3fffffffu, 0x40000000u, 0x7fffffffu, 0x80000000u,
            0x80000001u, 0xfffffffeu, 0xffffffffu};
        for (auto numerator : edges)
            for (auto divisor : edges) {
                division(numerator, divisor);
                for (auto end : edges) {
                    tail<true>(numerator, end, divisor);
                    tail<false>(numerator, end, divisor);
                }
            }
        uint32_t random = 0x4a85c9f1u;
        auto next = [&] { random = random * 1664525u + 1013904223u; return random; };
        for (unsigned i = 0; i < 100'000; ++i) {
            const uint32_t numerator = next(), divisor = next(), end = next();
            division(numerator, divisor);
            tail<true>(numerator, end, divisor);
            tail<false>(numerator, end, divisor);
        }
        // Quotient boundaries near exact multiples catch one-step errors which
        // uniform random large divisors otherwise rarely exercise.
        for (auto divisor : edges) if (divisor)
            for (uint64_t quotient : {1ull, 2ull, 31ull, 65535ull, 0x7fffffffull}) {
                const uint64_t multiple = quotient * divisor;
                if (multiple <= UINT32_MAX) {
                    division(uint32_t(multiple), divisor);
                    division(uint32_t(multiple - 1), divisor);
                    if (multiple < UINT32_MAX) division(uint32_t(multiple + 1), divisor);
                }
            }
        std::cout << "{\"passed\":true,\"integer_checks\":" << checks << ",\"gpu_used\":false}\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n'; return 1;
    }
}
