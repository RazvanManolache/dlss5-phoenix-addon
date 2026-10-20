#include "../../src/native/kernels/production/common/word_arithmetic.h"
#include "../../src/native/kernels/production/common/boundary_layout.h"
#include "../../src/native/model/window_kernel_abi.h"
#include <iostream>
#include <stdexcept>

static unsigned checks = 0;
static void check(bool valid) {
    ++checks;
    if (!valid) throw std::runtime_error("Kernel word contract changed");
}
template <unsigned Shift> static void check_signed_truncation(uint32_t bits) {
    const uint32_t sign = uint32_t(int32_t(bits) >> 31);
    const uint32_t biased = uint32_t((uint64_t(sign) << Shift) >> 32) + bits;
    const auto old_bits = uint32_t(int32_t(biased) >> Shift);
    check(truncate_signed_power_of_two<Shift>(bits) == old_bits);
    check(truncate_signed_power_of_two<Shift>(bits) ==
          uint32_t(int64_t(int32_t(bits)) / (int64_t(1) << Shift)));
}

static void check_boundary_layout(BoundaryVectorLayout layout, uint32_t component,
                                  uint32_t row, uint32_t column) {
    // Independent byte-domain reference: scale each term separately, with no
    // calls to the address/split/combine helpers under test.
    uint64_t expected = layout.base_address;
    expected += uint64_t(int64_t(int32_t(component)) *
                         int64_t(int32_t(layout.component_stride_words))) * 4u;
    expected += uint64_t(int64_t(int32_t(row)) *
                         int64_t(int32_t(layout.row_stride_words))) * 4u;
    const uint32_t absolute_column = layout.column_origin + column;
    expected += uint64_t(int64_t(int32_t(absolute_column))) * 16u;
    check(boundary_vector_address(layout, component, row, column) == expected);
}
int main() {
    try {
        check(signed_word_magnitude(0) == 0);
        check(signed_word_magnitude(0xffffffffu) == 1);
        check(signed_word_magnitude(0x80000000u) == 0x80000000u);
        uint32_t low = 0, high = 0;
        split_address_words(low, high, 0xfedcba9876543210ull);
        check(low == 0x76543210u && high == 0xfedcba98u);
        check(combine_words(low, high) == 0xfedcba9876543210ull);
        check(scaled_word_address(0, 0x3fffffffull) == 0xfffffffcull);
        check(scaled_word_address(0, 0x40000000ull) == 0x100000000ull);
        check(scaled_word_address(0, 0x40000001ull) == 0x100000004ull);
        check(scaled_word_address(0xfffffffcull, 0x40000001ull) == 0x200000000ull);
        check(scaled_word_address(0, 0x100000000ull) == 0x400000000ull);
        check(scaled_word_address(UINT64_MAX - 3, 1) == 0);
        check(boundary_vector_address({0, 0x40000000u, 0, 0}, 1, 0, 0) == 0x100000000ull);
        check(scaled_word_address(0x100000000ull, UINT64_MAX) == 0xfffffffcull);
        // Origin addition wraps in 32 bits before signed extension.
        check(boundary_vector_address({0, 0, 0, 0x7fffffffu}, 0, 0, 1) == 0xfffffff800000000ull);
        for (uint32_t bits : {0u, 1u, 0x3fffffffu, 0x40000000u, 0x7fffffffu, 0x80000000u, 0xffffffffu}) {
            check_boundary_layout({UINT64_MAX - 15u, bits, bits, bits}, bits, bits, bits);
        }
        using namespace phoenix_model::window_abi;
        check(compact.width == 28 && wide.width == 36);
        check(sizeof(Shift) == 8 && split_projection_shift == 64);
        uint32_t random = 0x94f2e175u;
        for (unsigned i = 0; i < 100'000; ++i) {
            random = random * 1664525u + 1013904223u;
            check_signed_truncation<0>(random);
            check_signed_truncation<1>(random);
            check_signed_truncation<2>(random);
            check_signed_truncation<5>(random);
            check_signed_truncation<7>(random);
            check_signed_truncation<30>(random);
            const uint32_t component = random & 63u;
            random = random * 1664525u + 1013904223u;
            const uint32_t row = random;
            random = random * 1664525u + 1013904223u;
            const uint32_t stride = random;
            random = random * 1664525u + 1013904223u;
            check_boundary_layout({combine_words(random, stride), stride, row, random},
                                  component, row, random);
        }
        for (uint32_t bits : {0u, 1u, 127u, 128u, 0x7fffffffu, 0x80000000u, 0xffffffffu}) {
            check_signed_truncation<2>(bits);
            check_signed_truncation<7>(bits);
        }
        std::cout << "{\"passed\":true,\"contracts\":" << checks << "}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
