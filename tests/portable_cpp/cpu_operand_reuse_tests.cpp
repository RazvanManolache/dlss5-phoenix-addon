// Differential CPU tile tests and paired timings against a pinned pre-change
// implementation. This exercises real operators, including padded/tail batches.
#include "../../native_runtime/backend.h"
#include "../../native_runtime/host_math.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <cstring>

using phoenix_cpp::Command;
using Tile = void (*)(const Command&, uint64_t);
extern "C" void phoenix_cpu_avx2(const Command&, uint64_t);
extern "C" void phoenix_cpu_avx512(const Command&, uint64_t);
extern "C" void control_avx2(const Command&, uint64_t);
extern "C" void control_avx512(const Command&, uint64_t);

int main() {
    phx::Kernel<HostMath> converter{};
    std::mt19937 random(90210);
    unsigned cases = 0;
    // Diverse layouts expose transposition, odd half-pair strides, incomplete
    // vector tiles and output padding. Sentinels detect writes outside outputs.
    for (unsigned width : {7u, 8u, 9u, 32u, 128u, 512u})
        for (unsigned height : {1u, 7u, 8u, 17u, 32u})
            for (unsigned flags : {0u, 1u, unsigned(phx::RightInputFp8), unsigned(phx::RightInputFp8) | 1u}) {
                const unsigned columns = width == 512 ? 128u : 13u;
                const unsigned left_stride = width + (height & 1u);
                const unsigned right_stride = (flags & 1u ? width : columns) + 1u;
                const unsigned right_rows = flags & 1u ? columns : width;
                std::vector<uint16_t> left(2ull * height * left_stride),
                    right(2ull * right_rows * right_stride),
                    expected(2ull * height * (columns + 3), 0x3555), actual = expected;
                for (auto& value : left)
                    value = uint16_t(converter.half_bits(float(int(random() % 129) - 64) / 32));
                for (auto& value : right)
                    value = flags & phx::RightInputFp8 ? uint16_t(random() % 256)
                         : uint16_t(converter.half_bits(float(int(random() % 97) - 48) / 64));
                Command command{};
                auto& op = command.operation;
                op.family = phx::MatrixMultiply;
                op.flags = flags | phx::StoreHalf;
                op.left_input = left.data(); op.right_input = right.data();
                op.row_count = height; op.column_count = columns; op.reduction_count = width;
                op.batch_count = 2;
                op.left_row_stride = left_stride; op.right_row_stride = right_stride;
                op.output_row_stride = columns + 3;
                op.left_batch_stride = height * left_stride;
                op.right_batch_stride = right_rows * right_stride;
                op.output_batch_stride = height * (columns + 3);
                for (unsigned vector_rows : {8u, 16u}) {
                    auto control = vector_rows == 8 ? control_avx2 : control_avx512;
                    auto candidate = vector_rows == 8 ? phoenix_cpu_avx2 : phoenix_cpu_avx512;
                    std::fill(expected.begin(), expected.end(), 0x3555);
                    std::fill(actual.begin(), actual.end(), 0x3555);
                    const uint64_t tiles = 2ull * ((height + vector_rows - 1) / vector_rows) *
                                           ((columns + 7) / 8);
                    op.output = expected.data();
                    for (uint64_t tile = 0; tile < tiles; ++tile) control(command, tile);
                    op.output = actual.data();
                    for (uint64_t tile = 0; tile < tiles; ++tile) candidate(command, tile);
                    if (actual != expected) throw std::runtime_error("CPU operand reuse changed output");
                    ++cases;
                }
            }
    std::cout << "exact_layout_cases=" << cases << '\n';
    for (unsigned vector_rows : {8u, 16u})
        for (unsigned reduction : {32u, 128u, 512u, 1024u}) {
            const unsigned rows = 256, columns = 128;
            std::vector<uint16_t> left(rows * reduction, 0x3800), output(rows * columns);
            std::vector<uint8_t> right(columns * reduction);
            for (auto& value : right) value = uint8_t(32 + random() % 48);
            Command command{};
            auto& op = command.operation;
            op.family = phx::MatrixMultiply;
            op.flags = 1 | phx::RightInputFp8 | phx::StoreHalf;
            op.left_input = left.data(); op.right_input = right.data(); op.output = output.data();
            op.row_count = rows; op.column_count = columns; op.reduction_count = reduction;
            op.left_row_stride = op.right_row_stride = reduction;
            op.output_row_stride = columns;
            const uint64_t tiles = uint64_t(rows / vector_rows) * (columns / 8);
            auto measure = [&](Tile function) {
                const auto started = std::chrono::steady_clock::now();
                unsigned repeats = 0;
                double elapsed;
                do {
                    for (uint64_t tile = 0; tile < tiles; ++tile) function(command, tile);
                    ++repeats;
                    elapsed = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - started).count();
                } while (elapsed < 40);
                return elapsed / repeats;
            };
            std::vector<double> controls, candidates;
            for (unsigned pair = 0; pair < 7; ++pair) {
                auto control = vector_rows == 8 ? control_avx2 : control_avx512;
                auto candidate = vector_rows == 8 ? phoenix_cpu_avx2 : phoenix_cpu_avx512;
                if (pair & 1) { candidates.push_back(measure(candidate)); controls.push_back(measure(control)); }
                else { controls.push_back(measure(control)); candidates.push_back(measure(candidate)); }
            }
            std::sort(controls.begin(), controls.end()); std::sort(candidates.begin(), candidates.end());
            std::cout << "rows_per_vector=" << vector_rows << " k=" << reduction
                      << " control_ms=" << controls[3] << " candidate_ms=" << candidates[3]
                      << " speedup=" << controls[3] / candidates[3] << '\n';
        }
}
