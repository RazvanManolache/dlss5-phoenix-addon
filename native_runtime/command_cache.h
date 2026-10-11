#pragma once
#include "backend.h"
#include <bit>

namespace phoenix_cpp {
struct CommandSpan {
    size_t begin, end;
    uint32_t block, layer;
};
inline std::vector<CommandSpan> command_spans(const std::vector<Command>& commands) {
    std::vector<CommandSpan> result;
    for (size_t iteration_index = 0; iteration_index < commands.size(); ++iteration_index) {
        const auto& command = commands[iteration_index];
        if (result.empty() || result.back().block != command.block ||
            result.back().layer != command.layer)
            result.push_back({iteration_index, iteration_index + 1, command.block, command.layer});
        else
            result.back().end = iteration_index + 1;
    }
    return result;
}
// Compare values, never struct padding or the address of a command vector.
// Captured GPU graphs are reusable only when every operand, stride, flag
// and scalar bit pattern matches. Pointer identity of the vector is irrelevant.
inline bool same_commands(const std::vector<Command>& left_commands,
                          const std::vector<Command>& right_commands) {
    if (left_commands.size() != right_commands.size())
        return false;
    for (size_t index = 0; index < left_commands.size(); ++index) {
        const auto& left_command = left_commands[index];
        const auto& right_command = right_commands[index];
        const auto& left_operation = left_command.operation;
        const auto& right_operation = right_command.operation;
        if (left_command.count != right_command.count ||
            left_command.block != right_command.block ||
            left_command.layer != right_command.layer ||
            left_operation.left_input != right_operation.left_input ||
            left_operation.right_input != right_operation.right_input ||
            left_operation.output != right_operation.output ||
            left_operation.auxiliary != right_operation.auxiliary ||
            left_operation.secondary_output != right_operation.secondary_output ||
            left_operation.family != right_operation.family ||
            left_operation.flags != right_operation.flags ||
            left_operation.row_count != right_operation.row_count ||
            left_operation.column_count != right_operation.column_count ||
            left_operation.reduction_count != right_operation.reduction_count ||
            left_operation.batch_count != right_operation.batch_count ||
            left_operation.left_batch_stride != right_operation.left_batch_stride ||
            left_operation.right_batch_stride != right_operation.right_batch_stride ||
            left_operation.output_batch_stride != right_operation.output_batch_stride ||
            left_operation.left_row_stride != right_operation.left_row_stride ||
            left_operation.right_row_stride != right_operation.right_row_stride ||
            left_operation.output_row_stride != right_operation.output_row_stride ||
            std::bit_cast<uint32_t>(left_operation.scale) !=
                std::bit_cast<uint32_t>(right_operation.scale))
            return false;
    }
    return true;
}
}
