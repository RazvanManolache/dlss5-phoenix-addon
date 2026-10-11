#include "../../native_runtime/scratch_plan.h"
#include "../../native_runtime/command_cache.h"
#include "../../native_runtime/parallel_rows.h"
#include <iostream>

using namespace phoenix_cpp;
int main() {
    try {
        ScratchPlan plan;
        auto first_allocation = plan.reserve(4096), small_allocation = plan.reserve(1024),
             last_allocation = plan.reserve(4096);
        plan.reserve(8192); // Unreferenced temporary must consume no arena space.
        plan.touch(first_allocation.address, 0);
        plan.touch(first_allocation.address, 1);
        plan.touch(small_allocation.address, 1);
        plan.touch(small_allocation.address, 3);
        plan.touch(last_allocation.address, 2);
        plan.touch(last_allocation.address, 3);
        demand(plan.layout() == 5120, "Scratch arena high-water extent");
        demand(plan.offset(first_allocation.address) == plan.offset(last_allocation.address),
               "Disjoint lifetimes should reuse storage");
        demand(plan.offset(small_allocation.address) != plan.offset(first_allocation.address),
               "Same-command input and output must stay disjoint");
        Allocation arena{reinterpret_cast<void*>(0x100000), 5120, 1};
        auto view = reinterpret_cast<void*>(uintptr_t(first_allocation.address) + 256);
        demand(plan.relocate(view, arena) == reinterpret_cast<void*>(0x100100),
               "Tensor subview relocation");
        demand(plan.relocate(nullptr, arena) == nullptr, "Null optional operands");
        ScratchPlan growing;
        auto small = growing.reserve(4096), large = growing.reserve(8192);
        growing.touch(small.address, 0);
        growing.touch(large.address, 1);
        demand(growing.layout() == 8192 && growing.offset(large.address) == 0,
               "A larger tensor should grow a freed tail range");

        std::vector<Command> original(1);
        original[0].operation.scale = 1;
        auto copy = original;
        demand(same_commands(original, copy), "Equivalent vectors must share a cached graph");
        copy[0].operation.secondary_output = reinterpret_cast<void*>(64);
        demand(!same_commands(original, copy), "Second output invalidation");
        copy = original;
        copy[0].operation.left_row_stride = 32;
        demand(!same_commands(original, copy), "Subview stride invalidation");
        copy = original;
        copy[0].operation.scale = -0.f;
        original[0].operation.scale = 0.f;
        demand(!same_commands(original, copy), "Floating-point bit invalidation");
        copy = original;
        copy[0].count = 256;
        demand(!same_commands(original, copy), "Dispatch count invalidation");

        RowExecutor rows;
        std::vector<unsigned> output(513);
        for (unsigned pass = 0; pass < 12; ++pass) {
            rows.run(unsigned(output.size()), [&](unsigned row) { output[row] = row * 17 + pass; });
            for (unsigned row = 0; row < output.size(); ++row)
                demand(output[row] == row * 17 + pass, "Parallel row publication");
        }
        bool caught = false;
        try {
            rows.run(513, [](unsigned row) {
                if (row == 63)
                    throw std::runtime_error("row failure");
            });
        } catch (const std::runtime_error&) {
            caught = true;
        }
        demand(caught, "Parallel exception propagation");
        rows.run(513, [&](unsigned row) { output[row] = row; });
        for (unsigned row = 0; row < output.size(); ++row)
            demand(output[row] == row, "Parallel pool recovery");
        std::cout
            << "Scratch lifetimes/subviews, command cache invalidation, persistent row pool and exception recovery passed\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
