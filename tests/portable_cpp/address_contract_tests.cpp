#include "../../native_runtime/scratch_plan.h"
#include "../../native_runtime/host_math.h"
#include "../../native_runtime/command_cache.h"
#include <type_traits>
#include <iostream>
using namespace phoenix_cpp;

int main() {
    try {
        static_assert(sizeof(phx::Operation) == 96);
        static_assert(offsetof(phx::Operation, family) == 40);
        static_assert(offsetof(phx::Operation, scale) == 88);
        static_assert(std::is_trivially_copyable_v<phx::Operation>);
        static_assert(std::is_standard_layout_v<phx::Operation>);
        uint16_t host[] = {0x3c00, 0x4000};
        phx::OperandAddress host_address = host;
        demand(static_cast<uint16_t*>(host_address) == host, "Host pointer roundtrip");
        phx::Operation op{};
        op.left_input = host_address;
        demand(phx::Kernel<HostMath>{op}.load_half(op.left_input, 1) == 2.f, "Host kernel pointer conversion");
        const uint64_t token = (uint64_t(7) << 40) + (uint64_t(3) << 30) + 128;
        auto wide = phx::address_from_bits(token);
        demand(phx::address_bits(wide) == token, "Wide device address roundtrip");
        demand((phx::address_bits(wide) >> 40) == 7, "Buffer ID preservation");
#if UINTPTR_MAX == UINT32_MAX
        bool pointer_rejected = false;
        try { (void)static_cast<void*>(wide); }
        catch (const std::overflow_error&) { pointer_rejected = true; }
        demand(pointer_rejected, "Wide device address silently converted to x86 host pointer");
#endif
        ScratchPlan plan;
        auto first = plan.reserve(uint64_t(6) << 30);
        auto second = plan.reserve(512);
        demand(phx::address_bits(first.address) == (uint64_t(1) << 62), "Scratch namespace");
        demand(first.address != second.address, "Scratch identities alias");
        auto subview = phx::address_from_bits(phx::address_bits(first.address) + (uint64_t(5) << 30));
        plan.touch(first.address, 0); plan.touch(subview, 1); plan.touch(second.address, 1);
        const auto bytes = plan.layout();
        demand(bytes == (uint64_t(6) << 30) + 512, "Inclusive lifetimes alias or extent truncated");
        Allocation arena{phx::address_from_bits(uint64_t(9) << 40), bytes, 9};
        demand(phx::address_bits(plan.relocate(subview, arena)) ==
                   (uint64_t(9) << 40) + (uint64_t(5) << 30), "Wide subview relocation");
        demand(plan.relocate(host_address, arena) == host_address, "Persistent pointer relocation");
        bool rejected = false;
        try { plan.offset(phx::address_from_bits(phx::address_bits(first.address) + first.bytes)); }
        catch (const std::exception&) { rejected = true; }
        demand(rejected, "One-past scratch view accepted");
        rejected = false;
        try { plan.relocate(subview, Allocation{phx::address_from_bits(UINT64_MAX - 1), bytes, 0}); }
        catch (const std::exception&) { rejected = true; }
        demand(rejected, "Relocation overflow accepted");
        ScratchPlan reused;
        auto a = reused.reserve(256), b = reused.reserve(256);
        reused.touch(a.address, 0); reused.touch(b.address, 1);
        demand(reused.layout() == 256 && reused.offset(a.address) == reused.offset(b.address), "Scratch reuse regression");
        Command command{op, 1, 0, 0};
        auto other = command;
        other.operation.left_input = wide;
        demand(!same_commands({command}, {other}), "Wide address omitted from command cache");
        other = command;
        demand(same_commands({command}, {other}), "Equal command cache regression");
        std::cout << "Address contracts passed; pointer_bits=" << sizeof(void*) * 8 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
