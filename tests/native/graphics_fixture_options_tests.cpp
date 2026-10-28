#include "graphics_fixture_options.h"
#include <cassert>
#include <iostream>
int main() {
    const char* msaa[]{"fixture", "d3d9", "--process-exit", "--msaa-control"};
    auto a = GraphicsFixtureOptions::parse(4, msaa);
    assert(a.process_exit && a.multisample && a.control);
    const char* queues[]{"fixture", "vulkan", "--secondary-queue", "--compute-present", "--process-exit"};
    auto b = GraphicsFixtureOptions::parse(5, queues);
    assert(b.process_exit && b.secondary_queue && b.compute_present);
    for (const char* invalid : {"--unknown", "--resize"}) {
        const char* args[]{"fixture", "d3d11", "--dynamic", invalid};
        bool rejected = false;
        try { GraphicsFixtureOptions::parse(4, args); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected);
    }
    std::cout << "{\"passed\":true,\"cpu_only\":true}\n";
}
