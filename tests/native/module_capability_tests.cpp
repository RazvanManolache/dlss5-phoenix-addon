#include "../../src/native/support/module_capability.h"
#include <cassert>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include "../../src/native/network/production_network.h"
#endif

int main() {
#ifdef _WIN32
    phoenix_native::Api unloaded(phoenix_native::Api::NoLoad{});
    assert(!unloaded.library && !unloaded.modules.fatbin_supported);
#endif
    bool queried = false;
    phoenix_kernel::require_native_module_capability(8, [&] { queried = true; return false; });
    assert(!queried);
    phoenix_kernel::require_native_module_capability(4, [&] { queried = true; return true; });
    assert(queried);
    bool rejected = false;
    try {
        phoenix_kernel::require_native_module_capability(4, [] { return false; });
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    assert(rejected);
}
