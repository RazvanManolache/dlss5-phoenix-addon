#include "../../src/native/network/production_network.h"
#include <iostream>

namespace {
unsigned initialized = 0, unloaded = 0;
NvAPI_Status __cdecl initialize_ok() { ++initialized; return NVAPI_OK; }
NvAPI_Status __cdecl initialize_failed() { return NVAPI_ERROR; }
NvAPI_Status __cdecl unload_ok() { ++unloaded; return NVAPI_OK; }
}
int main() try {
    using phoenix_native::Api;
    using phoenix_bundle::require;
    for (unsigned iteration = 0; iteration < 32; ++iteration) {
        Api api{Api::NoLoad{}};
        api.initialize_session(initialize_ok, unload_ok);
        if (iteration % 2) { api.close(); api.close(); }
    }
    require(initialized == 32 && unloaded == 32, "NVAPI lifecycle references unbalanced");
    bool rejected = false;
    try {
        Api api{Api::NoLoad{}};
        api.initialize_session(initialize_failed, unload_ok);
    } catch (...) { rejected = true; }
    require(rejected && initialized == 32 && unloaded == 32, "Failed initialization incorrectly unloaded NVAPI");
    try {
        Api api{Api::NoLoad{}};
        api.initialize_session(initialize_ok, unload_ok);
        throw std::runtime_error("Later interface query failed");
    } catch (...) {}
    require(initialized == 33 && unloaded == 33, "Partial setup leaked an initialized API");
    std::cout << nlohmann::json({{"passed", true}, {"gpu_used", false},
        {"references", Api::lifetime_report()}}).dump() << '\n';
    return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
