#define NOMINMAX
#include "../../src/integration/compatibility/transport_availability.h"
#include <iostream>

int wmain(int argc,wchar_t** argv) try {
    using namespace phoenix_wow64;driver_require(argc==2,"Runtime directory required");
    const auto root=fs::absolute(argv[1]);const auto driver=discover_native_driver();
#ifdef _WIN64
    const auto native_system=driver.windows/"System32";
#else
    const auto native_system=driver.windows/"Sysnative";
#endif
    const auto started=std::chrono::steady_clock::now();
    const auto first=phoenix_compat::inspect_native_transport(root,native_system);
    const auto initialized=std::chrono::steady_clock::now();
    const auto second=phoenix_compat::inspect_native_transport(root,native_system);
    const auto finished=std::chrono::steady_clock::now();
    driver_require(first.sha256==second.sha256&&second.validation_cached,"Validated driver result was not reused");
    std::cout<<DriverJson{{"passed",true},{"parent_bits",sizeof(void*)*8},{"cuda_sha256",driver.sha256},
        {"first_cached",first.validation_cached},{"second_cached",second.validation_cached},
        {"first_ms",std::chrono::duration<double,std::milli>(initialized-started).count()},
        {"cached_ms",std::chrono::duration<double,std::milli>(finished-initialized).count()},
        {"gpu_startup_probe",!first.validation_cached},{"model_inference",false},{"games_launched",false}}.dump();return 0;
}catch(const std::exception& error){std::cerr<<error.what();return 1;}
