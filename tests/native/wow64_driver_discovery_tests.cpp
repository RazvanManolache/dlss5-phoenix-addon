#define NOMINMAX
#include "../../src/integration/compatibility/wow64/driver_identity.h"
#include "../../src/integration/compatibility/wow64/driver_contract.h"
#include <iostream>

int main() try {
    using namespace phoenix_wow64;
    unsigned checks=0;auto expect=[&](bool ok){++checks;if(!ok)throw std::runtime_error("Driver discovery/cache assertion "+std::to_string(checks));};
    const fs::path windows=L"C:\\WINDOWS";
    for(const auto* image:{L"\\SystemRoot\\System32\\DriverStore\\FileRepository\\nv_test\\nvlddmkm.sys",
        L"\\??\\C:\\Windows\\System32\\DriverStore\\FileRepository\\nv_test\\nvlddmkm.sys",
        L"C:\\WINDOWS\\System32\\DriverStore\\FileRepository\\nv_test\\nvlddmkm.sys"}) {
        auto found=registered_driver_directory(image,windows);expect(found.filename()==L"nv_test");
#ifndef _WIN64
        expect(driver_fold(native_read_path(found/"nvcuda64.dll",windows).wstring()).find(L"sysnative")!=std::wstring::npos);
#endif
    }
    for(const auto* image:{L"D:\\user\\nvlddmkm.sys",L"C:\\Windows\\System32\\DriverStore\\FileRepository\\nv_test\\other.sys",
        L"C:\\Windows\\System32\\DriverStore\\FileRepository\\..\\nv_test\\nvlddmkm.sys",L"nvlddmkm.sys"}) {
        bool rejected=false;try{registered_driver_directory(image,windows);}catch(...){rejected=true;}expect(rejected);
    }
    DriverJson identity{{"contract",1},{"driver","active"},{"cuda_sha256","abc"},{"windows_sha256","win"},{"package_sha256","pkg"}};
    DriverJson proof{{"passed",true},{"contract",1},{"cuda_sha256","abc"},{"caller_bits",32},{"driver_bits",64},
        {"cuda_initialized",true},{"adapter_matched",true},{"nvapi_entries",7},{"d3d12_copy_verified",true},
        {"clean_retirement",true},{"model_inference",false}};
    DriverJson result{{"schema","phoenix-driver-compatibility/1"},{"identity",identity},{"checked",uint64_t(100)},
        {"passed",true},{"evidence",proof}};
    expect(reusable_driver_result(result,identity,1000000));
    for(const char* key:{"driver","cuda_sha256","windows_sha256","package_sha256","contract"}) {
        auto changed=identity;changed[key]="changed";expect(!reusable_driver_result(result,changed,101));
    }
    for(const char* key:{"cuda_initialized","adapter_matched","d3d12_copy_verified","clean_retirement"}) {
        auto broken=result;broken["evidence"][key]=false;expect(!reusable_driver_result(broken,identity,101));
    }
    auto broken=result;broken.erase("evidence");expect(!reusable_driver_result(broken,identity,101));
    broken=result;broken["evidence"]["caller_bits"]=64;expect(!reusable_driver_result(broken,identity,101));
    broken=result;broken["evidence"]["model_inference"]=true;expect(!reusable_driver_result(broken,identity,101));
    broken=result;broken["evidence"]["passed"]="true";expect(!reusable_driver_result(broken,identity,101));
    expect(!reusable_driver_result(result,identity,99));
    broken=result;broken["passed"]=false;expect(reusable_driver_result(broken,identity,219));expect(!reusable_driver_result(broken,identity,220));
    expect(!reusable_driver_result(DriverJson::array(),identity,100));
    phoenix_driver_probe::Channel channel;channel.caller_bits=32;channel.operation=phoenix_driver_probe::check_compatibility;
    channel.cuda_library=1024;channel.output=2048;channel.request_event=1;channel.response_event=2;
    expect(phoenix_driver_probe::valid_compatibility_check(channel));
    expect(!phoenix_driver_probe::valid_initialize(channel));
    channel.output=0;expect(!phoenix_driver_probe::valid_compatibility_check(channel));
    std::cout<<DriverJson{{"passed",true},{"checks",checks},{"bits",sizeof(void*)*8},{"gpu_used",false}}.dump();return 0;
}catch(const std::exception& error){std::cerr<<error.what();return 1;}
