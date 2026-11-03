#define NOMINMAX
#include "../../src/integration/compatibility/wow64/driver_compatibility.h"
#include <iostream>

// CPU child fixture; its synthetic reply is never used as a production cache.
// The parent exercises real job ownership, bounded waits and response rejection.
int wmain(int argc,wchar_t** argv) try {
    using namespace phoenix_wow64;
    if(argc==4) {
        std::string mode;{std::ifstream file(fs::path(argv[1])/"mode.txt");file>>mode;}
        {std::ofstream file(fs::path(argv[1])/"child.pid");file<<GetCurrentProcessId();}
        if(mode=="hang"){Sleep(10000);return 5;}if(mode=="failure")return 7;
        if(mode=="no_reply")return 0;
        const std::wstring wide=argv[2];const std::string sha(wide.begin(),wide.end());
        DriverJson proof{{"passed",true},{"contract",1},{"cuda_sha256",sha},{"caller_bits",32},{"driver_bits",64},
            {"cuda_initialized",true},{"adapter_matched",true},{"nvapi_entries",7},{"d3d12_copy_verified",true},
            {"clean_retirement",true},{"model_inference",false}};
        if(mode=="incomplete")proof.erase("d3d12_copy_verified");
        std::ofstream file(argv[3]);file<<proof.dump();return 0;
    }
    driver_require(argc==2,"Fixture directory required");fs::path root=fs::absolute(argv[1]);fs::create_directories(root);
    wchar_t module[32768]{};driver_require(GetModuleFileNameW(nullptr,module,32768)>0,"Fixture module unavailable");
    fs::copy_file(module,root/"phoenix_driver_probe32.exe",fs::copy_options::overwrite_existing);
    NativeDriverIdentity identity;identity.sha256=std::string(64,'a');unsigned checks=0;
    for(const auto* mode:{"good","no_reply","incomplete","failure","hang"}) {
        fs::remove(root/"child.pid");
        {std::ofstream file(root/"mode.txt");file<<mode;}
        const auto argument_root=std::string(mode)=="good"?root/"":root;
        bool succeeded=false;try{run_driver_probe(argument_root,identity,root/"reply.json",std::string(mode)=="hang"?500:5000);succeeded=true;}catch(...){}
        driver_require(succeeded==(std::string(mode)=="good"),"Process result was accepted/rejected incorrectly");++checks;
        DWORD pid=0;{std::ifstream file(root/"child.pid");file>>pid;}driver_require(pid>0,"Probe fixture never recorded its identity");
        if(auto process=OpenProcess(SYNCHRONIZE,FALSE,pid)) {
            const auto waited=WaitForSingleObject(process,1000);CloseHandle(process);
            driver_require(waited==WAIT_OBJECT_0,"Probe process survived result/timeout cleanup");
        }
        ++checks;
    }
    std::cout<<DriverJson{{"passed",true},{"checks",checks},{"bits",sizeof(void*)*8},{"gpu_used",false}}.dump();return 0;
}catch(const std::exception& error){std::cerr<<error.what();return 1;}
