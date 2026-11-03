// Model-free isolated compatibility check. Never opens a game or a window.
#define NOMINMAX
#include "client32.h"
#include <thread>

int wmain(int argc,wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX|SEM_NOOPENFILEERRORBOX);
    if(argc!=4)return 2;
    phoenix_wow64::DriverJson result{{"contract",1},{"passed",false},{"caller_bits",32}};
    int code=2;
    try {
        const auto root=std::filesystem::absolute(argv[1]);const std::wstring expected_wide=argv[2];
        const std::string expected(expected_wide.begin(),expected_wide.end());
        phoenix_wow64::driver_require(expected.size()==64&&expected.find_first_not_of("0123456789abcdef")==std::string::npos,"Invalid expected driver identity");
        // Production starts its native worker from a fresh x86 processing thread.
        // Reproduce that loader/TLS state instead of priming it on the main thread.
        std::exception_ptr failure;
        std::thread worker([&]{try {
            phoenix_driver_probe::Channel channel;phoenix_driver_probe::CompatibilityEvidence evidence;
            channel.caller_bits=32;channel.operation=phoenix_driver_probe::check_compatibility;
            channel.output=uint64_t(uintptr_t(&evidence));
            {
                phoenix_wow64::NativeThread native;native.start(root,channel,4,&channel,expected.c_str());
            }
            phoenix_wow64::driver_require(channel.status==0&&channel.native_compat_restored&&channel.driver_pid==GetCurrentProcessId()&&
                channel.driver_bits==64&&!channel.helper_pid&&!channel.phoenix64_loaded&&!channel.completed_frames,
                "Native driver probe did not retire cleanly in its x86 process");
            result.update({{"passed",true},{"cuda_sha256",expected},{"driver_bits",channel.driver_bits},
                {"cuda_initialized",evidence.cuda_initialized==1},{"adapter_matched",evidence.adapter_matched==1},
                {"cuda_version",evidence.cuda_version},{"cuda_devices",evidence.device_count},
                {"nvapi_entries",evidence.nvapi_entries},{"d3d12_copy_verified",evidence.d3d12_copy_verified==1},
                {"clean_retirement",true},{"model_inference",false}});
        }catch(...){failure=std::current_exception();}});
        worker.join();if(failure)std::rethrow_exception(failure);code=0;
    }catch(const std::exception& error){result["error"]=error.what();}
    std::ofstream output(argv[3]);output<<result.dump();output.flush();return output?code:3;
}
