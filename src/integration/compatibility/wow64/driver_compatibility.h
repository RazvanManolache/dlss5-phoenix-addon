#pragma once
#include "driver_identity.h"
#include <chrono>
#include <mutex>

namespace phoenix_wow64 {
inline uint64_t driver_cache_now() {
    return uint64_t(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
}
inline fs::path driver_cache_folder() {
    wchar_t folder[32768]{};const auto n=GetEnvironmentVariableW(L"LOCALAPPDATA",folder,32768);
    driver_require(n&&n<32768,"Local driver compatibility cache is unavailable");
    const auto path=fs::path(folder)/"PhoenixDLSS5/DriverCompatibility";fs::create_directories(path);return path;
}
inline std::wstring driver_quote(const fs::path& path) {
    const auto text=path.wstring();driver_require(text.find(L'"')==std::wstring::npos,"Invalid probe path");
    auto quoted=L"\""+text;
    // Windows argv parsing doubles trailing backslashes before the closing
    // quote. Folder-picker/manual paths may legitimately end in a separator.
    for(auto at=text.rbegin();at!=text.rend()&&*at==L'\\';++at)quoted+=L'\\';
    return quoted+L"\"";
}
struct DriverProbeHandles {
    HANDLE process{},thread{},job{};
    ~DriverProbeHandles() {
        // Closing the job is also the crash/exception/timeout cleanup path.
        if(job)CloseHandle(job);
        if(process){WaitForSingleObject(process,5000);CloseHandle(process);}
        if(thread)CloseHandle(thread);
    }
};
inline DriverJson run_driver_probe(const fs::path& root,const NativeDriverIdentity& driver,const fs::path& output,DWORD timeout_ms=20000) {
    driver_require(timeout_ms>0&&timeout_ms<=20000,"Invalid driver validation timeout");
    // A reused PID or interrupted previous attempt must never supply an old
    // successful reply when the new child exits before publishing its result.
    std::error_code removed;fs::remove(output,removed);
    driver_require(!removed,"Cannot retire the previous driver probe reply");
    const auto executable=root/"phoenix_driver_probe32.exe";
    DriverProbeHandles handles;handles.job=CreateJobObjectW(nullptr,nullptr);
    driver_require(handles.job!=nullptr,"Cannot create isolated driver validation job");
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    driver_require(SetInformationJobObject(handles.job,JobObjectExtendedLimitInformation,&limits,sizeof(limits))!=0,"Cannot bound driver validation job");
    auto command=driver_quote(executable)+L" "+driver_quote(root)+L" "+std::wstring(driver.sha256.begin(),driver.sha256.end())+L" "+driver_quote(output);
    STARTUPINFOW startup{};startup.cb=sizeof(startup);PROCESS_INFORMATION process{};
    driver_require(CreateProcessW(executable.c_str(),command.data(),nullptr,nullptr,FALSE,CREATE_NO_WINDOW|CREATE_SUSPENDED,
        nullptr,root.c_str(),&startup,&process)!=0,"Cannot start isolated x86 driver compatibility probe; update the native runtime package");
    handles.process=process.hProcess;handles.thread=process.hThread;
    if(!AssignProcessToJobObject(handles.job,handles.process)) {
        TerminateProcess(handles.process,125);throw std::runtime_error("Cannot isolate driver compatibility probe");
    }
    driver_require(ResumeThread(handles.thread)!=DWORD(-1),"Cannot resume driver compatibility probe");
    driver_require(WaitForSingleObject(handles.process,timeout_ms)==WAIT_OBJECT_0,"Isolated driver compatibility probe timed out");
    DWORD code=0;driver_require(GetExitCodeProcess(handles.process,&code)!=0,"Cannot read driver compatibility result");
    DriverJson reply;
    try {if(fs::file_size(output)<=65536){std::ifstream file(output);file>>reply;}}catch(...){}
    if(code!=0)throw std::runtime_error(reply.is_object()?reply.value("error","Driver compatibility probe failed"):
        "Driver compatibility probe exited with code "+std::to_string(code));
    driver_require(driver_probe_evidence_valid(reply,driver.sha256),
        "Driver compatibility probe returned incomplete evidence");
    return reply;
}
inline NativeDriverIdentity compatible_native_driver(const fs::path& root,const DriverJson& profile) {
    driver_require(profile.value("dynamic_driver_validation",0)==1&&profile.at("files").contains("phoenix_driver_probe32.exe"),
        "Update the native bridge runtime for dynamic driver validation");
    auto driver=discover_native_driver();const auto identity=driver_validation_identity(root,driver);
    const auto key=driver_text_digest(identity.dump());const auto folder=driver_cache_folder();const auto cache=folder/(key+".json");
    // Cross-process serialization prevents two starting games/probes validating
    // the same driver simultaneously. Neither waiting nor hashing is per-frame.
    const auto mutex_name=L"Local\\PhoenixDriverCompatibility-"+std::wstring(key.begin(),key.end());
    HANDLE mutex=CreateMutexW(nullptr,FALSE,mutex_name.c_str());driver_require(mutex!=nullptr,"Driver validation lock unavailable");
    struct Lock {HANDLE value;bool owned=false;~Lock(){if(owned)ReleaseMutex(value);CloseHandle(value);}} lock{mutex};
    const auto waited=WaitForSingleObject(mutex,25000);lock.owned=waited==WAIT_OBJECT_0||waited==WAIT_ABANDONED;
    driver_require(lock.owned,"Another driver validation is still running");
    DriverJson result;
    try {if(fs::file_size(cache)<=65536){std::ifstream file(cache);file>>result;}}catch(...){}
    const auto now=driver_cache_now();bool reuse=false;
    try {reuse=reusable_driver_result(result,identity,now);}catch(...){}
    if(!reuse) {
        const auto output=folder/(key+"-"+std::to_string(GetCurrentProcessId())+".probe");
        struct Remove {fs::path path;~Remove(){std::error_code ignored;fs::remove(path,ignored);}} cleanup{output};
        result={{"schema","phoenix-driver-compatibility/1"},{"identity",identity},{"checked",now},{"passed",false}};
        try {
            result["evidence"]=run_driver_probe(root,driver,output);
            // Do not bless the old identity if a driver update raced the probe.
            const auto after=discover_native_driver();
            driver_require(driver_validation_identity(root,after)==identity,"NVIDIA driver changed during compatibility validation; retry");
            result["passed"]=true;
        }catch(const std::exception& error){result["error"]=error.what();}
        const auto temporary=cache.wstring()+L"."+std::to_wstring(GetCurrentProcessId())+L".tmp";
        {std::ofstream file{fs::path(temporary)};file<<result.dump(2);file.flush();driver_require(bool(file),"Cannot save driver compatibility result");}
        if(!MoveFileExW(temporary.c_str(),cache.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH)) {
            std::error_code ignored;fs::remove(temporary,ignored);throw std::runtime_error("Cannot commit driver compatibility cache");
        }
    }
    if(!result.value("passed",false))throw std::runtime_error("Installed NVIDIA driver failed x86 bridge compatibility: "+result.value("error","validation failed"));
    driver.validation_cached=reuse;
    return driver;
}
}
