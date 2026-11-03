#pragma once
#include "../../../native/support/embedded_runtime.h"
#include <windows.h>
#include <bcrypt.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <algorithm>
#include <cwctype>
#pragma comment(lib,"advapi32.lib")
#pragma comment(lib,"bcrypt.lib")

// Driver discovery is data-only. Never load a driver on the presentation thread.
// Select the registered NVIDIA package, not the newest leftover DriverStore DLL.
namespace phoenix_wow64 {
namespace fs = std::filesystem;
using DriverJson = nlohmann::json;
inline void driver_require(bool ok,const char* error) { if(!ok)throw std::runtime_error(error); }
inline std::wstring driver_fold(std::wstring text) {
    std::transform(text.begin(),text.end(),text.begin(),[](wchar_t c){return wchar_t(towlower(c));});return text;
}
inline fs::path native_read_path(const fs::path& path,const fs::path& windows) {
#ifndef _WIN64
    auto normalized=path.lexically_normal();normalized.make_preferred();
    auto system=(windows/"System32").lexically_normal();system.make_preferred();
    const auto prefix=system.wstring()+L"\\";
    if(driver_fold(normalized.wstring()).starts_with(driver_fold(prefix)))
        return windows/"Sysnative"/normalized.wstring().substr(prefix.size());
#endif
    return path;
}
inline fs::path registered_driver_directory(std::wstring image,const fs::path& windows) {
    if(image.starts_with(L"\\??\\"))image.erase(0,4);
    if(driver_fold(image).starts_with(L"\\systemroot\\"))image=windows.wstring()+image.substr(11);
    wchar_t expanded[32768]{};
    const auto size=ExpandEnvironmentStringsW(image.c_str(),expanded,32768);
    driver_require(size&&size<=32768,"NVIDIA driver registry path is invalid");
    auto path=fs::path(expanded).lexically_normal();path.make_preferred();
    auto store=(windows/"System32/DriverStore/FileRepository").lexically_normal();store.make_preferred();
    driver_require(path.is_absolute()&&driver_fold(path.parent_path().parent_path().wstring())==driver_fold(store.wstring())&&
        driver_fold(path.filename().wstring())==L"nvlddmkm.sys","Registered NVIDIA driver is outside the Windows driver store");
    return path.parent_path();
}
inline std::string driver_text_digest(const std::string& text) {
    BCRYPT_HASH_HANDLE hash{};unsigned char bytes[32]{};char hex[65]{};
    driver_require(BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE,&hash,nullptr,0,nullptr,0,0)>=0,"Driver cache hash creation failed");
    const auto added=BCryptHashData(hash,reinterpret_cast<PUCHAR>(const_cast<char*>(text.data())),ULONG(text.size()),0);
    const auto done=added>=0?BCryptFinishHash(hash,bytes,sizeof(bytes),0):added;BCryptDestroyHash(hash);
    driver_require(done>=0,"Driver cache hash failed");
    for(unsigned i=0;i<32;++i)sprintf_s(hex+2*i,3,"%02x",unsigned(bytes[i]));return hex;
}
struct NativeDriverIdentity {
    fs::path windows,library,read_library;
    std::string sha256;
    bool validation_cached=false;
};
inline NativeDriverIdentity discover_native_driver() {
    wchar_t windows[32768]{};driver_require(GetWindowsDirectoryW(windows,32768)>0,"Windows directory unavailable");
    DWORD bytes=0,type=0;
    constexpr auto key=L"SYSTEM\\CurrentControlSet\\Services\\nvlddmkm";
    driver_require(RegGetValueW(HKEY_LOCAL_MACHINE,key,L"ImagePath",RRF_RT_REG_SZ|RRF_RT_REG_EXPAND_SZ|RRF_NOEXPAND,
        &type,nullptr,&bytes)==ERROR_SUCCESS&&bytes>2&&bytes<65536,"Installed NVIDIA driver registration unavailable");
    std::wstring image(bytes/sizeof(wchar_t),L'\0');
    driver_require(RegGetValueW(HKEY_LOCAL_MACHINE,key,L"ImagePath",RRF_RT_REG_SZ|RRF_RT_REG_EXPAND_SZ|RRF_NOEXPAND,
        &type,image.data(),&bytes)==ERROR_SUCCESS,"Cannot read installed NVIDIA driver registration");
    image.resize(wcsnlen(image.data(),image.size()));
    NativeDriverIdentity result;result.windows=windows;
    result.library=registered_driver_directory(image,result.windows)/"nvcuda64.dll";
    result.read_library=native_read_path(result.library,result.windows);
    std::ifstream stream(result.read_library,std::ios::binary);IMAGE_DOS_HEADER dos{};IMAGE_NT_HEADERS64 pe{};
    stream.read(reinterpret_cast<char*>(&dos),sizeof(dos));
    driver_require(bool(stream)&&dos.e_magic==IMAGE_DOS_SIGNATURE&&dos.e_lfanew>=sizeof(dos)&&dos.e_lfanew<1024*1024,
        "Registered NVIDIA package has no readable native CUDA DLL");
    stream.seekg(dos.e_lfanew);stream.read(reinterpret_cast<char*>(&pe),sizeof(pe));
    driver_require(bool(stream)&&pe.Signature==IMAGE_NT_SIGNATURE&&pe.FileHeader.Machine==IMAGE_FILE_MACHINE_AMD64&&
        pe.OptionalHeader.Magic==IMAGE_NT_OPTIONAL_HDR64_MAGIC&&(pe.FileHeader.Characteristics&IMAGE_FILE_DLL),
        "Registered CUDA driver is not an x64 DLL");
    result.sha256=phoenix_embedded::file_digest(result.read_library);return result;
}
// Invalidate on CUDA replacement, Windows bootstrap change, package/probe change,
// or protocol change. Paths are included as well as content identities.
inline DriverJson driver_validation_identity(const fs::path& root,const NativeDriverIdentity& driver) {
    const auto path=driver.library.generic_u8string();
    return {{"contract",1},{"driver",std::string(reinterpret_cast<const char*>(path.data()),path.size())}, {"cuda_sha256",driver.sha256},
        {"windows_sha256",phoenix_embedded::file_digest(native_read_path(driver.windows/"System32/ntdll.dll",driver.windows))},
        {"package_sha256",phoenix_embedded::file_digest(root/"bootstrap.json")}};
}
inline bool driver_probe_evidence_valid(const DriverJson& reply,const std::string& sha) {
    return reply.is_object()&&reply.value("passed",false)&&reply.value("contract",0)==1&&
        reply.value("cuda_sha256","")==sha&&reply.value("caller_bits",0)==32&&reply.value("driver_bits",0)==64&&
        reply.value("cuda_initialized",false)&&reply.value("adapter_matched",false)&&reply.value("nvapi_entries",0)==7&&
        reply.value("d3d12_copy_verified",false)&&reply.value("clean_retirement",false)&&
        reply.contains("model_inference")&&reply["model_inference"]==false;
}
inline bool reusable_driver_result(const DriverJson& value,const DriverJson& identity,uint64_t now) try {
    if(!value.is_object()||value.value("schema","")!="phoenix-driver-compatibility/1"||
        !value.contains("identity")||value["identity"]!=identity||!value.contains("passed")||!value["passed"].is_boolean())return false;
    if(!value.contains("checked")||!value["checked"].is_number_unsigned())return false;
    const auto checked=value["checked"].get<uint64_t>();if(checked>now)return false;
    // A transient startup failure is retried after two minutes. A positive
    // result remains valid until one of its content-bound identities changes.
    return value["passed"].get<bool>()?
        (value.contains("evidence")&&driver_probe_evidence_valid(value["evidence"],identity.at("cuda_sha256"))):now-checked<120;
}catch(...){return false;}
}
