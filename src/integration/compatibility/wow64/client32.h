#pragma once
#include "bootstrap_contract.h"
#include "gdi32.h"
#include "driver_contract.h"
#include "../../../native/model/bundle_loader.h"
#include <bcrypt.h>
#include <fstream>
#include <array>
#pragma comment(lib,"bcrypt.lib")
namespace phoenix_wow64 {
inline std::string file_digest(const std::filesystem::path& path) {
    std::ifstream file(path,std::ios::binary); phoenix_bundle::require(bool(file),"Cannot open native runtime file");
    BCRYPT_HASH_HANDLE hash=nullptr;
    phoenix_bundle::require(BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE,&hash,nullptr,0,nullptr,0,0)>=0,"Runtime digest create");
    unsigned char result[32]{}; char text[65]{}; std::array<char,65536> buffer{};
    try {
        while(file) { file.read(buffer.data(),buffer.size()); if(file.gcount()) phoenix_bundle::require(
            BCryptHashData(hash,reinterpret_cast<PUCHAR>(buffer.data()),ULONG(file.gcount()),0)>=0,"Runtime digest read"); }
        phoenix_bundle::require(file.eof()&&!file.bad(),"Short runtime file read");
        phoenix_bundle::require(BCryptFinishHash(hash,result,32,0)>=0,"Runtime digest finish");
    } catch(...) { BCryptDestroyHash(hash); throw; }
    BCryptDestroyHash(hash);
    for(unsigned i=0;i<32;++i) sprintf_s(text+i*2,3,"%02x",unsigned(result[i])); return text;
}
#ifndef _WIN64
class NativeThread {
    using Call=uint64_t(__cdecl*)(uint64_t,int,...);
    using Module=uint64_t(__cdecl*)(const wchar_t*);
    using Proc=uint64_t(__cdecl*)(uint64_t,const char*);
    HMODULE extension_=nullptr;
    HANDLE thread_=nullptr;
    PhoenixTlsTask task_{};
    phoenix_driver_probe::Channel* channel_=nullptr;
    std::wstring service_path_,cuda_path_;
    static uint64_t pointer(const void* p) { return uint64_t(uintptr_t(p)); }
  public:
    NativeThread()=default;
    NativeThread(const NativeThread&)=delete;
    ~NativeThread() { close(); }
    void close() noexcept {
        // This object is owned by the detached processing worker, never by the
        // render thread. Keep servicing native driver callbacks until retirement;
        // do not terminate a game or free packets while a native thread owns them.
        if(thread_) {
            channel_->operation=phoenix_driver_probe::shutdown;
            SetEvent(reinterpret_cast<HANDLE>(uintptr_t(channel_->request_event)));
            while(WaitForSingleObject(thread_,0)==WAIT_TIMEOUT) {
                HANDLE handles[]{thread_,reinterpret_cast<HANDLE>(uintptr_t(task_.gdi.request_event))};
                if(WaitForMultipleObjects(2,handles,FALSE,100)==WAIT_OBJECT_0+1) service_gdi(task_.gdi,extension_);
            }
            CloseHandle(thread_); thread_=nullptr;
        }
        for(auto handle:{task_.gdi.request_event,task_.gdi.response_event}) if(handle) CloseHandle(reinterpret_cast<HANDLE>(uintptr_t(handle)));
        task_.gdi.request_event=task_.gdi.response_event=0;
        if(channel_) {
            for(auto handle:{channel_->request_event,channel_->response_event}) if(handle) CloseHandle(reinterpret_cast<HANDLE>(uintptr_t(handle)));
            channel_->request_event=channel_->response_event=0;
        }
        // Native images/driver threads may retain imports. Libraries stay loaded
        // for the process lifetime rather than unloading under a callback.
    }
    void start(const std::filesystem::path& root,phoenix_driver_probe::Channel& channel,unsigned kind,void* packet) {
        using phoenix_bundle::require;
        const auto profile=phoenix_bundle::json_file(root/"bootstrap.json");
        require(profile.at("schema")=="phoenix-wow64-bootstrap/1","Unsupported native bootstrap profile");
        wchar_t windows[MAX_PATH]{}; require(GetWindowsDirectoryW(windows,MAX_PATH)!=0,"Windows path unavailable");
        const auto system=std::filesystem::path(windows);
        require(file_digest(system/"Sysnative/ntdll.dll")==profile.at("ntdll_sha256").get<std::string>(),
            "Native runtime profile does not match this Windows build; update its bootstrap profile or use the helper.");
        for(const char* name:{"wow64ext.dll","phoenix_bootstrap64.dll","phoenix_driver64.dll","phoenix-nvngx.dll","phoenix_native.dll"})
            require(profile.at("files").contains(name),"Incomplete native runtime manifest");
        for(auto item:profile.at("files").items()) {
            const auto name=std::filesystem::u8path(item.key()); require(name==name.filename(),"Invalid native runtime file name");
            require(file_digest(root/name)==item.value().get<std::string>(),"Native runtime package hash mismatch");
        }
        const auto driver=std::filesystem::u8path(profile.at("driver_directory").get<std::string>());
        require(driver==driver.filename(),"Invalid native driver directory");
        cuda_path_=(system/"System32/DriverStore/FileRepository"/driver/"nvcuda64.dll").wstring();
        require(file_digest(system/"Sysnative/DriverStore/FileRepository"/driver/"nvcuda64.dll")==profile.at("cuda_sha256").get<std::string>(),"Native driver profile changed; use the helper or rebuild the runtime profile.");
        service_path_=(root/(kind==4?"phoenix_driver64.dll":"phoenix-nvngx.dll")).wstring();
        extension_=LoadLibraryExW((root/"wow64ext.dll").c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_SYSTEM32);
        require(extension_!=nullptr,"Matching x86 WOW64 bridge library unavailable");
        auto call=reinterpret_cast<Call>(GetProcAddress(extension_,"X64Call"));
        auto module=reinterpret_cast<Module>(GetProcAddress(extension_,"GetModuleHandle64"));
        auto proc=reinterpret_cast<Proc>(GetProcAddress(extension_,"GetProcAddress64"));
        require(call&&module&&proc,"Invalid WOW64 bridge exports");
        const auto ntdll=module(L"ntdll.dll"); require(ntdll!=0,"Native loader unavailable");
        using Read=BOOL(__cdecl*)(HANDLE,uint64_t,void*,SIZE_T,SIZE_T*);
        const auto read=reinterpret_cast<Read>(GetProcAddress(extension_,"ReadProcessMemory64"));
        IMAGE_DOS_HEADER dos{};IMAGE_NT_HEADERS64 pe{};
        require(read&&read(GetCurrentProcess(),ntdll,&dos,sizeof(dos),nullptr)&&dos.e_magic==IMAGE_DOS_SIGNATURE&&
            dos.e_lfanew>0&&dos.e_lfanew<65536&&read(GetCurrentProcess(),ntdll+dos.e_lfanew,&pe,sizeof(pe),nullptr)&&
            pe.Signature==IMAGE_NT_SIGNATURE&&pe.FileHeader.Machine==IMAGE_FILE_MACHINE_AMD64&&
            pe.FileHeader.TimeDateStamp==profile.at("ntdll_timestamp").get<uint32_t>(),"Loaded native loader image mismatch");
        for(const char* field:{"initializer_rva","allocator_rva"}) {
            const auto rva=profile.at(field).get<uint32_t>();
            require(rva&&rva<pe.OptionalHeader.SizeOfImage,"Invalid native TLS routine offset");
        }
        const auto bootstrap=(root/"phoenix_bootstrap64.dll").wstring();
        struct String64 { uint16_t length,capacity; uint32_t padding; uint64_t data; } name{
            uint16_t(bootstrap.size()*2),uint16_t((bootstrap.size()+1)*2),0,pointer(bootstrap.c_str())};
        require(bootstrap.size()<30000,"Native runtime path too long"); uint64_t library=0;
        const auto loader=proc(ntdll,"LdrLoadDll"); require(loader!=0,"Native DLL loader export unavailable");
        const auto load_status=int32_t(call(loader,4,uint64_t(0),uint64_t(0),pointer(&name),pointer(&library)));
        if(load_status<0||!library) throw std::runtime_error("Native bootstrap load failed: "+std::to_string(uint32_t(load_status)));
        auto& tls=task_.tls; tls.module=library;
        tls.initializer=ntdll+profile.at("initializer_rva").get<uint32_t>();
        task_.thread_allocator=ntdll+profile.at("allocator_rva").get<uint32_t>();
        tls.loader_lock=proc(ntdll,"LdrLockLoaderLock"); tls.loader_unlock=proc(ntdll,"LdrUnlockLoaderLock");
        task_.kind=kind; task_.path=pointer(service_path_.c_str()); task_.kernel=pointer(packet);
        task_.gdi.request_event=pointer(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        task_.gdi.response_event=pointer(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        channel_=&channel; channel.cuda_library=pointer(cuda_path_.c_str());
        channel.ntdll_timestamp=profile.at("ntdll_timestamp");
        const auto offsets=profile.at("appcompat_offsets").get<std::array<uint32_t,4>>();
        std::copy(offsets.begin(),offsets.end(),channel.appcompat_offsets);
        channel.request_event=pointer(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        channel.response_event=pointer(CreateEventW(nullptr,FALSE,FALSE,nullptr));
        require(task_.gdi.request_event&&task_.gdi.response_event&&channel.request_event&&channel.response_event,"Native session events unavailable");
        const auto entry=proc(library,"PhoenixTlsRunTask64"), create=proc(ntdll,"NtCreateThreadEx");
        require(entry&&create,"Native bootstrap entry unavailable"); uint64_t handle=0;
        require(int32_t(call(create,11,pointer(&handle),uint64_t(0x1fffff),uint64_t(0),uint64_t(-1),entry,pointer(&task_),
            uint64_t(2),uint64_t(0),uint64_t(0),uint64_t(0),uint64_t(0)))>=0&&handle,"Cannot create native processing thread");
        thread_=reinterpret_cast<HANDLE>(uintptr_t(handle)); wait();
    }
    void wait() {
        using phoenix_bundle::require;
        for(;;) {
            HANDLE handles[]{thread_,reinterpret_cast<HANDLE>(uintptr_t(task_.gdi.request_event)),reinterpret_cast<HANDLE>(uintptr_t(channel_->response_event))};
            const auto waited=WaitForMultipleObjects(3,handles,FALSE,100);
            if(waited==WAIT_OBJECT_0+1) service_gdi(task_.gdi,extension_);
            else if(waited==WAIT_OBJECT_0+2) { require(channel_->status==0,channel_->error); return; }
            else if(waited==WAIT_OBJECT_0) {
                DWORD code=0;GetExitCodeThread(thread_,&code);
                if(channel_->error[0]) throw std::runtime_error(channel_->error);
                throw std::runtime_error("Native thread stopped: exit="+std::to_string(code)+" tls_stage="+
                    std::to_string(task_.tls.stage)+" init_status="+std::to_string(task_.tls.initializer_status)+
                    " vector="+std::to_string(task_.tls.vector_after)+" index="+std::to_string(task_.tls.index)+
                    " slot="+std::to_string(task_.tls.slot)+" loaded="+std::to_string(task_.loaded)+" load_error="+std::to_string(task_.load_error));
            }
            else require(waited==WAIT_TIMEOUT,"Native session wait failed");
        }
    }
    void frame() {
        channel_->operation=phoenix_driver_probe::frame;
        phoenix_bundle::require(SetEvent(reinterpret_cast<HANDLE>(uintptr_t(channel_->request_event)))!=0,"Native request signal failed"); wait();
    }
};
#endif
}
