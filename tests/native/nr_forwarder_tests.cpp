#include <windows.h>
#include <nvsdk_ngx.h>
#include <iostream>
#include <stdexcept>
int wmain(int argc, wchar_t** argv) {
    if (argc != 2) { std::cout << "{\"passed\":true,\"skipped\":\"No embedded wrapper\"}\n"; return 0; }
    unsigned checks = 0;
    auto check = [&](bool condition) { ++checks; if (!condition) throw std::runtime_error("NR wrapper contract failed"); };
    try {
        auto module = LoadLibraryExW(argv[1], nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        check(module != nullptr);
        auto load = reinterpret_cast<BOOL(WINAPI*)(const wchar_t*)>(GetProcAddress(module,"PhoenixNrLoad"));
        using Init = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long,const wchar_t*,ID3D12Device*,NVSDK_NGX_Version,const NVSDK_NGX_Parameter*);
        auto init = reinterpret_cast<Init>(GetProcAddress(module,"PhoenixNrInit"));
        using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
        auto release = reinterpret_cast<Release>(GetProcAddress(module,"PhoenixNrRelease"));
        check(load && init && release && GetProcAddress(module,"PhoenixNrCreate") && GetProcAddress(module,"PhoenixNrEvaluate"));
        check(!load(nullptr));
        check(!load(L""));
        check(!load(L"relative-dll-is-not-an-approved-model.dll"));
        check(init(0,nullptr,nullptr,NVSDK_NGX_Version_API,nullptr)==NVSDK_NGX_Result_FAIL_NotInitialized);
        check(release(nullptr)==NVSDK_NGX_Result_FAIL_NotInitialized);
        check(GetModuleHandleW(L"nvngx_dlssnr.dll")==nullptr);
        FreeLibrary(module);
        std::cout << "{\"passed\":true,\"cpu_only\":true,\"checks\":" << checks << "}\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
