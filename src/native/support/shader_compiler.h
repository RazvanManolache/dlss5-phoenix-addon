#pragma once
#include <windows.h>
#include <d3dcompiler.h>
#include <string>
#include <utility>

namespace phoenix_shader {
// A game's app-local D3DCompiler_47 may be years older than the OS compiler.
// Resolve an absolute Windows path, so in-process and helper rendering compile
// the same Phoenix shaders without replacing the game's own DLL or weakening
// warnings-as-errors. Loading/lookup is cached; there is no per-frame work.
inline HMODULE system_module() noexcept {
    static HMODULE module=[]() noexcept -> HMODULE {
        wchar_t directory[32768]{};
        const auto length=GetSystemDirectoryW(directory,DWORD(std::size(directory)));
        if(!length||length>=std::size(directory))return nullptr;
        try {
            const auto path=std::wstring(directory,length)+L"\\D3DCompiler_47.dll";
            // Keep one process-lifetime reference: compiler-created COM blobs
            // can outlive an individual add-on runtime or device generation.
            return LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
        }catch(...){return nullptr;}
    }();
    return module;
}
inline auto system_compile() noexcept {
    static auto proc=reinterpret_cast<decltype(&D3DCompile)>(
        system_module()?GetProcAddress(system_module(),"D3DCompile"):nullptr);
    return proc;
}
template<class... Args> inline HRESULT compile(Args&&... args) {
    const auto proc=system_compile();
    return proc?proc(std::forward<Args>(args)...):HRESULT_FROM_WIN32(ERROR_MOD_NOT_FOUND);
}
} // namespace phoenix_shader
