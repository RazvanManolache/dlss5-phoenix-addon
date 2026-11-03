// CPU compilation only. Optionally preload a read-only game-bundled compiler
// and prove Phoenix selects the separate Windows compiler by absolute path.
#include "../../src/native/support/shader_compiler.h"
#include <wrl/client.h>
#include <filesystem>
#include <iostream>
#include <stdexcept>

int wmain(int argc,wchar_t** argv) try {
    HMODULE game_compiler=nullptr;
    if(argc>1){game_compiler=LoadLibraryW(argv[1]);if(!game_compiler)throw std::runtime_error("Cannot load optional compiler fixture");}
    const auto system=phoenix_shader::system_module();
    if(!system||!phoenix_shader::system_compile())throw std::runtime_error("Windows compiler unavailable");
    wchar_t loaded[32768]{},directory[32768]{};
    if(!GetModuleFileNameW(system,loaded,DWORD(std::size(loaded)))||!GetSystemDirectoryW(directory,DWORD(std::size(directory))))
        throw std::runtime_error("Cannot verify compiler path");
    if(!std::filesystem::equivalent(loaded,std::filesystem::path(directory)/L"D3DCompiler_47.dll"))
        throw std::runtime_error("Phoenix used a compiler outside the Windows directory");
    if(game_compiler&&game_compiler==system)throw std::runtime_error("Game compiler was substituted for the Windows compiler");
    constexpr char shader[]="RWTexture2D<float4> O:register(u0);[numthreads(8,8,1)]void main(uint3 p:SV_DispatchThreadID){O[p.xy]=float4(1,0,0,1);}";
    Microsoft::WRL::ComPtr<ID3DBlob> bytes,errors;
    const auto result=phoenix_shader::compile(shader,sizeof(shader)-1,"phoenix-compiler-test",nullptr,nullptr,
        "main","cs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3|D3DCOMPILE_WARNINGS_ARE_ERRORS,0,&bytes,&errors);
    if(FAILED(result)||!bytes||!bytes->GetBufferSize())throw std::runtime_error("Windows shader compilation failed");
    if(game_compiler)FreeLibrary(game_compiler);
    std::cout<<"{\"passed\":true,\"cpu_only\":true,\"windows_compiler_verified\":true,\"foreign_compiler_preloaded\":"
        <<(argc>1?"true":"false")<<",\"bits\":"<<sizeof(void*)*8<<"}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
