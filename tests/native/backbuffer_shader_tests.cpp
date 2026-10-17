#include "../../src/integration/compatibility/host/phoenix-format-adapter.h"
#include <iostream>

int main() {
    unsigned shaders = 0;
    for (const auto* source : {kPhoenixUnpackShader, kPhoenixPackShader}) {
        Microsoft::WRL::ComPtr<ID3DBlob> code, errors;
        const auto result = D3DCompile(source, std::strlen(source), "backbuffer-conversion",
            nullptr, nullptr, "main", "cs_5_0", D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (FAILED(result) || !code || !code->GetBufferSize()) {
            if (errors) std::cerr.write(static_cast<const char*>(errors->GetBufferPointer()), errors->GetBufferSize());
            return 1;
        }
        ++shaders;
    }
    std::cout << "{\"passed\":true,\"cpu_only\":true,\"compiled_shaders\":" << shaders << "}\n";
}
