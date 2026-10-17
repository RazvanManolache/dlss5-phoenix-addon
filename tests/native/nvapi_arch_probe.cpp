// Driver entry-point availability only; no device, kernels, or inference.
#include <windows.h>
#include <iostream>

int main() {
    const auto name = sizeof(void*) == 8 ? L"nvapi64.dll" : L"nvapi.dll";
    const auto module = LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!module) return 2;
    const auto query = reinterpret_cast<void*(__cdecl*)(unsigned)>(
        GetProcAddress(module, "nvapi_QueryInterface"));
    bool available = query != nullptr;
    for (unsigned entry : {0xad1a677du, 0xe2436e22u, 0xdf295ea6u, 0x41c65285u,
                           0x24973538u, 0x329fe6e0u, 0x48f5b2eeu})
        available = available && query(entry) != nullptr;
    FreeLibrary(module);
    std::cout << "{\"bits\":" << sizeof(void*) * 8
              << ",\"native_entrypoints\":" << (available ? "true" : "false")
              << ",\"gpu_inference_executed\":false}\n";
    return available ? 0 : 3;
}
