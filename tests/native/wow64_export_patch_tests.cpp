// CPU-only regression for a worker restart that must reuse an immutable stub.
#include <windows.h>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include "umd_bridge.h"
#pragma comment(lib,"advapi32.lib")

static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    unsigned char* image = nullptr;
    void* high_image = nullptr;
    void* high_stub = nullptr;
    void* fragmented_image = nullptr;
    void* fragmented_stub = nullptr;
    unsigned char* imports_image = nullptr;
    wow64_umd_probe::ExportPatch patch{};
    try {
        // A private synthetic PE image forces the export-relative stub path;
        // no Windows, driver, or game image is modified by this test.
        for (uintptr_t address = 0x10000000; !image && address < 0x20000000; address += 65536)
            image = static_cast<unsigned char*>(VirtualAlloc(reinterpret_cast<void*>(address),
                4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE));
        check(image != nullptr, "Cannot allocate synthetic image");
        auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(image);
        dos->e_magic = IMAGE_DOS_SIGNATURE;
        dos->e_lfanew = 0x80;
        auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(image + dos->e_lfanew);
        nt->Signature = IMAGE_NT_SIGNATURE;
        nt->FileHeader.Machine = IMAGE_FILE_MACHINE_AMD64;
        nt->OptionalHeader.SizeOfImage = 4096;
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_EXPORT] = {0x200, 0x50};
        auto* exports = reinterpret_cast<IMAGE_EXPORT_DIRECTORY*>(image + 0x200);
        exports->NumberOfFunctions = exports->NumberOfNames = 1;
        exports->AddressOfFunctions = 0x280;
        exports->AddressOfNames = 0x290;
        exports->AddressOfNameOrdinals = 0x2a0;
        *reinterpret_cast<DWORD*>(image + 0x280) = 0x380;
        *reinterpret_cast<DWORD*>(image + 0x290) = 0x300;
        *reinterpret_cast<WORD*>(image + 0x2a0) = 0;
        std::memcpy(image + 0x300, "Probe", 6);
        const auto target = reinterpret_cast<void*>(uintptr_t(0x1000));
        auto* const slot = reinterpret_cast<DWORD*>(image + 0x280);
        void* first_stub = nullptr;
        unsigned char first_bytes[12]{};
        for (unsigned cycle = 0; cycle < 4; ++cycle) {
            check(wow64_umd_probe::redirect_export(reinterpret_cast<HMODULE>(image),
                "Probe", target, patch), "Export redirection failed");
            check(patch.before == 0x380, "Original export was not preserved");
            check(image + *slot == patch.thunk, "Export does not reference owned stub");
            MEMORY_BASIC_INFORMATION memory{};
            check(VirtualQuery(patch.thunk, &memory, sizeof(memory)) == sizeof(memory) &&
                memory.Protect == PAGE_EXECUTE_READ, "Published stub must remain RX");
            if (cycle == 0) {
                first_stub = patch.thunk;
                std::memcpy(first_bytes, patch.thunk, sizeof(first_bytes));
            } else {
                check(patch.thunk == first_stub, "Restart allocated a different stub");
                check(std::memcmp(first_bytes, patch.thunk, sizeof(first_bytes)) == 0,
                    "Restart changed the published stub");
            }
            // Mirror export restoration at native-worker retirement.
            DWORD previous = 0, ignored = 0;
            check(VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &previous) != 0,
                "Cannot restore synthetic export");
            *slot = patch.before;
            VirtualProtect(slot, sizeof(*slot), previous, &ignored);
            patch.slot = nullptr;
        }
        check(!wow64_umd_probe::redirect_export(reinterpret_cast<HMODULE>(image),
            "Probe", reinterpret_cast<void*>(uintptr_t(0x2000)), patch),
            "Changing an immutable stub target must be refused");
        // Native images and their 64-bit stubs need not fit below 2 GiB.
        high_image = VirtualAlloc(reinterpret_cast<void*>(uintptr_t(0x200000000ull)),
            4096, MEM_COMMIT|MEM_RESERVE, PAGE_READWRITE);
        check(high_image != nullptr, "Cannot allocate high synthetic image");
        std::memcpy(high_image, image, 4096);
        wow64_umd_probe::ExportPatch high_patch{};
        check(wow64_umd_probe::redirect_export(static_cast<HMODULE>(high_image),
            "Probe", target, high_patch), "High-image export redirection failed");
        high_stub = high_patch.thunk;
        check(uintptr_t(high_stub) > uintptr_t(high_image) &&
            uintptr_t(high_stub)-uintptr_t(high_image) <= UINT32_MAX,
            "High-image stub is outside the export RVA range");
        // Reserve (without committing RAM) the old scan's entire 128 MiB
        // window. A usable RVA beyond that arbitrary window must still work.
        constexpr SIZE_T crowded_bytes = 129ull*1024*1024;
        fragmented_image = VirtualAlloc(nullptr, crowded_bytes, MEM_RESERVE, PAGE_NOACCESS);
        check(fragmented_image != nullptr && VirtualAlloc(fragmented_image,4096,
            MEM_COMMIT,PAGE_READWRITE) == fragmented_image, "Cannot reserve crowded image");
        std::memcpy(fragmented_image, image, 4096);
        wow64_umd_probe::ExportPatch crowded_patch{};
        check(wow64_umd_probe::redirect_export(static_cast<HMODULE>(fragmented_image),
            "Probe", target, crowded_patch), "Fragmented-image export redirection failed");
        fragmented_stub = crowded_patch.thunk;
        check(uintptr_t(fragmented_stub)-uintptr_t(fragmented_image) >= crowded_bytes,
            "Crowded-image stub overlaps a reserved range");
        // A driver stack plus its directly loaded model can exceed 256 import
        // slots. Exercise real patch/restore behavior beyond that old ceiling.
        imports_image=static_cast<unsigned char*>(VirtualAlloc(nullptr,65536,
            MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE));
        check(imports_image!=nullptr,"Cannot allocate import fixture");
        std::memcpy(imports_image,image,4096);
        auto* imports_nt=reinterpret_cast<IMAGE_NT_HEADERS64*>(imports_image+dos->e_lfanew);
        imports_nt->OptionalHeader.SizeOfImage=65536;
        imports_nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT]={0x400,40};
        auto* descriptor=reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(imports_image+0x400);
        std::memset(descriptor,0,40);
        descriptor->Name=0x500;std::memcpy(imports_image+0x500,"kernel32.dll",13);
        descriptor->OriginalFirstThunk=0x1000;descriptor->FirstThunk=0x2000;
        auto* names=reinterpret_cast<IMAGE_THUNK_DATA64*>(imports_image+0x1000);
        auto* table=reinterpret_cast<IMAGE_THUNK_DATA64*>(imports_image+0x2000);
        auto* name=reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(imports_image+0x4000);
        std::memcpy(name->Name,"HeapAlloc",10);
        const auto original=uintptr_t(&HeapAlloc);
        for(unsigned i=0;i<300;++i){names[i].u1.AddressOfData=0x4000;table[i].u1.Function=original;}
        wow64_umd_probe::patch(reinterpret_cast<HMODULE>(imports_image));
        check(wow64_umd_probe::slot_count==300,"Native import adaptation stopped at the old slot ceiling");
        for(unsigned i=0;i<300;++i)check(table[i].u1.Function==uintptr_t(&wow64_runtime_diagnostics::heap_alloc),
            "An import after the old ceiling was not adapted");
        wow64_umd_probe::restore();
        for(unsigned i=0;i<300;++i)check(table[i].u1.Function==original,"Import restoration changed an original target");
        VirtualFree(imports_image,0,MEM_RELEASE);imports_image=nullptr;
        VirtualFree(fragmented_stub, 0, MEM_RELEASE);
        VirtualFree(fragmented_image, 0, MEM_RELEASE);
        VirtualFree(high_stub, 0, MEM_RELEASE);
        VirtualFree(high_image, 0, MEM_RELEASE);
        VirtualFree(patch.thunk, 0, MEM_RELEASE);
        VirtualFree(image, 0, MEM_RELEASE);
        std::cout << "{\"passed\":true,\"restart_cycles\":4,\"immutable_rx_stub\":true,\"high_image\":true,\"fragmented_image\":true,\"adapted_imports\":300}\n";
        return 0;
    } catch (const std::exception& error) {
        wow64_umd_probe::restore();
        if(imports_image)VirtualFree(imports_image,0,MEM_RELEASE);
        if (fragmented_stub) VirtualFree(fragmented_stub, 0, MEM_RELEASE);
        if (fragmented_image) VirtualFree(fragmented_image, 0, MEM_RELEASE);
        if (high_stub) VirtualFree(high_stub, 0, MEM_RELEASE);
        if (high_image) VirtualFree(high_image, 0, MEM_RELEASE);
        if (patch.thunk) VirtualFree(patch.thunk, 0, MEM_RELEASE);
        if (image) VirtualFree(image, 0, MEM_RELEASE);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
