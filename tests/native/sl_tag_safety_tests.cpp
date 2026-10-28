#define wmain packaged_load_main
#include "packaged_addon_load_tests.cpp"
#undef wmain
#include "../../third_party/streamline/sl_core_types.h"
#include <cstring>
struct Frame : sl::FrameToken { operator uint32_t() const override { return 7; } };
int wmain(int argc, wchar_t** argv) {
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
    try {
        require(argc == 3, "Usage: ADDON STREAMLINE_STUB");
        auto streamline = LoadLibraryW(argv[2]);
        require(streamline != nullptr, "Streamline stand-in failed to load");
        auto tag = reinterpret_cast<sl::Result(*)(const sl::FrameToken&, const sl::ViewportHandle&,
            const sl::ResourceTag*, uint32_t, sl::CommandBuffer*)>(GetProcAddress(streamline, "slSetTagForFrame"));
        auto old = reinterpret_cast<sl::Result(*)(const sl::ViewportHandle&, const sl::ResourceTag*,
            uint32_t, sl::CommandBuffer*)>(GetProcAddress(streamline, "slSetTag"));
        auto calls = reinterpret_cast<unsigned(*)()>(GetProcAddress(streamline, "ProbeCalls"));
        require(tag && old && calls, "Streamline stand-in exports missing");
        unsigned char original[16];
        std::memcpy(original, reinterpret_cast<void*>(tag), sizeof(original));
        auto addon = LoadLibraryW(argv[1]);
        require(addon != nullptr, "Addon failed to load");
        require(std::memcmp(original, reinterpret_cast<void*>(tag), sizeof(original)) != 0,
            "Streamline stand-in was not actually hooked");
        Frame frame;
        sl::ViewportHandle viewport(0u);
        // This is deliberately not a COM pointer: Vulkan image handles are opaque.
        sl::Resource resource(sl::ResourceType::eTex2d, reinterpret_cast<void*>(1));
        resource.width = 640; resource.height = 480;
        sl::ResourceTag tagged(&resource, 2u, sl::eValidUntilPresent);
        auto foreign_command = reinterpret_cast<sl::CommandBuffer*>(2);
        require(tag(frame, viewport, &tagged, 1, foreign_command) == sl::Result::eOk,
            "Frame tag was not preserved and forwarded");
        require(old(viewport, &tagged, 1, foreign_command) == sl::Result::eOk,
            "Legacy tag was not preserved and forwarded");
        require(calls() == 2, "Tag forwarding duplicated or skipped a call");
        auto uninit = reinterpret_cast<void(WINAPI*)(HMODULE, HMODULE)>(GetProcAddress(addon, "AddonUninit"));
        require(uninit != nullptr, "Uninit export missing");
        uninit(addon, GetModuleHandleW(nullptr));
        FreeLibrary(addon);
        std::cout << "{\"passed\":true,\"cpu_only\":true,\"foreign_image_not_dereferenced\":true,\"forwarded_calls\":2}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
