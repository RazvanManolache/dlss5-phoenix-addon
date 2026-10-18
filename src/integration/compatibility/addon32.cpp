// Only graphics transport and UI live in the 32-bit game. Neural code stays
// in the same 64-bit Phoenix implementation used by the normal addon.
#define IMGUI_DISABLE_INCLUDE_IMCONFIG_H
#include <imgui.h>
#include <reshade.hpp>
#include "controller.h"
extern "C" __declspec(dllexport) const char* NAME = "Phoenix DLSS5 (32-bit compatibility)";
extern "C" __declspec(dllexport) const char* DESCRIPTION =
    "32-bit game transport with a private 64-bit D3D12 neural-rendering helper.";
namespace {
bool attached = false;
bool attach(HMODULE module) {
    if (attached)
        return true;
    if (!reshade::register_addon(module))
        return false;
    attached = true;
    phoenix_compat::attach(module);
    reshade::register_overlay("Phoenix DLSS5", phoenix_compat::panel);
    reshade::register_overlay(nullptr, phoenix_compat::compact);
    return true;
}
void detach(HMODULE module) {
    if (!attached)
        return;
    attached = false;
    reshade::unregister_overlay("Phoenix DLSS5", phoenix_compat::panel);
    reshade::unregister_overlay(nullptr, phoenix_compat::compact);
    phoenix_compat::detach();
    reshade::unregister_addon(module);
}
}
extern "C" __declspec(dllexport) bool AddonInit(HMODULE module, HMODULE) {
    return attach(module);
}
extern "C" __declspec(dllexport) void AddonUninit(HMODULE module, HMODULE) {
    detach(module);
}
BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH)
        return attach(module);
    // AddonUninit owns normal unloading. During process termination, Windows
    // owns thread teardown; joining a worker under the loader lock deadlocks.
    if (reason == DLL_PROCESS_DETACH && !reserved && !attached)
        return TRUE;
    return TRUE;
}
