#pragma once
#include <Xinput.h>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <array>
#include "../../../third_party/minhook/include/MinHook.h"

// A test-only virtual pad, local to this process. No desktop input or focus
// changes. The external screenshot controller decides WHEN confirmation is
// appropriate; the addon accepts at most three short pulses per launch.
namespace phoenix_addon::menu_helper {
using GetState = DWORD(WINAPI*)(DWORD, XINPUT_STATE*);
inline std::array<GetState, 4> originals{};
inline std::array<void*, 4> targets{};
using GetCaps = DWORD(WINAPI*)(DWORD, DWORD, XINPUT_CAPABILITIES*);
inline std::array<GetCaps, 3> caps_originals{};
inline std::array<void*, 3> caps_targets{};
inline std::atomic<ULONGLONG> deadline{0}, pulse_end{0};
inline std::atomic<unsigned> polls{0}, pressed_polls{0}, packet{0};
inline std::atomic<unsigned> caps_polls{0};
inline std::atomic<bool> interrupted{false};
inline bool configured = false, stopped = false;
inline unsigned actions = 0;
inline int last_token = 0;
inline ULONGLONG next_tick = 0;
inline std::string status = "disabled";

template <size_t Slot>
DWORD WINAPI get_caps(DWORD index, DWORD flags, XINPUT_CAPABILITIES* output) {
    const auto result = caps_originals[Slot] ? caps_originals[Slot](index, flags, output)
                                             : ERROR_DEVICE_NOT_CONNECTED;
    if (index != 0 || !output || GetTickCount64() >= deadline.load() || interrupted.load())
        return result;
    ++caps_polls;
    if (result != ERROR_SUCCESS) {
        *output = {};
        output->Type = XINPUT_DEVTYPE_GAMEPAD;
        output->SubType = XINPUT_DEVSUBTYPE_GAMEPAD;
        output->Gamepad.wButtons = XINPUT_GAMEPAD_A;
    }
    return ERROR_SUCCESS;
}

template <size_t Slot> DWORD WINAPI get_state(DWORD index, XINPUT_STATE* output) {
    const auto result =
        originals[Slot] ? originals[Slot](index, output) : ERROR_DEVICE_NOT_CONNECTED;
    const auto now = GetTickCount64();
    if (index != 0 || !output || now >= deadline.load() || interrupted.load())
        return result;
    ++polls;
    if (result == ERROR_SUCCESS) {
        const auto& gamepad = output->Gamepad;
        if (gamepad.wButtons || gamepad.bLeftTrigger > 30 || gamepad.bRightTrigger > 30 ||
            abs(int(gamepad.sThumbLX)) > 8000 || abs(int(gamepad.sThumbLY)) > 8000 ||
            abs(int(gamepad.sThumbRX)) > 8000 || abs(int(gamepad.sThumbRY)) > 8000) {
            interrupted = true;
            pulse_end = 0;
            deadline = 0;
            return result; // A physical controller always wins.
        }
    }
    *output = {};
    output->dwPacketNumber = packet.load();
    if (now < pulse_end.load()) {
        output->Gamepad.wButtons = XINPUT_GAMEPAD_A;
        ++pressed_polls;
    } else
        output->dwPacketNumber += 1;
    return ERROR_SUCCESS;
}

inline void configure(bool enabled) {
    if (!enabled || configured)
        return;
    wchar_t exe[32768]{};
    GetModuleFileNameW(nullptr, exe, 32768);
    if (_wcsicmp(std::filesystem::path(exe).filename().c_str(), L"re9.exe"))
        return;
    configured = true;
    deadline = GetTickCount64() + 240000;
    status = "waiting_for_request";
}

inline void install() {
    const char* libraries[] = {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll",
                               "xinput1_4.dll"};
    void* detours[] = {
        reinterpret_cast<void*>(&get_state<0>), reinterpret_cast<void*>(&get_state<1>),
        reinterpret_cast<void*>(&get_state<2>), reinterpret_cast<void*>(&get_state<3>)};
    for (size_t iteration_index = 0; iteration_index < originals.size(); ++iteration_index) {
        if (targets[iteration_index])
            continue;
        auto module = GetModuleHandleA(libraries[iteration_index]);
        auto target =
            module ? reinterpret_cast<void*>(GetProcAddress(
                         module, iteration_index == 3 ? MAKEINTRESOURCEA(100) : "XInputGetState"))
                   : nullptr;
        if (!target)
            continue;
        bool duplicate = false;
        for (auto existing : targets)
            if (existing == target)
                duplicate = true;
        if (duplicate)
            continue;
        const auto init = MH_Initialize();
        if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED)
            continue;
        if (MH_CreateHook(target, detours[iteration_index],
                          reinterpret_cast<void**>(&originals[iteration_index])) != MH_OK)
            continue;
        if (MH_EnableHook(target) != MH_OK) {
            MH_RemoveHook(target);
            originals[iteration_index] = nullptr;
            continue;
        }
        targets[iteration_index] = target;
    }
    void* cap_detours[] = {reinterpret_cast<void*>(&get_caps<0>),
                           reinterpret_cast<void*>(&get_caps<1>),
                           reinterpret_cast<void*>(&get_caps<2>)};
    for (size_t iteration_index = 0; iteration_index < caps_originals.size(); ++iteration_index) {
        if (caps_targets[iteration_index])
            continue;
        auto module = GetModuleHandleA(libraries[iteration_index]);
        auto target = module
                          ? reinterpret_cast<void*>(GetProcAddress(module, "XInputGetCapabilities"))
                          : nullptr;
        if (!target)
            continue;
        bool duplicate = false;
        for (auto existing : caps_targets)
            if (existing == target)
                duplicate = true;
        if (duplicate)
            continue;
        if (MH_CreateHook(target, cap_detours[iteration_index],
                          reinterpret_cast<void**>(&caps_originals[iteration_index])) != MH_OK)
            continue;
        if (MH_EnableHook(target) != MH_OK) {
            MH_RemoveHook(target);
            caps_originals[iteration_index] = nullptr;
            continue;
        }
        caps_targets[iteration_index] = target;
    }
}

inline void tick(const std::filesystem::path& folder, const std::filesystem::path& output) {
    if (!configured)
        return;
    const auto now = GetTickCount64();
    if (now < next_tick)
        return;
    next_tick = now + 100;
    if (!stopped && (now >= deadline.load() || interrupted.load())) {
        stopped = true;
        pulse_end = 0;
        deadline = 0;
        status = interrupted.load() ? "physical_controller_cancelled" : "expired";
    }
    if (!stopped)
        install();
    try {
        std::ifstream file(folder / "PhoenixMenuRequest.json");
        if (file) {
            Json request;
            file >> request;
            const auto token = request.value("token", 0);
            if (request.value("pid", 0ul) == GetCurrentProcessId() && token > last_token) {
                last_token = token;
                const auto action = request.value("action", std::string());
                if (action == "stop") {
                    // Keep the idle virtual pad connected until this bounded
                    // test exits. Disconnecting here opens RE9's modal dialog.
                    // No later confirm request can reactivate a stopped helper.
                    stopped = true;
                    pulse_end = 0;
                    status = "stopped";
                } else if (!stopped && action == "confirm" && actions < 3 &&
                           now >= pulse_end.load() + 1000) {
                    ++actions;
                    packet += 2;
                    pulse_end = now + 300;
                    status = "confirm_pulse";
                } else
                    status = "request_refused";
            }
        }
    } catch (...) {
        status = "invalid_request";
    }
    unsigned hooks = 0;
    for (auto target : targets)
        if (target)
            ++hooks;
    const Json info{{"pid", GetCurrentProcessId()},
                    {"status", status},
                    {"token", last_token},
                    {"actions", actions},
                    {"hooks", hooks},
                    {"polls", polls.load()},
                    {"capability_polls", caps_polls.load()},
                    {"pressed_polls", pressed_polls.load()},
                    {"active", !stopped},
                    {"virtual_pad_connected", now < deadline.load()},
                    {"desktop_input", false}};
    const auto temp = output / "PhoenixMenuStatus.tmp";
    {
        std::ofstream file(temp);
        if (file)
            file << info.dump(2);
    }
    MoveFileExW(temp.c_str(), (output / "PhoenixMenuStatus.json").c_str(),
                MOVEFILE_REPLACE_EXISTING);
}

inline void detach() {
    deadline = 0;
    pulse_end = 0;
    for (auto target : targets)
        if (target) {
            MH_DisableHook(target);
            MH_RemoveHook(target);
        }
    for (auto target : caps_targets)
        if (target) {
            MH_DisableHook(target);
            MH_RemoveHook(target);
        }
}
}
