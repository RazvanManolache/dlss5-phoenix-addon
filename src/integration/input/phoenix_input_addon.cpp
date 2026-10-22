// Phoenix Input: background input injection companion for the Phoenix DLSS5
// bridge. Loaded by ReShade beside Phoenix.addon32/64; it shares the game's
// PhoenixOutput folder but owns its own mailbox (PhoenixOutput/input).
//
// Goals
//  * Let an automated harness drive a game while the user keeps working in
//    other windows: no SetForegroundWindow, no SendInput, no real cursor moves.
//  * "Virtual focus": the game believes it is the foreground/active window and
//    never sees focus loss (GetForegroundWindow/GetActiveWindow/GetFocus hooks,
//    WM_ACTIVATE/WM_ACTIVATEAPP/WM_KILLFOCUS suppression, DirectInput forced to
//    background cooperative level, ClipCursor/SetCursorPos virtualised).
//  * Virtual devices merged into every common polling path: window messages,
//    GetAsyncKeyState/GetKeyState/GetKeyboardState, GetCursorPos, Raw Input
//    (WM_INPUT + GetRawInputData), DirectInput 8 (GetDeviceState/GetDeviceData)
//    and XInput.
//  * Isolation: while virtual focus is on and the real foreground window is
//    not the game, real keyboard/mouse state is masked so the user's typing in
//    another window never leaks into the game.
//
// Nothing here is active until the harness enables it (mailbox "focus" op or
// PhoenixInput.json {"focus_override": true}). With virtual focus off, every
// hook is a pure pass-through.
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define DIRECTINPUT_VERSION 0x0800
#include <windows.h>
#include <dinput.h>
#include <Xinput.h>
#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <reshade.hpp>
#include <nlohmann/json.hpp>
#include "MinHook.h"

using Json = nlohmann::json;
namespace fs = std::filesystem;

extern "C" __declspec(dllexport) const char* NAME = "Phoenix Input";
extern "C" __declspec(dllexport) const char* DESCRIPTION =
    "Background input injection and virtual focus for unattended Phoenix DLSS5 testing.";

namespace {
// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------
HMODULE g_module = nullptr;
std::recursive_mutex g_lock;
std::atomic<HWND> g_hwnd{nullptr};
std::atomic<bool> g_focus{false};       // virtual focus / isolation enabled
std::atomic<bool> g_pad_enabled{false}; // virtual XInput pad connected
std::atomic<bool> g_stop{false};
std::thread g_worker;
fs::path g_folder;
std::string g_session;
std::string g_last_error;

std::array<std::atomic<uint8_t>, 256> g_vk{}; // virtual key/button state
POINT g_cursor{};                              // virtual cursor, client coords
bool g_cursor_valid = false;
long g_di_dx = 0, g_di_dy = 0, g_di_dz = 0;    // pending DirectInput deltas
XINPUT_GAMEPAD g_pad{};
DWORD g_pad_packet = 1;

struct Counters {
    std::atomic<uint64_t> foreground{0}, async_key{0}, key_state{0}, keyboard_state{0},
        cursor{0}, set_cursor{0}, raw_reads{0}, raw_injected{0}, dinput_state{0},
        dinput_data{0}, xinput{0}, messages{0}, focus_swallowed{0};
} g_count;

// Timed actions run on the worker thread.
struct Timed { ULONGLONG due; std::function<void()> run; };
std::vector<Timed> g_timed;

// DirectInput buffered events waiting to be appended to GetDeviceData.
struct DiEvent { DWORD ofs, data; };
std::deque<DiEvent> g_di_keyboard_events, g_di_mouse_events;
std::map<void*, int> g_di_kind; // device -> 1 keyboard, 2 mouse

// Raw input events fabricated for WM_INPUT.
constexpr uintptr_t kRawSentinel = 0x5EED0000u;
std::array<RAWINPUT, 256> g_raw_ring{};
uint32_t g_raw_next = 0;

// ---------------------------------------------------------------------------
// Original functions
// ---------------------------------------------------------------------------
decltype(&GetForegroundWindow) o_GetForegroundWindow = nullptr;
decltype(&GetActiveWindow) o_GetActiveWindow = nullptr;
decltype(&GetFocus) o_GetFocus = nullptr;
decltype(&GetAsyncKeyState) o_GetAsyncKeyState = nullptr;
decltype(&GetKeyState) o_GetKeyState = nullptr;
decltype(&GetKeyboardState) o_GetKeyboardState = nullptr;
decltype(&GetCursorPos) o_GetCursorPos = nullptr;
decltype(&SetCursorPos) o_SetCursorPos = nullptr;
decltype(&ClipCursor) o_ClipCursor = nullptr;
decltype(&GetRawInputData) o_GetRawInputData = nullptr;
using DiGetState = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, LPVOID);
using DiGetData = HRESULT(STDMETHODCALLTYPE*)(void*, DWORD, LPDIDEVICEOBJECTDATA, LPDWORD, DWORD);
using DiSetFormat = HRESULT(STDMETHODCALLTYPE*)(void*, LPCDIDATAFORMAT);
using DiSetCoop = HRESULT(STDMETHODCALLTYPE*)(void*, HWND, DWORD);
using DiAcquire = HRESULT(STDMETHODCALLTYPE*)(void*);
struct DiHooks { DiGetState state = nullptr; DiGetData data = nullptr; DiSetFormat format = nullptr;
                 DiSetCoop coop = nullptr; DiAcquire acquire = nullptr; };
std::array<DiHooks, 2> o_di{}; // [0] ANSI vtable, [1] wide vtable
using XGetState = DWORD(WINAPI*)(DWORD, XINPUT_STATE*);
std::array<XGetState, 4> o_xinput{};
std::array<void*, 4> xinput_targets{};
std::vector<void*> g_hook_targets;
WNDPROC o_wndproc = nullptr;
HWND g_subclassed = nullptr;
bool g_subclass_unicode = true;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
HWND game_root() {
    HWND hwnd = g_hwnd.load();
    if (!hwnd || !IsWindow(hwnd)) return nullptr;
    HWND root = GetAncestor(hwnd, GA_ROOT);
    return root ? root : hwnd;
}
bool real_foreground_is_game() {
    HWND root = game_root();
    if (!root) return true;
    HWND fg = o_GetForegroundWindow ? o_GetForegroundWindow() : GetForegroundWindow();
    if (!fg) return false;
    DWORD pid = 0;
    GetWindowThreadProcessId(fg, &pid);
    return pid == GetCurrentProcessId();
}
// True while real devices must be hidden from the game.
bool isolate() { return g_focus.load() && !real_foreground_is_game(); }
bool virtual_down(int vk) { return vk >= 0 && vk < 256 && (g_vk[size_t(vk)].load() & 0x80); }

int dik_for_vk(int vk) {
    UINT sc = MapVirtualKeyW(UINT(vk), MAPVK_VK_TO_VSC_EX);
    switch (vk) { // MapVirtualKey omits the extended prefix for these.
    case VK_UP: case VK_DOWN: case VK_LEFT: case VK_RIGHT: case VK_PRIOR: case VK_NEXT:
    case VK_HOME: case VK_END: case VK_INSERT: case VK_DELETE: case VK_RCONTROL: case VK_RMENU:
    case VK_DIVIDE: case VK_LWIN: case VK_RWIN: case VK_APPS:
        return int((sc & 0xFF) | 0x80);
    default: break;
    }
    if ((sc >> 8) == 0xE0 || (sc >> 8) == 0xE1) return int((sc & 0xFF) | 0x80);
    return int(sc & 0xFF);
}
bool extended_vk(int vk) { return dik_for_vk(vk) & 0x80; }

LPARAM key_lparam(int vk, bool down) {
    const UINT sc = MapVirtualKeyW(UINT(vk), MAPVK_VK_TO_VSC) & 0xFF;
    LPARAM value = 1 | (LPARAM(sc) << 16);
    if (extended_vk(vk)) value |= LPARAM(1) << 24;
    if (!down) value |= (LPARAM(1) << 30) | (LPARAM(1) << 31);
    return value;
}
void post(UINT message, WPARAM wparam, LPARAM lparam) {
    HWND hwnd = g_hwnd.load();
    if (!hwnd) return;
    ++g_count.messages;
    if (IsWindowUnicode(hwnd)) PostMessageW(hwnd, message, wparam, lparam);
    else PostMessageA(hwnd, message, wparam, lparam);
}
bool raw_registered(USHORT usage) {
    UINT count = 0;
    if (GetRegisteredRawInputDevices(nullptr, &count, sizeof(RAWINPUTDEVICE)) != 0 && count == 0) return false;
    if (count == 0) return false;
    std::vector<RAWINPUTDEVICE> devices(count);
    if (GetRegisteredRawInputDevices(devices.data(), &count, sizeof(RAWINPUTDEVICE)) == UINT(-1)) return false;
    for (UINT index = 0; index < count; ++index)
        if (devices[index].usUsagePage == 1 && devices[index].usUsage == usage) return true;
    return false;
}
void post_raw(const RAWINPUT& input) {
    std::lock_guard lock(g_lock);
    const uint32_t slot = g_raw_next++ & 0xFF;
    g_raw_ring[slot] = input;
    ++g_count.raw_injected;
    post(WM_INPUT, RIM_INPUT, LPARAM(kRawSentinel | slot));
}

// ---------------------------------------------------------------------------
// Virtual device actions (worker thread)
// ---------------------------------------------------------------------------
void client_center(POINT& point) {
    RECT rect{};
    HWND hwnd = g_hwnd.load();
    if (hwnd && GetClientRect(hwnd, &rect)) point = {(rect.right - rect.left) / 2, (rect.bottom - rect.top) / 2};
}
WPARAM mouse_flags() {
    WPARAM flags = 0;
    if (virtual_down(VK_LBUTTON)) flags |= MK_LBUTTON;
    if (virtual_down(VK_RBUTTON)) flags |= MK_RBUTTON;
    if (virtual_down(VK_MBUTTON)) flags |= MK_MBUTTON;
    if (virtual_down(VK_SHIFT)) flags |= MK_SHIFT;
    if (virtual_down(VK_CONTROL)) flags |= MK_CONTROL;
    return flags;
}
LPARAM cursor_lparam() {
    std::lock_guard lock(g_lock);
    if (!g_cursor_valid) { client_center(g_cursor); g_cursor_valid = true; }
    return MAKELPARAM(short(g_cursor.x), short(g_cursor.y));
}
void set_key(int vk, bool down) {
    if (vk <= 0 || vk > 255) return;
    const bool was = virtual_down(vk);
    g_vk[size_t(vk)] = down ? 0x80 : 0;
    // Generic modifiers mirror their left-hand variant for state queries.
    if (vk == VK_LSHIFT || vk == VK_RSHIFT) g_vk[VK_SHIFT] = down ? 0x80 : 0;
    if (vk == VK_LCONTROL || vk == VK_RCONTROL) g_vk[VK_CONTROL] = down ? 0x80 : 0;
    if (vk == VK_LMENU || vk == VK_RMENU) g_vk[VK_MENU] = down ? 0x80 : 0;
    const bool alt = virtual_down(VK_MENU);
    const UINT message = down ? (alt || vk == VK_MENU ? WM_SYSKEYDOWN : WM_KEYDOWN)
                              : (alt || vk == VK_MENU ? WM_SYSKEYUP : WM_KEYUP);
    LPARAM lparam = key_lparam(vk, down);
    if (down && was) lparam |= LPARAM(1) << 30; // auto-repeat
    post(message, WPARAM(vk), lparam);
    if (down) {
        UINT ch = MapVirtualKeyW(UINT(vk), MAPVK_VK_TO_CHAR) & 0x7FFF;
        if (vk == VK_RETURN) ch = '\r';
        if (ch >= 'A' && ch <= 'Z' && !virtual_down(VK_SHIFT)) ch = ch - 'A' + 'a';
        if (ch && vk != VK_ESCAPE && vk != VK_BACK && vk != VK_TAB) post(WM_CHAR, WPARAM(ch), lparam);
        else if (vk == VK_ESCAPE) post(WM_CHAR, 27, lparam);
    }
    if (!was || !down) {
        std::lock_guard lock(g_lock);
        const int dik = dik_for_vk(vk);
        if (dik) g_di_keyboard_events.push_back({DWORD(dik), DWORD(down ? 0x80 : 0)});
        while (g_di_keyboard_events.size() > 128) g_di_keyboard_events.pop_front();
    }
    if (raw_registered(6)) {
        RAWINPUT input{};
        input.header.dwType = RIM_TYPEKEYBOARD;
        input.header.dwSize = sizeof(RAWINPUT);
        input.header.wParam = RIM_INPUT;
        input.data.keyboard.MakeCode = USHORT(MapVirtualKeyW(UINT(vk), MAPVK_VK_TO_VSC) & 0xFF);
        input.data.keyboard.Flags = USHORT((down ? RI_KEY_MAKE : RI_KEY_BREAK) | (extended_vk(vk) ? RI_KEY_E0 : 0));
        input.data.keyboard.VKey = USHORT(vk);
        input.data.keyboard.Message = message;
        post_raw(input);
    }
}
void set_button(int vk, bool down) {
    g_vk[size_t(vk)] = down ? 0x80 : 0;
    const LPARAM where = cursor_lparam();
    UINT message = 0;
    USHORT raw_flags = 0;
    DWORD di_ofs = DIMOFS_BUTTON0;
    if (vk == VK_LBUTTON) { message = down ? WM_LBUTTONDOWN : WM_LBUTTONUP; raw_flags = down ? RI_MOUSE_LEFT_BUTTON_DOWN : RI_MOUSE_LEFT_BUTTON_UP; di_ofs = DIMOFS_BUTTON0; }
    if (vk == VK_RBUTTON) { message = down ? WM_RBUTTONDOWN : WM_RBUTTONUP; raw_flags = down ? RI_MOUSE_RIGHT_BUTTON_DOWN : RI_MOUSE_RIGHT_BUTTON_UP; di_ofs = DIMOFS_BUTTON1; }
    if (vk == VK_MBUTTON) { message = down ? WM_MBUTTONDOWN : WM_MBUTTONUP; raw_flags = down ? RI_MOUSE_MIDDLE_BUTTON_DOWN : RI_MOUSE_MIDDLE_BUTTON_UP; di_ofs = DIMOFS_BUTTON2; }
    if (!message) return;
    post(message, mouse_flags(), where);
    {
        std::lock_guard lock(g_lock);
        g_di_mouse_events.push_back({di_ofs, DWORD(down ? 0x80 : 0)});
        while (g_di_mouse_events.size() > 128) g_di_mouse_events.pop_front();
    }
    if (raw_registered(2)) {
        RAWINPUT input{};
        input.header.dwType = RIM_TYPEMOUSE;
        input.header.dwSize = sizeof(RAWINPUT);
        input.header.wParam = RIM_INPUT;
        input.data.mouse.usButtonFlags = raw_flags;
        post_raw(input);
    }
}
void move_mouse(long dx, long dy, bool absolute, long x, long y) {
    {
        std::lock_guard lock(g_lock);
        if (!g_cursor_valid) { client_center(g_cursor); g_cursor_valid = true; }
        if (absolute) { dx = x - g_cursor.x; dy = y - g_cursor.y; g_cursor = {x, y}; }
        else { g_cursor.x += dx; g_cursor.y += dy; }
        g_di_dx += dx; g_di_dy += dy;
        if (dx) g_di_mouse_events.push_back({DWORD(DIMOFS_X), DWORD(dx)});
        if (dy) g_di_mouse_events.push_back({DWORD(DIMOFS_Y), DWORD(dy)});
        while (g_di_mouse_events.size() > 128) g_di_mouse_events.pop_front();
    }
    post(WM_MOUSEMOVE, mouse_flags(), cursor_lparam());
    if (raw_registered(2) && (dx || dy)) {
        RAWINPUT input{};
        input.header.dwType = RIM_TYPEMOUSE;
        input.header.dwSize = sizeof(RAWINPUT);
        input.header.wParam = RIM_INPUT;
        input.data.mouse.usFlags = MOUSE_MOVE_RELATIVE;
        input.data.mouse.lLastX = dx;
        input.data.mouse.lLastY = dy;
        post_raw(input);
    }
}
void wheel(int delta) {
    {
        std::lock_guard lock(g_lock);
        g_di_dz += delta;
        g_di_mouse_events.push_back({DWORD(DIMOFS_Z), DWORD(delta)});
    }
    POINT screen{};
    {
        std::lock_guard lock(g_lock);
        screen = g_cursor;
    }
    if (HWND hwnd = g_hwnd.load()) ClientToScreen(hwnd, &screen);
    post(WM_MOUSEWHEEL, MAKEWPARAM(WORD(mouse_flags()), WORD(short(delta))), MAKELPARAM(short(screen.x), short(screen.y)));
    if (raw_registered(2)) {
        RAWINPUT input{};
        input.header.dwType = RIM_TYPEMOUSE;
        input.header.dwSize = sizeof(RAWINPUT);
        input.header.wParam = RIM_INPUT;
        input.data.mouse.usButtonFlags = RI_MOUSE_WHEEL;
        input.data.mouse.usButtonData = USHORT(short(delta));
        post_raw(input);
    }
}
void release_all() {
    for (int vk = 1; vk < 256; ++vk) {
        if (!virtual_down(vk)) continue;
        if (vk == VK_LBUTTON || vk == VK_RBUTTON || vk == VK_MBUTTON) set_button(vk, false);
        else set_key(vk, false);
    }
    std::lock_guard lock(g_lock);
    g_pad = {};
    ++g_pad_packet;
}
void announce_focus() {
    HWND hwnd = g_hwnd.load();
    if (!hwnd) return;
    // Wake games that paused before virtual focus was enabled.
    post(WM_ACTIVATEAPP, TRUE, 0);
    post(WM_NCACTIVATE, TRUE, 0);
    post(WM_ACTIVATE, WA_ACTIVE, 0);
    post(WM_SETFOCUS, 0, 0);
}

// ---------------------------------------------------------------------------
// Hooks
// ---------------------------------------------------------------------------
HWND WINAPI h_GetForegroundWindow() {
    ++g_count.foreground;
    if (g_focus.load()) if (HWND root = game_root()) return root;
    return o_GetForegroundWindow();
}
HWND WINAPI h_GetActiveWindow() {
    if (g_focus.load()) if (HWND root = game_root()) {
        DWORD pid = 0;
        if (GetWindowThreadProcessId(root, &pid) == GetCurrentThreadId()) return root;
    }
    return o_GetActiveWindow();
}
HWND WINAPI h_GetFocus() {
    if (g_focus.load()) if (HWND hwnd = g_hwnd.load()) {
        DWORD pid = 0;
        if (GetWindowThreadProcessId(hwnd, &pid) == GetCurrentThreadId()) return hwnd;
    }
    return o_GetFocus();
}
SHORT WINAPI h_GetAsyncKeyState(int vk) {
    ++g_count.async_key;
    SHORT value = isolate() ? 0 : o_GetAsyncKeyState(vk);
    if (virtual_down(vk)) value = SHORT(value | 0x8000);
    return value;
}
SHORT WINAPI h_GetKeyState(int vk) {
    ++g_count.key_state;
    SHORT value = o_GetKeyState(vk);
    if (isolate() && vk != VK_CAPITAL && vk != VK_NUMLOCK && vk != VK_SCROLL) value = 0;
    if (virtual_down(vk)) value = SHORT(value | 0x8000);
    return value;
}
BOOL WINAPI h_GetKeyboardState(PBYTE state) {
    ++g_count.keyboard_state;
    const BOOL result = o_GetKeyboardState(state);
    if (!result || !state) return result;
    const bool hide = isolate();
    for (int vk = 0; vk < 256; ++vk) {
        if (hide && vk != VK_CAPITAL && vk != VK_NUMLOCK && vk != VK_SCROLL) state[vk] &= 0x01;
        if (virtual_down(vk)) state[vk] |= 0x80;
    }
    return result;
}
BOOL WINAPI h_GetCursorPos(LPPOINT point) {
    ++g_count.cursor;
    if (point && isolate()) {
        HWND hwnd = g_hwnd.load();
        std::lock_guard lock(g_lock);
        if (!g_cursor_valid) { client_center(g_cursor); g_cursor_valid = true; }
        POINT screen = g_cursor;
        if (hwnd) ClientToScreen(hwnd, &screen);
        *point = screen;
        return TRUE;
    }
    return o_GetCursorPos(point);
}
BOOL WINAPI h_SetCursorPos(int x, int y) {
    ++g_count.set_cursor;
    if (isolate()) {
        POINT client{x, y};
        if (HWND hwnd = g_hwnd.load()) ScreenToClient(hwnd, &client);
        std::lock_guard lock(g_lock);
        g_cursor = client;
        g_cursor_valid = true;
        return TRUE; // Never steal the user's real cursor.
    }
    return o_SetCursorPos(x, y);
}
BOOL WINAPI h_ClipCursor(const RECT* rect) {
    if (rect && g_focus.load() && !real_foreground_is_game()) return TRUE;
    return o_ClipCursor(rect);
}
UINT WINAPI h_GetRawInputData(HRAWINPUT handle, UINT command, LPVOID data, PUINT size, UINT header_size) {
    ++g_count.raw_reads;
    const auto value = uintptr_t(handle);
    if ((value & ~uintptr_t(0xFF)) == kRawSentinel && size && header_size == sizeof(RAWINPUTHEADER)) {
        RAWINPUT input;
        {
            std::lock_guard lock(g_lock);
            input = g_raw_ring[value & 0xFF];
        }
        const UINT bytes = command == RID_HEADER ? UINT(sizeof(RAWINPUTHEADER)) : UINT(sizeof(RAWINPUT));
        if (!data) { *size = bytes; return 0; }
        if (*size < bytes) { *size = bytes; SetLastError(ERROR_INSUFFICIENT_BUFFER); return UINT(-1); }
        memcpy(data, &input, bytes);
        return bytes;
    }
    const UINT result = o_GetRawInputData(handle, command, data, size, header_size);
    if (data && result != UINT(-1) && result >= sizeof(RAWINPUTHEADER) && command == RID_INPUT && isolate()) {
        auto* input = static_cast<RAWINPUT*>(data);
        if (input->header.dwType == RIM_TYPEMOUSE) {
            input->data.mouse.lLastX = input->data.mouse.lLastY = 0;
            input->data.mouse.usButtonFlags = 0;
            input->data.mouse.usButtonData = 0;
        } else if (input->header.dwType == RIM_TYPEKEYBOARD) {
            input->data.keyboard.VKey = 0xFF; // "no key" per Raw Input docs
            input->data.keyboard.MakeCode = 0;
        }
    }
    return result;
}

template <size_t Wide> HRESULT STDMETHODCALLTYPE h_DiSetFormat(void* device, LPCDIDATAFORMAT format) {
    if (format) {
        std::lock_guard lock(g_lock);
        if (format->dwDataSize == 256) g_di_kind[device] = 1;
        else if (format->dwDataSize == sizeof(DIMOUSESTATE) || format->dwDataSize == sizeof(DIMOUSESTATE2)) g_di_kind[device] = 2;
    }
    return o_di[Wide].format(device, format);
}
template <size_t Wide> HRESULT STDMETHODCALLTYPE h_DiSetCoop(void* device, HWND hwnd, DWORD flags) {
    if (g_focus.load() && (flags & DISCL_FOREGROUND)) {
        // Foreground devices are unacquired whenever the game loses focus.
        flags = (flags & ~(DISCL_FOREGROUND | DISCL_EXCLUSIVE)) | DISCL_BACKGROUND | DISCL_NONEXCLUSIVE;
    }
    return o_di[Wide].coop(device, hwnd, flags);
}
int di_kind(void* device, DWORD bytes) {
    std::lock_guard lock(g_lock);
    auto found = g_di_kind.find(device);
    if (found != g_di_kind.end()) return found->second;
    if (bytes == 256) return 1;
    if (bytes == sizeof(DIMOUSESTATE) || bytes == sizeof(DIMOUSESTATE2)) return 2;
    return 0;
}
template <size_t Wide> HRESULT STDMETHODCALLTYPE h_DiGetState(void* device, DWORD bytes, LPVOID data) {
    ++g_count.dinput_state;
    HRESULT result = o_di[Wide].state(device, bytes, data);
    const int kind = di_kind(device, bytes);
    if (!data || !kind || !g_focus.load()) return result;
    if (FAILED(result)) {
        o_di[Wide].acquire(device);
        result = o_di[Wide].state(device, bytes, data);
        if (FAILED(result)) { memset(data, 0, bytes); result = DI_OK; }
    }
    const bool hide = isolate();
    if (kind == 1 && bytes >= 256) {
        auto* keys = static_cast<uint8_t*>(data);
        if (hide) memset(keys, 0, 256);
        for (int vk = 1; vk < 256; ++vk)
            if (virtual_down(vk)) if (int dik = dik_for_vk(vk)) keys[dik] = 0x80;
    } else if (kind == 2 && bytes >= sizeof(DIMOUSESTATE)) {
        auto* mouse = static_cast<DIMOUSESTATE*>(data);
        if (hide) memset(data, 0, bytes);
        std::lock_guard lock(g_lock);
        mouse->lX += g_di_dx; mouse->lY += g_di_dy; mouse->lZ += g_di_dz;
        g_di_dx = g_di_dy = g_di_dz = 0;
        if (virtual_down(VK_LBUTTON)) mouse->rgbButtons[0] = 0x80;
        if (virtual_down(VK_RBUTTON)) mouse->rgbButtons[1] = 0x80;
        if (virtual_down(VK_MBUTTON)) mouse->rgbButtons[2] = 0x80;
    }
    return result;
}
template <size_t Wide> HRESULT STDMETHODCALLTYPE h_DiGetData(void* device, DWORD object_bytes, LPDIDEVICEOBJECTDATA data, LPDWORD count, DWORD flags) {
    ++g_count.dinput_data;
    const DWORD capacity = count ? *count : 0;
    HRESULT result = o_di[Wide].data(device, object_bytes, data, count, flags);
    if (!count || !g_focus.load() || (flags & DIGDD_PEEK)) return result;
    const int kind = di_kind(device, 0);
    if (!kind) return result;
    if (FAILED(result)) { o_di[Wide].acquire(device); *count = 0; result = DI_OK; }
    if (isolate()) *count = 0;
    if (!data || object_bytes < 16) return result;
    std::lock_guard lock(g_lock);
    auto& queue = kind == 1 ? g_di_keyboard_events : g_di_mouse_events;
    DWORD written = *count;
    while (!queue.empty() && written < capacity) {
        auto* item = reinterpret_cast<DIDEVICEOBJECTDATA*>(reinterpret_cast<uint8_t*>(data) + size_t(written) * object_bytes);
        memset(item, 0, object_bytes);
        item->dwOfs = queue.front().ofs;
        item->dwData = queue.front().data;
        item->dwTimeStamp = GetTickCount();
        item->dwSequence = GetTickCount();
        queue.pop_front();
        ++written;
    }
    // Relative axes delivered as buffered data must not be counted twice.
    if (kind == 2 && written > *count) g_di_dx = g_di_dy = g_di_dz = 0;
    *count = written;
    return result;
}

template <size_t Slot> DWORD WINAPI h_XInputGetState(DWORD index, XINPUT_STATE* state) {
    ++g_count.xinput;
    const DWORD result = o_xinput[Slot] ? o_xinput[Slot](index, state) : ERROR_DEVICE_NOT_CONNECTED;
    if (index != 0 || !state || !g_pad_enabled.load()) return result;
    std::lock_guard lock(g_lock);
    state->dwPacketNumber = g_pad_packet;
    state->Gamepad = g_pad;
    return ERROR_SUCCESS;
}

LRESULT CALLBACK h_WndProc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam) {
    if (g_focus.load()) {
        switch (message) {
        case WM_ACTIVATEAPP:
            if (!wparam) { ++g_count.focus_swallowed; return 0; }
            break;
        case WM_ACTIVATE:
            if (LOWORD(wparam) == WA_INACTIVE) { ++g_count.focus_swallowed; return 0; }
            break;
        case WM_NCACTIVATE:
            if (!wparam) { ++g_count.focus_swallowed; return TRUE; }
            break;
        case WM_KILLFOCUS:
            ++g_count.focus_swallowed;
            return 0;
        default: break;
        }
    }
    return g_subclass_unicode ? CallWindowProcW(o_wndproc, hwnd, message, wparam, lparam)
                              : CallWindowProcA(o_wndproc, hwnd, message, wparam, lparam);
}

bool hook(void* target, void* detour, void** original) {
    if (!target) return false;
    for (auto existing : g_hook_targets) if (existing == target) return true;
    if (MH_CreateHook(target, detour, original) != MH_OK) return false;
    if (MH_EnableHook(target) != MH_OK) { MH_RemoveHook(target); return false; }
    g_hook_targets.push_back(target);
    return true;
}
template <class T> bool hook_export(const char* library, const char* name, T detour, T* original) {
    HMODULE module = GetModuleHandleA(library);
    if (!module) module = LoadLibraryA(library);
    return module && hook(reinterpret_cast<void*>(GetProcAddress(module, name)),
                          reinterpret_cast<void*>(detour), reinterpret_cast<void**>(original));
}
void install_dinput() {
    HMODULE dinput = LoadLibraryA("dinput8.dll");
    if (!dinput) return;
    using Create = HRESULT(WINAPI*)(HINSTANCE, DWORD, REFIID, LPVOID*, LPUNKNOWN);
    auto create = reinterpret_cast<Create>(GetProcAddress(dinput, "DirectInput8Create"));
    if (!create) return;
    for (size_t wide = 0; wide < 2; ++wide) {
        IUnknown* root = nullptr;
        if (FAILED(create(GetModuleHandleW(nullptr), DIRECTINPUT_VERSION,
                          wide ? IID_IDirectInput8W : IID_IDirectInput8A, reinterpret_cast<void**>(&root), nullptr)) || !root)
            continue;
        void* device = nullptr;
        HRESULT created = wide ? static_cast<IDirectInput8W*>(static_cast<void*>(root))->CreateDevice(GUID_SysKeyboard, reinterpret_cast<LPDIRECTINPUTDEVICE8W*>(&device), nullptr)
                               : static_cast<IDirectInput8A*>(static_cast<void*>(root))->CreateDevice(GUID_SysKeyboard, reinterpret_cast<LPDIRECTINPUTDEVICE8A*>(&device), nullptr);
        if (SUCCEEDED(created) && device) {
            void** table = *reinterpret_cast<void***>(device);
            o_di[wide].acquire = reinterpret_cast<DiAcquire>(table[7]);
            if (wide) {
                hook(table[9], reinterpret_cast<void*>(&h_DiGetState<1>), reinterpret_cast<void**>(&o_di[1].state));
                hook(table[10], reinterpret_cast<void*>(&h_DiGetData<1>), reinterpret_cast<void**>(&o_di[1].data));
                hook(table[11], reinterpret_cast<void*>(&h_DiSetFormat<1>), reinterpret_cast<void**>(&o_di[1].format));
                hook(table[13], reinterpret_cast<void*>(&h_DiSetCoop<1>), reinterpret_cast<void**>(&o_di[1].coop));
            } else {
                hook(table[9], reinterpret_cast<void*>(&h_DiGetState<0>), reinterpret_cast<void**>(&o_di[0].state));
                hook(table[10], reinterpret_cast<void*>(&h_DiGetData<0>), reinterpret_cast<void**>(&o_di[0].data));
                hook(table[11], reinterpret_cast<void*>(&h_DiSetFormat<0>), reinterpret_cast<void**>(&o_di[0].format));
                hook(table[13], reinterpret_cast<void*>(&h_DiSetCoop<0>), reinterpret_cast<void**>(&o_di[0].coop));
            }
            static_cast<IUnknown*>(device)->Release();
        }
        root->Release();
    }
    // If ANSI and wide share an implementation, one hook covers both tables.
    for (size_t wide = 0; wide < 2; ++wide) {
        auto& self = o_di[wide];
        const auto& other = o_di[1 - wide];
        if (!self.state) self.state = other.state;
        if (!self.data) self.data = other.data;
        if (!self.format) self.format = other.format;
        if (!self.coop) self.coop = other.coop;
    }
}
void install_xinput() {
    const char* libraries[] = {"xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll", "xinput1_2.dll"};
    void* detours[] = {reinterpret_cast<void*>(&h_XInputGetState<0>), reinterpret_cast<void*>(&h_XInputGetState<1>),
                       reinterpret_cast<void*>(&h_XInputGetState<2>), reinterpret_cast<void*>(&h_XInputGetState<3>)};
    for (size_t slot = 0; slot < 4; ++slot) {
        if (xinput_targets[slot]) continue;
        HMODULE module = GetModuleHandleA(libraries[slot]); // only hook what the game loaded
        if (!module) continue;
        void* target = reinterpret_cast<void*>(GetProcAddress(module, "XInputGetState"));
        bool duplicate = false;
        for (auto existing : xinput_targets) duplicate |= existing == target;
        if (target && !duplicate && hook(target, detours[slot], reinterpret_cast<void**>(&o_xinput[slot])))
            xinput_targets[slot] = target;
    }
}
void install_hooks() {
    const auto init = MH_Initialize();
    if (init != MH_OK && init != MH_ERROR_ALREADY_INITIALIZED) { g_last_error = "MinHook initialization failed"; return; }
    hook_export("user32.dll", "GetForegroundWindow", &h_GetForegroundWindow, &o_GetForegroundWindow);
    hook_export("user32.dll", "GetActiveWindow", &h_GetActiveWindow, &o_GetActiveWindow);
    hook_export("user32.dll", "GetFocus", &h_GetFocus, &o_GetFocus);
    hook_export("user32.dll", "GetAsyncKeyState", &h_GetAsyncKeyState, &o_GetAsyncKeyState);
    hook_export("user32.dll", "GetKeyState", &h_GetKeyState, &o_GetKeyState);
    hook_export("user32.dll", "GetKeyboardState", &h_GetKeyboardState, &o_GetKeyboardState);
    hook_export("user32.dll", "GetCursorPos", &h_GetCursorPos, &o_GetCursorPos);
    hook_export("user32.dll", "SetCursorPos", &h_SetCursorPos, &o_SetCursorPos);
    hook_export("user32.dll", "ClipCursor", &h_ClipCursor, &o_ClipCursor);
    hook_export("user32.dll", "GetRawInputData", &h_GetRawInputData, &o_GetRawInputData);
    install_dinput();
    install_xinput();
}

// ---------------------------------------------------------------------------
// Window discovery / subclassing
// ---------------------------------------------------------------------------
BOOL CALLBACK find_window(HWND hwnd, LPARAM best) {
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid != GetCurrentProcessId() || !IsWindowVisible(hwnd) || GetWindow(hwnd, GW_OWNER)) return TRUE;
    RECT rect{};
    GetClientRect(hwnd, &rect);
    const long area = (rect.right - rect.left) * (rect.bottom - rect.top);
    auto* result = reinterpret_cast<std::pair<HWND, long>*>(best);
    if (area > result->second) *result = {hwnd, area};
    return TRUE;
}
void ensure_window() {
    HWND hwnd = g_hwnd.load();
    if (!hwnd || !IsWindow(hwnd)) {
        std::pair<HWND, long> best{nullptr, 0};
        EnumWindows(find_window, reinterpret_cast<LPARAM>(&best));
        if (best.first) g_hwnd = hwnd = best.first;
    }
    if (hwnd && g_focus.load() && g_subclassed != hwnd) {
        g_subclass_unicode = IsWindowUnicode(hwnd) != FALSE;
        LONG_PTR previous = g_subclass_unicode
            ? SetWindowLongPtrW(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&h_WndProc))
            : SetWindowLongPtrA(hwnd, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(&h_WndProc));
        if (previous) { o_wndproc = reinterpret_cast<WNDPROC>(previous); g_subclassed = hwnd; }
    }
}
void restore_window() {
    if (g_subclassed && IsWindow(g_subclassed) && o_wndproc) {
        const auto current = g_subclass_unicode ? GetWindowLongPtrW(g_subclassed, GWLP_WNDPROC)
                                                : GetWindowLongPtrA(g_subclassed, GWLP_WNDPROC);
        if (current == reinterpret_cast<LONG_PTR>(&h_WndProc)) {
            if (g_subclass_unicode) SetWindowLongPtrW(g_subclassed, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(o_wndproc));
            else SetWindowLongPtrA(g_subclassed, GWLP_WNDPROC, reinterpret_cast<LONG_PTR>(o_wndproc));
        }
    }
    g_subclassed = nullptr;
}

// ---------------------------------------------------------------------------
// Mailbox
// ---------------------------------------------------------------------------
void write_json(const fs::path& path, const Json& value) {
    auto temporary = path;
    temporary += L".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return;
        output << value.dump(2);
    }
    MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING);
}
int key_code(const Json& value) {
    if (value.is_number_integer()) return value.get<int>();
    if (!value.is_string()) throw std::runtime_error("key must be a name or virtual-key code");
    std::string name = value.get<std::string>();
    for (auto& ch : name) ch = char(tolower(static_cast<unsigned char>(ch)));
    if (name.size() == 1 && ((name[0] >= 'a' && name[0] <= 'z') || (name[0] >= '0' && name[0] <= '9')))
        return toupper(name[0]);
    if (name.size() >= 2 && name[0] == 'f' && isdigit(static_cast<unsigned char>(name[1]))) {
        const int index = atoi(name.c_str() + 1);
        if (index >= 1 && index <= 24) return VK_F1 + index - 1;
    }
    static const std::map<std::string, int> names{
        {"enter", VK_RETURN}, {"return", VK_RETURN}, {"space", VK_SPACE}, {"escape", VK_ESCAPE}, {"esc", VK_ESCAPE},
        {"tab", VK_TAB}, {"backspace", VK_BACK}, {"up", VK_UP}, {"down", VK_DOWN}, {"left", VK_LEFT},
        {"right", VK_RIGHT}, {"shift", VK_SHIFT}, {"lshift", VK_LSHIFT}, {"rshift", VK_RSHIFT},
        {"ctrl", VK_CONTROL}, {"control", VK_CONTROL}, {"lctrl", VK_LCONTROL}, {"rctrl", VK_RCONTROL},
        {"alt", VK_MENU}, {"lalt", VK_LMENU}, {"ralt", VK_RMENU}, {"pageup", VK_PRIOR}, {"pagedown", VK_NEXT},
        {"home", VK_HOME}, {"end", VK_END}, {"insert", VK_INSERT}, {"delete", VK_DELETE}, {"pause", VK_PAUSE},
        {"capslock", VK_CAPITAL}, {"tilde", VK_OEM_3}, {"grave", VK_OEM_3}, {"minus", VK_OEM_MINUS},
        {"equals", VK_OEM_PLUS}, {"comma", VK_OEM_COMMA}, {"period", VK_OEM_PERIOD}, {"slash", VK_OEM_2},
        {"semicolon", VK_OEM_1}, {"quote", VK_OEM_7}, {"lbracket", VK_OEM_4}, {"rbracket", VK_OEM_6},
        {"backslash", VK_OEM_5}, {"numpad0", VK_NUMPAD0}, {"numpad5", VK_NUMPAD5}, {"numpadenter", VK_RETURN}};
    auto found = names.find(name);
    if (found == names.end()) throw std::runtime_error("unknown key name: " + name);
    return found->second;
}
int button_code(const std::string& name) {
    if (name == "left" || name.empty()) return VK_LBUTTON;
    if (name == "right") return VK_RBUTTON;
    if (name == "middle") return VK_MBUTTON;
    throw std::runtime_error("unknown mouse button: " + name);
}
WORD pad_buttons(const Json& list) {
    static const std::map<std::string, WORD> names{
        {"a", XINPUT_GAMEPAD_A}, {"b", XINPUT_GAMEPAD_B}, {"x", XINPUT_GAMEPAD_X}, {"y", XINPUT_GAMEPAD_Y},
        {"start", XINPUT_GAMEPAD_START}, {"back", XINPUT_GAMEPAD_BACK}, {"lb", XINPUT_GAMEPAD_LEFT_SHOULDER},
        {"rb", XINPUT_GAMEPAD_RIGHT_SHOULDER}, {"ls", XINPUT_GAMEPAD_LEFT_THUMB}, {"rs", XINPUT_GAMEPAD_RIGHT_THUMB},
        {"up", XINPUT_GAMEPAD_DPAD_UP}, {"down", XINPUT_GAMEPAD_DPAD_DOWN}, {"left", XINPUT_GAMEPAD_DPAD_LEFT},
        {"right", XINPUT_GAMEPAD_DPAD_RIGHT}};
    WORD result = 0;
    for (const auto& item : list) {
        auto found = names.find(item.get<std::string>());
        if (found == names.end()) throw std::runtime_error("unknown pad button");
        result |= found->second;
    }
    return result;
}
void schedule(ULONGLONG due, std::function<void()> run) {
    std::lock_guard lock(g_lock);
    g_timed.push_back({due, std::move(run)});
}
// Each operation may carry "at_ms" (offset from request arrival) so a whole
// macro arrives in one request and executes with stable timing in-process.
void execute(const Json& op, ULONGLONG base) {
    const std::string kind = op.at("op").get<std::string>();
    const ULONGLONG at = base + ULONGLONG(std::max(0, op.value("at_ms", 0)));
    const ULONGLONG hold = ULONGLONG(std::clamp(op.value("hold_ms", 80), 1, 60000));
    if (kind == "focus") {
        const bool enabled = op.value("enabled", true);
        schedule(at, [enabled] {
            g_focus = enabled;
            if (enabled) { ensure_window(); announce_focus(); }
            else { release_all(); restore_window(); }
        });
    } else if (kind == "key") {
        const int vk = key_code(op.at("key"));
        const std::string action = op.value("action", std::string("tap"));
        if (action == "down" || action == "tap") schedule(at, [vk] { set_key(vk, true); });
        if (action == "up") schedule(at, [vk] { set_key(vk, false); });
        if (action == "tap") schedule(at + hold, [vk] { set_key(vk, false); });
        if (action != "down" && action != "up" && action != "tap") throw std::runtime_error("key action must be down, up or tap");
    } else if (kind == "text") {
        const std::string text = op.at("text").get<std::string>();
        ULONGLONG when = at;
        for (unsigned char ch : text) {
            schedule(when, [ch] { post(WM_CHAR, WPARAM(ch), 1); });
            when += 15;
        }
    } else if (kind == "mouse_move") {
        const bool absolute = op.contains("x") || op.contains("y");
        const long x = op.value("x", 0L), y = op.value("y", 0L), dx = op.value("dx", 0L), dy = op.value("dy", 0L);
        const int steps = std::clamp(op.value("steps", 1), 1, 240);
        const ULONGLONG duration = ULONGLONG(std::max(0, op.value("duration_ms", 0)));
        if (absolute) schedule(at, [x, y] { move_mouse(0, 0, true, x, y); });
        else {
            long sent_x = 0, sent_y = 0;
            for (int step = 1; step <= steps; ++step) {
                const long target_x = dx * step / steps, target_y = dy * step / steps;
                const long part_x = target_x - sent_x, part_y = target_y - sent_y;
                sent_x = target_x; sent_y = target_y;
                schedule(at + duration * ULONGLONG(step - 1) / ULONGLONG(steps),
                         [part_x, part_y] { move_mouse(part_x, part_y, false, 0, 0); });
            }
        }
    } else if (kind == "mouse_button") {
        const int vk = button_code(op.value("button", std::string("left")));
        const std::string action = op.value("action", std::string("tap"));
        if (action == "down" || action == "tap") schedule(at, [vk] { set_button(vk, true); });
        if (action == "up") schedule(at, [vk] { set_button(vk, false); });
        if (action == "tap") schedule(at + hold, [vk] { set_button(vk, false); });
    } else if (kind == "wheel") {
        const int delta = op.value("delta", 120);
        schedule(at, [delta] { wheel(delta); });
    } else if (kind == "pad") {
        XINPUT_GAMEPAD pad{};
        pad.wButtons = op.contains("buttons") ? pad_buttons(op.at("buttons")) : 0;
        pad.sThumbLX = SHORT(std::clamp(op.value("lx", 0), -32768, 32767));
        pad.sThumbLY = SHORT(std::clamp(op.value("ly", 0), -32768, 32767));
        pad.sThumbRX = SHORT(std::clamp(op.value("rx", 0), -32768, 32767));
        pad.sThumbRY = SHORT(std::clamp(op.value("ry", 0), -32768, 32767));
        pad.bLeftTrigger = BYTE(std::clamp(op.value("lt", 0), 0, 255));
        pad.bRightTrigger = BYTE(std::clamp(op.value("rt", 0), 0, 255));
        const bool held = op.contains("hold_ms");
        schedule(at, [pad] { std::lock_guard lock(g_lock); g_pad_enabled = true; g_pad = pad; ++g_pad_packet; });
        if (held) schedule(at + hold, [] { std::lock_guard lock(g_lock); g_pad = {}; ++g_pad_packet; });
    } else if (kind == "pad_connect") {
        const bool enabled = op.value("enabled", true);
        schedule(at, [enabled] { g_pad_enabled = enabled; });
    } else if (kind == "release_all") {
        schedule(at, [] { release_all(); });
    } else if (kind != "status") {
        throw std::runtime_error("unknown input operation: " + kind);
    }
}
Json status_json() {
    Json hooks = Json::object();
    hooks["user32"] = o_GetForegroundWindow != nullptr;
    hooks["raw_input"] = o_GetRawInputData != nullptr;
    hooks["dinput8"] = o_di[0].state != nullptr || o_di[1].state != nullptr;
    int xinput = 0;
    for (auto target : xinput_targets) xinput += target != nullptr;
    hooks["xinput_modules"] = xinput;
    HWND hwnd = g_hwnd.load();
    size_t pending = 0;
    {
        std::lock_guard lock(g_lock);
        pending = g_timed.size();
    }
    return {{"schema", "phoenix-input/1"}, {"session", g_session}, {"pid", GetCurrentProcessId()},
            {"tick_ms", GetTickCount64()}, {"hwnd", uint64_t(reinterpret_cast<uintptr_t>(hwnd))},
            {"focus_override", g_focus.load()}, {"real_foreground_is_game", real_foreground_is_game()},
            {"subclassed", g_subclassed != nullptr}, {"pad_connected", g_pad_enabled.load()},
            {"pending_actions", pending}, {"hooks", hooks}, {"last_error", g_last_error},
            {"operations", {"status", "focus", "key", "text", "mouse_move", "mouse_button", "wheel", "pad",
                            "pad_connect", "release_all"}},
            // Which input APIs the game actually polls: useful to pick a strategy.
            {"api_calls", {{"GetForegroundWindow", g_count.foreground.load()},
                           {"GetAsyncKeyState", g_count.async_key.load()},
                           {"GetKeyState", g_count.key_state.load()},
                           {"GetKeyboardState", g_count.keyboard_state.load()},
                           {"GetCursorPos", g_count.cursor.load()}, {"SetCursorPos", g_count.set_cursor.load()},
                           {"GetRawInputData", g_count.raw_reads.load()},
                           {"DirectInputGetDeviceState", g_count.dinput_state.load()},
                           {"DirectInputGetDeviceData", g_count.dinput_data.load()},
                           {"XInputGetState", g_count.xinput.load()}}},
            {"injected", {{"messages", g_count.messages.load()}, {"raw_input", g_count.raw_injected.load()},
                          {"focus_messages_swallowed", g_count.focus_swallowed.load()}}}};
}
void poll_mailbox() {
    const auto incoming = g_folder / "request.json", claimed = g_folder / "processing.json";
    std::error_code error;
    if (!fs::exists(incoming, error)) return;
    if (!MoveFileExW(incoming.c_str(), claimed.c_str(), MOVEFILE_REPLACE_EXISTING)) return;
    Json response{{"schema", "phoenix-input/1"}, {"session", g_session}, {"id", nullptr}, {"ok", false}};
    try {
        if (fs::file_size(claimed) > 262144) throw std::runtime_error("request exceeds 256 KiB");
        std::ifstream input(claimed, std::ios::binary);
        const Json request = Json::parse(input);
        if (request.contains("id")) response["id"] = request["id"];
        if (request.value("schema", std::string()) != "phoenix-input/1") throw std::runtime_error("schema mismatch");
        if (request.value("session", std::string()) != g_session) throw std::runtime_error("session changed; reconnect");
        const auto& ops = request.at("ops");
        if (!ops.is_array() || ops.size() > 4096) throw std::runtime_error("ops must be an array of at most 4096 items");
        const ULONGLONG base = GetTickCount64();
        for (const auto& op : ops) execute(op, base); // validate+schedule; throws before partial use is rare
        response["ok"] = true;
        response["accepted"] = ops.size();
    } catch (const std::exception& failure) {
        response["error"] = failure.what();
    }
    fs::remove(claimed, error);
    response["status"] = status_json();
    write_json(g_folder / "response.json", response);
}
void run_due() {
    std::vector<std::function<void()>> ready;
    {
        std::lock_guard lock(g_lock);
        const ULONGLONG now = GetTickCount64();
        std::stable_sort(g_timed.begin(), g_timed.end(), [](const Timed& a, const Timed& b) { return a.due < b.due; });
        size_t count = 0;
        while (count < g_timed.size() && g_timed[count].due <= now) ++count;
        for (size_t index = 0; index < count; ++index) ready.push_back(std::move(g_timed[index].run));
        g_timed.erase(g_timed.begin(), g_timed.begin() + ptrdiff_t(count));
    }
    for (auto& run : ready) {
        try { run(); } catch (const std::exception& failure) { g_last_error = failure.what(); }
    }
}
void load_startup_config() {
    // The harness may request virtual focus from the first frame, before the
    // game window can observe any focus change.
    try {
        std::ifstream file(g_folder.parent_path().parent_path() / "PhoenixInput.json");
        if (!file) return;
        const Json config = Json::parse(file);
        g_focus = config.value("focus_override", false);
        g_pad_enabled = config.value("virtual_pad", false);
    } catch (const std::exception& failure) {
        g_last_error = std::string("PhoenixInput.json: ") + failure.what();
    }
}
void worker() {
    install_hooks();
    load_startup_config();
    ULONGLONG next_poll = 0, next_status = 0, next_scan = 0;
    while (!g_stop.load()) {
        const ULONGLONG now = GetTickCount64();
        if (now >= next_scan) {
            next_scan = now + 1000;
            ensure_window();
            install_xinput();
        }
        if (now >= next_poll) { next_poll = now + 20; poll_mailbox(); }
        run_due();
        if (now >= next_status) {
            next_status = now + 1000;
            write_json(g_folder / "endpoint.json", status_json());
        }
        Sleep(2);
    }
}

void on_init_swapchain(reshade::api::swapchain* swapchain, bool) {
    if (!swapchain) return;
    if (auto hwnd = static_cast<HWND>(swapchain->get_hwnd())) g_hwnd = hwnd;
}

bool g_attached = false;
bool attach(HMODULE module) {
    if (g_attached) return true;
    if (!reshade::register_addon(module)) return false;
    g_attached = true;
    g_module = module;
    wchar_t path[32768]{};
    GetModuleFileNameW(module, path, 32768);
    g_folder = fs::path(path).parent_path() / L"PhoenixOutput" / L"input";
    std::error_code error;
    fs::create_directories(g_folder, error);
    FILETIME created{}, exited{}, kernel{}, user{};
    GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user);
    g_session = std::to_string(GetCurrentProcessId()) + "-" +
                std::to_string((uint64_t(created.dwHighDateTime) << 32) | created.dwLowDateTime);
    reshade::register_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
    g_stop = false;
    // Hooks are installed on the worker, outside the loader lock.
    g_worker = std::thread(worker);
    return true;
}
// can_join: false inside DllMain (loader lock held; joining would deadlock).
// process_exit: Windows is tearing the process down; leave hooks alone.
void detach(HMODULE module, bool can_join, bool process_exit) {
    if (!g_attached) return;
    g_attached = false;
    g_stop = true;
    if (g_worker.joinable()) {
        if (can_join) g_worker.join();
        else g_worker.detach();
    }
    if (!process_exit) {
        g_focus = false;
        restore_window();
        for (auto target : g_hook_targets) { MH_DisableHook(target); MH_RemoveHook(target); }
        g_hook_targets.clear();
    }
    reshade::unregister_event<reshade::addon_event::init_swapchain>(on_init_swapchain);
    reshade::unregister_addon(module);
}
} // namespace

extern "C" __declspec(dllexport) bool AddonInit(HMODULE module, HMODULE) { return attach(module); }
extern "C" __declspec(dllexport) void AddonUninit(HMODULE module, HMODULE) { detach(module, true, false); }
BOOL APIENTRY DllMain(HMODULE module, DWORD reason, LPVOID reserved) {
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(module);
        return attach(module);
    }
    // AddonUninit owns normal unloading; this is only a last-resort cleanup.
    if (reason == DLL_PROCESS_DETACH) detach(module, false, reserved != nullptr);
    return TRUE;
}
