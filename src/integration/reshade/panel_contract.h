#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
namespace phoenix_ui {
// Transport commands contain values, never UI pointers or native objects.
enum class Action : uint32_t {
    ResetHistory = 1,
    CaptureInputs,
    ReloadOverrides,
    Marker,
    ClearTiming,
    ValidateModel
};
inline bool valid_action(uint32_t value) { return value >= 1 && value <= 6; }

// Keep key-binding rules independent of bitness and of ReShade's input source.
// Windows virtual-key values are used here to avoid a windows.h dependency.
enum class BindingResult { Accepted, Cancelled, Reserved, Duplicate, Invalid };
inline bool bindable_key(unsigned key) {
    return key >= 8 && key <= 255 && key != 0x1b && key != 0x24 && key != 0x7b;
}
inline unsigned sanitize_bindings(std::array<int, 6> &bindings) {
    std::array<bool, 256> seen{};
    unsigned cleared = 0;
    for (auto &key : bindings) {
        if (!key)
            continue;
        if (!bindable_key(unsigned(key)) || seen[unsigned(key)]) {
            key = 0;
            ++cleared;
        } else
            seen[unsigned(key)] = true;
    }
    return cleared;
}
inline BindingResult capture_binding(std::array<int, 6> &bindings, int slot, unsigned key) {
    if (slot < 0 || slot >= int(bindings.size()) || key > 255 || key == 0)
        return BindingResult::Invalid;
    if (key == 0x1b) // Escape cancels without changing the binding.
        return BindingResult::Cancelled;
    if (!bindable_key(key)) // Mouse keys, Home, F12.
        return BindingResult::Reserved;
    for (std::size_t index = 0; index < bindings.size(); ++index)
        if (int(index) != slot && bindings[index] == int(key))
            return BindingResult::Duplicate;
    bindings[std::size_t(slot)] = int(key);
    return BindingResult::Accepted;
}
inline const char *binding_error(BindingResult result) {
    switch (result) {
    case BindingResult::Reserved: return "This key is reserved for the game or ReShade overlay.";
    case BindingResult::Duplicate: return "This key is already bound to another action.";
    case BindingResult::Invalid: return "Choose a keyboard key.";
    default: return "";
    }
}
} // namespace phoenix_ui
