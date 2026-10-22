#pragma once
#include <string>
#include <stdexcept>
#include <utility>

namespace phoenix_model_tools {
// Keep the requested source separate from the authenticated model in a live
// executor. An empty model id means the configured default, not Original.
struct ModelSelectionState {
    std::string source, folder, dll, model_id, search_folder;

    template <class Settings> static ModelSelectionState read(const Settings& settings) {
        return {settings.model_source, settings.model_folder, settings.model_dll,
                settings.model_id, settings.nvidia_dll_folder};
    }
    template <class Settings> void restore(Settings& settings) const {
        settings.model_source = source;
        settings.model_folder = folder;
        settings.model_dll = dll;
        settings.model_id = model_id;
        settings.nvidia_dll_folder = search_folder;
    }
    bool operator==(const ModelSelectionState&) const = default;
};

// Construction must finish before publication. In particular, exceptions must
// not destroy the last working executor or publish a candidate's model label.
template <class Owner, class Factory, class Retire>
bool replace_idle_generation(Owner& active, Factory make_candidate, Retire retire) {
    if (active && !active->idle())
        return false;
    auto candidate = make_candidate();
    if (!candidate)
        throw std::runtime_error("Replacement executor was not constructed");
    if (active)
        retire(*active);
    active = std::move(candidate);
    return true;
}
} // namespace phoenix_model_tools
