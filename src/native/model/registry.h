#pragma once
#include "models/original/definition.h"
#include "models/small_fp8/definition.h"
#include <array>
#include <span>
#include <stdexcept>
#include <string>

namespace phoenix_model {
inline constexpr std::array definitions{models::original, models::small_fp8};

// Non-owning view of process-lifetime compiled definitions. No fixed model count.
class Registry {
    std::span<const Definition> entries_;

  public:
    explicit Registry(std::span<const Definition> entries) : entries_(entries) {
        for (size_t index = 0; index < entries.size(); ++index) {
            const auto& model = entries[index];
            if (model.id.empty() || model.architecture.empty() || model.resource.empty() ||
                model.layout_sha256.size() != 64 || !model.record_count ||
                (model.implemented && !model.decoded_tensor_count) ||
                (!model.implemented && model.unavailable_reason.empty()))
                throw std::runtime_error("Incomplete Phoenix DLSS5 model definition");
            for (size_t previous = 0; previous < index; ++previous)
                if (entries[previous].id == model.id)
                    throw std::runtime_error("Duplicate Phoenix DLSS5 model ID: " + std::string(model.id));
        }
    }
    std::span<const Definition> entries() const { return entries_; }
    const Definition& find(std::string_view id) const {
        for (const auto& model : entries_)
            if (model.id == id)
                return model;
        throw std::runtime_error("Unknown Phoenix DLSS5 model: " + std::string(id));
    }
};

inline const Registry& registry() {
    static const Registry result{definitions};
    return result;
}

inline const Definition& definition(std::string_view id = models::original.id) {
    return registry().find(id);
}

inline void require_implemented(const Definition& model) {
    if (!model.implemented)
        throw std::runtime_error(std::string(model.unavailable_reason));
}

inline void require_graph_model(const Definition& model, std::string_view architecture) {
    require_implemented(model);
    if (model.architecture != architecture)
        throw std::runtime_error("Selected Phoenix DLSS5 model does not match the graph architecture");
}
} // namespace phoenix_model
