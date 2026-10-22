#pragma once
#include "../native/model/model_library.h"
#include <optional>
namespace phoenix_pipeline {
inline constexpr const char* tiny_weights_sha256="ac3835ecbde43bd2934cded28da99678649864fb7414ee50f660dc8a232d68e8";
inline constexpr unsigned tiny_weights_bytes=375800;
inline std::optional<std::filesystem::path> find_tiny_model(const std::filesystem::path& root) {
    if(root.empty()||!root.is_absolute())return std::nullopt;
    try {
        const auto index=phoenix_model::library_json(root/"tiny-upscaler.json");
        if(index.at("schema")!="phoenix-tiny-2x/1")return std::nullopt;
        const auto weights=phoenix_model::folder_resource_path(root,{{"path",index.at("path")}});
        if(std::filesystem::file_size(weights)!=tiny_weights_bytes ||
            phoenix_model::mapped_file_digest(weights)!=tiny_weights_sha256)return std::nullopt;
        return weights;
    }catch(const std::exception&){return std::nullopt;}
}
}
