#pragma once
#include "../../third_party/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <optional>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#endif

namespace phoenix_pipeline {
using Json = nlohmann::json;
#ifdef _WIN32
inline std::filesystem::path application_recovery_path() {
    std::wstring executable(32768, L'\0'), local(32768, L'\0');
    const auto length = GetModuleFileNameW(nullptr, executable.data(), DWORD(executable.size()));
    const auto local_length = GetEnvironmentVariableW(L"LOCALAPPDATA", local.data(), DWORD(local.size()));
    if (!length || length >= executable.size() || !local_length || local_length >= local.size())
        throw std::runtime_error("Phoenix recovery location is unavailable");
    executable.resize(length); local.resize(local_length);
    unsigned char digest[32]{};
    if (BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0,
        reinterpret_cast<PUCHAR>(executable.data()), ULONG(executable.size() * sizeof(wchar_t)), digest, 32) < 0)
        throw std::runtime_error("Cannot identify Phoenix recovery application");
    std::string name;
    for (auto byte : digest) { name += "0123456789abcdef"[byte >> 4]; name += "0123456789abcdef"[byte & 15]; }
    return std::filesystem::path(local) / "Phoenix" / "Recovery" / (name + ".json");
}
#endif
// Each session owns one journal. Persist a candidate before initializing a
// provider; only completed frames can promote it to last-working configuration.
class RecoveryJournal {
    std::filesystem::path path_;
    Json document_;
    unsigned healthy_ = 0;
    void write() {
        std::filesystem::create_directories(path_.parent_path());
        auto temporary = path_; temporary += ".pending";
        const auto bytes = document_.dump(2);
#ifdef _WIN32
        const HANDLE file = CreateFileW(temporary.c_str(), GENERIC_WRITE, 0, nullptr,
            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("Cannot write Phoenix recovery journal");
        DWORD written = 0;
        const bool saved = WriteFile(file, bytes.data(), DWORD(bytes.size()), &written, nullptr) &&
            written == bytes.size() && FlushFileBuffers(file);
        CloseHandle(file);
        if (!saved || !MoveFileExW(temporary.c_str(), path_.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::runtime_error("Cannot commit Phoenix recovery journal");
#else
        { std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
          file << bytes; file.flush(); if (!file) throw std::runtime_error("Recovery journal write failed"); }
        std::filesystem::rename(temporary, path_);
#endif
    }
public:
    RecoveryJournal(std::filesystem::path path, Json defaults) : path_(std::move(path)) {
        document_ = {{"schema", "phoenix-recovery/1"}, {"working", defaults},
                     {"pending", nullptr}, {"failures", 0}, {"reason", ""}};
        if (!std::filesystem::exists(path_)) return;
        if (std::filesystem::is_symlink(path_)) throw std::runtime_error("Recovery journal must be a regular file");
        std::error_code ec;
        const auto size = std::filesystem::file_size(path_, ec);
        if (ec || size > 256 * 1024) throw std::runtime_error("Invalid Phoenix recovery journal size");
        std::ifstream file(path_); Json saved;
        try { file >> saved; } catch (const Json::exception&) { throw std::runtime_error("Invalid Phoenix recovery journal JSON"); }
        if (saved.value("schema", "") != "phoenix-recovery/1" || !saved.contains("working") ||
            !saved["working"].is_object() || !saved.contains("pending") ||
            (!saved["pending"].is_null() && !saved["pending"].is_object()) ||
            (saved.contains("requested")&&!saved["requested"].is_null()&&!saved["requested"].is_object()))
            throw std::runtime_error("Invalid Phoenix recovery journal schema");
        document_ = std::move(saved);
    }
    bool pending() const { return !document_.at("pending").is_null(); }
    const Json& working() const { return document_.at("working"); }
    const Json& report() const { return document_; }
    std::optional<Json> take_requested(){
        if(!document_.contains("requested")||document_["requested"].is_null())return std::nullopt;
        auto selected=document_["requested"];document_["requested"]=nullptr;write();return selected;
    }
    std::optional<Json> recover_interrupted() {
        if (pending()) {
            auto restored=failure("Initialization or settings trial did not complete");
            document_["restore_required"]=false;write();return restored;
        }
        if (document_.value("restore_required", !document_.value("reason", std::string()).empty())) {
            document_["restore_required"] = false; write(); return working();
        }
        return std::nullopt;
    }
    void begin(Json candidate) {
        if (!candidate.is_object()) throw std::invalid_argument("Recovery candidate must be an object");
        if (pending() && document_["pending"] == candidate) return;
        if(!pending()&&document_["working"]==candidate)return;
        document_["pending"] = std::move(candidate); healthy_ = 0; write();
    }
    std::optional<Json> failure(const std::string& reason) {
        if (!pending()) return std::nullopt;
        document_["rejected"]=document_["pending"];
        document_["pending"] = nullptr;
        document_["restore_required"] = true;
        document_["reason"] = reason.substr(0, 4096);
        document_["failures"] = document_.value("failures", uint64_t(0)) + 1;
        healthy_ = 0; write(); return working();
    }
    bool completed_frame(unsigned threshold = 30) {
        if (!pending() || ++healthy_ < std::max(1u, threshold)) return false;
        document_["working"] = document_["pending"];
        document_["pending"] = nullptr; document_["reason"] = "";
        document_["restore_required"]=false;
        write(); return true;
    }
};
} // namespace phoenix_pipeline
