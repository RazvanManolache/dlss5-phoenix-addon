#pragma once
#include <windows.h>
#include <filesystem>
#include <stdexcept>
#include <string>
#include "../../third_party/nlohmann/json.hpp"

namespace phoenix_pipeline {
// Publish only after the image is saved. CREATE_NEW preserves existing receipts,
// including receipts for an older image that the user subsequently overwrote.
inline std::filesystem::path save_screenshot_receipt(const std::filesystem::path& image,
    const std::string& sha256, nlohmann::json facts) {
    if (sha256.size()!=64 || sha256.find_first_not_of("0123456789abcdef")!=std::string::npos ||
        !std::filesystem::is_regular_file(image))
        throw std::invalid_argument("Screenshot receipt requires a saved image and SHA-256");
    const auto filename=image.filename().u8string();
    facts["schema"]="phoenix-screenshot/1";
    facts["image"]={{"filename",std::string(filename.begin(),filename.end())},
        {"sha256",sha256},{"bytes",std::filesystem::file_size(image)}};
    FILETIME now{};GetSystemTimeAsFileTime(&now);
    facts["saved_utc_filetime_100ns"]=(uint64_t(now.dwHighDateTime)<<32)|now.dwLowDateTime;
    const auto text=facts.dump(2)+"\n";
    if(text.size()>4*1024*1024)throw std::runtime_error("Screenshot receipt exceeds 4 MiB");
    for(unsigned suffix=0;suffix!=1024;++suffix){
        auto path=image;path+=L".phoenix"+(suffix?L"-"+std::to_wstring(suffix):L"")+L".json";
        HANDLE file=CreateFileW(path.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
        if(file==INVALID_HANDLE_VALUE){
            if(GetLastError()==ERROR_FILE_EXISTS||GetLastError()==ERROR_ALREADY_EXISTS)continue;
            throw std::runtime_error("Cannot create screenshot receipt");
        }
        DWORD written=0;
        const bool ok=WriteFile(file,text.data(),DWORD(text.size()),&written,nullptr)&&
            written==text.size()&&FlushFileBuffers(file);
        CloseHandle(file);
        if(!ok){DeleteFileW(path.c_str());throw std::runtime_error("Cannot finish screenshot receipt");}
        return path;
    }
    throw std::runtime_error("Too many receipts for this screenshot filename");
}
}
