// Read model files as data on Unix hosts; never load or execute Windows DLL code.
#pragma once
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace phoenix_model {
// SHA-256 keeps model authentication available without an extra crypto runtime
// on Android/macOS. Process one block at a time, including the final padding.
inline std::string portable_sha256(const uint8_t *data, size_t size) {
    constexpr uint32_t constants[] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    uint32_t state[] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    if (size > (UINT64_MAX / 8) - 128)
        throw std::runtime_error("SHA-256 input extent");
    const size_t blocks = size / 64 + ((size % 64) < 56 ? 1 : 2);
    for (size_t block = 0; block < blocks; ++block) {
        uint32_t words[64]{};
        for (size_t byte = 0; byte < 64; ++byte) {
            const auto position = block * 64 + byte;
            uint8_t value = position < size ? data[position] : position == size ? 0x80 : 0;
            if (block == blocks - 1 && byte >= 56)
                value = uint8_t((uint64_t(size) * 8) >> (8 * (63 - byte)));
            words[byte / 4] |= uint32_t(value) << (24 - 8 * (byte % 4));
        }
        for (unsigned index = 16; index < 64; ++index) {
            const auto left = words[index - 15], right = words[index - 2];
            const auto first = std::rotr(left, 7) ^ std::rotr(left, 18) ^ (left >> 3);
            const auto second = std::rotr(right, 17) ^ std::rotr(right, 19) ^ (right >> 10);
            words[index] = words[index - 16] + first + words[index - 7] + second;
        }
        auto a = state[0], b = state[1], c = state[2], d = state[3];
        auto e = state[4], f = state[5], g = state[6], h = state[7];
        for (unsigned index = 0; index < 64; ++index) {
            const auto sum = h + (std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25)) +
                             ((e & f) ^ (~e & g)) + constants[index] + words[index];
            const auto majority = (a & b) ^ (a & c) ^ (b & c);
            const auto rotate = std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            h = g;
            g = f;
            f = e;
            e = d + sum;
            d = c;
            c = b;
            b = a;
            a = sum + rotate + majority;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
    constexpr char hex[] = "0123456789abcdef";
    std::string result;
    for (auto word : state)
        for (int shift = 28; shift >= 0; shift -= 4)
            result += hex[(word >> shift) & 15];
    return result;
}
#ifdef _WIN32
// Hash a reconstructed address space with bounded scratch storage. Readers may
// supply file records plus zero padding rather than one materialized buffer.
class WindowsWeightDigest {
    BCRYPT_HASH_HANDLE hash_ = nullptr;
  public:
    WindowsWeightDigest() {
        if (BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &hash_, nullptr, 0, nullptr, 0, 0) < 0)
            throw std::runtime_error("Weight SHA-256 initialization failed");
    }
    WindowsWeightDigest(const WindowsWeightDigest&) = delete;
    WindowsWeightDigest& operator=(const WindowsWeightDigest&) = delete;
    ~WindowsWeightDigest() { if (hash_) BCryptDestroyHash(hash_); }
    void append(const uint8_t* data, size_t bytes) {
        if (!hash_ || (bytes && !data)) throw std::runtime_error("Invalid weight digest input");
        while (bytes) {
            const auto count = ULONG(bytes > UINT32_MAX ? UINT32_MAX : bytes);
            if (BCryptHashData(hash_, const_cast<PUCHAR>(data), count, 0) < 0)
                throw std::runtime_error("Weight SHA-256 hashing failed");
            data += count;
            bytes -= count;
        }
    }
    std::string finish() {
        std::array<uint8_t, 32> digest{};
        if (!hash_ || BCryptFinishHash(hash_, digest.data(), ULONG(digest.size()), 0) < 0)
            throw std::runtime_error("Weight SHA-256 finalization failed");
        BCryptDestroyHash(hash_); hash_ = nullptr;
        constexpr char hex[] = "0123456789abcdef";
        std::string result(64, '0');
        for (size_t index = 0; index < digest.size(); ++index) {
            result[2 * index] = hex[digest[index] >> 4];
            result[2 * index + 1] = hex[digest[index] & 15];
        }
        return result;
    }
};
template<class Read>
inline std::string windows_reader_digest(size_t bytes, Read read) {
    WindowsWeightDigest hash;
    std::array<uint8_t, 64 * 1024> scratch{};
    for (size_t offset = 0; offset < bytes;) {
        const auto count = bytes - offset < scratch.size() ? bytes - offset : scratch.size();
        read(offset, scratch.data(), count);
        hash.append(scratch.data(), count);
        offset += count;
    }
    return hash.finish();
}
// Hash the already-open, write-excluded file without reserving its full extent.
inline std::string windows_file_digest(HANDLE file, uint64_t offset = 0, uint64_t count = 0) {
    struct Hash {
        BCRYPT_HASH_HANDLE handle = nullptr;
        ~Hash() {
            if (handle)
                BCryptDestroyHash(handle);
        }
    } hash;
    LARGE_INTEGER size{}, start{};
    if (!GetFileSizeEx(file, &size) || size.QuadPart <= 0 ||
        uint64_t(size.QuadPart) > UINT32_MAX || offset > uint64_t(size.QuadPart))
        throw std::runtime_error("Model file is unreadable or exceeds addressable extent");
    if (!count)
        count = uint64_t(size.QuadPart) - offset;
    if (count > uint64_t(size.QuadPart) - offset)
        throw std::runtime_error("Weight SHA-256 range exceeds file");
    start.QuadPart = offset;
    if (!SetFilePointerEx(file, start, nullptr, FILE_BEGIN))
        throw std::runtime_error("Model file is unreadable or exceeds addressable extent");
    if (BCryptCreateHash(BCRYPT_SHA256_ALG_HANDLE, &hash.handle, nullptr, 0, nullptr, 0, 0) < 0)
        throw std::runtime_error("Weight SHA-256 initialization failed");
    std::array<uint8_t, 64 * 1024> buffer{};
    uint64_t total = 0;
    while (total < count) {
        const DWORD request = DWORD(count - total < buffer.size() ? count - total : buffer.size());
        DWORD bytes = 0;
        if (!ReadFile(file, buffer.data(), request, &bytes, nullptr) || bytes != request)
            throw std::runtime_error("Weight SHA-256 file read failed");
        total += bytes;
        if (BCryptHashData(hash.handle, buffer.data(), bytes, 0) < 0)
            throw std::runtime_error("Weight SHA-256 hashing failed");
    }
    std::array<uint8_t, 32> digest{};
    if (total != count ||
        BCryptFinishHash(hash.handle, digest.data(), ULONG(digest.size()), 0) < 0)
        throw std::runtime_error("Weight SHA-256 finalization failed");
    constexpr char hex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t index = 0; index < digest.size(); ++index) {
        result[2 * index] = hex[digest[index] >> 4];
        result[2 * index + 1] = hex[digest[index] & 15];
    }
    return result;
}
#endif
// Locate a named PE RCDATA resource without Windows headers. Every offset stays inside
// both its containing resource table and the physical file section.
// A bounded reader also supports finding a resource without mapping the DLL's
// code and unrelated resources into a fragmented 32-bit address space.
template <class Read>
inline std::pair<size_t, size_t> pe_weight_resource_range(size_t file_bytes, Read read,
                                                 std::string_view resource_name = "WEIGHTS_HT") {
    if (resource_name.empty() || resource_name.find('\0') != std::string_view::npos)
        throw std::runtime_error("Invalid model resource name");
    auto range = [&](size_t offset, size_t size) {
        if (offset > file_bytes || size > file_bytes - offset)
            throw std::runtime_error("Truncated PE model file");
    };
    auto word = [&](size_t offset) {
        range(offset, 2);
        uint8_t bytes[2];
        read(offset, bytes, sizeof(bytes));
        return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8);
    };
    auto dword = [&](size_t offset) {
        range(offset, 4);
        return word(offset) | (word(offset + 2) << 16);
    };
    if (word(0) != 0x5a4d)
        throw std::runtime_error("Model DLL needs a PE header");
    const size_t pe = dword(0x3c);
    if (dword(pe) != 0x4550)
        throw std::runtime_error("Invalid PE signature");
    range(pe, 24);
    const auto sections = word(pe + 6), optional_size = word(pe + 20);
    const size_t optional = pe + 24;
    range(optional, optional_size);
    const auto magic = word(optional);
    const size_t directory = magic == 0x20b ? 112 : magic == 0x10b ? 96 : 0;
    if (!directory || optional_size < directory + 24 || dword(optional + directory - 4) < 3)
        throw std::runtime_error("PE resource directory unavailable");
    const auto resource_rva = dword(optional + directory + 16);
    const auto resource_size = dword(optional + directory + 20);
    const size_t section_table = optional + optional_size;
    range(section_table, size_t(sections) * 40);
    auto physical = [&](uint32_t rva, uint32_t length) -> size_t {
        for (unsigned index = 0; index < sections; ++index) {
            const auto section = section_table + index * 40;
            const auto address = dword(section + 12), raw_size = dword(section + 16),
                       raw = dword(section + 20);
            if (rva >= address && uint64_t(rva) - address <= raw_size &&
                length <= raw_size - (rva - address)) {
                const uint64_t physical_offset = uint64_t(raw) + (rva - address);
                if (physical_offset > SIZE_MAX)
                    throw std::runtime_error("PE resource offset overflow");
                const size_t offset = size_t(physical_offset);
                range(offset, length);
                return offset;
            }
        }
        throw std::runtime_error("PE resource RVA outside file sections");
    };
    const auto base = physical(resource_rva, resource_size);
    auto table_range = [&](size_t relative, size_t size) {
        if (relative > resource_size || size > resource_size - relative)
            throw std::runtime_error("PE resource table bounds");
    };
    size_t table = 0;
    for (unsigned level = 0; level < 3; ++level) {
        table_range(table, 16);
        const auto count = word(base + table + 12) + word(base + table + 14);
        table_range(table + 16, size_t(count) * 8);
        bool found = false;
        for (unsigned index = 0; index < count; ++index) {
            const auto entry = base + table + 16 + index * 8;
            const auto name = dword(entry), target = dword(entry + 4);
            bool match = level == 2;
            if (level == 0)
                match = name == 10; // RT_RCDATA
            if (level == 1 && (name & 0x80000000u)) {
                const size_t text = name & 0x7fffffffu;
                table_range(text, 2);
                const auto length = word(base + text);
                table_range(text + 2, size_t(length) * 2);
                match = length == resource_name.size();
                for (unsigned character = 0; match && character < length; ++character)
                    match = word(base + text + 2 + character * 2) ==
                            static_cast<unsigned char>(resource_name[character]);
            }
            if (!match)
                continue;
            if (level < 2) {
                if (!(target & 0x80000000u))
                    throw std::runtime_error("Invalid PE resource tree");
                table = target & 0x7fffffffu;
            } else {
                if (target & 0x80000000u)
                    throw std::runtime_error("Invalid PE resource leaf");
                table_range(target, 16);
                const auto rva = dword(base + target), length = dword(base + target + 4);
                if (!length)
                    throw std::runtime_error("Empty model resource");
                return {physical(rva, length), length};
            }
            found = true;
            break;
        }
        if (!found)
            throw std::runtime_error("DLL has no selected model resource: " + std::string(resource_name));
    }
    throw std::runtime_error("Invalid PE resource tree");
}
inline std::span<const uint8_t> pe_weight_resource(std::span<const uint8_t> file,
                                                 std::string_view resource_name = "WEIGHTS_HT") {
    const auto [offset, bytes] = pe_weight_resource_range(file.size(),
        [&](size_t position, void* output, size_t count) {
            std::memcpy(output, file.data() + position, count);
        }, resource_name);
    return file.subspan(offset, bytes);
}
#ifdef _WIN32
class ReadOnlyModelFile {
    HANDLE file_ = INVALID_HANDLE_VALUE, mapping_ = nullptr;
    const uint8_t* data_ = nullptr;
    const void* view_ = nullptr;
    size_t size_ = 0, file_offset_ = 0;
    void close() noexcept {
        if (view_) UnmapViewOfFile(view_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
    }
  public:
    explicit ReadOnlyModelFile(const std::filesystem::path& path, std::string_view resource_name = {},
                               std::string_view expected_digest = {}, bool map_view = true) {
        try {
            file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
            LARGE_INTEGER size{};
            if (file_ == INVALID_HANDLE_VALUE || !GetFileSizeEx(file_, &size) ||
                size.QuadPart <= 0 || uint64_t(size.QuadPart) > UINT32_MAX ||
                uint64_t(size.QuadPart) > SIZE_MAX)
                throw std::runtime_error("Model file is unreadable or exceeds addressable extent");
            // Authentication uses this same write-excluded file handle, closing
            // the path replacement gap between an earlier digest and this view.
            if (!expected_digest.empty() && windows_file_digest(file_) != expected_digest)
                throw std::runtime_error("Mapped DLL weight source SHA-256 mismatch");
            size_t offset = 0;
            size_ = size_t(size.QuadPart);
            if (!resource_name.empty()) {
                const auto region = pe_weight_resource_range(size_,
                    [&](size_t position, void* output, size_t count) {
                        LARGE_INTEGER location{};
                        location.QuadPart = position;
                        DWORD actual = 0;
                        if (count > MAXDWORD ||
                            !SetFilePointerEx(file_, location, nullptr, FILE_BEGIN) ||
                            !ReadFile(file_, output, DWORD(count), &actual, nullptr) || actual != count)
                            throw std::runtime_error("Cannot read PE resource directory");
                    }, resource_name);
                offset = region.first;
                size_ = region.second;
            }
            file_offset_ = offset;
            if (!map_view)
                return;
            SYSTEM_INFO system{};
            GetSystemInfo(&system);
            const size_t aligned = offset - offset % system.dwAllocationGranularity;
            const size_t prefix = offset - aligned;
            mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_) {
                view_ = MapViewOfFile(mapping_, FILE_MAP_READ, DWORD(uint64_t(aligned) >> 32),
                                      DWORD(aligned), prefix + size_);
                if (view_)
                    data_ = static_cast<const uint8_t*>(view_) + prefix;
            }
            if (!data_)
                throw std::runtime_error("Model read-only mapping failed (Windows error " +
                    std::to_string(GetLastError()) + "; requested bytes " +
                    std::to_string(prefix + size_) + ")");
        } catch (...) { close(); throw; }
    }
    ReadOnlyModelFile(const ReadOnlyModelFile&) = delete;
    ReadOnlyModelFile& operator=(const ReadOnlyModelFile&) = delete;
    ~ReadOnlyModelFile() { close(); }
    std::span<const uint8_t> bytes() const {
        if (!data_)
            throw std::runtime_error("Model resource view already released");
        return {data_, size_};
    }
    size_t resource_size() const { return size_; }
    std::string resource_digest(size_t offset, size_t count) const {
        if (offset > size_ || !count || count > size_ - offset)
            throw std::runtime_error("Model digest outside selected extent");
        return windows_file_digest(file_, uint64_t(file_offset_) + offset, count);
    }
    // Keep the authenticated file open while releasing its address range.
    // Native packing can then allocate its destination without holding two
    // model-sized contiguous regions at once in a fragmented x86 process.
    void release_view() {
        if (view_ && !UnmapViewOfFile(view_))
            throw std::runtime_error("Cannot release model resource view");
        view_ = nullptr;
        data_ = nullptr;
    }
    void read_resource(size_t offset, void* output, size_t count) {
        if (offset > size_ || count > size_ - offset || (!output && count))
            throw std::runtime_error("Model resource read outside selected extent");
        LARGE_INTEGER position{};
        position.QuadPart = uint64_t(file_offset_) + offset;
        if (!SetFilePointerEx(file_, position, nullptr, FILE_BEGIN))
            throw std::runtime_error("Cannot seek selected model resource");
        auto* destination = static_cast<uint8_t*>(output);
        while (count) {
            const DWORD request = DWORD(count > 1024 * 1024 ? 1024 * 1024 : count);
            DWORD actual = 0;
            if (!ReadFile(file_, destination, request, &actual, nullptr) || actual != request)
                throw std::runtime_error("Cannot read selected model resource");
            destination += actual;
            count -= actual;
        }
    }
};
#else
class ReadOnlyModelFile {
    int descriptor_ = -1;
    void *mapping_ = MAP_FAILED;
    size_t size_ = 0;

  public:
    explicit ReadOnlyModelFile(const std::filesystem::path &path) {
        descriptor_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        struct stat info{};
        if (descriptor_ < 0 || fstat(descriptor_, &info) || !S_ISREG(info.st_mode) ||
            info.st_size <= 0 || uint64_t(info.st_size) > UINT32_MAX) {
            if (descriptor_ >= 0)
                close(descriptor_);
            throw std::runtime_error("Model file is unreadable or exceeds 4 GiB");
        }
        size_ = size_t(info.st_size);
        mapping_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
        if (mapping_ == MAP_FAILED) {
            close(descriptor_);
            throw std::runtime_error("Model read-only mapping failed");
        }
    }
    ReadOnlyModelFile(const ReadOnlyModelFile &) = delete;
    ReadOnlyModelFile& operator=(const ReadOnlyModelFile&) = delete;
    ~ReadOnlyModelFile() {
        munmap(mapping_, size_);
        close(descriptor_);
    }
    std::span<const uint8_t> bytes() const {
        return {static_cast<const uint8_t *>(mapping_), size_};
    }
};
#endif
} // namespace phoenix_model
