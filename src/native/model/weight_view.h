#pragma once
#include "portable_model_io.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace phoenix_model {
// Immutable weights need not occupy a contiguous CPU allocation. A reader can
// reconstruct a requested range directly into bounded GPU upload storage.
class WeightView {
    size_t bytes_ = 0;
    const void* identity_ = nullptr;
    std::function<void(size_t, void*, size_t)> read_;
    std::string digest_;
  public:
    WeightView() = default;
    WeightView(size_t bytes, const void* identity,
               std::function<void(size_t, void*, size_t)> read)
        : bytes_(bytes), identity_(identity), read_(std::move(read)) {
        if (bytes && (!identity || !read_)) throw std::invalid_argument("Missing weight reader");
    }
    // Existing capture/probe callers retain their vector through construction.
    WeightView(const std::vector<uint8_t>& bytes)
        : WeightView(bytes.size(), bytes.data(), [data=bytes.data()](size_t offset, void* output, size_t count) {
            std::memcpy(output, data + offset, count);
        }) {}
    static WeightView owned(std::vector<uint8_t> bytes) {
        auto storage = std::make_shared<std::vector<uint8_t>>(std::move(bytes));
        return {storage->size(), storage.get(), [storage](size_t offset, void* output, size_t count) {
            std::memcpy(output, storage->data() + offset, count);
        }};
    }
    size_t size() const { return bytes_; }
    bool empty() const { return bytes_ == 0; }
    const void* identity() const { return identity_; }
    const std::string& expected_digest() const { return digest_; }
    WeightView authenticated(std::string digest) const {
        if (digest.size() != 64 || digest.find_first_not_of("0123456789abcdef") != std::string::npos)
            throw std::invalid_argument("Invalid packed weight digest");
        auto result = *this;
        result.digest_ = std::move(digest);
        return result;
    }
    void copy_to(size_t offset, void* output, size_t count) const {
        if (offset > bytes_ || count > bytes_ - offset || (count && !output))
            throw std::out_of_range("Weight read outside packed extent");
        if (count) read_(offset, output, count);
    }
};
#ifdef _WIN32
// Verify the bytes actually copied, including when a previously writable file
// mapping could change after source authentication. Never read GPU upload memory
// back for hashing: hash bounded CPU chunks immediately before their copies.
class WeightTransfer {
    const WeightView& source_;
    size_t copied_ = 0;
    std::unique_ptr<WindowsWeightDigest> digest_;
  public:
    explicit WeightTransfer(const WeightView& source) : source_(source) {
        if (!source.expected_digest().empty()) digest_ = std::make_unique<WindowsWeightDigest>();
    }
    void copy_to(size_t offset, void* output, size_t count) {
        if (offset != copied_ || offset > source_.size() || count > source_.size() - offset || (count && !output))
            throw std::out_of_range("Incomplete or unordered weight upload");
        if (!digest_) {
            source_.copy_to(offset, output, count);
            copied_ += count;
            return;
        }
        std::array<uint8_t, 64 * 1024> scratch{};
        auto* destination = static_cast<uint8_t*>(output);
        for (size_t done = 0; done < count;) {
            const auto bytes = std::min(scratch.size(), count - done);
            source_.copy_to(offset + done, scratch.data(), bytes);
            digest_->append(scratch.data(), bytes);
            std::memcpy(destination + done, scratch.data(), bytes);
            done += bytes;
        }
        copied_ += count;
    }
    void finish() {
        if (copied_ != source_.size()) throw std::runtime_error("Incomplete native weight upload");
        if (digest_ && digest_->finish() != source_.expected_digest())
            throw std::runtime_error("Weight resource changed during GPU upload");
    }
};
#endif
} // namespace phoenix_model
