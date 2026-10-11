#pragma once
#include "../session/session.h"
#include "diagnostic_compare.h"

namespace phoenix_addon::diagnostic {
// Only the owned candidate sees this proxy. Native receives the original object
// unchanged, including its Output pointer and any private non-virtual fields.
class CandidateParameters final : public NVSDK_NGX_Parameter {
    const NVSDK_NGX_Parameter& source_;
    ID3D12Resource* output_;

  public:
    CandidateParameters(const NVSDK_NGX_Parameter& source, ID3D12Resource* output)
        : source_(source), output_(output) {
        check(output != nullptr, "phoenix diagnostic missing independent output");
    }
#define PHOENIX_DIAG_PARAMETER(T)                                                                  \
    void Set(const char*, T) override {                                                            \
        throw std::runtime_error("phoenix diagnostic candidate parameters are read-only");         \
    }                                                                                              \
    NVSDK_NGX_Result Get(const char* key, T* value) const override {                               \
        return source_.Get(key, value);                                                            \
    }
    PHOENIX_DIAG_PARAMETER(unsigned long long)
    PHOENIX_DIAG_PARAMETER(float)
    PHOENIX_DIAG_PARAMETER(double)
    PHOENIX_DIAG_PARAMETER(unsigned)
    PHOENIX_DIAG_PARAMETER(int)
    PHOENIX_DIAG_PARAMETER(ID3D11Resource*)
#undef PHOENIX_DIAG_PARAMETER
    void Set(const char*, ID3D12Resource*) override {
        throw std::runtime_error("phoenix diagnostic read-only parameters");
    }
    void Set(const char*, void*) override {
        throw std::runtime_error("phoenix diagnostic read-only parameters");
    }
    NVSDK_NGX_Result Get(const char* key, ID3D12Resource** value) const override {
        if (key && !std::strcmp(key, "DLSSNR.Output")) {
            *value = output_;
            return NVSDK_NGX_Result_Success;
        }
        return source_.Get(key, value);
    }
    NVSDK_NGX_Result Get(const char* key, void** value) const override {
        if (key && !std::strcmp(key, "DLSSNR.Output")) {
            *value = output_;
            return NVSDK_NGX_Result_Success;
        }
        return source_.Get(key, value);
    }
    void Reset() override {
        throw std::runtime_error("phoenix diagnostic read-only parameters");
    }
};

class Pair {
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12Resource> candidate_, native_, candidate_readback_, native_readback_;
    ComPtr<ID3D12Fence> fence_;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint_{};
    UINT rows_ = 0;
    UINT64 row_bytes_ = 0, allocation_ = 0;
    ID3D12GraphicsCommandList* command_;
    bool recorded_candidate_ = false, recorded_native_ = false;
    bool submitted_ = false, completed_ = false, failed_ = false;
    bool native_available_ = true;
    unsigned long long native_code_ = 0;
    Json result_;

    ComPtr<ID3D12Resource> readback() {
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = allocation_;
        desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> resource;
        phoenix_native::hr(device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                            D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                            IID_PPV_ARGS(&resource)));
        return resource;
    }
    void capture(ID3D12GraphicsCommandList* cmd, ID3D12Resource* source,
                 D3D12_RESOURCE_STATES state, ID3D12Resource* destination) {
        phoenix_native::transition(cmd, source, state, D3D12_RESOURCE_STATE_COPY_SOURCE);
        D3D12_TEXTURE_COPY_LOCATION from{};
        from.pResource = source;
        from.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        D3D12_TEXTURE_COPY_LOCATION to{};
        to.pResource = destination;
        to.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        to.PlacedFootprint = footprint_;
        cmd->CopyTextureRegion(&to, 0, 0, 0, &from, nullptr);
        phoenix_native::transition(cmd, source, D3D12_RESOURCE_STATE_COPY_SOURCE, state);
    }
    std::vector<uint8_t> mapped(ID3D12Resource* resource) {
        void* data = nullptr;
        D3D12_RANGE read{0, size_t(allocation_)};
        phoenix_native::hr(resource->Map(0, &read, &data));
        const D3D12_RANGE written{0, 0};
        try {
            auto packed = pack_rows(static_cast<const uint8_t*>(data), size_t(allocation_),
                                    size_t(footprint_.Offset), footprint_.Footprint.RowPitch,
                                    size_t(row_bytes_), rows_);
            resource->Unmap(0, &written);
            return packed;
        } catch (...) {
            resource->Unmap(0, &written);
            throw;
        }
    }
    static std::string digest(const std::vector<uint8_t>& bytes) {
        check(bytes.size() <= ULONG_MAX, "phoenix diagnostic hash extent");
        unsigned char hash[32]{};
        check(BCryptHash(BCRYPT_SHA256_ALG_HANDLE, nullptr, 0, const_cast<PUCHAR>(bytes.data()),
                         ULONG(bytes.size()), hash, 32) >= 0,
              "phoenix diagnostic hash failed");
        constexpr char hex[] = "0123456789abcdef";
        std::string text(64, '0');
        for (size_t iteration_index = 0; iteration_index < 32; ++iteration_index) {
            text[2 * iteration_index] = hex[hash[iteration_index] >> 4];
            text[2 * iteration_index + 1] = hex[hash[iteration_index] & 15];
        }
        return text;
    }
    static void save(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
        check(!std::filesystem::exists(path), "phoenix diagnostic refuses capture overwrite");
        std::ofstream stream(path, std::ios::binary);
        check(bool(stream), "phoenix diagnostic capture open failed");
        stream.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        stream.close();
        check(bool(stream), "phoenix diagnostic capture write failed");
    }

  public:
    Pair(ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* command, ID3D12Resource* output,
         uint64_t index)
        : queue_(queue), native_(output), command_(command) {
        check(queue && command && output, "phoenix diagnostic null pair resource");
        const auto choice = resolve_allocation_device(queue, output);
        check(choice.resolved, "phoenix diagnostic could not resolve an allocation device");
        device_ = choice.device;
        ComPtr<ID3D12Device> owner;
        phoenix_native::hr(output->GetDevice(IID_PPV_ARGS(&owner)));
        check(phoenix_native::device_owns(owner.Get(), device_.Get()),
              "phoenix diagnostic engine output belongs to another device");
        const auto desc = output->GetDesc();
        check(true && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D &&
                  desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT && desc.Width &&
                  desc.Width <= 16384 && desc.Height && desc.Height <= 16384 &&
                  desc.DepthOrArraySize == 1 && desc.MipLevels == 1 && desc.SampleDesc.Count == 1 &&
                  desc.SampleDesc.Quality == 0,
              "phoenix diagnostic requires single-subresource RGBA16F output");
        device_->GetCopyableFootprints(&desc, 0, 1, 0, &footprint_, &rows_, &row_bytes_,
                                       &allocation_);
        check(row_bytes_ == desc.Width * 8 && rows_ == desc.Height && allocation_ &&
                  allocation_ <= SIZE_MAX,
              "phoenix diagnostic unexpected output footprint");
        // UAV-capable: the HDR bridge denormalizes into this texture through
        // an unordered-access write, and native NR drivers may require the
        // capability on their output. States and copies are unchanged, so
        // SDR pair behavior is identical; only the heap flags grow.
        candidate_ = phoenix_native::image(device_.Get(), unsigned(desc.Width), desc.Height,
                                           desc.Format, D3D12_RESOURCE_STATE_COPY_DEST, true);
        candidate_readback_ = readback();
        native_readback_ = readback();
        phoenix_native::hr(device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_)));
        result_ = {{"pair", index},
                   {"width", desc.Width},
                   {"height", desc.Height},
                   {"format", "RGBA16F"},
                   {"submitted", false},
                   {"completed", false},
                   {"comparison", "pending"},
                   {"independent_outputs", true},
                   {"independent_histories", true},
                   {"candidate_output", uint64_t(candidate_.Get())},
                   {"native_output", uint64_t(output)}};
    }
    ID3D12Resource* candidate_output() const noexcept {
        return candidate_.Get();
    }
    ID3D12Resource* native_output() const noexcept {
        return native_.Get();
    }
    ID3D12GraphicsCommandList* command() const noexcept {
        return command_;
    }
    bool completed() const noexcept {
        return completed_;
    }
    bool submitted() const noexcept {
        return submitted_;
    }
    const Json& result() const noexcept {
        return result_;
    }
    void capture_candidate(ID3D12GraphicsCommandList* cmd) {
        check(cmd == command_ && !recorded_candidate_,
              "phoenix diagnostic candidate recording identity");
        capture(cmd, candidate_.Get(), D3D12_RESOURCE_STATE_COPY_DEST, candidate_readback_.Get());
        recorded_candidate_ = true;
    }
    // The native refused this frame (code carried for forensics) with the
    // frame controls that were in force. The pair degrades to candidate-only:
    // no native bytes, no comparison -- but the candidate side still
    // completes, submits, and reports statistics. Never terminal by itself;
    // the run decides.
    void mark_native_unavailable(unsigned long long code, const Json& controls,
                                 const char* reason) {
        native_available_ = false;
        native_code_ = code;
        result_["controls"] = controls;
        result_["native_unavailable_reason"] = reason;
    }
    void capture_native(ID3D12GraphicsCommandList* cmd, D3D12_RESOURCE_STATES observed_state,
                        const Json& controls) {
        check(cmd == command_ && !recorded_native_, "phoenix diagnostic native recording identity");
        capture(cmd, native_.Get(), observed_state, native_readback_.Get());
        recorded_native_ = true;
        result_["native_output_state_restored"] = unsigned(observed_state);
        result_["controls"] = controls;
    }
    void after_submit(ID3D12CommandQueue* queue, ID3D12GraphicsCommandList* cmd) {
        check(recorded_candidate_ && (recorded_native_ || !native_available_) && !submitted_ &&
                  queue == queue_.Get() && cmd == command_,
              "phoenix diagnostic submission mismatch");
        phoenix_native::hr(queue->Signal(fence_.Get(), 1));
        phoenix_native::hr(device_->GetDeviceRemovedReason());
        submitted_ = true;
        result_["submitted"] = true;
    }
    // Never wait on the GPU or map a merely recorded/submitted result.
    // `save_raw` bounds how much evidence a long run writes. A mismatch is
    // always saved in full regardless, because that is the evidence that
    // matters, and every pair is compared and hashed either way.
    bool poll(const std::filesystem::path& directory, bool save_raw = true) {
        if (completed_ || failed_ || !submitted_)
            return false;
        try {
            const auto value = fence_->GetCompletedValue();
            check(value != UINT64_MAX, "phoenix diagnostic fence reports device loss");
            if (value < 1)
                return false;
            phoenix_native::hr(device_->GetDeviceRemovedReason());
            const auto candidate = mapped(candidate_readback_.Get());
            const auto prefix = "pair_" + std::to_string(result_.at("pair").get<uint64_t>());
            if (!native_available_) {
                // Candidate-only: the native refused, so there is no A/B. The
                // candidate side still proves the meter/normalize/network/
                // denormalize chain executed on game input and shows exactly
                // what it produced. Always saved, never compared, never
                // terminal by itself; the run decides.
                const auto stats = candidate_statistics(candidate);
                save(directory / (prefix + "_candidate.rgba16f"), candidate);
                result_["raw_captures_saved"] = true;
                result_["candidate_sha256"] = digest(candidate);
                result_["compared_bytes"] = candidate.size();
                result_["candidate_finite"] = stats.finite;
                result_["candidate_channels"] = stats.channels;
                result_["candidate_max_abs"] = stats.max_abs;
                result_["candidate_mean_abs"] = stats.mean_abs;
                result_["candidate_min_value"] = stats.min_value;
                result_["candidate_max_value"] = stats.max_value;
                result_["native_available"] = false;
                result_["native_code"] = native_code_;
                result_["comparison"] = "native_unavailable";
                result_["completed"] = true;
                std::ofstream report(directory / (prefix + ".json"), std::ios::binary);
                report << result_.dump(2) << '\n';
                report.close();
                check(bool(report), "phoenix diagnostic comparison report failed");
                completed_ = true;
                candidate_.Reset();
                native_.Reset();
                candidate_readback_.Reset();
                native_readback_.Reset();
                return true;
            }
            const auto native = mapped(native_readback_.Get());
            const auto difference = compare_rgba16f(candidate, native);
            if (save_raw || !difference.exact()) {
                save(directory / (prefix + "_candidate.rgba16f"), candidate);
                save(directory / (prefix + "_native.rgba16f"), native);
                result_["raw_captures_saved"] = true;
            } else
                result_["raw_captures_saved"] = false;
            result_["candidate_sha256"] = digest(candidate);
            result_["native_sha256"] = digest(native);
            result_["compared_bytes"] = candidate.size();
            result_["different_bytes"] = difference.bytes;
            result_["different_pixels"] = difference.pixels;
            if (difference.exact()) {
                result_["comparison"] = "byte_exact";
            } else {
                // Not byte-exact, so measure instead of only counting: the HDR
                // bridge path can never be exact (fp16 normalize/denormalize
                // stores, different network), and its acceptance is bounded
                // error, never silence. Mismatch stays terminal below.
                const auto bounded = bounded_difference(candidate, native);
                result_["max_abs_err"] = bounded.max_abs;
                result_["max_rel_err"] = bounded.max_rel;
                result_["alpha_max_abs_err"] = bounded.alpha_max_abs;
                result_["nonfinite_mismatches"] = bounded.nonfinite_mismatches;
                result_["matched_nonfinite_channels"] = bounded.matched_nonfinite;
                result_["error_ceiling"] = hdr_sanity_ceiling;
                result_["error_ceiling_status"] = "interim sanity ceiling; pin from RE9 evidence";
                const bool within = bounded.within(hdr_sanity_ceiling);
                result_["within_error_ceiling"] = within;
                result_["comparison"] = within ? "bounded_error" : "mismatch";
            }
            if (!difference.exact()) {
                const auto width = result_.at("width").get<size_t>();
                result_["first_difference"] = {{"byte", difference.first_byte},
                                               {"x", difference.first_byte / 8 % width},
                                               {"y", difference.first_byte / 8 / width},
                                               {"channel", difference.first_byte % 8 / 2}};
            }
            result_["completed"] = true;
            std::ofstream report(directory / (prefix + ".json"), std::ios::binary);
            report << result_.dump(2) << '\n';
            report.close();
            check(bool(report), "phoenix diagnostic comparison report failed");
            completed_ = true;
            candidate_.Reset();
            native_.Reset();
            candidate_readback_.Reset();
            native_readback_.Reset();
            return true;
        } catch (const std::exception& exception) {
            failed_ = true;
            result_["comparison"] = "capture_failed";
            result_["completed"] = false;
            result_["error"] = exception.what();
            throw;
        } catch (...) {
            failed_ = true;
            result_["comparison"] = "capture_failed";
            result_["completed"] = false;
            throw;
        }
    }
};
} // namespace production_game41::diagnostic
