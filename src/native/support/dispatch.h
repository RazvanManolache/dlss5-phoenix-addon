#pragma once
#include "../network/network.h"
#include "controls.h"
#include "dispatch_audit.h"

namespace phoenix_addon {
// Tail truncation is an approximate architecture, not stale-workspace ablation.
// Require the captured eight-block chain and the repack ABI before enabling it.
inline void validate_phoenix_depth_plan(const phoenix_bundle::Json& frame) {
    const auto& launches = frame.at("launches");
    check(launches.size() == 156, "Phoenix DLSS5 depth plan extent");
    auto pointer = [](const auto& launch, unsigned offset, const char* role) {
        for (const auto& relocation : launch.at("relocations"))
            if (relocation.at("argument_offset") == offset) {
                check(relocation.at("arena") == "workspace" && relocation.at("bytes") == 8 &&
                          relocation.at("role") == role &&
                          !relocation.at("relative_offset").is_null(),
                      "Phoenix DLSS5 depth pointer contract");
                return relocation.at("relative_offset").template get<uint64_t>();
            }
        check(false, "Phoenix DLSS5 depth missing pointer");
        return uint64_t(0);
    };
    uint64_t previous = pointer(launches.at(57), 8, "output");
    const auto dimensions = launches.at(57).at("scalar_u32_le");
    check(launches.at(57).at("name") == "cc_vit_1d_repack_2d_to_1d_fp8" &&
              launches.at(98).at("name") == "cc_vit_1d_repack_1d_to_2d_fp8" &&
              launches.at(98).at("argument_bytes") == 24 &&
              launches.at(98).at("scalar_u32_le") == dimensions,
          "Phoenix DLSS5 depth repack contract");
    for (unsigned block_index = 0; block_index < 8; ++block_index) {
        const unsigned first = 58 + block_index * 5;
        const char* names[]{block_index ? "cc_vit_1d_ffn_expand_chained_fp8"
                                        : "cc_vit_1d_ffn_expand_publish_fp8",
                            "cc_vit_1d_ffn_contract_chained_fp8", "cc_vit_1d_qkv_chained_fp8",
                            "cc_vit_1d_attention_chained_fp8",
                            block_index == 7 ? "cc_vit_1d_projection_wait_fp8"
                                             : "cc_vit_1d_projection_chained_fp8"};
        for (unsigned iteration_index = 0; iteration_index < 5; ++iteration_index)
            check(launches.at(first + iteration_index).at("name") == names[iteration_index],
                  "Phoenix DLSS5 depth block sequence");
        check(pointer(launches.at(first), 0, "input") == previous,
              "Phoenix DLSS5 depth disconnected prefix");
        const auto& projection = launches.at(first + 4);
        check(projection.at("argument_bytes") == 72, "Phoenix DLSS5 depth projection ABI");
        previous = pointer(projection, 16, "output");
    }
    check(pointer(launches.at(98), 0, "input") == previous,
          "Phoenix DLSS5 depth final output contract");
}
// Dedicated API table: never changes the parent's/shared Api or launches any
// native kernel. Only verified dynamic fields in own stem/head arguments vary.
class FrameDispatch {
    phoenix_native::Api api_;
    decltype(api_.dispatch) real_ = api_.dispatch;
    inline static thread_local FrameDispatch* active_ = nullptr;
    ID3D12GraphicsCommandList* command_ = nullptr;
    ResolvedControls controls_{};
    unsigned position_ = 0;
    uint64_t launches_ = 0;
    uint32_t skip_mask_ = 0;
    unsigned expected_launches_ = kLaunchCount;
    uint64_t frame_launches_ = 0;
    unsigned transformer_blocks_ = 8;
    uint64_t retained_output_ = 0;
    std::array<int, 8> repeats_{{1, 1, 1, 1, 1, 1, 1, 1}};
    bool repeating_ = false;
    bool bypassing_ = false;
    uint64_t upstream_publication_ = 0;
    uint64_t repeat_stride_ = 0;
    phoenix_native::ComPtr<ID3D12Resource> repeat_counters_, repeat_clear_;
    std::array<std::array<std::vector<std::pair<unsigned, uint64_t>>, 5>, 8> repeat_sync_;
    std::array<NVAPI_CU_KERNEL_LAUNCH_PARAMS, 5> repeat_launches_{};
    std::array<std::array<uint8_t, 80>, 5> repeat_arguments_{};

    void remember_repeat(unsigned position, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& launch) {
        if (!repeating_ || position < 57 || position >= 97)
            return;
        const unsigned block_index = (position - 57) / 5, stage_index = (position - 57) % 5;
        if (repeats_[block_index] <= 1 || block_index >= transformer_blocks_)
            return;
        check(launch.pParams && launch.paramSize <= 80, "Phoenix DLSS5 repeat argument extent");
        repeat_launches_[stage_index] = launch;
        memcpy(repeat_arguments_[stage_index].data(), launch.pParams, launch.paramSize);
    }
    NvAPI_Status repeat_block(unsigned position) {
        if (!repeating_ || position < 61 || position >= 97 || (position - 61) % 5)
            return NVAPI_OK;
        const unsigned block_index = (position - 61) / 5;
        if (repeats_[block_index] <= 1 || block_index >= transformer_blocks_)
            return NVAPI_OK;
        uint64_t feedback = 0;
        memcpy(&feedback, repeat_arguments_[4].data() + 16, 8);
        check(feedback && repeat_counters_, "Phoenix DLSS5 repeat missing feedback/counters");
        for (int iteration = 1; iteration < repeats_[block_index]; ++iteration) {
            // Every repeated block gets fresh synchronization words. Its first
            // wait still names the completed upstream block; all internal
            // waits/reduction counters/publications use this iteration's bank.
            phoenix_native::uav(command_);
            const uint64_t bank = repeat_counters_->GetGPUVirtualAddress() +
                                  repeat_stride_ * (block_index * 3 + unsigned(iteration - 1));
            for (unsigned stage_index = 0; stage_index < 5; ++stage_index) {
                auto bytes = repeat_arguments_[stage_index];
                auto launch = repeat_launches_[stage_index];
                if (stage_index == 0)
                    memcpy(bytes.data(), &feedback, 8);
                if (stage_index == 1)
                    memcpy(bytes.data() + 8, &feedback, 8);
                for (const auto& field : repeat_sync_[block_index][stage_index]) {
                    const uint64_t address = bank + field.second;
                    memcpy(bytes.data() + field.first, &address, 8);
                }
                launch.pParams = bytes.data();
                dispatch_audit::OwnCall own_dispatch;
                const auto result = real_(command_, &launch, 1);
                ++launches_;
                ++frame_launches_;
                if (result != NVAPI_OK)
                    return result;
                // Repeated output reuses the original feature buffers. Full
                // barriers prevent overlap with readers from the prior use.
                phoenix_native::uav(command_);
            }
        }
        return NVAPI_OK;
    }

    bool omitted(unsigned position) const noexcept {
        return position >= 57 && position < 97 &&
               ((position - 57) / 5 >= transformer_blocks_ || repeats_[(position - 57) / 5] == 0);
    }
    void retain(unsigned position, const NVAPI_CU_KERNEL_LAUNCH_PARAMS& launch) {
        if (bypassing_) {
            if (position == 56) {
                check(launch.pParams && launch.paramSize == 24,
                      "Phoenix DLSS5 bypass input repack ABI");
                memcpy(&retained_output_, static_cast<const uint8_t*>(launch.pParams) + 8, 8);
                return;
            }
            if (position < 61 || position >= 97 || (position - 61) % 5)
                return;
            check(launch.pParams && launch.paramSize == 72, "Phoenix DLSS5 bypass projection ABI");
            memcpy(&upstream_publication_, static_cast<const uint8_t*>(launch.pParams) + 56, 8);
        } else if (transformer_blocks_ == 8 || position != 56 + 5 * transformer_blocks_)
            return;
        check(launch.pParams && launch.paramSize == 72, "Phoenix DLSS5 depth projection ABI");
        std::memcpy(&retained_output_, static_cast<const uint8_t*>(launch.pParams) + 16, 8);
        check(retained_output_, "Phoenix DLSS5 depth missing current-frame output");
    }
    void repack(unsigned position, NVAPI_CU_KERNEL_LAUNCH_PARAMS& launch,
                std::array<uint8_t, 24>& bytes) {
        if ((!bypassing_ && transformer_blocks_ == 8) || position != 97)
            return;
        check(retained_output_ && launch.pParams && launch.paramSize == bytes.size(),
              "Phoenix DLSS5 depth repack ABI");
        std::memcpy(bytes.data(), launch.pParams, bytes.size());
        std::memcpy(bytes.data(), &retained_output_, 8);
        launch.pParams = bytes.data();
    }

    static NvAPI_Status __cdecl dispatch(ID3D12GraphicsCommandList* cmd,
                                         const NVAPI_CU_KERNEL_LAUNCH_PARAMS* source, NvU32 count) {
        auto* self = active_;
        check(self && cmd == self->command_ && source && count &&
                  count <= kLaunchCount - self->position_,
              "phoenix dispatch outside verified frame scope");
        if (self->repeating_ && count > 1) {
            for (NvU32 iteration_index = 0; iteration_index < count; ++iteration_index) {
                const auto result = dispatch(cmd, source + iteration_index, 1);
                if (result != NVAPI_OK)
                    return result;
                phoenix_native::uav(cmd);
            }
            return NVAPI_OK;
        }
        // The captured plan issues one launch per call, 154 calls per frame.
        // Only positions 0 and 153 have their arguments patched, so the 152
        // interior launches are forwarded exactly as the Network recorded
        // them: no descriptor copy and no allocation. Handles, grids, blocks,
        // shared memory, argument bytes, order, count and the audit are
        // unchanged. The multi-launch body below still handles chaining.
        if (count == 1) {
            const unsigned position = self->position_;
            if (self->omitted(position) ||
                (self->skip_mask_ & (1u << group_of_position(position)))) {
                self->position_ += 1;
                return NVAPI_OK;
            }
            self->retain(position, source[0]);
            // Reconnect to the current frame's last executed output. Chained
            // expansion dereferences its wait pointer even for a skipped
            // prefix, which therefore uses a dedicated zero-filled ready bank.
            NVAPI_CU_KERNEL_LAUNCH_PARAMS bypass_launch = source[0];
            std::array<uint8_t, 80> bypass_bytes{};
            if (self->bypassing_ && position >= 57 && position < 97 && (position - 57) % 5 < 2) {
                check(source[0].pParams && source[0].paramSize == 72 && self->retained_output_,
                      "Phoenix DLSS5 bypass block ABI");
                memcpy(bypass_bytes.data(), source[0].pParams, source[0].paramSize);
                const unsigned stage_index = (position - 57) % 5;
                memcpy(bypass_bytes.data() + (stage_index ? 8 : 0), &self->retained_output_, 8);
                if (!stage_index) {
                    const uint64_t ready = self->upstream_publication_
                                               ? self->upstream_publication_
                                               : self->repeat_counters_->GetGPUVirtualAddress() +
                                                     self->repeat_stride_ * 24;
                    memcpy(bypass_bytes.data() + 48, &ready, 8);
                    phoenix_native::uav(cmd);
                }
                bypass_launch.pParams = bypass_bytes.data();
                source = &bypass_launch;
            }
            self->remember_repeat(position, source[0]);
            if (position != 0 && position != 153 &&
                !((self->bypassing_ || self->transformer_blocks_ < 8) && position == 97)) {
                dispatch_audit::OwnCall own_dispatch;
                const auto result = self->real_(cmd, source, 1);
                self->position_ += 1;
                self->launches_ += 1;
                self->frame_launches_ += 1;
                return result == NVAPI_OK ? self->repeat_block(position) : result;
            }
            NVAPI_CU_KERNEL_LAUNCH_PARAMS launch = source[0];
            std::array<uint8_t, 264> stem{};
            std::array<uint8_t, 184> head{};
            std::array<uint8_t, 24> repack_bytes{};
            if (position == 0) {
                check(launch.pParams && launch.paramSize == stem.size(),
                      "phoenix missing stem ABI");
                std::memcpy(stem.data(), launch.pParams, stem.size());
                patch_stem(stem.data(), stem.size(), self->controls_);
                launch.pParams = stem.data();
            } else if (position == 153) {
                check(launch.pParams && launch.paramSize == head.size(),
                      "phoenix missing head ABI");
                std::memcpy(head.data(), launch.pParams, head.size());
                patch_head(head.data(), head.size(), self->controls_);
                launch.pParams = head.data();
            }
            self->repack(position, launch, repack_bytes);
            dispatch_audit::OwnCall own_dispatch;
            const auto result = self->real_(cmd, &launch, 1);
            self->position_ += 1;
            self->launches_ += 1;
            self->frame_launches_ += 1;
            return result;
        }
        std::vector<NVAPI_CU_KERNEL_LAUNCH_PARAMS> launches(source, source + count);
        std::array<uint8_t, 264> stem{};
        std::array<uint8_t, 184> head{};
        std::array<uint8_t, 24> repack_bytes{};
        for (NvU32 iteration_index = 0; iteration_index < count; ++iteration_index) {
            const unsigned position = self->position_ + iteration_index;
            auto& launch = launches[iteration_index];
            self->retain(position, launch);
            self->repack(position, launch, repack_bytes);
            if (position == 0) {
                check(launch.pParams && launch.paramSize == stem.size(),
                      "phoenix missing stem ABI");
                std::memcpy(stem.data(), launch.pParams, stem.size());
                patch_stem(stem.data(), stem.size(), self->controls_);
                launch.pParams = stem.data();
            } else if (position == 153) {
                check(launch.pParams && launch.paramSize == head.size(),
                      "phoenix missing head ABI");
                std::memcpy(head.data(), launch.pParams, head.size());
                patch_head(head.data(), head.size(), self->controls_);
                launch.pParams = head.data();
            }
        }
        NvU32 forwarded = 0;
        for (NvU32 iteration_index = 0; iteration_index < count; ++iteration_index)
            if (!self->omitted(self->position_ + iteration_index) &&
                !(self->skip_mask_ & (1u << group_of_position(self->position_ + iteration_index))))
                launches[forwarded++] = launches[iteration_index];
        self->position_ += count;
        if (!forwarded)
            return NVAPI_OK;
        dispatch_audit::OwnCall own_dispatch;
        const auto result = self->real_(cmd, launches.data(), forwarded);
        self->launches_ += forwarded;
        self->frame_launches_ += forwarded;
        return result;
    }

  public:
    void configure_repeats(ID3D12Device* device, const phoenix_bundle::Json& frame,
                           const std::array<int, 8>& counts) {
        check(!repeat_counters_, "Phoenix DLSS5 repeat configuration is immutable per generation");
        repeats_ = counts;
        for (int repeat_count : counts) {
            check(repeat_count >= 0 && repeat_count <= 4,
                  "Phoenix DLSS5 block repeat count outside 0..4");
            repeating_ |= repeat_count != 1;
            bypassing_ |= repeat_count == 0;
        }
        if (!repeating_)
            return;
        validate_phoenix_depth_plan(frame);
        repeat_stride_ = uint64_t(phoenix_native::policy_detail::word(
                             phoenix_bundle::hex(
                                 frame.at("launches").at(0).at("template_hex").get<std::string>()),
                             8)) *
                         4;
        check(repeat_stride_ && repeat_stride_ < 16 * 1024 * 1024,
              "Phoenix DLSS5 repeat counter extent");
        for (unsigned block_index = 0; block_index < 8; ++block_index)
            for (unsigned stage_index = 0; stage_index < 5; ++stage_index)
                for (const auto& relocation : frame.at("launches")
                                                  .at(58 + block_index * 5 + stage_index)
                                                  .at("relocations")) {
                    const auto role = relocation.at("role").get<std::string>();
                    if (relocation.at("relative_offset").is_null() ||
                        !phoenix_native::policy_detail::synchronization(role) ||
                        (stage_index == 0 && role == "wait_optional"))
                        continue;
                    const auto offset = relocation.at("relative_offset").get<uint64_t>();
                    check(relocation.at("arena") == "workspace" && offset < repeat_stride_,
                          "Phoenix DLSS5 repeat counter outside prefix");
                    repeat_sync_[block_index][stage_index].push_back(
                        {relocation.at("argument_offset").get<unsigned>(), offset});
                }
        const uint64_t size = repeat_stride_ * 25;
        repeat_counters_ = phoenix_native::buffer(device, size, D3D12_HEAP_TYPE_DEFAULT,
                                                  D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        repeat_clear_ = phoenix_native::buffer(device, size, D3D12_HEAP_TYPE_UPLOAD,
                                               D3D12_RESOURCE_STATE_GENERIC_READ);
        void* data = nullptr;
        D3D12_RANGE empty{};
        phoenix_native::hr(repeat_clear_->Map(0, &empty, &data));
        memset(data, 0xff, size_t(repeat_stride_ * 24));
        memset(static_cast<uint8_t*>(data) + repeat_stride_ * 24, 0, size_t(repeat_stride_));
        repeat_clear_->Unmap(0, nullptr);
    }
    void prepare_repeats(ID3D12GraphicsCommandList* cmd) {
        if (!repeating_)
            return;
        phoenix_native::transition(cmd, repeat_counters_.Get(),
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                   D3D12_RESOURCE_STATE_COPY_DEST);
        cmd->CopyBufferRegion(repeat_counters_.Get(), 0, repeat_clear_.Get(), 0,
                              repeat_stride_ * 25);
        phoenix_native::transition(cmd, repeat_counters_.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    FrameDispatch() {
        api_.dispatch = dispatch;
    }
    // CPU-test construction only. Substitutes the real entry point so this
    // exact wrapper body can be exercised differentially against the frozen
    // pre-optimization copy without loading NVAPI. The table stays inert:
    // every other entry is null, so nothing can dispatch through it.
    explicit FrameDispatch(decltype(api_.dispatch) injected)
        : api_(phoenix_native::Api::NoLoad{}), real_(injected) {
        check(real_, "phoenix null injected dispatch");
        api_.dispatch = dispatch;
    }
    static decltype(api_.dispatch) dispatch_function() noexcept {
        return dispatch;
    }
    phoenix_native::Api& api() noexcept {
        return api_;
    }
    uint64_t launches() const noexcept {
        return launches_;
    }
    unsigned expected_launches() const noexcept {
        return expected_launches_;
    }
    const ResolvedControls& controls() const noexcept {
        return controls_;
    }
    class Scope {
        FrameDispatch& owner_;

      public:
        Scope(FrameDispatch& owner, ID3D12GraphicsCommandList* cmd,
              const ResolvedControls& controls, unsigned transformer_blocks = 8)
            : owner_(owner) {
            check(!active_, "phoenix nested candidate dispatch");
            check(transformer_blocks >= 1 && transformer_blocks <= 8,
                  "Phoenix DLSS5 transformer depth outside 1..8");
            check(transformer_blocks == 8 || skip_groups_mask.load(std::memory_order_relaxed) == 0,
                  "Phoenix DLSS5 depth cannot combine with stale-tensor ablation");
            owner.command_ = cmd;
            owner.controls_ = controls;
            owner.position_ = 0;
            active_ = &owner;
            owner.transformer_blocks_ = transformer_blocks;
            owner.retained_output_ = 0;
            owner.upstream_publication_ = 0;
            owner.skip_mask_ = skip_groups_mask.load(std::memory_order_relaxed);
            owner.expected_launches_ =
                kLaunchCount - skipped_launches(owner.skip_mask_) - 5 * (8 - transformer_blocks);
            if (owner.repeating_) {
                check(!owner.skip_mask_,
                      "Phoenix DLSS5 repeats cannot combine with stale-tensor ablation");
                owner.expected_launches_ = 114;
                for (unsigned iteration_index = 0; iteration_index < transformer_blocks;
                     ++iteration_index)
                    owner.expected_launches_ += 5 * unsigned(owner.repeats_[iteration_index]);
            }
            owner.frame_launches_ = 0;
        }
        Scope(const Scope&) = delete;
        ~Scope() {
            active_ = nullptr;
            owner_.command_ = nullptr;
        }
        void finish() const {
            check(owner_.position_ == kLaunchCount, "phoenix incomplete candidate frame");
            check(owner_.frame_launches_ == owner_.expected_launches_,
                  "phoenix skip mask/launch count mismatch");
        }
    };
};
} // namespace production_game41
