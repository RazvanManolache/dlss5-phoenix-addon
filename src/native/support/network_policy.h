#pragma once
#include "../model/bundle_loader.h"
#include "../runtime/kernel_module.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <set>

// Pure CPU preflight. No file reads, driver calls, allocations of GPU objects,
// or production_network include. Binary identity remains load_catalog's job.
namespace phoenix_native {
// Native Reset or absent motion suppresses BOTH history and motion bindings.
inline constexpr bool temporal_reuse(bool have_history, bool reset, bool have_motion) noexcept {
    return have_history && !reset && have_motion;
}

struct PreflightInfo {
    unsigned candidate_modules = 41, neural_launches = 154;
    uint32_t clear_words = 0;
    uint64_t counter_bytes = 0;
    unsigned synchronization_bindings = 0;
    // Exclusive endpoints in the 154-element neural array (clear excluded).
    std::vector<unsigned> group_ends;
    std::array<bool, 154> barrier_after{};
};

namespace policy_detail {
using phoenix_bundle::Json;
inline void check(bool valid, const char* message) {
    if (!valid)
        throw std::invalid_argument(message);
}
inline uint64_t integer(const Json& value, uint64_t maximum = UINT64_MAX) {
    check(value.is_number_integer() && (value.is_number_unsigned() || value.get<int64_t>() >= 0),
          "policy integer type/range");
    const auto integer_value = value.get<uint64_t>();
    check(integer_value <= maximum, "policy integer overflow");
    return integer_value;
}
inline uint32_t word(const std::vector<uint8_t>& bytes, size_t offset) {
    check(offset <= bytes.size() && bytes.size() - offset >= 4, "policy scalar extent");
    uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, 4);
    return value;
}
inline bool starts(const std::string& name, const char* prefix) {
    return name.rfind(prefix, 0) == 0;
}
struct Abi {
    std::string name;
    unsigned bytes, block_width, block_height, z;
};
inline std::vector<Abi> schedule() {
    std::vector<Abi> launch_schedule;
    auto add = [&](std::string name, unsigned bytes, unsigned block_height = 1,
                   unsigned block_width = 32) {
        launch_schedule.push_back({std::move(name), bytes, block_width, block_height, 1});
    };
    add("cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8", 264);
    for (unsigned heads : {1u, 2u, 4u, 8u}) {
        const auto prefix = "cc_tinlayout_fused_swin_" + std::to_string(heads) + "h_" +
                            std::to_string(heads * 32) + "_" + std::to_string(heads) + "_";
        const unsigned bytes = heads == 1 ? 96 : 88;
        add(prefix + "inpview_tilesync_fp8", bytes, heads);
        for (unsigned iteration_index = 0; iteration_index < (heads < 4    ? 2u
                                                              : heads == 4 ? 4u
                                                                           : 6u);
             ++iteration_index)
            add(prefix + "chained_fp8", bytes, heads);
        add(prefix + "ds_wait_fp8", bytes, heads);
    }
    const std::string split = "cc_split_swin_16h_";
    add(split + "ffwd_inpview_512_tilesync_fp8", 56, 4);
    add(split + "ffwd_proj_inpview_512_chained_fp8", 72, 4);
    for (unsigned iteration_index = 0; iteration_index < 8; ++iteration_index) {
        add(split + "qkv_512_chained_fp8", 56, 4);
        add(split + (iteration_index == 7 ? "proj_pool_512_chained_fp8" : "proj_512_chained_fp8"),
            iteration_index == 7 ? 80 : 72, iteration_index == 7 ? 4 : 8);
        if (iteration_index < 7) {
            add(split + "ffwd_512_chained_fp8", 56, 8);
            add(split + "ffwd_proj_512_chained_fp8", 72, 4);
        }
    }
    add(split + "final_head_512_wait_fp8", 40, 8);
    add("cc_vit_1d_repack_2d_to_1d_fp8", 24, 1, 256);
    for (unsigned iteration_index = 0; iteration_index < 8; ++iteration_index) {
        add(iteration_index == 0 ? "cc_vit_1d_ffn_expand_publish_fp8"
                                 : "cc_vit_1d_ffn_expand_chained_fp8",
            72, 4);
        add("cc_vit_1d_ffn_contract_chained_fp8", 72, 4);
        add("cc_vit_1d_qkv_chained_fp8", 80, 4);
        add("cc_vit_1d_attention_chained_fp8", 64, 4);
        add(iteration_index == 7 ? "cc_vit_1d_projection_wait_fp8"
                                 : "cc_vit_1d_projection_chained_fp8",
            72, 4);
    }
    add("cc_vit_1d_repack_1d_to_2d_fp8", 24, 1, 256);
    add("cc_dec_input_upsample_1024_512_tilesync_fp8", 80, 2);
    for (unsigned iteration_index = 0; iteration_index < 8; ++iteration_index) {
        add(split + "ffwd_512_chained_fp8", 56, 8);
        add(split + "ffwd_proj_512_chained_fp8", 72, 4);
        add(split + "qkv_512_chained_fp8", 56, 4);
        add(split + (iteration_index == 7 ? "proj_512_outview_wait_fp8" : "proj_512_chained_fp8"),
            72, iteration_index == 7 ? 4 : 8);
    }
    for (unsigned heads : {8u, 4u, 2u, 1u}) {
        const auto prefix = "cc_tinlayout_fused_swin_" + std::to_string(heads) + "h_" +
                            std::to_string(heads * 32) + "_" + std::to_string(heads) + "_";
        const unsigned bytes = heads == 1 ? 96 : 88;
        add(prefix + "upsample_tilesync_fp8", bytes, heads);
        for (unsigned iteration_index = 0; iteration_index < (heads < 4    ? 2u
                                                              : heads == 4 ? 4u
                                                                           : 6u);
             ++iteration_index)
            add(prefix + "chained_fp8", bytes, heads);
        add(prefix + "outview_wait_fp8", bytes, heads);
    }
    add("cc_tinlayout_fused_post_block_swin_1h_32_fp8", 184);
    return launch_schedule;
}

struct Slot {
    unsigned offset;
    const char* role;
    const char* arena = "workspace";
};
struct Slots {
    std::vector<Slot> pointers, descriptors;
};
// Describe the pointer and texture-handle slots for each launch ABI.
// Offsets are bytes in the argument packet; they are independent of
// the current model arena addresses and are checked before relocation.
inline Slots slots(const std::string& kernel_name) {
    Slots bindings;
    if (kernel_name == "cc_cb_clear")
        bindings.pointers = {{0, "synchronization_arena"}};
    else if (starts(kernel_name, "cc_tinlayout_fused_pre_block_")) {
        bindings.pointers = {{216, "output_full_stem_skip"},
                             {224, "trained_weights", "weights"},
                             {232, "publication_optional"},
                             {248, "output_downsample"}};
        bindings.descriptors = {{0, "color"},
                                {8, "history_previous"},
                                {16, "motion"},
                                {24, "depth_optional"},
                                {32, "aux_optional"}};
    } else if (starts(kernel_name, "cc_tinlayout_fused_post_block_")) {
        bindings.pointers = {{0, "input_low"},
                             {8, "input_stem_skip"},
                             {24, "trained_weights", "weights"},
                             {104, "blend_scale", "weights"}};
        bindings.descriptors = {
            {16, "neural_output_surface"}, {56, "color"}, {88, "history_previous"}, {96, "motion"}};
    } else if (kernel_name == "cg2r_copy_kernel")
        bindings.descriptors = {{0, "post_copy_source_texture"},
                                {8, "post_copy_destination_surface"}};
    else if (starts(kernel_name, "cc_vit_1d_repack_"))
        bindings.pointers = {{0, "input"}, {8, "output"}};
    else if (starts(kernel_name, "cc_vit_1d_ffn_") || starts(kernel_name, "cc_vit_1d_projection_"))
        bindings.pointers = {{0, "input"},
                             {8, "input_skip_optional"},
                             {16, "output"},
                             {24, "trained_weights", "weights"},
                             {32, "reduction_counter_optional"},
                             {40, "reduction_scratch_optional"},
                             {48, "wait_optional"},
                             {56, "publication_optional"}};
    else if (starts(kernel_name, "cc_vit_1d_qkv_"))
        bindings.pointers = {{0, "input"},
                             {8, "output_q"},
                             {16, "output_k"},
                             {24, "output_v"},
                             {32, "trained_weights", "weights"},
                             {40, "reduction_counter"},
                             {48, "reduction_scratch"},
                             {56, "wait"},
                             {64, "publication"}};
    else if (starts(kernel_name, "cc_vit_1d_attention_"))
        bindings.pointers = {{0, "input_q"},
                             {8, "input_k"},
                             {16, "input_v"},
                             {24, "output"},
                             {32, "trained_weights", "weights"},
                             {40, "wait"},
                             {48, "publication"}};
    else if (kernel_name == "cc_dec_input_upsample_1024_512_tilesync_fp8")
        bindings.pointers = {{0, "input_low"},
                             {8, "input_encoder_skip"},
                             {16, "output"},
                             {24, "scratch"},
                             {32, "counter"},
                             {40, "publication"},
                             {48, "scratch_secondary"},
                             {56, "trained_weights", "weights"}};
    else if (kernel_name == "cc_split_swin_16h_proj_pool_512_chained_fp8")
        bindings.pointers = {{0, "input"},
                             {8, "input_skip"},
                             {16, "output_encoder_skip"},
                             {24, "output_pool"},
                             {32, "trained_weights", "weights"},
                             {40, "wait"},
                             {48, "publication"}};
    else if (starts(kernel_name, "cc_split_swin_16h_final_head_"))
        bindings.pointers = {
            {0, "input"}, {8, "output"}, {16, "trained_weights", "weights"}, {24, "wait"}};
    else if (starts(kernel_name, "cc_split_swin_16h_")) {
        if (kernel_name.find("_proj") != std::string::npos)
            bindings.pointers = {{0, "input"},
                                 {8, "input_skip"},
                                 {16, "output"},
                                 {24, "trained_weights", "weights"},
                                 {48, "publication_optional"},
                                 {56, "wait"}};
        else
            bindings.pointers = {
                {0, "input"},
                {8, "output"},
                {16, "trained_weights", "weights"},
                {kernel_name.find("_ffwd_512_") != std::string::npos ? 32u : 40u, "wait_optional"},
                {48, "publication"}};
    } else if (starts(kernel_name, "cc_tinlayout_fused_swin_")) {
        bindings.pointers = {{0, "input"}, {8, "output"}, {16, "trained_weights", "weights"}};
        const std::vector<Slot> rest = kernel_name.find("_1h_32_") != std::string::npos
                                           ? std::vector<Slot>{{40, "wait_optional"},
                                                               {56, "publication_optional"},
                                                               {64, "output_downsample_optional"},
                                                               {80, "input_upsample_skip_optional"}}
                                           : std::vector<Slot>{{24, "input_upsample_skip_optional"},
                                                               {48, "wait_optional"},
                                                               {64, "publication_optional"},
                                                               {72, "output_downsample_optional"}};
        bindings.pointers.insert(bindings.pointers.end(), rest.begin(), rest.end());
    } else
        throw std::invalid_argument("policy unknown kernel ABI");
    return bindings;
}
inline bool synchronization(const std::string& role) {
    return starts(role, "wait") || starts(role, "publication") ||
           starts(role, "reduction_counter") || role == "counter" ||
           role == "synchronization_arena";
}
inline void rectangle(const std::vector<uint8_t>& bytes, unsigned offset, unsigned width,
                      unsigned height) {
    const float expected[]{
        0, 0, float(width), float(height), 1.f / float(width), 1.f / float(height)};
    for (unsigned iteration_index = 0; iteration_index < 6; ++iteration_index) {
        uint32_t bits = 0;
        std::memcpy(&bits, expected + iteration_index, 4);
        check(word(bytes, offset + iteration_index * 4) == bits,
              "policy unsupported source/history/motion rectangle");
    }
}
} // namespace policy_detail

inline PreflightInfo preflight(const phoenix_bundle::Json& frame,
                               const std::vector<phoenix_kernel::Spec>& candidates) {
    using namespace policy_detail;
    static const auto graph = schedule();
    check(graph.size() == 154 && candidates.size() == 41, "policy candidate/launch count");
    std::map<std::string, Abi> expected;
    for (const auto& abi : graph)
        expected.emplace(abi.name, abi);
    check(expected.size() == 41, "policy internal catalog count");
    std::set<std::string> seen;
    for (const auto& spec : candidates) {
        const auto expected_entry = expected.find(spec.name);
        check(expected_entry != expected.end() && seen.insert(spec.name).second &&
                  !spec.entry.empty(),
              "policy candidate catalog");
        const auto& expected_abi = expected_entry->second;
        check(spec.bytes == expected_abi.bytes && spec.x == expected_abi.block_width &&
                  spec.y == expected_abi.block_height && spec.z == expected_abi.z,
              "policy candidate ABI");
    }
    const auto width = unsigned(integer(frame.at("width"), 16384)),
               height = unsigned(integer(frame.at("height"), 16384));
    check(width && height, "policy frame extent");
    const auto workspace = integer(frame.at("arenas").at("workspace").at("bytes"), 1ull << 36);
    const auto weights = integer(frame.at("arenas").at("weights").at("bytes"), 1ull << 36);
    check(workspace && weights, "policy arena extent");
    const auto& launches = frame.at("launches");
    check(launches.is_array() && launches.size() == 156, "policy launch count");
    check(launches[0].at("name") == "cc_cb_clear" && launches[155].at("name") == "cg2r_copy_kernel",
          "policy boundary symbols");
    check(phoenix_bundle::has_barrier_policy(frame), "policy missing barrier declaration");
    constexpr std::array<unsigned, 15> positions{1,  2,  6,   10,  16,  24,  57, 58,
                                                 98, 99, 132, 140, 146, 150, 154};
    const auto& barriers = frame.at("neural_region_barriers");
    check(barriers.is_array() && barriers.size() == positions.size(), "policy barrier coverage");
    PreflightInfo info;
    for (size_t iteration_index = 0; iteration_index < positions.size(); ++iteration_index) {
        const auto& binding_record = barriers[iteration_index];
        check(integer(binding_record.at("after_launch")) == positions[iteration_index] &&
                  integer(binding_record.at("resource")) == 0 &&
                  integer(binding_record.at("before")) == 8 &&
                  integer(binding_record.at("after")) == 8,
              "policy barrier order/semantics");
        if (positions[iteration_index] > 1) {
            info.group_ends.push_back(positions[iteration_index] - 1);
            info.barrier_after[positions[iteration_index] - 2] = true;
        }
    }
    info.group_ends.push_back(154);
    const auto clear = phoenix_bundle::hex(launches[0].at("template_hex").get<std::string>());
    check(clear.size() == 16, "policy clear ABI");
    info.clear_words = word(clear, 8);
    info.counter_bytes = uint64_t(info.clear_words) * 4;
    check(info.clear_words && info.counter_bytes < workspace, "policy clear extent");
    const std::array<phoenix_launch_plan::Arena, 2> arenas{
        {{0x1000000000ull, workspace}, {0x8000000000ull, weights}}};
    for (unsigned index = 0; index < 156; ++index) {
        const auto& source = launches[index];
        check(integer(source.at("index")) == index, "policy launch order");
        const std::string name = source.at("name");
        const unsigned extent = index == 0 ? 16 : index == 155 ? 72 : graph[index - 1].bytes;
        check(integer(source.at("argument_bytes")) == extent &&
                  integer(source.at("dynamic_shared_bytes")) == 0,
              "policy launch ABI");
        check(source.at("block").is_array() && source.at("block").size() == 3 &&
                  source.at("grid").is_array() && source.at("grid").size() == 3,
              "policy launch shape");
        for (unsigned axis = 0; axis < 3; ++axis) {
            check(integer(source.at("grid")[axis], UINT32_MAX) > 0, "policy launch grid");
            check(integer(source.at("block")[axis], 1024) > 0, "policy launch block");
        }
        if (index > 0 && index < 155) {
            const auto& expected_abi = graph[index - 1];
            check(name == expected_abi.name &&
                      source.at("block") ==
                          Json::array({expected_abi.block_width, expected_abi.block_height,
                                       expected_abi.z}),
                  "policy ordered neural ABI");
        }
        const auto raw = phoenix_bundle::hex(source.at("template_hex").get<std::string>());
        check(raw.size() == extent, "policy template ABI");
        const auto abi = slots(name);
        std::vector<bool> covered(extent);
        auto claim = [&](unsigned offset, unsigned count) {
            check(offset <= extent && count <= extent - offset, "policy slot extent");
            for (unsigned iteration_index = offset; iteration_index < offset + count;
                 ++iteration_index) {
                check(!covered[iteration_index], "policy overlapping slots");
                covered[iteration_index] = true;
            }
        };
        auto bindings = [&](const Json& rows, const std::vector<Slot>& specs, bool descriptor) {
            check(rows.is_array() && rows.size() == specs.size(), "policy binding coverage");
            std::set<unsigned> offsets;
            for (const auto& binding_record : rows) {
                const auto offset =
                    unsigned(integer(binding_record.at("argument_offset"), UINT32_MAX));
                check(integer(binding_record.at("bytes")) == 8 && offsets.insert(offset).second,
                      "policy pointer ABI");
                const auto expected_entry =
                    std::find_if(specs.begin(), specs.end(), [&](const Slot& parameter_slot) {
                        return parameter_slot.offset == offset;
                    });
                check(expected_entry != specs.end() &&
                          binding_record.at("role") == expected_entry->role,
                      "policy binding role/slot");
                claim(offset, 8);
                for (unsigned iteration_index = 0; iteration_index < 8; ++iteration_index)
                    check(raw[offset + iteration_index] == 0, "policy pointer template");
                if (descriptor) {
                    check(binding_record.at("encoding") == "low32_descriptor_high32_sampler",
                          "policy descriptor encoding");
                    const bool optional = std::string(expected_entry->role) == "depth_optional" ||
                                          std::string(expected_entry->role) == "aux_optional";
                    if (optional)
                        check(binding_record.at("binding").is_null(),
                              "policy unsupported optional descriptor");
                    else
                        check(binding_record.at("binding").is_string() &&
                                  !binding_record.at("binding").get<std::string>().empty(),
                              "policy unsupported captured-null descriptor");
                } else {
                    check(binding_record.at("arena") == expected_entry->arena,
                          "policy binding arena");
                    const std::string role = expected_entry->role;
                    if (binding_record.at("relative_offset").is_null()) {
                        check(role.find("optional") != std::string::npos,
                              "policy null required pointer");
                        continue;
                    }
                    const auto offset_in_arena = integer(binding_record.at("relative_offset"));
                    const bool sync = synchronization(role);
                    if (sync) {
                        check(offset_in_arena % 4 == 0 && offset_in_arena < info.counter_bytes &&
                                  info.counter_bytes - offset_in_arena >= 4,
                              "policy counter prefix bounds");
                        ++info.synchronization_bindings;
                    } else if (std::string(expected_entry->arena) == "workspace")
                        check(offset_in_arena >= info.counter_bytes,
                              "policy feature overlaps counter prefix");
                    if (role == "synchronization_arena")
                        check(offset_in_arena == 0, "policy clear offset");
                    if (role == "output_full_stem_skip" || role == "input_stem_skip")
                        check(offset_in_arena == info.counter_bytes,
                              "policy counter prefix coverage");
                }
            }
        };
        bindings(source.at("relocations"), abi.pointers, false);
        bindings(source.at("descriptor_bindings"), abi.descriptors, true);
        for (const auto& scalar : source.at("scalar_u32_le")) {
            const auto offset = unsigned(integer(scalar.at("argument_offset"), UINT32_MAX));
            check(offset % 4 == 0, "policy scalar alignment");
            claim(offset, 4);
            const auto bits = phoenix_bundle::hex(scalar.at("bits_hex").get<std::string>());
            check(bits.size() == 4 && std::equal(bits.begin(), bits.end(), raw.begin() + offset),
                  "policy scalar bits");
        }
        check(std::all_of(covered.begin(), covered.end(), [](bool value) { return value; }),
              "policy scalar coverage");
        auto launch = phoenix_bundle::launch(source, [](const Json&) { return uint64_t(1); });
        phoenix_launch_plan::materialize(launch, arenas);
        if (index == 1 || index == 154) {
            for (unsigned offset : index == 1 ? std::array<unsigned, 3>{40, 64, 136}
                                              : std::array<unsigned, 3>{64, 116, 140})
                rectangle(raw, offset, width, height);
            // Source-built plans use the tiled geometry rules, including the
            // minimum canvas and shifted-window margin. Captured legacy plans
            // keep their original, narrower padding contract.
            const auto stem_raw =
                phoenix_bundle::hex(launches[1].at("template_hex").get<std::string>());
            const unsigned padded_width = word(stem_raw, 244), padded_height = word(stem_raw, 240);
            if (frame.value("barrier_policy", std::string()) == "phoenix-native-graph/1") {
                const phoenix_generated::Geometry geometry(width, height);
                check(padded_width == geometry.values.at("W0") &&
                          padded_height == geometry.values.at("H0"),
                      "policy generated padded extent");
            } else
                check(padded_width >= width && padded_width - width < 128 &&
                          padded_height >= height && padded_height - height < 128 &&
                          padded_width % 64 == 0 && padded_height % 64 == 0,
                      "policy padded extent");
            check(source.at("grid") ==
                      (index == 1 ? Json::array({padded_width / 8, padded_height / 8, 1u})
                                  : Json::array({padded_width / 8 + 1, padded_height / 8 + 1, 1u})),
                  "policy stem/head grid extent");
            if (index == 1)
                check(word(raw, 208) == height && word(raw, 212) == width &&
                          word(raw, 240) == padded_height && word(raw, 244) == padded_width &&
                          word(raw, 256) == padded_height / 2 && word(raw, 260) == padded_width / 2,
                      "policy stem extent");
            else
                check(word(raw, 32) == padded_height && word(raw, 36) == padded_width &&
                          word(raw, 172) == width && word(raw, 176) == height,
                      "policy head extent");
        }
    }
    return info;
}

inline PreflightInfo preflight(const phoenix_bundle::Json& frame,
                               const std::vector<phoenix_kernel::Binary>& candidates) {
    std::vector<phoenix_kernel::Spec> specs;
    specs.reserve(candidates.size());
    for (const auto& binary : candidates) {
        policy_detail::check(!binary.bytes.empty() && binary.bytes.size() <= 16 * 1024 * 1024,
                             "policy candidate binary extent");
        specs.push_back(binary.spec);
    }
    return preflight(frame, specs);
}
} // namespace production41
