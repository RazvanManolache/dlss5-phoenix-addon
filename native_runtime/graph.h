#pragma once

#include "model.h"
#include "scratch_plan.h"

#include <array>
#include <functional>

namespace phoenix_cpp {

// A tensor is a view of an allocation. Views share ownership but can select
// different byte offsets and shapes. Dimensions are [height, width, channels].
struct Tensor {
    std::shared_ptr<Allocation> memory;
    size_t byte_offset = 0;
    unsigned height = 1, width = 1, channels = 1, element_bytes = 2;
    bool fp8_grid = false; // Proven publication, not a per-frame value scan.

    phx::OperandAddress ptr() const {
        demand(memory && byte_offset < memory->bytes &&
                   phx::address_bits(memory->address) <= UINT64_MAX - byte_offset,
               "Tensor address overflow");
        return phx::address_from_bits(phx::address_bits(memory->address) + byte_offset);
    }

    size_t size() const {
        return size_t(height) * width * channels;
    }
};

// Records the model as backend-independent operations, then assigns temporary
// tensors to a shared scratch arena. GPU/CPU kernels execute these operations.
class Graph {
    Backend& backend_;
    Model& model_;
    std::vector<Allocation> owned_allocations_;
    ScratchPlan scratch_;
    std::map<std::string, Tensor> weight_cache_;
    unsigned current_block_ = 0, current_layer_ = 0;
    bool fused_attention_ = false;
    unsigned window_batch_limit_ = 128;
    bool device_features_ = false;
    bool fused_residual_ = false;
    bool device_noise_cache_ = false;
    bool noise_uploaded_ = false;
    unsigned uploaded_noise_seed_ = 0;
    std::array<Tensor, 4> frame_planes_;

    static bool publishes_fp8(unsigned publication) {
        return publication == 1 || publication == 3 || publication == 5 || publication == 6 ||
               publication == 7;
    }

    static unsigned align_extent(unsigned extent, unsigned alignment) {
        return (extent + alignment - 1) / alignment * alignment;
    }

    Tensor allocate_tensor(unsigned height, unsigned width, unsigned channels,
                           unsigned element_bytes = 2, bool recyclable = true) {
        uint64_t bytes = uint64_t(height) * width * channels * element_bytes;
        demand(bytes && bytes <= uint64_t(16) * 1024 * 1024 * 1024,
               "Tensor exceeds allocation budget");
        demand(bytes <= SIZE_MAX, "Tensor exceeds host addressable extent");
        Allocation allocation;
        if (recyclable) {
            allocation = scratch_.reserve(bytes);
        } else {
            allocation = backend_.allocate(bytes);
            owned_allocations_.push_back(allocation);
        }
        return {
            std::make_shared<Allocation>(allocation), 0, height, width, channels, element_bytes};
    }

    void commit_scratch() {
        // Logical addresses identify temporary tensors during recording.
        // Mark every use before choosing which lifetimes can share storage.
        for (size_t index = 0; index < commands.size(); ++index) {
            auto& operation = commands[index].operation;
            for (phx::OperandAddress address : {operation.left_input, operation.right_input, operation.output,
                                  operation.auxiliary, operation.secondary_output})
                scratch_.touch(address, index + 1);
        }
        scratch_.touch(input.ptr(), 0);
        scratch_.touch(output.ptr(), commands.size() + 1);
        for (const auto& point : checkpoints)
            scratch_.touch(
                phx::address_from_bits(phx::address_bits(point.allocation.address) + point.offset),
                point.after + 1);

        for (const auto& plane : frame_planes_)
            if (plane.memory) {
                scratch_.touch(plane.ptr(), 0);
                // Cached noise must survive every neural command and subsequent
                // frames. Other upload planes remain short-lived scratch views.
                if (device_noise_cache_ && &plane == &frame_planes_[2])
                    scratch_.touch(plane.ptr(), commands.size() + 1);
            }
        auto arena = backend_.allocate(scratch_.layout());
        owned_allocations_.push_back(arena);

        // Replace logical addresses with offsets in the final arena. Weight
        // allocations are persistent and are left unchanged by relocate().
        for (auto& command : commands) {
            auto& operation = command.operation;
            operation.left_input = scratch_.relocate(operation.left_input, arena);
            operation.right_input = scratch_.relocate(operation.right_input, arena);
            operation.output = scratch_.relocate(operation.output, arena);
            operation.auxiliary = scratch_.relocate(operation.auxiliary, arena);
            operation.secondary_output = scratch_.relocate(operation.secondary_output, arena);
        }
        for (auto& point : checkpoints) {
            if (scratch_.contains(point.allocation.address)) {
                point.offset += scratch_.offset(point.allocation.address);
                point.allocation = arena;
            }
        }
        auto bind_to_arena = [&](Tensor& tensor) {
            auto byte_offset = scratch_.offset(tensor.ptr());
            tensor.memory = std::make_shared<Allocation>(arena);
            tensor.byte_offset = byte_offset;
        };
        bind_to_arena(input);
        bind_to_arena(output);
        for (auto& plane : frame_planes_)
            if (plane.memory)
                bind_to_arena(plane);
    }

    static Allocation tensor_allocation(const Tensor& tensor) {
        return {tensor.ptr(), tensor.size() * tensor.element_bytes, tensor.memory->token};
    }

    Tensor tensor_view(Tensor tensor, size_t element_offset, unsigned height, unsigned width,
                       unsigned channels) {
        tensor.byte_offset += element_offset * tensor.element_bytes;
        tensor.height = height;
        tensor.width = width;
        tensor.channels = channels;
        demand(tensor.byte_offset + tensor.size() * tensor.element_bytes <= tensor.memory->bytes,
               "Tensor view bounds");
        return tensor;
    }

    Tensor upload_weight(std::string key, std::function<std::vector<float>()> make_values,
                         bool store_half = true) {
        if (weight_cache_.contains(key))
            return weight_cache_.at(key);
        auto values = make_values();
        demand(!values.empty(), "Empty weight");
        Tensor tensor = allocate_tensor(1, 1, unsigned(values.size()), store_half ? 2 : 4, false);
        if (store_half) {
            std::vector<uint16_t> half_values(values.size());
            phx::Kernel<HostMath> converter{};
            for (size_t index = 0; index < half_values.size(); ++index)
                half_values[index] = uint16_t(converter.half_bits(values[index]));
            backend_.upload(*tensor.memory, half_values.data(), half_values.size() * 2);
        } else {
            backend_.upload(*tensor.memory, values.data(), values.size() * 4);
        }
        weight_cache_[key] = tensor;
        return tensor;
    }

    Tensor upload_packed_weight(const std::string& key,
                                std::function<std::vector<uint8_t>()> make_bytes) {
        if (weight_cache_.contains(key))
            return weight_cache_.at(key);
        auto bytes = make_bytes();
        demand(!bytes.empty(), "Empty FP8 weight");
        auto tensor = allocate_tensor(1, 1, unsigned(bytes.size()), 1, false);
        backend_.upload(*tensor.memory, bytes.data(), bytes.size());
        weight_cache_[key] = tensor;
        return tensor;
    }

    Tensor model_weight(int block, int layer, const std::string& field, bool packed_fp8 = true) {
        auto key = std::to_string(block) + '.' + std::to_string(layer) + '.' + field;
        return packed_fp8
                   ? upload_packed_weight(key, [&] { return model_.raw(block, layer, field); })
                   : upload_weight(key, [&] { return model_.half(block, layer, field); });
    }

    Tensor residual_gain(int block, int layer, const std::string& field, unsigned channels) {
        return upload_weight(
            std::to_string(block) + '.' + std::to_string(layer) + field + "gain",
            [&] {
                auto values = field == "residual_scale" ? model_.gain(block)
                                                        : model_.half(block, layer, field);
                demand(values.size() >= channels, "Gain tensor too short");
                values.resize(channels);
                return values;
            },
            false);
    }

    static uint32_t dispatch_count(size_t elements) {
        demand(elements <= UINT32_MAX, "Dispatch index overflow");
        return uint32_t(elements);
    }

    void emit(phx::Operation operation, uint64_t elements) {
        demand(elements > 0, "Empty operation");
        commands.push_back({operation, elements, current_block_, current_layer_});
    }

    // Operation is the shared backend descriptor; see docs/OPERATOR_ABI.md.
    // Dimension and stride slots carry operation-specific geometry for pointwise work.
    // StoreHalf controls destination precision; Left/RightInputHalf describe input storage.
    Tensor pointwise(Tensor source, unsigned flags, unsigned height = 0, unsigned width = 0,
                     unsigned channels = 0, Tensor* secondary = nullptr,
                     Tensor* auxiliary = nullptr, unsigned batch = 0, unsigned stride_a = 0,
                     unsigned stride_b = 0, unsigned stride_c = 0, unsigned padding = 0,
                     float scale = 1) {
        if (!height) {
            height = source.height;
            width = source.width;
            channels = source.channels;
        }
        auto result = allocate_tensor(height, width, channels, flags & phx::StoreHalf ? 2 : 4);
        phx::Operation operation{};
        operation.left_input = source.ptr();
        operation.right_input = secondary ? secondary->ptr() : source.ptr();
        operation.output = result.ptr();
        operation.auxiliary = auxiliary ? auxiliary->ptr() : source.ptr();
        operation.flags = flags | (source.element_bytes == 2 ? phx::LeftInputHalf : 0) |
                          (secondary && secondary->element_bytes == 2 ? phx::RightInputHalf : 0);
        operation.row_count = dispatch_count(result.size());
        operation.column_count = channels;
        operation.batch_count = batch;
        operation.left_batch_stride = stride_a;
        operation.right_batch_stride = stride_b;
        operation.output_batch_stride = stride_c;
        operation.reduction_count = padding;
        operation.scale = scale;
        emit(operation, result.size());
        const auto publication = (flags >> 8) & 15u;
        const unsigned kind = flags & 255u;
        const bool keeps_grid = kind == phx::CopyValues || kind == phx::SpatialPad ||
                                kind == phx::GatherWindowBatch || kind == phx::GatherWindows ||
                                kind == phx::ScatterWindowBatch;
        result.fp8_grid =
            publishes_fp8(publication) || (publication == 0 && keeps_grid && source.fp8_grid);
        return result;
    }

    // Publication preserves explicit intermediate rounding. Epilogue 1 rounds
    // to E4M3, 4 to FP16, and 5 to FP16 followed by E4M3 (stored as FP16).
    Tensor publish(Tensor source, unsigned epilogue = 5) {
        return pointwise(source, (epilogue << 8) | phx::StoreHalf | 4);
    }

    Tensor matrix_multiply(Tensor source, Tensor matrix, unsigned output_channels,
                           unsigned epilogue = 0, bool store_half = false, unsigned batch = 1,
                           unsigned stride_a = 0, unsigned stride_b = 0, unsigned stride_c = 0,
                           unsigned input_row_stride = 0, unsigned output_row_stride = 0) {
        demand(source.element_bytes == 2 &&
                   (matrix.element_bytes == 1 || matrix.element_bytes == 2),
               "GEMM operand precision");
        auto result =
            allocate_tensor(source.height, source.width, output_channels, store_half ? 2 : 4);
        phx::Operation operation{};
        operation.family = phx::MatrixMultiply;
        operation.left_input = source.ptr();
        operation.right_input = matrix.ptr();
        operation.output = result.ptr();
        operation.row_count = dispatch_count(size_t(source.height) * source.width);
        operation.column_count = output_channels;
        operation.reduction_count = source.channels;
        operation.batch_count = batch;
        operation.left_batch_stride = stride_a;
        operation.right_batch_stride = stride_b;
        operation.output_batch_stride = stride_c;
        operation.left_row_stride = input_row_stride ? input_row_stride : source.channels;
        operation.right_row_stride = source.channels;
        operation.output_row_stride = output_row_stride ? output_row_stride : output_channels;
        // phx::RightInputFp8 selects byte-packed FP8 weights; the accumulation and
        // publication order is otherwise the same as in the original graph.
        operation.flags = 1 | (epilogue << 8) | (store_half ? phx::StoreHalf : 0) |
                          (matrix.element_bytes == 1 ? phx::RightInputFp8 : 0) |
                          (source.fp8_grid ? phx::LeftInputFp8Grid : 0);
        emit(operation, uint64_t(operation.row_count) * operation.column_count * batch);
        result.fp8_grid = publishes_fp8(epilogue);
        return result;
    }

    Tensor residual(Tensor source, Tensor skip, Tensor gains, unsigned epilogue = 5,
                    bool store_half = true) {
        // A packed-weight GEMM can publish its scaled residual directly. Keep
        // the same FP32 multiply/add and half/FP8 boundaries, without materializing
        // the unconsumed projection as a separate tensor in global memory.
        if (fused_residual_ && source.element_bytes == 4 && !commands.empty()) {
            auto& previous = commands.back().operation;
            if (previous.family == phx::MatrixMultiply && previous.output == source.ptr() &&
                (previous.flags & phx::RightInputFp8) && ((previous.flags >> 8) & 15) == 0 &&
                previous.output_row_stride == source.channels) {
                auto result = allocate_tensor(source.height, source.width, source.channels,
                                              store_half ? 2 : 4);
                previous.output = result.ptr();
                previous.auxiliary = skip.ptr();
                previous.secondary_output = gains.ptr();
                result.fp8_grid = publishes_fp8(epilogue);
                previous.flags |= phx::MatrixResidual | (epilogue << 8) |
                                  (store_half ? phx::StoreHalf : 0) |
                                  (skip.element_bytes == 2 ? phx::MatrixSkipHalf : 0);
                return result;
            }
        }
        // Operator 5 computes source + skip * gains, then publishes it.
        return pointwise(source,
                         phx::ScaledResidual | (epilogue << 8) | (store_half ? phx::StoreHalf : 0),
                         0, 0, 0, &skip, &gains);
    }

    Tensor average_pool(Tensor source) {
        return pointwise(source, phx::AveragePool | (1 << 8) | phx::StoreHalf, source.height / 2,
                         source.width / 2, source.channels, nullptr, nullptr, 0, source.height,
                         source.width);
    }

    Tensor pad_spatial(Tensor source, unsigned height, unsigned width) {
        if (height == source.height && width == source.width)
            return source;
        demand(height >= source.height && width >= source.width, "Spatial padding cannot crop");
        return pointwise(source, phx::SpatialPad | phx::StoreHalf, height, width, source.channels,
                         nullptr, nullptr, 0, source.height, source.width, width);
    }

    Tensor qkv_weight(int block, int layer, unsigned channels, bool global_attention) {
        return upload_packed_weight(std::to_string(block) + "qkv", [&] {
            std::vector<uint8_t> combined;
            if (global_attention) {
                for (auto field : {"q", "k", "v"}) {
                    auto matrix = model_.raw(block, layer, field);
                    combined.insert(combined.end(), matrix.begin(), matrix.end());
                }
                return combined;
            }
            // Window records interleave Q/K/V by head. GEMM expects all Q
            // rows first, then K rows, then V rows.
            auto source = model_.raw(block, layer, "qkv");
            combined.resize(source.size());
            for (unsigned head = 0; head < channels / 32; ++head)
                for (unsigned part = 0; part < 3; ++part)
                    for (unsigned head_channel = 0; head_channel < 32; ++head_channel)
                        for (unsigned input_channel = 0; input_channel < channels; ++input_channel)
                            combined[size_t(part * channels + head * 32 + head_channel) * channels +
                                     input_channel] =
                                source[size_t(head * 96 + part * 32 + head_channel) * channels +
                                       input_channel];
            return combined;
        });
    }

    Tensor attention_windows(Tensor gathered, int block, int layer, bool global_attention) {
        current_layer_ = layer;
        unsigned batch = gathered.height, tokens = gathered.width;
        unsigned channels = gathered.channels, head_count = channels / 32;
        auto qkv_matrix = qkv_weight(block, layer, channels, global_attention);
        auto head_scale = upload_weight(
            std::to_string(block) + "headscale",
            [&] {
                auto values = model_.floats(block, layer,
                                            global_attention  ? "header"
                                            : channels == 512 ? "tail"
                                                              : "scalars_b");
                values.resize(head_count);
                return values;
            },
            false);
        auto position_bias =
            global_attention ? head_scale
                             : upload_weight(
                                   std::to_string(block) + "bias",
                                   [&] { return model_.bias(block, layer, head_count); }, false);

        auto projected = matrix_multiply(gathered, qkv_matrix, 3 * channels);
        std::array<Tensor, 3> qkv;
        for (unsigned part = 0; part < 3; ++part) {
            qkv[part] = allocate_tensor(batch, head_count * tokens, 32);
            phx::Operation operation{};
            operation.left_input = projected.ptr();
            operation.output = qkv[part].ptr();
            operation.auxiliary = head_scale.ptr();
            operation.row_count = batch * head_count * tokens;
            operation.column_count = tokens;
            operation.batch_count = head_count;
            if (part < 2) {
                // Attention-family operators normalize/prepare Q and K.
                operation.family = phx::RowReduction;
                operation.flags =
                    phx::StoreHalf | 0x20000 | (part << 18) | (global_attention ? 0x80000 : 0);
                operation.reduction_count = part == 0;
                emit(operation, operation.row_count);
            } else {
                // V needs only head separation and the required publication.
                operation.flags = phx::SplitQkvHeads | phx::StoreHalf | (5 << 8);
                operation.column_count = channels;
                operation.left_batch_stride = tokens;
                operation.right_batch_stride = 2;
                operation.batch_count = head_count;
                emit(operation, uint64_t(batch) * tokens * channels);
            }
        }

        // A fused local-attention dispatch keeps the 64x64 score/probability
        // tiles in shared memory and publishes directly in token-major order.
        // Global attention retains its different normalization contract below.
        if (fused_attention_ && !global_attention && tokens == 64) {
            auto merged = allocate_tensor(batch, tokens, channels);
            phx::Operation operation{};
            operation.family = phx::WindowAttention;
            operation.left_input = qkv[0].ptr();
            operation.right_input = qkv[1].ptr();
            operation.auxiliary = qkv[2].ptr();
            operation.secondary_output = position_bias.ptr(); // Read-only relative-position bias.
            operation.output = merged.ptr();
            operation.batch_count = batch * head_count;
            operation.column_count = head_count;
            emit(operation, operation.batch_count);
            merged.fp8_grid = true;
            return merged;
        }
        if (global_attention)
            current_layer_ = 3;
        // Q * transpose(K) produces one tokens-by-tokens score matrix per head.
        auto scores = allocate_tensor(batch * head_count, tokens, tokens, 4);
        phx::Operation operation{};
        operation.family = phx::MatrixMultiply;
        operation.left_input = qkv[0].ptr();
        operation.right_input = qkv[1].ptr();
        operation.output = scores.ptr();
        operation.flags = 1;
        operation.row_count = tokens;
        operation.column_count = tokens;
        operation.reduction_count = 32;
        operation.batch_count = batch * head_count;
        operation.left_batch_stride = operation.right_batch_stride = tokens * 32;
        operation.output_batch_stride = tokens * tokens;
        operation.left_row_stride = operation.right_row_stride = 32;
        operation.output_row_stride = tokens;
        emit(operation, uint64_t(batch) * head_count * tokens * tokens);

        auto probabilities = allocate_tensor(batch * head_count, tokens, tokens);
        auto inverse_normalizer = allocate_tensor(batch * head_count, tokens, 1, 4);
        operation = {};
        operation.family = phx::RowReduction;
        operation.left_input = scores.ptr();
        operation.right_input = position_bias.ptr();
        operation.output = probabilities.ptr();
        operation.auxiliary = global_attention ? inverse_normalizer.ptr() : scores.ptr();
        operation.flags = phx::StoreHalf | (global_attention ? 6 : 1 | 0x2000);
        operation.row_count = batch * head_count * tokens;
        operation.column_count = tokens;
        operation.batch_count = head_count;
        operation.left_batch_stride = tokens;
        emit(operation, operation.row_count);

        // Multiply normalized scores by V, then merge the head dimension.
        auto context = allocate_tensor(batch * head_count, tokens, 32, 4);
        operation = {};
        operation.family = phx::MatrixMultiply;
        operation.left_input = probabilities.ptr();
        operation.right_input = qkv[2].ptr();
        operation.output = context.ptr();
        operation.row_count = tokens;
        operation.column_count = 32;
        operation.reduction_count = tokens;
        operation.batch_count = batch * head_count;
        operation.left_batch_stride = tokens * tokens;
        operation.right_batch_stride = operation.output_batch_stride = tokens * 32;
        operation.left_row_stride = tokens;
        operation.right_row_stride = operation.output_row_stride = 32;
        emit(operation, uint64_t(batch) * head_count * tokens * 32);
        auto merged = pointwise(
            context,
            phx::MergeAttentionHeads | phx::StoreHalf | (5 << 8) | (global_attention ? 0x20000 : 0),
            batch, tokens, channels, nullptr, global_attention ? &inverse_normalizer : nullptr,
            head_count, tokens);
        return merged;
    }

    Tensor attention(Tensor source, int block, int layer, int shift_x, int shift_y,
                     bool global_attention) {
        if (global_attention) {
            auto merged = attention_windows(
                tensor_view(source, 0, 1, source.height * source.width, source.channels), block,
                layer, true);
            return tensor_view(merged, 0, source.height, source.width, source.channels);
        }

        unsigned channels = source.channels;
        unsigned top_padding = unsigned(-shift_y * 4), left_padding = unsigned(-shift_x * 4);
        unsigned windows_across = align_extent(source.width + left_padding, 8) / 8;
        unsigned window_count = align_extent(source.height + top_padding, 8) / 8 * windows_across;
        auto result = allocate_tensor(source.height, source.width, channels);

        // Stream bounded window batches; the configured limit controls temporary activations.
        for (unsigned first_window = 0; first_window < window_count;
             first_window += window_batch_limit_) {
            unsigned batch = std::min(window_batch_limit_, window_count - first_window);
            auto gathered =
                pointwise(source, phx::GatherWindowBatch | phx::StoreHalf, batch, 64, channels,
                          nullptr, nullptr, first_window, source.height, source.width,
                          windows_across, (top_padding << 16) | left_padding);
            auto merged = attention_windows(gathered, block, layer, false);
            phx::Operation operation{};
            operation.left_input = merged.ptr();
            operation.output = result.ptr();
            operation.flags = phx::ScatterWindowBatch | phx::StoreHalf | phx::LeftInputHalf;
            operation.column_count = channels;
            operation.batch_count = first_window;
            operation.left_batch_stride = source.height;
            operation.right_batch_stride = source.width;
            operation.output_batch_stride = windows_across;
            operation.reduction_count = (top_padding << 16) | left_padding;
            emit(operation, merged.size());
        }
        result.fp8_grid = true;
        return result;
    }

    Tensor shifted_window_block(Tensor source, int block, int shift_x, int shift_y,
                                bool keep_half = false, Tensor* pooled = nullptr,
                                Tensor* skip = nullptr, Tensor* blend_gains = nullptr) {
        current_layer_ = 0;
        bool input_stem = block == 0 && source.channels == 16;
        unsigned channels = input_stem ? 32 : source.channels, head_count = channels / 32;
        unsigned height = skip ? skip->height : source.height;
        unsigned width = skip ? skip->width : source.width;
        unsigned top_padding = unsigned(-shift_y * 4), left_padding = unsigned(-shift_x * 4);
        unsigned windows_across = align_extent(width + left_padding, 8) / 8;
        unsigned window_count = align_extent(height + top_padding, 8) / 8 * windows_across;
        auto result = allocate_tensor(height, width, channels);
        Tensor pooled_result;
        if (pooled)
            pooled_result = allocate_tensor(height / 2, width / 2, channels);

        for (unsigned first_window = 0; first_window < window_count;
             first_window += window_batch_limit_) {
            unsigned batch = std::min(window_batch_limit_, window_count - first_window);
            Tensor gathered;
            if (skip) {
                demand(blend_gains, "Post blend gain missing");
                gathered = pointwise(source, phx::GatherUpsampleBlendBatch | phx::StoreHalf, batch,
                                     64, channels, skip, blend_gains, first_window, height, width,
                                     windows_across, (top_padding << 16) | left_padding,
                                     float(source.width));
            } else {
                gathered = pointwise(source, phx::GatherWindowBatch | phx::StoreHalf, batch, 64,
                                     source.channels, nullptr, nullptr, first_window, height, width,
                                     windows_across, (top_padding << 16) | left_padding);
            }
            if (input_stem)
                gathered = matrix_multiply(
                    gathered, upload_weight("lift", [&] { return model_.lift(); }), 32, 4, true);

            auto quantized = publish(gathered, 1);
            auto hidden = matrix_multiply(quantized, model_weight(block, 0, "mlp_expand"),
                                          4 * channels, 6, true);
            if (head_count > 1) {
                auto middle_matrix = model_weight(block, 0, "mlp_mid");
                auto per_head_result = allocate_tensor(batch, 64, channels);
                per_head_result.fp8_grid = true;
                // Heads are independent 128->32 projections. Represent them as
                // one strided batch instead of recording a launch per head.
                // A batch advances across channels; rows retain the original
                // interleaved token stride. No transpose or extra tensor is needed.
                {
                    auto head_input = tensor_view(hidden, 0, batch, 64, 128);
                    auto head_matrix = tensor_view(middle_matrix, 0, 1, 32, 128);
                    phx::Operation operation{};
                    operation.family = phx::MatrixMultiply;
                    operation.flags = 1 | phx::StoreHalf | (5 << 8) | phx::RightInputFp8 |
                                      (head_input.fp8_grid ? phx::LeftInputFp8Grid : 0);
                    operation.left_input = head_input.ptr();
                    operation.right_input = head_matrix.ptr();
                    operation.output = per_head_result.ptr();
                    operation.row_count = batch * 64;
                    operation.column_count = 32;
                    operation.reduction_count = 128;
                    operation.batch_count = head_count;
                    operation.left_batch_stride = 128;
                    operation.right_batch_stride = 32 * 128;
                    operation.output_batch_stride = 32;
                    operation.left_row_stride = 4 * channels;
                    operation.right_row_stride = 128;
                    operation.output_row_stride = channels;
                    emit(operation, uint64_t(operation.row_count) * 32 * head_count);
                }
                hidden = per_head_result;
            }
            auto feed_forward =
                matrix_multiply(hidden, model_weight(block, 0, "mlp_contract"), channels);
            auto residual_features =
                residual(feed_forward, gathered,
                         residual_gain(block, 0, "residual_scale", channels), 0, false);
            auto quantized_residual = publish(residual_features, 5);
            auto context = attention_windows(quantized_residual, block, 0, false);
            auto projected =
                matrix_multiply(context, model_weight(block, 0, "attn_out_proj"), channels);
            auto accumulator =
                residual(projected, head_count == 1 ? residual_features : quantized_residual,
                         residual_gain(block, 0, "attn_residual_scale", channels), 0, false);

            // Scatter windows back into image order. The pooled variant also
            // preserves a full-resolution skip output for the decoder.
            phx::Operation operation{};
            operation.left_input = accumulator.ptr();
            operation.column_count = channels;
            operation.batch_count = first_window;
            operation.left_batch_stride = height;
            operation.right_batch_stride = width;
            operation.output_batch_stride = windows_across;
            operation.reduction_count = (top_padding << 16) | left_padding;
            if (pooled) {
                operation.output = pooled_result.ptr();
                operation.secondary_output = result.ptr();
                operation.flags = phx::ScatterWindowBatchAndPool | phx::StoreHalf | (1 << 8);
            } else {
                operation.output = result.ptr();
                operation.flags =
                    phx::ScatterWindowBatch | phx::StoreHalf | ((keep_half ? 4 : 5) << 8);
            }
            emit(operation, accumulator.size());
        }
        result.fp8_grid = pooled || !keep_half;
        if (pooled) {
            pooled_result.fp8_grid = true;
            *pooled = pooled_result;
        }
        return result;
    }

    Tensor split_window_block(Tensor source, int block, int shift_x, int shift_y,
                              Tensor* accumulator = nullptr) {
        current_layer_ = 0;
        auto projections = model_.split(block);
        std::array<Tensor, 3> matrices;
        for (unsigned index = 0; index < 3; ++index)
            matrices[index] =
                upload_packed_weight(std::to_string(block) + "split" + std::to_string(index),
                                     [&, index] { return projections[index]; });

        auto combined = allocate_tensor(source.height, source.width, 512);
        combined.fp8_grid = true;
        for (unsigned group = 0; group < 8; ++group) {
            auto reduced = matrix_multiply(
                source, tensor_view(matrices[0], group * 64 * 512, 1, 64, 512), 64, 5, true);
            auto hidden = matrix_multiply(
                reduced, tensor_view(matrices[1], group * 256 * 64, 1, 256, 64), 256, 3, true);
            auto contract_matrix = tensor_view(matrices[2], group * 64 * 256, 1, 64, 256);
            phx::Operation operation{};
            operation.family = phx::MatrixMultiply;
            operation.left_input = hidden.ptr();
            operation.right_input = contract_matrix.ptr();
            operation.output =
                tensor_view(combined, group * 64, source.height, source.width, 64).ptr();
            operation.flags = 1 | phx::StoreHalf | (5 << 8) |
                              (contract_matrix.element_bytes == 1 ? phx::RightInputFp8 : 0) |
                              (hidden.fp8_grid ? phx::LeftInputFp8Grid : 0);
            operation.row_count = source.height * source.width;
            operation.column_count = 64;
            operation.reduction_count = 256;
            operation.batch_count = 1;
            operation.left_row_stride = operation.right_row_stride = 256;
            operation.output_row_stride = 512;
            emit(operation, uint64_t(operation.row_count) * 64);
        }
        current_layer_ = 1;
        auto residual_features =
            residual(matrix_multiply(combined, model_weight(block, 1, "weight"), 512), source,
                     residual_gain(block, 1, "skip_weight", 512));
        auto context = attention(residual_features, block, 2, shift_x, shift_y, false);
        current_layer_ = 3;
        auto accumulated =
            residual(matrix_multiply(context, model_weight(block, 3, "weight"), 512),
                     residual_features, residual_gain(block, 3, "skip_weight", 512), 4, false);
        if (accumulator)
            *accumulator = accumulated;
        return publish(accumulated);
    }

    Tensor transformer_block(Tensor source, int block) {
        current_layer_ = 0;
        auto hidden = matrix_multiply(source, model_weight(block, 0, "weight"), 4096, 7, true);
        current_layer_ = 1;
        auto residual_features =
            residual(matrix_multiply(hidden, model_weight(block, 1, "weight"), 1024), source,
                     residual_gain(block, 1, "skip_weight", 1024));
        auto context = attention(residual_features, block, 2, 0, 0, true);
        current_layer_ = 4;
        return residual(matrix_multiply(context, model_weight(block, 4, "weight"), 1024),
                        residual_features, residual_gain(block, 4, "skip_weight", 1024));
    }

    Tensor upsample(Tensor source, Tensor skip, int block, bool decoder_transition) {
        current_layer_ = 0;
        auto projected = matrix_multiply(
            source, model_weight(block, 0, decoder_transition ? "weight" : "resample"),
            skip.channels, 4, false);
        Tensor gains;
        if (decoder_transition) {
            gains = residual_gain(block, 0, "skip_weight", skip.channels);
        } else {
            gains = upload_weight(
                std::to_string(block) + "upsamplegain",
                [&] {
                    auto values = model_.half(block, 0, "upsample_gain");
                    if (values.size() < skip.channels) {
                        auto fallback = model_.half(block, 0, "residual_scale");
                        values.insert(values.begin(),
                                      fallback.end() - (skip.channels - values.size()),
                                      fallback.end());
                    }
                    values.resize(skip.channels);
                    return values;
                },
                false);
        }
        return pointwise(
            projected,
            phx::UpsampleResidual | phx::StoreHalf | ((skip.channels == 32 ? 4 : 5) << 8),
            skip.height, skip.width, skip.channels, &skip, &gains, 0, skip.width, source.width);
    }

  public:
    // Checkpoint field names remain compatible with the existing probe tools.
    // after is the command boundary; offset/bytes select the tensor's storage.
    struct Checkpoint {
        size_t after;
        Allocation allocation;
        size_t offset, bytes;
        unsigned height, width, channels, element;
        std::string name;
    };

    std::vector<Checkpoint> checkpoints;

    void checkpoint(const std::string& name, const Tensor& tensor) {
        checkpoints.push_back({commands.size(), *tensor.memory, tensor.byte_offset,
                               tensor.size() * tensor.element_bytes, tensor.height, tensor.width,
                               tensor.channels, tensor.element_bytes, name});
    }

    std::vector<Command> commands;
    Tensor input, output;
    unsigned width = 0, height = 0;
    bool cached_noise(unsigned seed) const {
        return device_noise_cache_ && noise_uploaded_ && uploaded_noise_seed_ == seed;
    }

    Graph(Backend& backend, Model& model, bool fused_attention = false,
          unsigned window_batch_limit = 128, bool device_features = false,
          bool fused_residual = false, bool device_noise_cache = false)
        : backend_(backend), model_(model), fused_attention_(fused_attention),
          window_batch_limit_(window_batch_limit), device_features_(device_features),
          fused_residual_(fused_residual), device_noise_cache_(device_noise_cache && device_features) {
        demand(window_batch_limit >= 16 && window_batch_limit <= 8192,
               "Window batch limit must be 16..8192");
    }

    ~Graph() {
        commands.clear();
        input = {};
        output = {};
        weight_cache_.clear();
        try {
            backend_.wait();
        } catch (...) {
        }
        for (auto allocation : owned_allocations_)
            try {
                backend_.release(allocation);
            } catch (...) {
            }
    }

    void build(unsigned frame_width, unsigned frame_height,
               const std::array<unsigned, 8>& repeats) {
        demand(commands.empty(), "Graph is immutable after recording");
        width = frame_width;
        height = frame_height;
        auto input_owner = allocate_tensor(frame_height, frame_width, 16);
        input = input_owner;
        input.memory = std::make_shared<Allocation>(*input_owner.memory);
        current_block_ = 0;

        // Stem: lift the 16 input features to 32 channels, retain the first
        // full-resolution skip, and start the spatial encoder.
        auto features = input_owner;
        Tensor accumulator;
        auto stem = shifted_window_block(features, 0, 0, 0, true, &accumulator);
        std::map<int, Tensor> skips;
        skips[0] = stem;
        features = pad_spatial(accumulator, align_extent(frame_height / 2, 4),
                               align_extent(frame_width / 2, 4));
        input_owner = {};
        checkpoint("block0", features);
        checkpoint("skip0", skips[0]);

        const int shifts[4][2] = {{0, 0}, {-1, -1}, {-1, 0}, {0, -1}};
        int block = 1;
        // The last block at each encoder level also pools its accumulator.
        for (auto level_blocks : {4, 4, 6, 8}) {
            for (int index = 0; index < level_blocks; ++index, ++block) {
                current_block_ = block;
                features = shifted_window_block(features, block, shifts[index % 4][0],
                                                shifts[index % 4][1], false,
                                                index == level_blocks - 1 ? &accumulator : nullptr);
                if (index == level_blocks - 1) {
                    skips[block] = features;
                    auto pooled = accumulator;
                    features = matrix_multiply(pooled, model_weight(block, 0, "resample"),
                                               features.channels * 2, 5, true);
                    features = pad_spatial(features, align_extent(features.height, 4),
                                           align_extent(features.width, 4));
                }
                checkpoint("block" + std::to_string(block), features);
            }
        }

        for (block = 23; block <= 30; ++block) {
            current_block_ = block;
            features = split_window_block(features, block, shifts[(block - 23) % 4][0],
                                          shifts[(block - 23) % 4][1], &accumulator);
            checkpoint("block" + std::to_string(block), features);
        }
        skips[30] = features;
        auto pooled = pad_spatial(average_pool(accumulator), align_extent(features.height / 2, 4),
                                  align_extent(features.width / 2, 4));
        current_block_ = 30;
        current_layer_ = 4;
        features = matrix_multiply(pooled, model_weight(30, 4, "weight"), 1024, 5, true);
        checkpoint("pooled30", features);

        // Eight global transformer blocks operate on the coarsest features.
        for (block = 31; block <= 38; ++block) {
            demand(repeats[block - 31] <= 8, "Transformer repeat count exceeds eight");
            current_block_ = block;
            for (unsigned repetition = 0; repetition < repeats[block - 31]; ++repetition)
                features = transformer_block(features, block);
            checkpoint("block" + std::to_string(block), features);
        }

        // Decoder: restore resolution and combine the matching encoder skips.
        current_block_ = 39;
        features = upsample(features, skips[30], 39, true);
        checkpoint("block39", features);
        for (block = 40; block <= 47; ++block) {
            current_block_ = block;
            features = split_window_block(features, block, shifts[(block - 40) % 4][0],
                                          shifts[(block - 40) % 4][1]);
            checkpoint("block" + std::to_string(block), features);
        }
        const int level_sizes[] = {8, 6, 4, 4}, skip_blocks[] = {22, 14, 8, 4};
        block = 48;
        for (int level = 0; level < 4; ++level) {
            for (int index = 0; index < level_sizes[level]; ++index, ++block) {
                current_block_ = block;
                if (index == 0)
                    features = upsample(features, skips[skip_blocks[level]], block, false);
                int shift = (index + (level == 1 ? 2 : 0)) % 4;
                features =
                    shifted_window_block(features, block, shifts[shift][0], shifts[shift][1]);
                checkpoint("block" + std::to_string(block), features);
            }
            skips.erase(skip_blocks[level]);
        }

        // Blend the final decoder features with the stem skip, then project
        // the 32 feature channels to the four-channel output head.
        current_block_ = 70;
        current_layer_ = 0;
        auto blend_gains = upload_weight(
            "post_sincos",
            [&] {
                auto main_gain = model_.half(70, 0, "main_gain");
                auto skip_gain = model_.half(70, 0, "skip_gain");
                main_gain.insert(main_gain.end(), skip_gain.begin(), skip_gain.end());
                return main_gain;
            },
            false);
        features =
            shifted_window_block(features, 70, -1, -1, true, nullptr, &skips[0], &blend_gains);
        output = matrix_multiply(features,
                                 upload_weight("output_head",
                                               [&] {
                                                   auto values = model_.half(70, 0, "out_project");
                                                   values.resize(4 * 32);
                                                   return values;
                                               }),
                                 4);
        checkpoint("head", output);
        if (device_features_) {
            // These upload views live only before the first neural command and
            // share its existing scratch arena rather than adding resident buffers.
            frame_planes_[0] = allocate_tensor(height, width, 3, 4);
            frame_planes_[1] = allocate_tensor(height, width, 3, 4);
            frame_planes_[2] = allocate_tensor(height, width, 3, 2);
            // Validity bytes followed by an aligned optional RGBA32F map.
            frame_planes_[3] = allocate_tensor(height, width, 5, 4);
        }
        commit_scratch();
    }

    void run_from_planes(const float* color, size_t color_count, const std::vector<float>& prior,
                         const std::vector<uint8_t>& validity, const std::vector<uint16_t>& noise,
                         unsigned active_width, unsigned active_height, unsigned style, float tone,
                         float structure, float skin, bool auto_mask, std::vector<float>& head,
                         std::vector<Event>& events, unsigned noise_seed = 0,
                         const float* control_mask = nullptr) {
        demand(device_features_ && color_count == size_t(active_width) * active_height * 3 &&
                   prior.size() == color_count && validity.size() * 3 == color_count &&
                   (noise.size() == size_t(width) * height * 3 || cached_noise(noise_seed)),
               "Device feature plane extent");
        std::vector<uint8_t> mapped_validity;
        if (control_mask) {
            const size_t offset = (validity.size() + 3) & ~size_t(3);
            mapped_validity.resize(offset + validity.size() * 4 * sizeof(float));
            std::memcpy(mapped_validity.data(), validity.data(), validity.size());
            std::memcpy(mapped_validity.data() + offset, control_mask, validity.size() * 4 * sizeof(float));
        }
        const void* sources[] = {color, prior.data(), noise.data(),
                                control_mask ? mapped_validity.data() : validity.data()};
        const uint64_t bytes[] = {color_count * 4, prior.size() * 4, noise.size() * 2,
                                  control_mask ? mapped_validity.size() : validity.size()};
        // Invalidate the old seed before any upload can fail: a partially
        // replaced noise plane must never be mistaken for the previous seed.
        if (uploaded_noise_seed_ != noise_seed)
            noise_uploaded_ = false;
        for (unsigned plane = 0; plane < 4; ++plane) {
            if (plane == 2 && cached_noise(noise_seed))
                continue;
            backend_.upload(tensor_allocation(frame_planes_[plane]), sources[plane], bytes[plane]);
        }
        if (device_noise_cache_) {
            noise_uploaded_ = true;
            uploaded_noise_seed_ = noise_seed;
        }
        phx::Operation pack{};
        pack.family = phx::FrameFeatures;
        pack.left_input = frame_planes_[0].ptr();
        pack.right_input = frame_planes_[1].ptr();
        pack.auxiliary = frame_planes_[2].ptr();
        pack.secondary_output = frame_planes_[3].ptr();
        pack.output = input.ptr();
        pack.row_count = active_height;
        pack.column_count = active_width;
        pack.reduction_count = width;
        pack.batch_count = height;
        pack.left_batch_stride = style;
        pack.right_batch_stride = std::bit_cast<uint32_t>(tone);
        pack.output_batch_stride = std::bit_cast<uint32_t>(structure);
        pack.left_row_stride = std::bit_cast<uint32_t>(skin);
        pack.flags = unsigned(auto_mask) | (control_mask ? 2u : 0u);
        std::vector<Event> ignored;
        backend_.execute({{pack, size_t(width) * height, UINT32_MAX, 0}}, ignored);
        backend_.execute(commands, events);
        head.resize(output.size());
        backend_.download(tensor_allocation(output), head.data(), head.size() * 4);
    }

    void run(std::vector<uint16_t>& features, std::vector<float>& head,
             std::vector<Event>& events) {
        demand(features.size() == input.size(), "Input feature extent");
        backend_.upload(tensor_allocation(input), features.data(), features.size() * 2);
        // Every backend consumes the source before upload returns. The input
        // arena now owns the features, so this frame-sized host copy need not
        // overlap execution, head download, resolve and next-history allocation.
        std::vector<uint16_t>().swap(features);
        backend_.execute(commands, events);
        head.resize(output.size());
        backend_.download(tensor_allocation(output), head.data(), head.size() * 4);
    }

    uint64_t memory_bytes() const {
        uint64_t bytes = 0;
        for (auto allocation : owned_allocations_)
            bytes += allocation.bytes;
        return bytes;
    }

    uint64_t weight_bytes() const {
        uint64_t bytes = 0;
        for (const auto& [key, tensor] : weight_cache_)
            bytes += tensor.memory->bytes;
        return bytes;
    }
};

} // namespace phoenix_cpp
