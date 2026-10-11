// DLSS5 Phoenix resident operator reference. Arithmetic follows the corrected Vulkan
// provider; explicit publication and reduction order are shared by HIP/SYCL.
#pragma once
#include <cstdint>
#include <cstddef>
namespace phx {
using u32 = uint32_t;
using u64 = uint64_t;
// Backend-independent descriptor. Member order and widths are a binary ABI:
// CPU, CUDA, HIP, SYCL and the Vulkan adapter must agree on the same layout.
// Matrix operations use C = A * B (or A * transpose(B)); strides count elements,
// not bytes. Elementwise/row operations reuse the dimension slots as documented
// in docs/OPERATOR_ABI.md. Do not interpret their fields as GEMM dimensions.
enum OperationFamily : u32 { Elementwise = 0, MatrixMultiply = 1, RowReduction = 2 };
// The low byte is interpreted within its family. For example, the value 1
// means an activation gate in Elementwise, but transposes B in MatrixMultiply.
enum ElementwiseKind : u32 {
    QuantizeFp8 = 0,
    ActivationGate = 1,
    RoundHalf = 2,
    FloatToHalf = 3,
    CopyValues = 4,
    ScaledResidual = 5,
    ScaleHalf = 6,
    GatherWindows = 7,
    ScatterWindows = 8,
    AddPositionBias = 9,
    SplitQkvHeads = 10,
    MergeAttentionHeads = 11,
    AveragePool = 12,
    NearestUpsample = 13,
    ChannelGain = 14,
    AddInputs = 15,
    SpatialPad = 16,
    GateFp8ToHalf = 17,
    Fp8ToHalf = 18,
    GateToHalf = 19,
    UpsampleBlendAndPublish = 20,
    UpsampleResidual = 21,
    PoolAndPublish = 22,
    GatherWindowBatch = 25,
    ScatterWindowBatch = 26,
    ScatterWindowBatchAndPool = 27,
    GatherUpsampleBlendBatch = 28
};
enum Publication : u32 {
    KeepFloat = 0,
    PublishFp8 = 1,
    PublishGate = 2,
    PublishGateFp8 = 3,
    PublishHalf = 4,
    PublishHalfThenFp8 = 5,
    PublishFusedGateFp8 = 6,
    PublishGateThenHalfFp8 = 7
};
enum OperationFlag : u32 {
    StoreHalf = 0x1000,
    AddAttentionBias = 0x2000,
    ResidualInWindowOrder = 0x4000,
    LeftInputHalf = 0x8000,
    RightInputHalf = 0x10000,
    InterleavedQkv = 0x20000,
    GlobalAttention = 0x80000,
    RightInputFp8 = 0x2000000
};
struct Operation {
    void* left_input = nullptr;       // A: activations, scores or source image.
    void* right_input = nullptr;      // B: weights, second input or attention bias.
    void* output = nullptr;           // C: primary destination.
    void* auxiliary = nullptr;        // D: scales, inverse sums or extra input.
    void* secondary_output = nullptr; // Optional second destination (pooled/full output).
    u32 family = Elementwise;
    u32 flags = 0;           // Low byte: operator; bits 8..11: publication.
    u32 row_count = 0;       // GEMM M; otherwise rows or output elements.
    u32 column_count = 0;    // GEMM N; otherwise channels or tokens per row.
    u32 reduction_count = 0; // GEMM K; otherwise padding/scaling selector.
    u32 batch_count = 0;
    u32 left_batch_stride = 0;
    u32 right_batch_stride = 0;
    u32 output_batch_stride = 0;
    u32 left_row_stride = 0;
    u32 right_row_stride = 0;
    u32 output_row_stride = 0;
    float scale = 1;
};
static_assert(sizeof(Operation) == 96, "Resident operation ABI size changed");
static_assert(offsetof(Operation, family) == 40, "Resident pointer ABI changed");
static_assert(offsetof(Operation, scale) == 88, "Resident scalar ABI changed");
// Math supplies bit casts, fma, sqrt, round-even and min/max on the selected
// execution device. No subgroup width or matrix extension is assumed.
template <class Math> struct Kernel {
    Operation operation;
    // Round to binary16 using ties-to-even, but return float32 storage. Explicit
    // rounding boundaries reproduce the device arithmetic even on scalar CPU paths.
    PHX_DEVICE float half(float value) const {
        u32 bits = Math::bits(value), sign = bits & 0x80000000u,
            magnitude_bits = bits & 0x7fffffffu;
        if (magnitude_bits >= 0x7f800000u)
            return value;
        if (magnitude_bits >= 0x38800000u) {
            u32 rounded = magnitude_bits + 0xfffu + ((magnitude_bits >> 13u) & 1u);
            return Math::value(sign |
                               (rounded >= 0x47800000u ? 0x7f800000u : rounded & 0xffffe000u));
        }
        if (magnitude_bits < 0x33000000u)
            return Math::value(sign);
        u32 shift = 126u - (magnitude_bits >> 23u),
            mantissa_bits = (magnitude_bits & 0x7fffffu) | 0x800000u;
        u32 rounded =
            (mantissa_bits + ((1u << (shift - 1u)) - 1u) + ((mantissa_bits >> shift) & 1u)) >>
            shift;
        return Math::value(sign | Math::bits(float(rounded) * 5.9604644775390625e-8f));
    }
    // Encode a rounded value as IEEE binary16, including subnormals and NaNs.
    PHX_DEVICE u32 half_bits(float value) const {
        u32 bits = Math::bits(half(value)), sign = (bits >> 16u) & 0x8000u,
            magnitude_bits = bits & 0x7fffffffu;
        if (magnitude_bits >= 0x7f800000u)
            return sign | 0x7c00u | (magnitude_bits > 0x7f800000u ? 0x200u : 0u);
        if (!magnitude_bits)
            return sign;
        u32 exponent = magnitude_bits >> 23u;
        if (exponent < 113u)
            return sign | (((magnitude_bits & 0x7fffffu) | 0x800000u) >> (126u - exponent));
        return sign | ((exponent - 112u) << 10u) | ((magnitude_bits >> 13u) & 0x3ffu);
    }
    // Decode IEEE binary16 bits without requiring a host/device half type.
    PHX_DEVICE float half_value(u32 bits) const {
        u32 sign = (bits & 0x8000u) << 16u, exponent = (bits >> 10u) & 31u, mantissa = bits & 1023u;
        if (!exponent)
            return Math::value(sign | Math::bits(float(mantissa) * 5.9604644775390625e-8f));
        if (exponent == 31u)
            return Math::value(sign | 0x7f800000u | (mantissa << 13u));
        return Math::value(sign | ((exponent + 112u) << 23u) | (mantissa << 13u));
    }
    PHX_DEVICE float load_half(void* storage, u64 element_index) const {
        return half_value(static_cast<uint16_t*>(storage)[element_index]);
    }
    // Weights stay as one-byte E4M3 when flagged; decode on load rather than
    // keeping a second expanded model in memory.
    PHX_DEVICE float load_matrix_right(u64 element_index) const {
        if (!(operation.flags & RightInputFp8))
            return load_half(operation.right_input, element_index);
        u32 code = static_cast<uint8_t*>(operation.right_input)[element_index],
            exponent = (code >> 3u) & 15u, mantissa = code & 7u;
        u32 sign = (code & 128u) << 24u;
        return Math::value(sign | (exponent ? ((exponent + 120u) << 23u) | (mantissa << 20u)
                                            : Math::bits(float(mantissa) * 0.001953125f)));
    }
    PHX_DEVICE float load_float(void* storage, u64 element_index) const {
        return static_cast<float*>(storage)[element_index];
    }
    PHX_DEVICE float load_left(u64 element_index) const {
        return (operation.flags & 0x8000u) ? load_half(operation.left_input, element_index)
                                           : load_float(operation.left_input, element_index);
    }
    PHX_DEVICE float load_right(u64 element_index) const {
        return (operation.flags & 0x10000u) ? load_half(operation.right_input, element_index)
                                            : load_float(operation.right_input, element_index);
    }
    PHX_DEVICE float clamp(float value, float minimum, float maximum) const {
        return Math::min(Math::max(value, minimum), maximum);
    }
    // Quantize to the finite E4M3 grid, saturating magnitude at 448. The returned
    // float carries the quantized value; it is not the encoded one-byte weight.
    PHX_DEVICE float quantize_e4m3(float value) const {
        float magnitude = Math::min(Math::abs(value), 448.f);
        u32 exponent = (Math::bits(magnitude) >> 23) & 255u;
        exponent = (exponent > 121u ? exponent : 121u) - 3u;
        float step = Math::value(exponent << 23), reciprocal = Math::value((254u - exponent) << 23);
        float rounded = Math::even(magnitude * reciprocal) * step;
        return value < 0 ? -rounded : rounded;
    }
    // Evaluate the cubic gate with half rounding after each polynomial stage.
    // The clamped value controls the gate; the original value supplies its amplitude.
    PHX_DEVICE float activation_gate(float value) const {
        float wide = half(value), clamped_value = clamp(wide, -4, 4);
        float gate_factor = half(Math::abs(clamped_value) * -0.055908203125f + 0.447265625f);
        gate_factor = half(gate_factor * clamped_value + 0.89453125f);
        return half(wide * gate_factor);
    }
    // Publication controls the exact gate/half/FP8 rounding after an operator.
    // These sequences are part of numerical parity, not interchangeable casts.
    PHX_DEVICE float publish(u32 epilogue, float value) const {
        if (epilogue == 6u) {
            float clamped_value = clamp(value, -4, 4);
            return quantize_e4m3(value * Math::fma(clamped_value,
                                                   Math::fma(Math::abs(clamped_value),
                                                             -0.055908203125f, 0.447265625f),
                                                   0.89453125f));
        }
        if (epilogue == 7u) {
            float clamped_value = clamp(value, -4, 4),
                  cubic_term = -0.055908203125f * Math::abs(clamped_value) * clamped_value,
                  linear_term = 0.447265625f * clamped_value;
            return quantize_e4m3(half(value * ((cubic_term + linear_term) + 0.89453125f)));
        }
        if (epilogue == 2u || epilogue == 3u)
            value = activation_gate(value);
        if (epilogue == 1u || epilogue == 3u)
            value = quantize_e4m3(value);
        if (epilogue == 4u)
            value = half(value);
        if (epilogue == 5u)
            value = quantize_e4m3(half(value));
        return value;
    }
    PHX_DEVICE void store(u64 element_index, float value) const {
        value = publish((operation.flags >> 8u) & 15u, value);
        if (operation.flags & StoreHalf)
            static_cast<uint16_t*>(operation.output)[element_index] = uint16_t(half_bits(value));
        else
            static_cast<float*>(operation.output)[element_index] = value;
    }
    PHX_DEVICE float add_half(float left, float right) const {
        return half(left + right);
    }
    PHX_DEVICE float multiply_half(float left, float right) const {
        return half(left * right);
    }
    // Correct the float reciprocal once using its FMA residual. Infinite inputs
    // map to zero, matching the normalization path's original edge behavior.
    PHX_DEVICE float reciprocal(float value) const {
        if (Math::inf(value))
            return 0;
        float estimate = 1.f / value, error = Math::fma(-estimate, value, 1.f);
        return Math::fma(error, estimate, estimate);
    }
    // Map an image element into 8x8 window storage. Tokens are ordered as four
    // 4x4 quadrants, rather than ordinary 8x8 row-major pixels.
    PHX_DEVICE u64 window_index(u64 index) const {
        u32 channel = index % operation.column_count, pixel_index = index / operation.column_count,
            padded_y =
                pixel_index / operation.right_batch_stride + (operation.reduction_count >> 16),
            padded_x =
                pixel_index % operation.right_batch_stride + (operation.reduction_count & 65535u);
        u32 token = ((padded_y % 8u) / 4u) * 32u + ((padded_x % 8u) / 4u) * 16u +
                    (padded_y % 4u) * 4u + padded_x % 4u;
        return (((padded_y / 8u) * operation.output_batch_stride + padded_x / 8u) * 64u + token) *
                   u64(operation.column_count) +
               channel;
    }
    // Dispatch elementwise and spatial operators using the low flag byte.
    // Spatial operators reuse stride fields for image geometry (OPERATOR_ABI.md).
    PHX_DEVICE void unary(u64 index) const {
        u32 kind = operation.flags & 255u;
        // A work item identifies one channel of a quadrant-ordered window token.
        // The first window is batch_count; image bounds exclude padded tokens.
        if (kind >= GatherWindowBatch && kind <= GatherUpsampleBlendBatch) {
            u32 channel = index % operation.column_count,
                pixel_or_token_index = index / operation.column_count,
                token = pixel_or_token_index % 64u,
                window_index = operation.batch_count + pixel_or_token_index / 64u;
            int pixel_y = int((window_index / operation.output_batch_stride) * 8u +
                              (token / 32u) * 4u + (token % 16u) / 4u) -
                          int(operation.reduction_count >> 16);
            int pixel_x = int((window_index % operation.output_batch_stride) * 8u +
                              ((token / 16u) % 2u) * 4u + token % 4u) -
                          int(operation.reduction_count & 65535u);
            bool valid = pixel_x >= 0 && pixel_y >= 0 &&
                         u32(pixel_y) < operation.left_batch_stride &&
                         u32(pixel_x) < operation.right_batch_stride;
            u64 source_offset = valid
                                    ? (u64(pixel_y) * operation.right_batch_stride + u32(pixel_x)) *
                                              operation.column_count +
                                          channel
                                    : 0;
            if (kind == GatherWindowBatch) {
                store(index, valid ? load_left(source_offset) : 0.f);
                return;
            }
            if (kind == ScatterWindowBatch) {
                if (valid)
                    store(source_offset, load_left(index));
                return;
            }
            // Read the coarse pixel at floor(x/2,y/2), scale it per channel, then
            // fuse the fine-resolution skip contribution before half publication.
            if (kind == GatherUpsampleBlendBatch) {
                float value = 0;
                if (valid) {
                    u64 above = (u64(pixel_y / 2) * u32(operation.scale) + u32(pixel_x / 2)) *
                                    operation.column_count +
                                channel;
                    float scaled =
                        half(load_left(above) * load_float(operation.auxiliary, channel));
                    value = half(Math::fma(
                        load_half(operation.right_input, source_offset),
                        load_float(operation.auxiliary, operation.column_count + channel), scaled));
                }
                store(index, value);
                return;
            }
            // Publish the full-resolution skip and pool each aligned 2x2 tile once.
            // Offsets 0,1,4,5 select adjacent tokens inside a 4x4 quadrant.
            if (valid) {
                static_cast<uint16_t*>(operation.secondary_output)[source_offset] =
                    uint16_t(half_bits(quantize_e4m3(half(load_left(index)))));
                if (!(u32(pixel_y) & 1u) && !(u32(pixel_x) & 1u) &&
                    u32(pixel_y) + 1u < operation.left_batch_stride &&
                    u32(pixel_x) + 1u < operation.right_batch_stride) {
                    float top = half(load_left(index) + load_left(index + operation.column_count));
                    float bottom = half(load_left(index + 4u * operation.column_count) +
                                        load_left(index + 5u * operation.column_count));
                    store((u64(pixel_y / 2) * (operation.right_batch_stride / 2u) +
                           u32(pixel_x / 2)) *
                                  operation.column_count +
                              channel,
                          half(top + bottom) * .25f);
                }
            }
            return;
        }
        // Convert scaled float input directly to binary16 storage; this branch
        // deliberately bypasses the optional store() epilogue.
        if (kind == FloatToHalf) {
            static_cast<uint16_t*>(operation.output)[index] =
                uint16_t(half_bits(load_float(operation.left_input, index) * operation.scale));
            return;
        }
        // Decode binary16 input, scale in float, then apply the selected
        // publication and destination storage format.
        if (kind == ScaleHalf) {
            store(index, load_half(operation.left_input, index) * operation.scale);
            return;
        }
        // The three conversion variants independently select the cubic gate
        // and E4M3 quantization, then encode a binary16 destination.
        if (kind >= GateFp8ToHalf && kind <= GateToHalf) {
            float value = load_float(operation.left_input, index);
            if (kind != Fp8ToHalf)
                value = activation_gate(value);
            if (kind != GateToHalf)
                value = quantize_e4m3(value);
            static_cast<uint16_t*>(operation.output)[index] = uint16_t(half_bits(value));
            return;
        }
        // Add a per-channel-scaled skip to the main input. The main input may
        // be in quadrant window order while the skip remains image ordered.
        if (kind == ScaledResidual) {
            u64 source_offset = (operation.flags & 0x4000u) ? window_index(index) : index;
            float product =
                load_right(index) * load_float(operation.auxiliary, index % operation.column_count);
            store(index, load_left(source_offset) + product);
            return;
        }
        // Gather maps image pixels into four 4x4 quadrants per 8x8 window,
        // zero-filling padding; scatter applies the inverse index map.
        if (kind == GatherWindows || kind == ScatterWindows) {
            u32 channel = index % operation.column_count,
                pixel_or_token_index = index / operation.column_count;
            if (kind == GatherWindows) {
                u32 token = pixel_or_token_index % 64u, window_index = pixel_or_token_index / 64u;
                int pixel_y = int((window_index / operation.output_batch_stride) * 8u +
                                  (token / 32u) * 4u + (token % 16u) / 4u) -
                              int(operation.reduction_count >> 16);
                int pixel_x = int((window_index % operation.output_batch_stride) * 8u +
                                  ((token / 16u) % 2u) * 4u + token % 4u) -
                              int(operation.reduction_count & 65535u);
                store(index,
                      (pixel_x < 0 || pixel_y < 0 || u32(pixel_y) >= operation.left_batch_stride ||
                       u32(pixel_x) >= operation.right_batch_stride)
                          ? 0
                          : load_left((u64(pixel_y) * operation.right_batch_stride + u32(pixel_x)) *
                                          operation.column_count +
                                      channel));
            } else
                store(index, load_left(window_index(index)));
            return;
        }
        // Average a spatial 2x2 tile. Round each horizontal pair and their
        // sum to half before multiplying by 1/4; optionally retain an FP8 skip.
        if (kind == AveragePool || kind == PoolAndPublish) {
            u32 channel = index % operation.column_count,
                pixel_or_token_index = index / operation.column_count,
                pixel_x = pixel_or_token_index % (operation.right_batch_stride / 2u),
                pixel_y = pixel_or_token_index / (operation.right_batch_stride / 2u);
            u64 base = (u64(2u * pixel_y) * operation.right_batch_stride + 2u * pixel_x) *
                           operation.column_count +
                       channel;
            float top = half(load_left(base) + load_left(base + operation.column_count)),
                  bottom = half(
                      load_left(base + u64(operation.right_batch_stride) * operation.column_count) +
                      load_left(base +
                                u64(operation.right_batch_stride + 1u) * operation.column_count));
            store(index, half(top + bottom) * .25f);
            if (kind == PoolAndPublish) {
                u64 taps[4] = {base, base + operation.column_count,
                               base + u64(operation.right_batch_stride) * operation.column_count,
                               base +
                                   u64(operation.right_batch_stride + 1u) * operation.column_count};
                for (u32 tap_index = 0; tap_index < 4; tap_index++)
                    static_cast<uint16_t*>(operation.secondary_output)[taps[tap_index]] =
                        uint16_t(half_bits(quantize_e4m3(half(load_left(taps[tap_index])))));
            }
            return;
        }
        // Replicate each coarse pixel into a 2x2 destination region. Fused
        // variants mix in the fine skip and may publish a second FP8 output.
        if (kind == NearestUpsample || kind == UpsampleBlendAndPublish ||
            kind == UpsampleResidual) {
            u32 channel = index % operation.column_count,
                pixel_or_token_index = index / operation.column_count,
                pixel_x = pixel_or_token_index % operation.left_batch_stride,
                pixel_y = pixel_or_token_index / operation.left_batch_stride;
            float value =
                load_left((u64(pixel_y / 2u) * operation.right_batch_stride + pixel_x / 2u) *
                              operation.column_count +
                          channel);
            if (kind == UpsampleBlendAndPublish) {
                float scaled = half(value * load_float(operation.auxiliary, channel));
                value = half(Math::fma(
                    load_right(index),
                    load_float(operation.auxiliary, operation.column_count + channel), scaled));
                static_cast<float*>(operation.output)[index] = value;
                static_cast<uint16_t*>(operation.secondary_output)[index] =
                    uint16_t(half_bits(quantize_e4m3(value)));
                return;
            }
            // The 512-channel stage rounds the scaled skip before addition;
            // other stages fuse the multiply-add. Preserve this model-specific boundary.
            if (kind == UpsampleResidual) {
                float projected = half(value);
                value = operation.column_count == 512u
                            ? half(projected + half(load_right(index) *
                                                    load_float(operation.auxiliary, channel)))
                            : half(Math::fma(load_right(index),
                                             load_float(operation.auxiliary, channel), projected));
            }
            store(index, value);
            if (kind == UpsampleResidual && operation.secondary_output)
                static_cast<uint16_t*>(operation.secondary_output)[index] =
                    uint16_t(half_bits(quantize_e4m3(value)));
            return;
        }
        // Copy the active source rectangle into a wider destination and fill
        // its right/bottom padding with zero; channels remain contiguous.
        if (kind == SpatialPad) {
            u32 channel = index % operation.column_count,
                pixel_or_token_index = index / operation.column_count,
                pixel_x = pixel_or_token_index % operation.output_batch_stride,
                pixel_y = pixel_or_token_index / operation.output_batch_stride;
            store(index,
                  (pixel_y < operation.left_batch_stride && pixel_x < operation.right_batch_stride)
                      ? load_left((u64(pixel_y) * operation.right_batch_stride + pixel_x) *
                                      operation.column_count +
                                  channel)
                      : 0);
            return;
        }
        // Broadcast one learned gain per channel across every spatial position.
        if (kind == ChannelGain) {
            store(index, load_left(index) *
                             load_float(operation.auxiliary, index % operation.column_count));
            return;
        }
        // Add equally shaped tensors before the selected publication epilogue.
        if (kind == AddInputs) {
            store(index, load_left(index) + load_right(index));
            return;
        }
        // Select Q, K or V from token-major interleaved projections, then
        // write contiguous 32-channel rows grouped by window and attention head.
        if (kind == SplitQkvHeads) {
            u32 head_channel = index % 32u, pixel_or_token_index = index / 32u,
                token = pixel_or_token_index % operation.left_batch_stride;
            pixel_or_token_index /= operation.left_batch_stride;
            u32 head = pixel_or_token_index % operation.batch_count,
                window_index = pixel_or_token_index / operation.batch_count;
            store(index, load_left((u64(window_index) * operation.left_batch_stride + token) * 3u *
                                       operation.column_count +
                                   operation.right_batch_stride * operation.column_count +
                                   head * 32u + head_channel));
            return;
        }
        // Undo the head split into token-major channel rows. Global attention
        // can first apply the inverse score-row normalization from auxiliary.
        if (kind == MergeAttentionHeads) {
            u32 channel = index % operation.column_count,
                pixel_or_token_index = index / operation.column_count,
                token = pixel_or_token_index % operation.left_batch_stride,
                window_index = pixel_or_token_index / operation.left_batch_stride;
            u64 source_offset = (u64(window_index) * operation.batch_count + channel / 32u) *
                                    operation.left_batch_stride +
                                token;
            float value = load_left(source_offset * 32u + channel % 32u);
            if (operation.flags & 0x20000u)
                value = half(half(value) * load_float(operation.auxiliary, source_offset));
            store(index, value);
            return;
        }
        // Broadcast the learned tokens-by-tokens bias for each head across
        // windows, preserving the query/key coordinates within a score plane.
        if (kind == AddPositionBias) {
            u64 plane = u64(operation.column_count) * operation.column_count;
            store(index,
                  load_left(index) + load_float(operation.auxiliary,
                                                ((index / plane) % operation.batch_count) * plane +
                                                    index % plane));
            return;
        }
        // Remaining modes scale an independent element, then quantize, gate,
        // round to half, or copy it before the common publication epilogue.
        float value = load_left(index) * operation.scale;
        store(index, kind == QuantizeFp8      ? quantize_e4m3(value)
                     : kind == ActivationGate ? activation_gate(value)
                     : kind == RoundHalf      ? half(value)
                                              : value);
    }
    // Reference batched matrix multiplication. Accumulate one output in ascending
    // K order; publication happens only after the final multiply-add.
    PHX_DEVICE void gemm(u64 index) const {
        u32 column = index % operation.column_count,
            row = (index / operation.column_count) % operation.row_count,
            batch_index = index / (u64(operation.column_count) * operation.row_count);
        if ((operation.flags & 0x10000u) && column >= 4u)
            return;
        u32 left_stride =
                operation.left_row_stride ? operation.left_row_stride : operation.reduction_count,
            right_stride =
                operation.right_row_stride
                    ? operation.right_row_stride
                    : ((operation.flags & 1u) ? operation.reduction_count : operation.column_count),
            output_stride =
                operation.output_row_stride ? operation.output_row_stride : operation.column_count;
        float sum = 0;
        for (u32 reduction_index = 0; reduction_index < operation.reduction_count;
             reduction_index++)
            sum = Math::fma(
                load_half(operation.left_input, u64(batch_index) * operation.left_batch_stride +
                                                    u64(row) * left_stride + reduction_index),
                load_matrix_right(u64(batch_index) * operation.right_batch_stride +
                                  ((operation.flags & 1u)
                                       ? u64(column) * right_stride + reduction_index
                                       : u64(reduction_index) * right_stride + column)),
                sum);
        store(u64(batch_index) * operation.output_batch_stride + u64(row) * output_stride + column,
              sum);
    }
    // Independent outputs share operand loads, while each accumulator still
    // visits k in ascending order with the original single FP32 FMA.
    PHX_DEVICE void gemm_columns(u64 index) const {
        u32 width = (operation.flags & 0x10000u) ? 4u : operation.column_count,
            tiles = (width + 7u) / 8u;
        u32 first_column = (index % tiles) * 8u, row = (index / tiles) % operation.row_count;
        u64 batch_index = index / (u64(tiles) * operation.row_count);
        float sums[8] = {};
        for (u32 reduction_index = 0; reduction_index < operation.reduction_count;
             reduction_index++) {
            float left_value = load_half(
                operation.left_input, batch_index * operation.left_batch_stride +
                                          u64(row) * operation.left_row_stride + reduction_index);
            for (u32 column_in_tile = 0; column_in_tile < 8u; column_in_tile++)
                if (first_column + column_in_tile < width)
                    sums[column_in_tile] = Math::fma(
                        left_value,
                        load_matrix_right(
                            batch_index * operation.right_batch_stride +
                            ((operation.flags & 1u)
                                 ? u64(first_column + column_in_tile) * operation.right_row_stride +
                                       reduction_index
                                 : u64(reduction_index) * operation.right_row_stride +
                                       first_column + column_in_tile)),
                        sums[column_in_tile]);
        }
        for (u32 column_in_tile = 0; column_in_tile < 8u; column_in_tile++)
            if (first_column + column_in_tile < width)
                store(batch_index * operation.output_batch_stride +
                          u64(row) * operation.output_row_stride + first_column + column_in_tile,
                      sums[column_in_tile]);
    }
    // Select one 32-channel query/key head from interleaved projection output.
    PHX_DEVICE float load_query_or_key(u32 row, u32 channel) const {
        if (operation.flags & 0x20000u) {
            u32 head = (row / operation.column_count) % operation.batch_count,
                token = row % operation.column_count,
                window_index = row / (operation.column_count * operation.batch_count),
                query_or_key = (operation.flags >> 18) & 1u;
            return load_float(
                operation.left_input,
                ((u64(window_index) * operation.column_count + token) * 3u * operation.batch_count +
                 query_or_key * operation.batch_count + head) *
                        32u +
                    channel);
        }
        return load_left(u64(row) * 32u + channel);
    }
    // Cosine normalization follows the head-specific reduction tree. FMA order
    // and half-rounding differ for fused windows versus split/global attention.
    PHX_DEVICE void norm(u32 row) const {
        float head_values[32];
        for (u32 channel = 0; channel < 32; channel++)
            head_values[channel] = load_query_or_key(row, channel);
        bool split_heads = operation.batch_count == 16u,
             global_attention = (operation.flags & 0x80000u) != 0;
        float inverse;
        if (!split_heads && !global_attention) {
            float partial[4][2];
            for (u32 column_pair = 0; column_pair < 4; column_pair++)
                for (u32 parity = 0; parity < 2; parity++) {
                    float partial_sum = 0;
                    for (u32 row_half = 0; row_half < 2; row_half++)
                        for (u32 fragment = 0; fragment < 2; fragment++) {
                            u32 channel =
                                fragment * 16u + row_half * 8u + column_pair * 2u + parity;
                            partial_sum =
                                Math::fma(head_values[channel], head_values[channel], partial_sum);
                        }
                    partial[column_pair][parity] = partial_sum;
                }
            float even_sum =
                half((partial[0][0] + partial[1][0]) + (partial[2][0] + partial[3][0]));
            float odd_sum = half((partial[0][1] + partial[1][1]) + (partial[2][1] + partial[3][1]));
            float total = Math::max(half(even_sum + odd_sum), 0.00006198883056640625f),
                  root = Math::sqrt(total);
            float error = Math::fma(-root, root, total);
            root += error / (2.f * root);
            inverse = Math::inf(total) ? 0 : reciprocal(root);
            if (operation.reduction_count)
                inverse *= load_float(operation.auxiliary,
                                      (row / operation.column_count) % operation.batch_count);
            for (u32 channel = 0; channel < 32; channel++)
                store(u64(row) * 32u + channel, quantize_e4m3(head_values[channel] * inverse));
            return;
        }
        float reduction[16];
        for (u32 channel = 0; channel < 32; channel++)
            head_values[channel] = half(head_values[channel]);
        for (u32 channel = 0; channel < 16; channel++)
            reduction[channel] =
                add_half(multiply_half(head_values[channel], head_values[channel]),
                         multiply_half(head_values[channel + 16], head_values[channel + 16]));
        for (u32 width = 8; width; width /= 2)
            for (u32 channel = 0; channel < width; channel++)
                reduction[channel] = add_half(reduction[channel], reduction[channel + width]);
        inverse = half(1.f / Math::sqrt(Math::max(reduction[0], 0.00006198883056640625f)));
        float scale = operation.reduction_count
                          ? half(load_float(operation.auxiliary,
                                            (row / operation.column_count) % operation.batch_count))
                          : 1.f;
        for (u32 channel = 0; channel < 32; channel++) {
            float value = multiply_half(global_attention && operation.reduction_count
                                            ? multiply_half(head_values[channel], 5.65625f)
                                            : head_values[channel],
                                        inverse);
            store(u64(row) * 32u + channel,
                  quantize_e4m3(operation.reduction_count ? multiply_half(value, scale) : value));
        }
    }
    // Global-attention softmax approximation: a clamped affine ramp is remapped
    // through half bits. Replacing it with exp() would implement a different model.
    PHX_DEVICE float global_attention_weight(float score) const {
        float ramp = half(score) * .08953857421875f;
        ramp += 1.708984375f;
        ramp = clamp(half(ramp), 1.439453125f, 1.9775390625f);
        return half_value(((half_bits(ramp) << 4) + 0x4000u) & 65535u);
    }
    // Publish unnormalized global weights and store the inverse row sum separately;
    // the merge stage applies it. Padded lanes must not contribute to the denominator.
    PHX_DEVICE void global_attention_softmax(u32 row) const {
        u32 stride =
                operation.left_batch_stride ? operation.left_batch_stride : operation.column_count,
            blocks = (operation.column_count + 63u) / 64u;
        u64 base = u64(row) * stride;
        float zero = global_attention_weight(0), total = 0;
        for (u32 token_block = 0; token_block < blocks; token_block++) {
            float column[8];
            for (u32 column_group = 0; column_group < 8; column_group++) {
                float pair[4];
                for (u32 pair_index = 0; pair_index < 4; pair_index++) {
                    u32 first_token = token_block * 64u + column_group + 16u * pair_index,
                        second_token = first_token + 8u;
                    pair[pair_index] =
                        add_half(first_token < operation.column_count
                                     ? global_attention_weight(
                                           load_float(operation.left_input, base + first_token))
                                     : zero,
                                 second_token < operation.column_count
                                     ? global_attention_weight(
                                           load_float(operation.left_input, base + second_token))
                                     : zero);
                }
                column[column_group] =
                    add_half(add_half(add_half(pair[0], pair[1]), pair[2]), pair[3]);
            }
            float even = add_half(add_half(add_half(column[0], column[2]), column[4]), column[6]);
            float odd = add_half(add_half(add_half(column[1], column[3]), column[5]), column[7]);
            total = add_half(total, add_half(even, odd));
        }
        u32 padding = blocks * 64u - operation.column_count;
        if (padding)
            total = add_half(total, -multiply_half(zero, float(padding)));
        static_cast<float*>(operation.auxiliary)[row] = half(1.f / total);
        for (u32 token_index = 0; token_index < stride; token_index++)
            store(base + token_index, token_index < operation.column_count
                                          ? quantize_e4m3(global_attention_weight(load_float(
                                                operation.left_input, base + token_index)))
                                          : 0);
    }
    // Each 8x8 window has 64 keys. Add relative-position bias, approximate the
    // weights, then preserve the model-specific reduction tree before normalization.
    PHX_DEVICE void softmax(u32 row) const {
        u64 base = u64(row) * 64u,
            bias = u64(row / 64u % operation.batch_count) * 4096u + u64(row % 64u) * 64u;
        float weights[64];
        bool split_heads = operation.batch_count == 16u;
        for (u32 token_index = 0; token_index < 64; token_index++) {
            float position_bias = (operation.flags & 0x2000u)
                                      ? load_float(operation.right_input, bias + token_index)
                                      : 0.f,
                  score = load_float(operation.left_input, base + token_index);
            float ramp =
                split_heads ? half(Math::fma(half(score + position_bias), .044921875f, 1.30078125f))
                            : half(Math::fma(score, .044921875f,
                                             Math::fma(position_bias, .044921875f, 1.30078125f)));
            ramp = clamp(ramp, 1.03125f, 1.5693359375f);
            weights[token_index] = half_value((half_bits(ramp) & 1023u) << 5u);
        }
        float total;
        if (split_heads) {
            float halves[2];
            for (u32 half_index = 0; half_index < 2; half_index++) {
                float partial_sums[4];
                for (u32 pair = 0; pair < 4; pair++) {
                    float even_value = weights[half_index * 8u + pair * 2u],
                          odd_value = weights[half_index * 8u + pair * 2u + 1u];
                    for (u32 block = 1; block < 4; block++) {
                        even_value += weights[block * 16u + half_index * 8u + pair * 2u];
                        odd_value += weights[block * 16u + half_index * 8u + pair * 2u + 1u];
                    }
                    partial_sums[pair] = even_value + odd_value;
                }
                halves[half_index] =
                    (partial_sums[0] + partial_sums[1]) + (partial_sums[2] + partial_sums[3]);
            }
            total = halves[0] + halves[1];
        } else {
            float members[2];
            for (u32 member = 0; member < 2; member++) {
                float halves[2];
                for (u32 parity = 0; parity < 2; parity++) {
                    float pairs = 0;
                    for (u32 pair = 0; pair < 4; pair++) {
                        float blocksum = weights[pair * 4u + member * 2u + parity];
                        for (u32 block = 1; block < 4; block++)
                            blocksum = add_half(
                                blocksum, weights[block * 16u + pair * 4u + member * 2u + parity]);
                        pairs = pair ? add_half(pairs, blocksum) : blocksum;
                    }
                    halves[parity] = pairs;
                }
                members[member] = add_half(halves[0], halves[1]);
            }
            total = add_half(members[0], members[1]);
        }
        float inverse_sum = reciprocal(Math::max(total, 0.00006103515625f));
        for (u32 token_index = 0; token_index < 64; token_index++)
            store(base + token_index, quantize_e4m3(half(weights[token_index] * inverse_sum)));
    }
    PHX_DEVICE void operator()(u64 iteration_index) const {
        if (operation.family == phx::Elementwise)
            unary(iteration_index);
        else if (operation.family == phx::MatrixMultiply)
            gemm(iteration_index);
        else if ((operation.flags & 255u) == 0u)
            norm(u32(iteration_index));
        else if ((operation.flags & 255u) == 6u)
            global_attention_softmax(u32(iteration_index));
        else
            softmax(u32(iteration_index));
    }
};
}
