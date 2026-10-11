// Phoenix Swin arithmetic, transcribed from the existing portable reference.
// Inputs are FP32 Q/K accumulators. Unlike split/ViT, Swin normalizes before half.
float phoenix_reciprocal(float denominator) {
    if (isinf(denominator))
        return 0.0;
    precise float inverse = 1.0 / denominator;
    precise float error = fma(-inverse, denominator, 1.0);
    return fma(error, inverse, inverse);
}

float phoenix_swin_norm(float head_values[32]) {
    precise float partial[4][2];
    for (uint c = 0u; c < 4u; c++)
        for (uint pair_sums = 0u; pair_sums < 2u; pair_sums++) {
            float total = 0.0;
            for (uint g = 0u; g < 2u; g++)
                for (uint f = 0u; f < 2u; f++) {
                    uint iteration_index = f * 16u + g * 8u + c * 2u + pair_sums;
                    total = fma(head_values[iteration_index], head_values[iteration_index], total);
                }
            partial[c][pair_sums] = total;
        }
    precise float even_accumulator =
        half_round((partial[0][0] + partial[1][0]) + (partial[2][0] + partial[3][0]));
    precise float odd_accumulator =
        half_round((partial[0][1] + partial[1][1]) + (partial[2][1] + partial[3][1]));
    float norm = max(half_round(even_accumulator + odd_accumulator), 0.00006198883056640625);
    if (isinf(norm))
        return 0.0;
    precise float root = sqrt(norm);
    precise float error = fma(-root, root, norm);
    root += error / (2.0 * root);
    return phoenix_reciprocal(root);
}

void phoenix_swin_cosine(uint row, uint local, bool scaled, FloatBuf scales) {
    uint base = local * 32u;
    float head_values[32];
    for (uint iteration_index = 0u; iteration_index < 32u; iteration_index++)
        head_values[iteration_index] = stage[base + iteration_index];
    precise float scale = phoenix_swin_norm(head_values);
    if (scaled)
        scale *= scales.v[(row / pc.n) % pc.batch];
    for (uint iteration_index = 0u; iteration_index < 32u; iteration_index++)
        stage[base + iteration_index] = e4m3(head_values[iteration_index] * scale);
}

void phoenix_swin_softmax(uint row, uint local) {
    uint base = local * 64u, bias = bias_offset(row);
    float weights[64];
    for (uint iteration_index = 0u; iteration_index < 64u; iteration_index++) {
        float bias_value =
            (operation_flags() & 0x2000u) != 0u ? pc.b.v[bias + iteration_index] : 0.0;
        float affine_bias = fma(bias_value, 0.044921875, 1.30078125);
        float ramp = clamp(half_round(fma(stage[base + iteration_index], 0.044921875, affine_bias)),
                           1.03125, 1.5693359375);
        uint bits = (phoenix_half_bits(ramp) & 0x3ffu) << 5u;
        weights[iteration_index] = phoenix_unpack_half(bits).x;
    }
    float members[2];
    for (uint member = 0u; member < 2u; member++) {
        float halves[2];
        for (uint parity = 0u; parity < 2u; parity++) {
            float pair_sum = 0.0;
            for (uint pair = 0u; pair < 4u; pair++) {
                float block_sum = weights[pair * 4u + member * 2u + parity];
                for (uint block = 1u; block < 4u; block++)
                    block_sum = half_round(block_sum +
                                           weights[block * 16u + pair * 4u + member * 2u + parity]);
                pair_sum = pair == 0u ? block_sum : half_round(pair_sum + block_sum);
            }
            halves[parity] = pair_sum;
        }
        members[member] = half_round(halves[0] + halves[1]);
    }
    float reciprocal =
        phoenix_reciprocal(max(half_round(members[0] + members[1]), 0.00006103515625));
    for (uint iteration_index = 0u; iteration_index < 64u; iteration_index++)
        stage[base + iteration_index] = e4m3(half_round(weights[iteration_index] * reciprocal));
}

// Split and global blocks square and reduce in half, unlike Swin.
void phoenix_half_cosine(uint row, uint local, bool scaled, FloatBuf scales, bool vit) {
    uint base = local * 32u;
    float head_values[32], squared_sums[16];
    for (uint iteration_index = 0u; iteration_index < 32u; iteration_index++)
        head_values[iteration_index] = half_round(stage[base + iteration_index]);
    for (uint iteration_index = 0u; iteration_index < 16u; iteration_index++)
        squared_sums[iteration_index] =
            hadd(hmul(head_values[iteration_index], head_values[iteration_index]),
                 hmul(head_values[iteration_index + 16u], head_values[iteration_index + 16u]));
    for (uint width = 8u; width > 0u; width /= 2u)
        for (uint iteration_index = 0u; iteration_index < width; iteration_index++)
            squared_sums[iteration_index] =
                hadd(squared_sums[iteration_index], squared_sums[iteration_index + width]);
    float inverse = half_round(inversesqrt(max(squared_sums[0], 0.00006198883056640625)));
    float scale = scaled ? half_round(scales.v[(row / pc.n) % pc.batch]) : 1.0;
    for (uint iteration_index = 0u; iteration_index < 32u; iteration_index++) {
        float normalized_value = hmul(vit && scaled ? hmul(head_values[iteration_index], 5.65625)
                                                    : head_values[iteration_index],
                                      inverse);
        stage[base + iteration_index] =
            e4m3(scaled ? hmul(normalized_value, scale) : normalized_value);
    }
}

void phoenix_split_softmax(uint row, uint local) {
    uint base = local * 64u, bias = bias_offset(row);
    float weights[64];
    for (uint iteration_index = 0u; iteration_index < 64u; iteration_index++) {
        float bias_value =
            (operation_flags() & 0x2000u) != 0u ? pc.b.v[bias + iteration_index] : 0.0;
        float ramp = clamp(half_round(fma(half_round(stage[base + iteration_index] + bias_value),
                                          0.044921875, 1.30078125)),
                           1.03125, 1.5693359375);
        weights[iteration_index] = phoenix_unpack_half((phoenix_half_bits(ramp) & 0x3ffu) << 5u).x;
    }
    precise float halves[2];
    for (uint half_index = 0u; half_index < 2u; half_index++) {
        precise float pair_sums[4];
        for (uint pair = 0u; pair < 4u; pair++) {
            precise float even_accumulator = weights[half_index * 8u + pair * 2u],
                          odd_accumulator = weights[half_index * 8u + pair * 2u + 1u];
            for (uint block = 1u; block < 4u; block++) {
                even_accumulator += weights[block * 16u + half_index * 8u + pair * 2u];
                odd_accumulator += weights[block * 16u + half_index * 8u + pair * 2u + 1u];
            }
            pair_sums[pair] = even_accumulator + odd_accumulator;
        }
        halves[half_index] = (pair_sums[0] + pair_sums[1]) + (pair_sums[2] + pair_sums[3]);
    }
    float inverse = phoenix_reciprocal(max(halves[0] + halves[1], 0.00006103515625));
    for (uint iteration_index = 0u; iteration_index < 64u; iteration_index++)
        stage[base + iteration_index] = e4m3(half_round(weights[iteration_index] * inverse));
}
