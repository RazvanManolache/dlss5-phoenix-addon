// Eight lanes cooperate on a row. Preserve the existing scalar reduction order.
// The selected pipeline has 256 lanes and handles 32 rows per workgroup.
float phx_lane(float value, uint lane) {
    return subgroupShuffle(value, (gl_SubgroupInvocationID & ~7u) + lane);
}
float phx_qk_load(uint row, uint channel) {
    uint flags = operation_flags();
    if ((flags & 0x20000u) != 0u) {
        uint head = (row / pc.n) % pc.batch, token = row % pc.n, window = row / (pc.n * pc.batch);
        uint part = (flags >> 18) & 1u;
        return pc.a
            .v[((window * pc.n + token) * 3u * pc.batch + part * pc.batch + head) * 32u + channel];
    }
    return (flags & 0x8000u) != 0u ? float(HalfBuf(uint64_t(pc.a)).v[row * 32u + channel])
                                   : pc.a.v[row * 32u + channel];
}
// Eight lanes each square four channels of one 32-channel head. Preserve
// the even/odd subgroup reduction and half-rounded norm before publication.
void phx_norm_parallel(uint row, uint lane) {
    float head_values[4];
    head_values[0] = phx_qk_load(row, lane);
    head_values[1] = phx_qk_load(row, lane + 16u);
    head_values[2] = phx_qk_load(row, lane + 8u);
    head_values[3] = phx_qk_load(row, lane + 24u);
    precise float partial_sum = 0.;
    for (uint iteration_index = 0u; iteration_index < 4u; iteration_index++)
        partial_sum = fma(head_values[iteration_index], head_values[iteration_index], partial_sum);
    precise float even_sum = half_round((phx_lane(partial_sum, 0u) + phx_lane(partial_sum, 2u)) +
                                        (phx_lane(partial_sum, 4u) + phx_lane(partial_sum, 6u)));
    precise float odd_sum = half_round((phx_lane(partial_sum, 1u) + phx_lane(partial_sum, 3u)) +
                                       (phx_lane(partial_sum, 5u) + phx_lane(partial_sum, 7u)));
    float total = max(half_round(even_sum + odd_sum), 0.00006198883056640625);
    precise float scale = 0.;
    if (!isinf(total)) {
        precise float root = sqrt(total);
        precise float error = fma(-root, root, total);
        root += error / (2.0 * root);
        scale = phoenix_reciprocal(root);
    }
    if (pc.k != 0u)
        scale *= pc.d.v[(row / pc.n) % pc.batch];
    store(row * 32u + lane, e4m3(head_values[0] * scale));
    store(row * 32u + lane + 16u, e4m3(head_values[1] * scale));
    store(row * 32u + lane + 8u, e4m3(head_values[2] * scale));
    store(row * 32u + lane + 24u, e4m3(head_values[3] * scale));
}
void phx_softmax_parallel(uint row, uint lane) {
    float weights[8];
    uint bias = bias_offset(row);
    bool split = pc.batch == 16u;
    for (uint iteration_index = 0u; iteration_index < 8u; iteration_index++) {
        uint key_index = lane + 8u * iteration_index;
        float score = pc.a.v[row * 64u + key_index];
        float position_bias = (operation_flags() & 0x2000u) != 0u ? pc.b.v[bias + key_index] : 0.;
        float ramp =
            split
                ? half_round(fma(half_round(score + position_bias), 0.044921875, 1.30078125))
                : half_round(fma(score, 0.044921875, fma(position_bias, 0.044921875, 1.30078125)));
        ramp = clamp(ramp, 1.03125, 1.5693359375);
        weights[iteration_index] = phoenix_unpack_half((phoenix_half_bits(ramp) & 0x3ffu) << 5u).x;
    }
    precise float total;
    if (split) {
        precise float even_sum = ((weights[0] + weights[2]) + weights[4]) + weights[6],
                      odd_sum = ((weights[1] + weights[3]) + weights[5]) + weights[7];
        precise float even_total = ((phx_lane(even_sum, 0u) + phx_lane(even_sum, 1u)) +
                                    (phx_lane(even_sum, 2u) + phx_lane(even_sum, 3u))) +
                                   ((phx_lane(even_sum, 4u) + phx_lane(even_sum, 5u)) +
                                    (phx_lane(even_sum, 6u) + phx_lane(even_sum, 7u)));
        precise float odd_total = ((phx_lane(odd_sum, 0u) + phx_lane(odd_sum, 1u)) +
                                   (phx_lane(odd_sum, 2u) + phx_lane(odd_sum, 3u))) +
                                  ((phx_lane(odd_sum, 4u) + phx_lane(odd_sum, 5u)) +
                                   (phx_lane(odd_sum, 6u) + phx_lane(odd_sum, 7u)));
        total = even_total + odd_total;
    } else {
        float even_sum = hadd(hadd(hadd(weights[0], weights[2]), weights[4]), weights[6]);
        float odd_sum = hadd(hadd(hadd(weights[1], weights[3]), weights[5]), weights[7]);
        float parity_sum = hadd(hadd(hadd(even_sum, phx_lane(even_sum, (lane % 4u) + 4u)), odd_sum),
                                phx_lane(odd_sum, (lane % 4u) + 4u));
        total = hadd(hadd(phx_lane(parity_sum, 0u), phx_lane(parity_sum, 1u)),
                     hadd(phx_lane(parity_sum, 2u), phx_lane(parity_sum, 3u)));
    }
    float inverse = phoenix_reciprocal(max(total, 0.00006103515625));
    for (uint iteration_index = 0u; iteration_index < 8u; iteration_index++)
        store(row * 64u + lane + 8u * iteration_index,
              e4m3(half_round(weights[iteration_index] * inverse)));
}
// Eight columns cooperate on a global row. The same half-rounded column,
// parity and block sums are evaluated in the original order, without the
// strided scalar reads of 32 unrelated rows in each subgroup.
// Reduce global scores in 64-key blocks. Padding uses the zero-score
// weight during the fixed reduction tree, then its contribution is removed
// from the denominator; padded values are published as zero.
void phx_vit_parallel(uint row, uint lane) {
    uint stride = pc.sa != 0u ? pc.sa : pc.n, base = row * stride, blocks = (pc.n + 63u) / 64u;
    float zero = vit_weight(0.), total = 0.;
    for (uint key_block = 0u; key_block < blocks; key_block++) {
        float pair[4];
        for (uint inner_index = 0u; inner_index < 4u; inner_index++) {
            uint key_index = key_block * 64u + lane + 16u * inner_index;
            float first_weight = key_index < pc.n ? vit_weight(pc.a.v[base + key_index]) : zero;
            float second_weight =
                key_index + 8u < pc.n ? vit_weight(pc.a.v[base + key_index + 8u]) : zero;
            pair[inner_index] = hadd(first_weight, second_weight);
            if (key_index < stride)
                store(base + key_index, key_index < pc.n ? e4m3(first_weight) : 0.);
            if (key_index + 8u < stride)
                store(base + key_index + 8u, key_index + 8u < pc.n ? e4m3(second_weight) : 0.);
        }
        float column = hadd(hadd(hadd(pair[0], pair[1]), pair[2]), pair[3]);
        float even =
            hadd(hadd(hadd(phx_lane(column, 0u), phx_lane(column, 2u)), phx_lane(column, 4u)),
                 phx_lane(column, 6u));
        float odd =
            hadd(hadd(hadd(phx_lane(column, 1u), phx_lane(column, 3u)), phx_lane(column, 5u)),
                 phx_lane(column, 7u));
        total = hadd(total, hadd(even, odd));
    }
    uint padding = blocks * 64u - pc.n;
    if (padding > 0u)
        total = hadd(total, -hmul(zero, float(padding)));
    if (lane == 0u)
        pc.d.v[row] = half_round(1. / total);
    // General callers may provide additional stride padding beyond the last
    // denominator block. It contributes no value in the following product.
    for (uint key_index = blocks * 64u + lane; key_index < stride; key_index += 8u)
        store(base + key_index, 0.);
}
