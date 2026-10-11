// Bounded window gather/scatter. Token order matches Phoenix's 4x4 subtiles.
if (kind >= 25u && kind <= 28u) {
    uint channels = pc.n, c = index % channels, rest = index / channels;
    uint token = rest % 64u, win = pc.batch + rest / 64u;
    int pixel_y =
        int((win / pc.sc) * 8u + (token / 32u) * 4u + (token % 16u) / 4u) - int(pc.k >> 16u);
    int pixel_x =
        int((win % pc.sc) * 8u + ((token / 16u) % 2u) * 4u + token % 4u) - int(pc.k & 65535u);
    bool valid = pixel_x >= 0 && pixel_y >= 0 && uint(pixel_y) < pc.sa && uint(pixel_x) < pc.sb;
    uint at = valid ? (uint(pixel_y) * pc.sb + uint(pixel_x)) * channels + c : 0u;
    if (kind == 25u) {
        store(index, valid ? load_a(at) : 0.);
        return;
    }
    if (kind == 26u) {
        if (valid)
            store(at, load_a(index));
        return;
    }
    if (kind == 28u) {
        float merged = 0.;
        if (valid) {
            uint above = ((uint(pixel_y) / 2u) * uint(pc.p0) + uint(pixel_x) / 2u) * channels + c;
            precise float scaled = load_a(above) * pc.d.v[c];
            merged = half_round(fma(float(HalfBuf(uint64_t(pc.b)).v[at]), pc.d.v[channels + c],
                                    half_round(scaled)));
        }
        store(index, merged);
        return;
    }
    if (valid) {
        HalfBuf(pc.second).v[at] = float16_t(e4m3(half_round(load_a(index))));
        if ((uint(pixel_y) & 1u) == 0u && (uint(pixel_x) & 1u) == 0u &&
            uint(pixel_y) + 1u < pc.sa && uint(pixel_x) + 1u < pc.sb) {
            precise float top = half_round(load_a(index) + load_a(index + channels));
            precise float bottom =
                half_round(load_a(index + 4u * channels) + load_a(index + 5u * channels));
            precise float sum = half_round(top + bottom);
            store(((uint(pixel_y) / 2u) * (pc.sb / 2u) + uint(pixel_x) / 2u) * channels + c,
                  sum * .25);
        }
    }
    return;
}
