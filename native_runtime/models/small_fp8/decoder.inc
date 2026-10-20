// CC_SMALL_FP8 uses the recovered packed families and its own 302-tensor contract.
    void decode_small_fp8(const uint8_t* bytes, size_t size) {
        decode_layout(bytes, size, [](size_t count, auto visitor) {
            small_fp8_layouts::visit_layout(count, visitor);
        });
    }
