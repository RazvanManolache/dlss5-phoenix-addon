// Original model keeps its independent layout table and identity checks.
    void decode_original(const uint8_t* bytes, size_t size) {
        decode_layout(bytes, size, [](size_t count, auto visitor) { visit_layout(count, visitor); });
    }
