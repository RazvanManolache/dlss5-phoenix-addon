// Platform/data-loader checks and host simulation of generated Metal arithmetic.
// Host simulation never substitutes for Apple's compiler or a real Metal run.
#include "native_runtime/backend.h"
#include "src/native/model/weight_source.h"
#define PHOENIX_METAL_HOST_TEST
#include "phoenix_reference.metal"
#include <iostream>
#include <vector>
using namespace phoenix_cpp;
int main() {
    try {
        using phoenix_model::portable_sha256;
        demand(portable_sha256(nullptr, 0) ==
                   "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
               "SHA empty");
        const std::string abc = "abc";
        demand(portable_sha256(reinterpret_cast<const uint8_t *>(abc.data()), abc.size()) ==
                   "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
               "SHA abc");
        const std::string million(1000000, 'a');
        demand(portable_sha256(reinterpret_cast<const uint8_t *>(million.data()), million.size()) ==
                   "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
               "SHA million a");
        // SHA-256 padding boundaries against the host's production hash on Windows.
#ifdef _WIN32
        for (size_t size : {55, 56, 63, 64, 65, 119, 120, 127, 128, 4096}) {
            std::vector<uint8_t> values(size);
            for (size_t index = 0; index < size; ++index)
                values[index] = uint8_t(index * 137);
            demand(portable_sha256(values.data(), size) ==
                       phoenix_model::weight_digest(values.data(), size),
                   "SHA padding boundary");
        }
#endif
        std::vector<uint8_t> pe(1024);
        auto word = [&](size_t offset, uint32_t value) {
            pe.at(offset) = uint8_t(value);
            pe.at(offset + 1) = uint8_t(value >> 8);
        };
        auto dword = [&](size_t offset, uint32_t value) {
            word(offset, value);
            word(offset + 2, value >> 16);
        };
        word(0, 0x5a4d);
        dword(0x3c, 64);
        dword(64, 0x4550);
        word(70, 1);
        word(84, 240);
        word(88, 0x20b);
        dword(88 + 108, 16);
        dword(88 + 128, 0x1000);
        dword(88 + 132, 256);
        dword(328 + 12, 0x1000);
        dword(328 + 16, 512);
        dword(328 + 20, 512);
        word(512 + 14, 1);
        dword(528, 10);
        dword(532, 0x80000020);
        word(544 + 12, 1);
        dword(560, 0x80000080);
        dword(564, 0x80000040);
        word(576 + 14, 1);
        dword(592, 1033);
        dword(596, 96);
        dword(608, 0x1100);
        dword(612, 4);
        const std::string name = "WEIGHTS_HT";
        word(640, unsigned(name.size()));
        for (size_t index = 0; index < name.size(); ++index)
            word(642 + index * 2, name[index]);
        pe[768] = 1;
        pe[769] = 2;
        pe[770] = 3;
        pe[771] = 4;
        auto resource = phoenix_model::pe_weight_resource(pe);
        demand(resource.size() == 4 && resource.data() == pe.data() + 768, "PE resource selection");
        auto reject = [&](auto bytes) {
            bool rejected = false;
            try {
                phoenix_model::pe_weight_resource(bytes);
            } catch (const std::exception &) {
                rejected = true;
            }
            demand(rejected, "Malformed PE accepted");
        };
        for (size_t length : {0, 1, 63, 87, 200, 511, 700})
            reject(std::span<const uint8_t>(pe.data(), length));
        dword(612, 1024);
        reject(std::span<const uint8_t>(pe));
        dword(612, 4);
        dword(596, 0x80000060);
        reject(std::span<const uint8_t>(pe));
        dword(596, 96);
        dword(532, 0xfffffff0);
        reject(std::span<const uint8_t>(pe));
        dword(532, 0x80000020);
        word(642, 'X');
        reject(std::span<const uint8_t>(pe));
        word(642, 'W');
        phx::Kernel<HostMath> reference{};
        phoenix_metal::Kernel<MetalMath> metal{};
        // Cover all binary16 inputs including subnormals, signed zero and infinities.
        for (uint32_t bits = 0; bits < 65536; ++bits) {
            const auto value = reference.half_value(bits);
            if (std::isnan(value))
                continue;
            demand(HostMath::bits(value) == HostMath::bits(metal.half_value(bits)),
                   "Metal half decode");
            demand(reference.half_bits(value) == metal.half_bits(value), "Metal half encode");
            for (unsigned publication = 0; publication < 8; ++publication) {
                if (!std::isfinite(value))
                    continue;
                auto expected = reference.publish(publication, value),
                     actual = metal.publish(publication, value);
                demand(HostMath::bits(expected) == HostMath::bits(actual),
                       "Metal publication adaptation");
            }
        }
        std::cout << "Portable SHA/PE/Metal host arithmetic checks passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
