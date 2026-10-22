// Compare every packed byte against the existing loader while denying large
// C++ allocations in the streaming path. No GPU or vendor code is executed.
#include "../../src/native/model/bundle_loader.h"
#include <cstdlib>
#include <iostream>
#include <new>

static bool bounded_allocations = false;
static size_t largest_request = 0;
void* operator new(size_t bytes) {
    if (bounded_allocations) {
        largest_request = std::max(largest_request, bytes);
        if (bytes > 1024 * 1024) throw std::bad_alloc();
    }
    if (auto* value = std::malloc(bytes ? bytes : 1)) return value;
    throw std::bad_alloc();
}
void* operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
void operator delete(void* value, size_t) noexcept { std::free(value); }
void operator delete[](void* value, size_t) noexcept { std::free(value); }

static void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Work> static void rejects(Work work) {
    try { work(); } catch (const std::exception&) { return; }
    throw std::runtime_error("Invalid weight request accepted");
}

static void upload_integrity() {
    std::vector<uint8_t> input(64 * 1024 + 17, 37), output(input.size());
    const auto expected = phoenix_model::weight_digest(input.data(), input.size());
    auto view = phoenix_model::WeightView(input).authenticated(expected);
    phoenix_model::WeightTransfer good(view);
    good.copy_to(0, output.data(), 64 * 1024);
    good.copy_to(64 * 1024, output.data() + 64 * 1024, 17);
    good.finish();
    check(input == output, "Verified upload changed pixels");
    phoenix_model::WeightTransfer incomplete(view);
    incomplete.copy_to(0, output.data(), 16);
    rejects([&] { incomplete.finish(); });
    rejects([&] { incomplete.copy_to(17, output.data(), 1); });
    auto changing = phoenix_model::WeightView(input.size(), input.data(),
        [&](size_t offset, void* destination, size_t bytes) {
            std::memcpy(destination, input.data() + offset, bytes);
            if (offset) static_cast<uint8_t*>(destination)[0] ^= 1;
        }).authenticated(expected);
    phoenix_model::WeightTransfer altered(changing);
    altered.copy_to(0, output.data(), output.size());
    rejects([&] { altered.finish(); });
}

int main(int argc, char** argv) {
    try {
        check(argc == 4, "Expected DLL, bundle and model ID");
        upload_integrity();
        const auto path = std::filesystem::absolute(std::filesystem::u8path(argv[1]));
        const auto digest = phoenix_model::mapped_file_digest(path);
        phoenix_model::WeightSource source{path, digest};
        source.model_id = argv[3];
        // The pre-existing contiguous algorithm is an independent byte oracle.
        auto reference = phoenix_model::packed_model(source, phoenix_model::mapped_file_digest);
        if (source.model_id == "original")
            check(reference.catalogue.at("sha256") == "a5513b1845c98a486985ed04f38e66a1854cce33c2aba3a505866028bd4ee3e5", "Original packed golden changed");
        bounded_allocations = true;
        phoenix_bundle::Bundle bundle(std::filesystem::u8path(argv[2]), phoenix_model::mapped_file_digest, source, true);
        check(bundle.embedded_weights.empty(), "Streaming bundle retained CPU model bytes");
        check(bundle.weights == reference.catalogue, "Streaming catalogue differs");
        auto view = bundle.load_weight_view();
        check(view.size() == reference.bytes.size(), "Streaming extent differs");
        check(bundle.streamed_weights.empty(), "Bundle retained an upload source after consumption");
        auto retained = view;
        check(retained.identity() == view.identity(), "Copied view lost its reader identity");
        std::array<uint8_t, 64 * 1024> scratch{};
        phoenix_model::WeightTransfer upload(view);
        for (size_t offset = 0; offset < view.size();) {
            const auto count = std::min(scratch.size(), view.size() - offset);
            upload.copy_to(offset, scratch.data(), count);
            check(std::memcmp(scratch.data(), reference.bytes.data() + offset, count) == 0, "Streamed native bytes differ");
            offset += count;
        }
        upload.finish();
        for (size_t offset : {size_t(1), size_t(511), size_t(513), view.size() - 1025}) {
            view.copy_to(offset, scratch.data(), 1025);
            check(std::memcmp(scratch.data(), reference.bytes.data() + offset, 1025) == 0, "Unaligned packed read differs");
        }
        view.copy_to(view.size(), nullptr, 0);
        rejects([&] { view.copy_to(view.size(), scratch.data(), 1); });
        rejects([&] { view.copy_to(0, nullptr, 1); });
        rejects([&] { view.copy_to(SIZE_MAX, scratch.data(), 1); });
        auto known = phoenix_model::streamed_packed_model(source, phoenix_model::mapped_file_digest, &reference.catalogue);
        check(known.catalogue == reference.catalogue, "Known catalogue reconstructed differently");
        auto invalid = reference.catalogue;
        invalid["records"][0]["sha256"] = std::string(64, '0');
        rejects([&] { phoenix_model::streamed_packed_model(source, phoenix_model::mapped_file_digest, &invalid); });
        invalid = reference.catalogue;
        invalid["records"][1]["offset"] = 0;
        rejects([&] { phoenix_model::streamed_packed_model(source, phoenix_model::mapped_file_digest, &invalid); });
        invalid = reference.catalogue;
        invalid["sha256"] = std::string(64, '0');
        rejects([&] { phoenix_model::streamed_packed_model(source, phoenix_model::mapped_file_digest, &invalid); });
        check(!GetModuleHandleW(L"nvapi64.dll") && !GetModuleHandleW(L"nvngx_dlssnr.dll"), "CPU test loaded vendor code");
        check(phoenix_model::mapped_file_digest(path) == digest, "Model source changed");
        check(largest_request <= 1024 * 1024, "Streaming attempted a model-sized host allocation");
        bounded_allocations = false;
        std::cout << nlohmann::json({{"passed",true},{"model",source.model_id},
            {"packed_bytes",view.size()},{"packed_sha256",reference.catalogue.at("sha256")},
            {"largest_streaming_cpp_allocation",largest_request},{"allocation_ceiling",1024*1024},
            {"all_bytes_equal",true},{"gpu_used",false}}).dump() << '\n';
    } catch (const std::exception& error) {
        bounded_allocations = false;
        std::cerr << error.what() << '\n'; return 1;
    }
}
