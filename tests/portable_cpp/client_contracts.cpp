#include "../../native_runtime/native_execution_client.h"
#include <iostream>
#include <chrono>
using namespace phoenix_cpp;
using Json = nlohmann::json;
void demand(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
void temporal_rejection_contracts(const std::filesystem::path& root, Json config, Json& checks) {
    constexpr unsigned width = 96, height = 64;
    constexpr size_t pixels = size_t(width) * height;
    config["cpu_threads"] = 8;
    RuntimeApi subject, control;
    subject.open(root, config.dump(), width, height);
    control.open(root, config.dump(), width, height);
    std::vector<float> color(pixels * 3), motion(pixels * 2, 0.f), depth(pixels, .5f),
        validity(pixels, 1.f), actual(color.size()), expected(color.size());
    Json metadata = {{"repeats", {0, 0, 0, 0, 0, 0, 0, 0}},
                     {"has_motion", true},
                     {"has_depth", true},
                     {"has_history_validity", true}};
    unsigned frame_index = 0;
    auto advance = [&] {
        for (size_t index = 0; index < color.size(); ++index)
            color[index] = .15f + float((index * 17 + frame_index * 41) % 601) / 1000.f;
        ++frame_index;
        for (auto* api : {&subject, &control})
            demand(!api->frame(api->instance, color.data(), motion.data(), depth.data(),
                               validity.data(), metadata.dump().c_str(),
                               api == &subject ? actual.data() : expected.data()),
                   api->error());
        demand(actual == expected, "Rejected request changed continuation pixels");
        const auto subject_status = Json::parse(subject.status_json()),
                   control_status = Json::parse(control.status_json());
        for (const char* field : {"history_used", "history_pre_valid_pixels",
                                  "history_post_valid_pixels", "passes", "repeats"})
            demand(subject_status.at(field) == control_status.at(field),
                   "Rejected request changed continuation history status");
        if (frame_index > 1)
            demand(subject_status.at("history_used") == true &&
                       subject_status.at("history_pre_valid_pixels") == pixels &&
                       subject_status.at("history_post_valid_pixels") == pixels,
                   "Continuation comparison did not exercise history");
    };
    advance();
    advance();
    // Each rejected request is followed by a real inference continuation in both
    // sessions. Only the subject sees the rejection; output and history use must match.
    auto reject_then_continue = [&](Json invalid, const float* requested_motion,
                                    const float* requested_depth, const float* requested_validity,
                                    const char* label) {
        std::fill(actual.begin(), actual.end(), -123.f);
        const auto status_before = subject.status_json();
        demand(subject.frame(subject.instance, color.data(), requested_motion, requested_depth,
                              requested_validity, invalid.dump().c_str(), actual.data()) != 0,
               "Invalid temporal request accepted");
        demand(std::all_of(actual.begin(), actual.end(),
                           [](float value) { return value == -123.f; }),
               "Rejected temporal request published output");
        demand(subject.status_json() == status_before,
               "Rejected temporal request changed completed status");
        advance();
        checks.push_back(label);
    };
    auto invalid = metadata;
    invalid.update({{"reset", true}, {"controls", {{"intensity", 3}}}});
    reject_then_continue(invalid, motion.data(), depth.data(), validity.data(),
                         "rejected reset with invalid controls preserves continuation");
    invalid = metadata;
    invalid.update({{"reset", true}, {"repeats", {5, 0, 0, 0, 0, 0, 0, 0}}});
    reject_then_continue(invalid, motion.data(), depth.data(), validity.data(),
                         "rejected reset with invalid repeats preserves continuation");
    invalid = metadata;
    invalid.update({{"repeats", {1, 0, 0, 0, 0, 0, 0, 0}},
                    {"controls", {{"style", 1}}}});
    reject_then_continue(invalid, nullptr, depth.data(), validity.data(),
                         "rejected graph and control change with missing motion preserves continuation");
    invalid = metadata;
    invalid["passes"] = 2;
    reject_then_continue(invalid, motion.data(), nullptr, validity.data(),
                         "rejected pass change with missing depth preserves continuation");
    invalid = metadata;
    invalid["reset"] = true;
    motion[7] = NAN;
    // Restore the caller input before the continuation.
    auto bad_motion = motion;
    motion[7] = 0;
    reject_then_continue(invalid, bad_motion.data(), depth.data(), validity.data(),
                         "rejected reset with nonfinite motion preserves continuation");
    auto bad_depth = depth;
    bad_depth[11] = NAN;
    reject_then_continue(invalid, motion.data(), bad_depth.data(), validity.data(),
                         "rejected reset with nonfinite depth preserves continuation");
    auto bad_validity = validity;
    bad_validity[13] = .5f;
    reject_then_continue(invalid, motion.data(), depth.data(), bad_validity.data(),
                         "rejected reset with nonbinary validity preserves continuation");
    invalid = metadata;
    invalid["motion_from_frame"] = "invalid";
    reject_then_continue(invalid, motion.data(), depth.data(), validity.data(),
                         "rejected status metadata preserves output and continuation");
}
int wmain(int argc, wchar_t** argv) {
    try {
        if (argc != 4)
            throw std::runtime_error("Usage: RUNTIME_ROOT WEIGHT_SOURCE_JSON NEW_REPORT");
        // The runtime and Windows DLL search policy require an absolute package root.
        auto root = std::filesystem::absolute(std::filesystem::path(argv[1]));
        Json source;
        std::ifstream(argv[2]) >> source;
        const auto selected_source = phoenix_model::WeightSource::config(source);
        demand(
            phoenix_model::WeightSource::config(selected_source.configuration()).configuration() ==
                source,
            "Model source UTF-8 configuration roundtrip failed");
        Json config = {{"runtime_root", path_utf8(root)},
                       {"backend", "cpu"},
                       {"device", "cpu"},
                       {"execution", "in_process"},
                       {"allow_unvalidated", true},
                       {"weight_source", source}};
        Json checks = Json::array();
        auto expect_error = [&](auto operation, const char* label) {
            bool caught = false;
            try {
                operation();
            } catch (const std::exception&) {
                caught = true;
            }
            demand(caught, label);
            checks.push_back(label);
        };
        expect_error(
            [&] {
                RuntimeApi api;
                auto bad = config;
                bad["backend"] = "missing";
                api.open(root, bad.dump(), 96, 64);
            },
            "unknown backend rejected");
        expect_error(
            [&] {
                RuntimeApi api;
                auto bad = config;
                bad["weight_source"]["sha256"] = std::string(64, '0');
                api.open(root, bad.dump(), 96, 64);
            },
            "incorrect model digest rejected");
        expect_error(
            [&] {
                RuntimeApi api;
                auto bad = config;
                bad["device"] = "missing";
                api.open(root, bad.dump(), 96, 64);
            },
            "unknown device rejected");
        for (const auto& value : {Json(-1), Json(65536), Json(1.5), Json(true), Json("32")})
            expect_error(
                [&] {
                    RuntimeApi api;
                    auto bad = config;
                    bad["cpu_threads"] = value;
                    api.open(root, bad.dump(), 96, 64);
                },
                "invalid CPU thread setting rejected");
        for (unsigned requested : {0u, 1u, 8u, 16u, 32u, 65535u}) {
            RuntimeApi api;
            auto count = config;
            count["cpu_threads"] = requested;
            api.open(root, count.dump(), 96, 64);
            const auto status = Json::parse(api.status_json());
            demand(status.at("cpu_threads") == phoenix_threads::resolve(requested),
                   "CPU worker count was ignored");
            demand(status.at("cpu_threads_available") == phoenix_threads::available(),
                   "CPU availability was hardcoded");
        }
        checks.push_back("CPU thread counts auto/manual/clamped and reported");
        RuntimeApi api;
        api.open(root, config.dump(), 96, 64);
        std::vector<float> color(96 * 64 * 3, .3f), output(color.size(), -123.f),
            validity(96 * 64, 1.f);
        auto reject = [&](const Json& metadata) {
            demand(api.frame(api.instance, color.data(), nullptr, nullptr, validity.data(),
                             metadata.dump().c_str(), output.data()) != 0,
                   "Invalid request accepted");
            demand(std::all_of(output.begin(), output.end(),
                               [](float sample_value) { return sample_value == -123.f; }),
                   "Failed frame modified caller output");
        };
        color[0] = NAN;
        reject(Json::object());
        color[0] = .3f;
        checks.push_back("nonfinite color rejected without publishing output");
        reject({{"repeats", {5, 1, 1, 1, 1, 1, 1, 1}}});
        checks.push_back("invalid repetition rejected");
        reject({{"has_history_validity", true}});
        checks.push_back("validity without motion rejected");
        reject({{"controls", {{"intensity", 3}}}});
        checks.push_back("invalid controls rejected");
        demand(!api.frame(api.instance, color.data(), nullptr, nullptr, nullptr, "{\"reset\":true}",
                          output.data()),
               api.error());
        demand(std::all_of(output.begin(), output.end(),
                           [](float sample_value) { return std::isfinite(sample_value); }) &&
                   output != color,
               "Recovery did not run the model");
        checks.push_back("valid reset recovers and changes input");
        temporal_rejection_contracts(root, config, checks);
        config["execution"] = "broker";
        {
            NativeSession session;
            session.open(root, config, 96, 64);
            Handle process(
                OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, session.process_id()));
            demand(process.value && session.process_id() != GetCurrentProcessId(),
                   "Owned broker unavailable");
            demand(TerminateProcess(process, 77), "Owned broker termination failed");
            expect_error(
                [&] {
                    session.frame(color.data(), nullptr, nullptr, nullptr, "{}", output.data());
                },
                "broker death detected");
        }
        double cancel_ms = 0;
        DWORD child = 0;
        {
            Handle cancel(require_handle(CreateEventW(nullptr, TRUE, FALSE, nullptr)));
            NativeSession session;
            session.set_cancellation(cancel);
            session.open(root, config, 96, 64);
            child = session.process_id();
            bool cancelled = false;
            std::thread worker([&] {
                try {
                    session.frame(color.data(), nullptr, nullptr, nullptr, "{}", output.data());
                } catch (const std::exception&) {
                    cancelled = true;
                }
            });
            Sleep(50);
            auto started_at = std::chrono::steady_clock::now();
            SetEvent(cancel);
            worker.join();
            cancel_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                                  started_at)
                            .count();
            demand(cancelled && cancel_ms < 2000, "Broker cancellation did not return promptly");
            checks.push_back("in-flight broker cancellation returns promptly");
        }
        Handle process(OpenProcess(SYNCHRONIZE, FALSE, child));
        demand(!process.value || WaitForSingleObject(process, 5000) == WAIT_OBJECT_0,
               "Cancelled broker leaked");
        checks.push_back("broker lifetime bound to owning session");
        Json report = {
            {"passed", true},
            {"checks", checks},
            {"cancellation_ms", cancel_ms},
            {"scope",
             "Compiled CPU/runtime contracts and owned C++ broker loss/cancellation; no Python"}};
        demand(!std::filesystem::exists(argv[3]), "Fresh report required");
        std::ofstream(argv[3]) << report.dump(2);
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
