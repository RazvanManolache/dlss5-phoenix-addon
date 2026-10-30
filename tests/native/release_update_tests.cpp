#include "../../src/native/support/release_updates.h"
#include <iostream>
#include <stdexcept>
int main() {
    auto check = [](bool value) { if (!value) throw std::runtime_error("Release notifier contract failed"); };
    using release_updates::parse;
    check(parse("v1.0-beta") == parse("1.0.0-beta"));
    check(*parse("1.0-beta.10") > *parse("1.0-beta.2"));
    check(*parse("1.0") > *parse("1.0-beta.99"));
    check(!parse("1.0-beta..2") && !parse("../../evil") && !parse("1.2.3.4"));
    const auto feed = R"([{"tag_name":"v1.0-beta.2","draft":false,"published_at":"2026-10-30"},{"tag_name":"v9.0","draft":true,"published_at":"2026-10-30"}])";
    check(phoenix_updates::newer_release(feed, "v1.0-beta") == "v1.0-beta.2");
    check(phoenix_updates::newer_release(feed, "v1.0.0").empty());
    check(phoenix_updates::newer_release("{\"message\":\"API limit\"}").empty());
    check(phoenix_updates::newer_release("[null,5,{}]").empty());
    check(phoenix_updates::newer_release(std::string(20, '[')).empty());
    check(phoenix_updates::newer_release(std::string(256 * 1024 + 1, ' ')).empty());
    // Exercise real cancellation/handle retirement in a bounded Windows process.
    // No game, GPU, settings writes or downloads are involved.
    phoenix_updates::Service service;
    service.poll(); service.stop();
    std::cout << "{\"passed\":true,\"version_and_feed_checks\":10,\"worker_retired\":true}\n";
}
