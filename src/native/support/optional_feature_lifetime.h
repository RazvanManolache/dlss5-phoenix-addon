#pragma once

namespace phoenix_model_tools {
// The caller proves the last GPU submission has finished. Keep a failed
// release's handle for retry; successful handles must never be released twice.
template <class Feature, class Release>
bool retire_optional_feature(Feature& feature, bool idle, Release release) {
    if (!idle)
        return false;
    if (feature.handle) {
        release(feature.handle);
        feature.handle = nullptr;
    }
    for (auto& pass : feature.refinement) {
        if (pass.handle) {
            release(pass.handle);
            pass.handle = nullptr;
        }
    }
    feature.refinement.clear();
    feature.output.Reset();
    feature.device.Reset();
    feature.inited = false;
    feature.failed = false;
    feature.attempts = 0;
    feature.warmup = 0;
    feature.first = true;
    feature.abandons = 0;
    feature.retry_after_tick = 0;
    feature.requested_contract.clear();
    return true;
}
} // namespace phoenix_model_tools
