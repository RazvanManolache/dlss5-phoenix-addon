// Resolves explicit or automatic CPU worker counts against this machine.
// Configuration values are checked before allocating a persistent worker pool.

#pragma once
#include "../src/native/support/cpu_threads.h"
#include "../third_party/nlohmann/json.hpp"
#include <stdexcept>

namespace phoenix_cpp {
inline unsigned cpu_thread_request(const nlohmann::json& config) {
    if (!config.contains("cpu_threads"))
        return 0;
    const auto& value = config.at("cpu_threads");
    if (!value.is_number_integer() || value < 0 || value > 65535)
        throw std::runtime_error("CPU threads must be an integer from 0 (automatic) to 65535");
    return value.get<unsigned>();
}
}
