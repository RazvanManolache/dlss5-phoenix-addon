#pragma once
#include <string>
#include <vector>
namespace phoenix_addon::phoenix {
inline bool valid_execution_device(const std::string& device_id) {
    if (device_id == "native" || device_id == "cpu" || device_id == "sycl:cpu")
        return true;
    size_t prefix = 0;
    if (device_id.rfind("vulkan:", 0) == 0)
        prefix = 7;
    else if (device_id.rfind("hip:", 0) == 0)
        prefix = 4;
    else if (device_id.rfind("sycl:", 0) == 0)
        prefix = 5;
    if (prefix) {
        if (device_id.size() != prefix + 32)
            return false;
        for (size_t iteration_index = prefix; iteration_index < device_id.size(); ++iteration_index)
            if (!((device_id[iteration_index] >= '0' && device_id[iteration_index] <= '9') ||
                  (device_id[iteration_index] >= 'a' && device_id[iteration_index] <= 'f')))
                return false;
        return true;
    }
    if (device_id.size() != 45 || device_id.rfind("cuda:GPU-", 0) != 0)
        return false;
    for (size_t iteration_index = 9; iteration_index < device_id.size(); ++iteration_index) {
        const char character = device_id[iteration_index];
        if (iteration_index == 17 || iteration_index == 22 || iteration_index == 27 ||
            iteration_index == 32) {
            if (character != '-')
                return false;
        } else if (!((character >= '0' && character <= '9') ||
                     (character >= 'a' && character <= 'f')))
            return false;
    }
    return true;
}
struct ExecutionDevice {
    std::string id, label;
};
}
