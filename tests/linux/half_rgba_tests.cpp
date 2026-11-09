#include "../../native_runtime/half_rgba.h"
#include "../../native_runtime/host_math.h"
#define PHX_DEVICE
#include "../../src/backends/hip_sycl/compute_ops.h"
#include <vector>
#include <iostream>
int main() {
    phx::Kernel<HostMath> reference;
    for (size_t count : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(65537)}) {
        std::vector<uint16_t> source(count * 4);
        for (size_t i = 0; i < source.size(); ++i)
            source[i] = uint16_t(i/4 + (i%4)*19337);
        std::vector<float> scalar(count * 3 + 1, 42), actual = scalar;
        phoenix_native_cuda::half_rgba_scalar(source.data(), scalar.data(), count);
        phoenix_native_cuda::half_rgba_to_rgb(source.data(), actual.data(), count);
        if (actual.back() != 42 || scalar.back() != 42)
            return 1;
        for (size_t p = 0; p < count; ++p)
            for (unsigned c = 0; c < 3; ++c) {
                const float expected = reference.half_value(source[p * 4 + c]);
                for (float value : {scalar[p * 3 + c], actual[p * 3 + c]})
                    if (std::isnan(expected) ? !std::isnan(value)
                                             : HostMath::bits(value) != HostMath::bits(expected))
                        return 2;
            }
    }
    std::cout << "RGBA16F conversion matches reference across all half codes and odd tails\n";
}
