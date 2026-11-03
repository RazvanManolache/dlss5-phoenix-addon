#include "../../src/integration/compatibility/retirement_serial.h"
#include <array>
#include <cstdlib>
#include <iostream>

namespace {
void check(bool value) { if (!value) std::abort(); }

// Reproduce the relevant DX9 timeline-ring contract without a GPU. A query
// retains its previous completed state until its slot is issued again.
struct Queries {
    std::array<bool, 8> done{};
    uint64_t latest = 0;
    void issue(uint64_t value) { latest = value; done[value % done.size()] = false; }
    void finish(uint64_t value) { done[value % done.size()] = true; }
    uint64_t completed() const {
        for (uint64_t value = latest, offset = 0; value && offset < done.size(); --value, ++offset)
            if (done[value % done.size()]) return value;
        return 0;
    }
};
}

int main() {
    using phoenix_compat::reserve_retirement_serial;
    uint64_t serial = 0, pending = 0;
    for (uint64_t batch = 1; batch <= 1000; ++batch) {
        const auto upload = reserve_retirement_serial(serial, pending);
        const auto capture = reserve_retirement_serial(serial, pending);
        check(upload == batch && capture == upload && serial == batch);
        // A failed signal retains the reservation until a successful retry.
        check(reserve_retirement_serial(serial, pending) == batch);
        pending = 0;
    }
    Queries skipped;
    for (uint64_t value : {1ull, 2ull, 4ull, 6ull, 8ull}) {
        skipped.issue(value); skipped.finish(value);
    }
    skipped.issue(10);
    check(skipped.completed() == 9); // Phantom progress through stale slot1.

    Queries sequential;
    serial = pending = 0;
    for (uint64_t batch = 1; batch <= 1000; ++batch) {
        const auto value = reserve_retirement_serial(serial, pending);
        check(reserve_retirement_serial(serial, pending) == value);
        sequential.issue(value);
        check(sequential.completed() < value); // Current batch is not retired.
        sequential.finish(value);
        check(sequential.completed() == value);
        pending = 0;
    }
    std::cout << "{\"passed\":true,\"batches\":2000,\"phantom_retirement_regression\":true}\n";
}
