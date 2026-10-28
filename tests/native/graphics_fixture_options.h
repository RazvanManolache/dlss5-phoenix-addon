#pragma once
#include <stdexcept>
#include <string_view>

// Scenario flags compose with the independent shutdown check. Parse this
// before creating any device so invalid requests never produce GPU evidence.
struct GraphicsFixtureOptions {
    bool dynamic = false, process_exit = false, multiwindow = false;
    bool control = false, resize = false, secondary_queue = false;
    bool compute_present = false, multisample = false;

    static GraphicsFixtureOptions parse(int argc, const char* const* argv) {
        if (argc < 2) throw std::runtime_error("Specify a graphics API");
        GraphicsFixtureOptions result;
        for (int i = 2; i < argc; ++i) {
            const std::string_view option = argv[i];
            if (option == "--dynamic") result.dynamic = true;
            else if (option == "--process-exit") result.process_exit = true;
            else if (option == "--multiwindow") result.multiwindow = true;
            else if (option == "--control") result.control = true;
            else if (option == "--resize") result.resize = true;
            else if (option == "--secondary-queue") result.secondary_queue = true;
            else if (option == "--compute-present") result.compute_present = true;
            else if (option == "--msaa") result.multisample = true;
            else if (option == "--msaa-control") result.multisample = result.control = true;
            else throw std::runtime_error("Unknown fixture option");
        }
        // These scenarios have separate fixture setup paths.
        if (int(result.dynamic) + int(result.multiwindow) + int(result.resize) > 1)
            throw std::runtime_error("Conflicting fixture scenarios");
        return result;
    }
};
