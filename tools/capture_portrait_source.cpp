// Reuse the gallery renderer without modifying its working build or source.
#define wmain unused_gallery_main
#include "../../dlss5-phoenix-gallery/scene.cpp"
#undef wmain

int wmain(int argc, wchar_t** argv) {
    try {
        ensure(argc == 4, "Expected gallery root, raw input path, and receipt path");
        hr(CoInitializeEx(nullptr, COINIT_MULTITHREADED));
        App app{fs::path(argv[1])};
        app.enabled = false;
        app.init(false, true);
        app.deterministicAnimation = true;
        app.clientSize(1920, 1080);
        app.qualityPreset(2);
        app.characterDetail = 0;
        app.enabled = false;
        app.materialScope = false;
        app.displayMode = 0;
        app.sceneTime = .32f;
        app.lightPhase = .63f;
        app.focusPerson = 0;
        app.focusFraming = 2;
        app.focusAngle = .3f;
        app.focusZoom = 1;
        app.facialAnimation = false;
        app.render(cases[0], {}, false, false);
        app.begin();
        capture(app.cmd.Get(), app.scene.Get(), app.rbInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        app.submit();
        save(app.rbInput, fs::path(argv[2]));
        auto row = app.report["frames"].back();
        ensure(!row["network_enabled"].get<bool>() && row["own_launches"] == 0, "Source must bypass all neural processing");
        ensure(row["d3d12_errors"] == 0 && !IsWindowVisible(app.window), "Hidden error-free render required");
        row["source_only"] = true;
        row["presentation"] = false;
        row["windows_hidden"] = true;
        row["desktop_capture"] = false;
        const auto rawPath = fs::path(argv[2]);
        app.begin();
        capture(app.cmd.Get(), app.gAlbedo.Get(), app.rbInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        app.submit();
        save(app.rbInput, rawPath.parent_path() / "woman-albedo.rgba16f");
        app.detailFlags |= 8192; // Existing gallery ablation: bypass skin diffusion.
        app.reset = true;
        app.render(cases[0], {}, false, false);
        app.begin();
        capture(app.cmd.Get(), app.scene.Get(), app.rbInput, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        app.submit();
        save(app.rbInput, rawPath.parent_path() / "woman-no-diffusion.rgba16f");
        row["no_diffusion_diagnostic"] = app.report["frames"].back();
        std::ofstream(fs::path(argv[3])) << row.dump(2);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
