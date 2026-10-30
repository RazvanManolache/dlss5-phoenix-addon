// Pure selection/manifest contracts: no driver loading, device or model weights.
#include "../../src/native/model/kernel_style_catalog.h"
#include "../../src/native/support/kernel_style_device.h"
#include "../../src/integration/compatibility/wow64/bridge_policy.h"
#include <iostream>
using Json = nlohmann::json;
static unsigned checks = 0;
static void require(bool condition) {
    ++checks;
    if (!condition) throw std::runtime_error("Kernel style test failed");
}
template<class Action> static void refuses(Action action) {
    bool failed = false;
    try { action(); } catch (const std::exception&) { failed = true; }
    require(failed);
}
int main() {
    using namespace phoenix_kernel;
    const LUID first{0x1234, 7}, second{0x1234, 8};
    remember_device_architecture(first, 120); remember_device_architecture(second, 89);
    require(adapter_identity(first) != adapter_identity(second));
    require(device_architecture_cache().values.at(adapter_identity(first)) == 120);
    require(device_architecture_cache().values.at(adapter_identity(second)) == 89);
    phoenix_compat::Settings settings;
    require(settings.valid() && settings.native_kernel_style == "auto");
    const auto automatic = settings;
    settings.native_kernel_style = "rtx20";
    require(settings.valid() && phoenix_compat::helper_settings_changed(automatic, settings));
    require(phoenix_compat::settings_json(settings).at("NativeKernelStyle") == "rtx20");
    require(!phoenix_compat::live_tuning_json(settings).contains("NativeKernelStyle"));
    require(*phoenix_compat::native_settings_unavailable_reason(settings, false));
    require(*phoenix_compat::native_settings_unavailable_reason(settings, true));
    settings.mode = phoenix_addon::phoenix::nvidia;
    require(std::string(phoenix_compat::native_settings_unavailable_reason(settings, false)).find("NVIDIA DLSS5") != std::string::npos);
    settings.mode = phoenix_addon::phoenix::ours;
    settings.native_kernel_style = "invalid"; require(!settings.valid());
    for (unsigned arch : {0u, 75u, 86u, 89u, 120u, 999u}) {
        require(style_supported("auto", arch));
        for (const char* style : {"rtx20", "rtx30", "rtx40"})
            require(style_supported(style, arch) == ((arch == 86 || arch == 89 || arch == 120) && style_architecture(style) < arch));
    }
    require(!valid_style("rtx50") && !valid_style("../rtx20"));
    Json manifest{{"kernels", Json::array()}};
    for (unsigned i = 0; i < 41; ++i)
        manifest["kernels"].push_back({{"name", std::to_string(i)}, {"entry", "kernel"},
            {"parameter_bytes", 8}, {"block", {32, 1, 1}}, {"binary", "native.cubin"}});
    require(&phoenix_bundle::style_catalog(manifest, "auto", 0) == &manifest["kernels"]);
    manifest["kernel_style"] = "rtx20";
    refuses([&] { phoenix_bundle::style_catalog(manifest, "auto", 120); });
    manifest.erase("kernel_style");
    refuses([&] { phoenix_bundle::style_catalog(manifest, "rtx20", 120); });
    manifest["kernel_styles"]["rtx20"] = {{"architectures", {"sm_86", "sm_89", "sm_120"}}, {"kernels", manifest["kernels"]}};
    require(phoenix_bundle::available_kernel_styles(manifest, 120) == Json::array({"rtx20"}));
    require(phoenix_bundle::available_kernel_styles(manifest, 75).empty());
    manifest["kernel_styles"]["unknown"] = 1;
    manifest["kernel_styles"]["rtx40"] = {{"architectures", "broken"}};
    require(phoenix_bundle::available_kernel_styles(manifest, 120) == Json::array({"rtx20"}));
    manifest["kernel_styles"].erase("unknown"); manifest["kernel_styles"].erase("rtx40");
    for (unsigned arch : {86u, 89u, 120u}) require(phoenix_bundle::style_catalog(manifest, "rtx20", arch).size() == 41);
    for (unsigned arch : {0u, 75u, 999u}) refuses([&] { phoenix_bundle::style_catalog(manifest, "rtx20", arch); });
    auto& rows = manifest["kernel_styles"]["rtx20"]["kernels"];
    const auto good = rows;
    rows[0]["block"] = {64, 1, 1};
    refuses([&] { phoenix_bundle::style_catalog(manifest, "rtx20", 120); });
    rows = good; rows[0]["large_input_entry"] = "wrong";
    refuses([&] { phoenix_bundle::style_catalog(manifest, "rtx20", 120); });
    rows = good; rows[1] = rows[0];
    refuses([&] { phoenix_bundle::style_catalog(manifest, "rtx20", 120); });
    rows = good; rows.erase(rows.begin());
    refuses([&] { phoenix_bundle::style_catalog(manifest, "rtx20", 120); });
    rows = good; manifest["kernel_styles"]["rtx20"]["architectures"] = {"sm_86"};
    refuses([&] { phoenix_bundle::style_catalog(manifest, "rtx20", 120); });
    std::cout << Json{{"passed", true}, {"cpu_only", true}, {"checks", checks}}.dump() << '\n';
}
