// Real ImGui navigation and layout checks; no rendering device or inference.
// Optional captures export the actual vertex stream for a CPU-only preview.
#pragma once
inline nlohmann::json run_dashboard_tests(phoenix_ui::Context& context) {
    auto& io = ImGui::GetIO();
    unsigned checks = 0, saves = 0, adapter_views = 0, frames = 0, paired_actions=0;
    context.action=[&](phoenix_ui::Action action) { if(action==phoenix_ui::Action::SaveScreenshotPair)++paired_actions; };
    const auto require = [&](bool passed, const char* reason) {
        ++checks;
        if (!passed) throw std::runtime_error(std::string("Dashboard: ") + reason);
    };
    const auto snapshot = [&] {
        nlohmann::json result;
        phoenix_ui::phoenix::visit_settings(context.panel, [&](const char* name, const auto& value) {
            result[name] = value;
        });
        return result;
    };
    char capture_path[4096]{};
    GetEnvironmentVariableA("PHOENIX_UI_CAPTURE_DIR", capture_path, sizeof(capture_path));
    const std::filesystem::path captures(capture_path);
    if (!captures.empty()) std::filesystem::create_directories(captures);
    const auto capture = [&](const std::string& name) {
        if (captures.empty()) return;
        auto* data = ImGui::GetDrawData();
        nlohmann::json output{{"size", {data->DisplaySize.x, data->DisplaySize.y}},
            {"lists", nlohmann::json::array()}};
        for (const auto* list : data->CmdLists) {
            nlohmann::json vertices = nlohmann::json::array(), indices = nlohmann::json::array(), commands = nlohmann::json::array();
            for (const auto& vertex : list->VtxBuffer)
                vertices.push_back({vertex.pos.x, vertex.pos.y, vertex.uv.x, vertex.uv.y, vertex.col});
            for (auto index : list->IdxBuffer) indices.push_back(index);
            for (const auto& command : list->CmdBuffer) {
                if (command.UserCallback || !command.ElemCount) continue;
                commands.push_back({{"clip", {command.ClipRect.x, command.ClipRect.y, command.ClipRect.z, command.ClipRect.w}},
                    {"count", command.ElemCount}, {"index", command.IdxOffset}, {"vertex", command.VtxOffset}});
            }
            output["lists"].push_back({{"vertices", vertices}, {"indices", indices}, {"commands", commands}});
        }
        std::ofstream(captures / (name + ".json")) << output.dump();
        unsigned char* pixels; int width, height;
        io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);
        std::ofstream(captures / "font.json") << nlohmann::json({{"width",width},{"height",height}}).dump();
        std::ofstream(captures / "font.rgba", std::ios::binary).write(reinterpret_cast<char*>(pixels), size_t(width) * height * 4);
    };
    for (int width : {360, 700, 1100}) {
        context.panel = {};
        context.panel.mode = phoenix_ui::phoenix::ours;
        context.panel.controls.enabled = true;
        context.processing_enabled = nullptr;
        context.portable_backbuffer = false;
        context.input_capture = true;
        context.direct_game = true;
        context.real_guides = true;
        context.frame_services_available = false;
        context.loaded_model_id = "original";
        context.active_model_source = context.panel.model_source;
        context.active_model_id = context.panel.model_id;
        context.active_model_folder.clear(); context.active_model_dll.clear();
        context.direct_runner = phoenix_ui::Preview{{1920,1080,1280,720,1280,720}, "Spatial"};
        context.last_mode = phoenix_ui::phoenix::ours;
        context.timing_provenance = "UI fixture: no GPU samples";
        context.inference_process = false;
        context.active_route = "Native D3D12 / game DLSS textures";
        context.active_device = "Game graphics device";
        context.processing_active = true;
        context.screenshot_pair_available=true;
        context.status = "UI fixture: no game or inference running";
        context.error.clear();
        context.save = [&] { ++saves; };
        context.setup_details = [&] { ++adapter_views; ImGui::TextWrapped("Transport details from the active adapter."); };
        context.performance_details = [&] { ImGui::TextUnformatted("Adapter execution details"); };
        const auto initial = snapshot();
        const auto original_style = ImGui::GetStyle();
        io.DisplaySize = {float(width), 900};
        const std::string window = "Dashboard " + std::to_string(width);
        const auto frame = [&] {
            ++frames;
            widgets.clear();
            ImGui::NewFrame();
            ImGui::SetNextWindowPos({0,0});
            ImGui::SetNextWindowSize(io.DisplaySize);
            ImGui::Begin(window.c_str(), nullptr, ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove);
            const auto stack = ImGui::GetCurrentWindow()->IDStack.Size;
            record_widgets = true;
            // Mirror both production adapters: metadata/action views are new
            // every frame; only the settings and ImGui window state persist.
            auto frame_context = context;
            frame_context.requested_page.reset();
            const bool changed = phoenix_ui::draw_controls(frame_context);
            record_widgets = false;
            require(ImGui::GetCurrentWindow()->IDStack.Size == stack, "ID stack leaked into ReShade");
            require(ImGui::GetStyle().FrameRounding == original_style.FrameRounding &&
                    ImGui::GetStyle().ItemSpacing.y == original_style.ItemSpacing.y &&
                    ImGui::GetStyle().WindowPadding.x == original_style.WindowPadding.x &&
                    std::memcmp(ImGui::GetStyle().Colors,original_style.Colors,sizeof(original_style.Colors)) == 0,
                    "Local styling escaped the panel");
            ImGui::End();
            ImGui::Render();
            const auto footer = std::find_if(widgets.begin(), widgets.end(), [](const Widget& item) {
                return item.label.find("/metrics_") != std::string::npos;
            });
            const auto* footer_window = footer == widgets.end() ? nullptr : ImGui::FindWindowByName(footer->label.c_str());
            require(footer_window && footer_window->Pos.y > 700 && footer_window->Pos.y + footer_window->Size.y <= 900,
                "Timing footer escaped the visible parent window");
            require(!changed && snapshot() == initial, "Navigation changed rendering settings");
        };
        const auto bounds = [&](const char* name) {
            auto found = std::find_if(widgets.begin(), widgets.end(), [&](const Widget& item) { return item.label == name; });
            if (found == widgets.end()) {
                capture("failure");
                std::string labels; for (const auto& widget : widgets) if (!widget.label.empty()) labels += widget.label + " | ";
                throw std::runtime_error(std::string("Dashboard missing ") + name + ": " + labels);
            }
            require(found != widgets.end(), (std::string("Missing dashboard widget: ") + name).c_str());
            return found->rect;
        };
        const auto click = [&](const char* name) {
            frame(); frame();
            auto rect = bounds(name);
            const bool fixed = std::strcmp(name,"Save configuration")==0 || std::strcmp(name,"Advanced details")==0 ||
                std::strcmp(name,"Play")==0 || std::strcmp(name,"Image")==0 || std::strcmp(name,"Performance")==0 || std::strcmp(name,"Setup")==0;
            if (!fixed) {
                const float footer_top = width < 420 ? 755.f : 835.f;
                for (int attempt=0; attempt<8 && (rect.Max.y > footer_top || rect.Min.y < 370); ++attempt) {
                    io.AddMousePosEvent(float(width)*.75f,footer_top-60);
                    io.AddMouseWheelEvent(0,rect.Max.y > footer_top ? -2.f : 2.f);
                    frame(); frame(); frame(); frame();
                    rect = bounds(name);
                }
            }
            require(rect.Min.x >= 0 && rect.Max.x <= width && rect.Min.y >= 0 && rect.Max.y < 900,
                    (std::string("Control outside panel: ") + name + " " + std::to_string(rect.Min.x) + "," + std::to_string(rect.Min.y)).c_str());
            io.AddMousePosEvent(rect.GetCenter().x, rect.GetCenter().y); frame();
            io.AddMouseButtonEvent(0, true); frame();
            io.AddMouseButtonEvent(0, false); frame(); frame(); frame(); frame();
        };
        frame(); frame();
        click("Save configuration");
        require(saves == unsigned(width == 360 ? 1 : width == 700 ? 2 : 3), "Save callback was not called exactly once");
        const auto save_rect = bounds("Save configuration");
        click("Models and runner setup");
        bounds("NVIDIA DLL search folder");
        click("Play");
        capture("play-default-"+std::to_string(width));
        click("Skin and automatic mask");
        bounds("Skin structure");
        capture("play-" + std::to_string(width));
        for (const char* tab : {"Image", "Performance", "Setup", "Play"}) {
            click(tab);
            const auto next_save = bounds("Save configuration");
            require(save_rect.Min.y == next_save.Min.y, "Save moved with tab contents");
            if (std::strcmp(tab,"Image") == 0) bounds("Advanced image controls");
            if (std::strcmp(tab,"Performance") == 0) bounds("Clear GPU timing");
            if (std::strcmp(tab,"Setup") == 0) bounds("NVIDIA DLL search folder");
            if (width == 700) capture(std::string(tab) + "-700");
        }
        if (width==700) {
            // Reach lower cards with ordinary wheel events, as a user would.
            io.AddMousePosEvent(600,700); io.AddMouseWheelEvent(0,-8);
            frame(); frame(); frame(); frame();
            capture("play-details-700");
        }
        click("Play");
        const auto before_pair=paired_actions;
        for(int scroll=0;scroll<12 && std::none_of(widgets.begin(),widgets.end(),[](const Widget& item){return item.label=="Save original + modified";});++scroll) {
            io.AddMousePosEvent(float(width)*.75f,650);io.AddMouseWheelEvent(0,-2);frame();frame();frame();frame();
        }
        click("Save original + modified");
        require(paired_actions==before_pair+1,"Paired screenshot button did not dispatch exactly once");
        context.screenshot_pair_available=false;
        click("Save original + modified");
        require(paired_actions==before_pair+1,"Unavailable pair capture dispatched an action");
        context.screenshot_pair_available=true;
        click("Advanced details");
        click("Setup");
        require(adapter_views > 0, "Adapter-specific Setup controls disappeared");
        // Scroll the body while the header stays in the parent window.
        const auto* save_widget = &*std::find_if(widgets.begin(), widgets.end(), [](const Widget& w) {return w.label=="Save configuration";});
        const auto header_y = save_widget->rect.Min.y;
        io.AddMousePosEvent(float(width) * .4f, 600); io.AddMouseWheelEvent(0,-8);
        frame(); frame();
        require(bounds("Save configuration").Min.y == header_y, "Body scrolling moved Save");
        // Saved transport failures remain readable, without inventing a running state.
        context.processing_active = false;
        context.route_notice = "Saved route unavailable. Open Setup to choose an eligible route.";
        context.error = "Example missing model file";
        frame(); frame();
        if (width == 700) capture("attention-700");
        context.error.clear(); context.route_notice.clear();
    }
    context.setup_details = {}; context.performance_details = {}; context.save = {};
    return {{"passed",true},{"checks",checks},{"frames",frames},{"widths",{360,700,1100}},
            {"save_actions",saves},{"paired_screenshot_actions",paired_actions},{"gpu_executed",false}};
}
