# ReShade overlay

The native, compatibility and x86 addons share one dashboard. ReShade supplies
the font and host window. The dashboard applies a local slate palette, rounded
cards, blue navigation, mint live controls, violet setup accents and amber
experimental controls. Style changes are restored when drawing ends; the compact
panel keeps the host theme colors.

## Everyday controls

- **Header:** Save configuration, enable, a tinted processing-status card, actual route,
  loaded model, geometry and actionable errors. Save retains the adapter's
  existing persistence behavior. Editing controls still autosaves.
- **Play:** output, Original/Lightweight selection, tone and structure, skin
  mask, comparisons, presets, reset history and quick resolution controls.
- **Image:** resolution and reconstruction, motion, processing chain,
  appearance, engine guides and temporal-history policy.
- **Performance:** pipeline responsiveness, network passes, approximate-block
  budgets, retained timing comparisons, GPU stages and worker details.
- **Setup:** model files, Browse and Validate, devices and CPU threads,
  execution mechanisms and eligibility, recovery, hotkeys and diagnostics.

The header includes three measured-summary tiles: neural GPU (or worker wall
time), capture-to-output age when supplied by the active pipeline, and the
loaded processing canvas. Unavailable timings remain `--`; the UI does not
enable monitoring itself. Source, Look, Resolution, Compare, Presets, image
controls, workload budgets, retained timings and setup details have separate
cards. Quick links move between tabs without editing rendering settings.

The body scrolls independently of Save and the measurement footer. Wide timing
tables can scroll horizontally. Narrow panels stack the footer measurements.
Advanced details is local presentation state, not an inference setting.

The Output selector replaces the old Backend label. Processing passes and
Network passes per frame previously edited the same setting; the editable
control now lives under Performance, with a summary in the processing drawer.
Timing clears require confirmation. Reset history still resets temporal state
immediately and does not remove timing comparisons.

## Availability and measurements

Current route capabilities continue to gate controls. Saved unavailable routes
remain inspectable; opening a tab does not select a fallback or overwrite a
recipe. Requested and loaded models are separate, including pending and failed
switches. The experimental x86 driver retains its Original-only restriction.

Game FPS counts presentation callbacks. Neural and addon GPU intervals have
different scopes; worker wall time is identified separately. Missing recent
measurements display `--`. Processing/submission status does not prove that a
game has displayed the final texture. Detailed output status remains under
Performance and a live visual check is still required.

## Source and checks

- `shared_panel.h`: dashboard/navigation and compact shared entry point.
- `panel_context.h`: adapter metadata and existing action/persistence contract.
- `panel_sections.h`: task renderers using the existing backing settings.
- `panel_chrome.h`: status, summary tiles and measurement footer.
- `panel_style.h`: scoped palette, rounded cards, headings and helper text.
- `panel_timing.h`: retained worker/GPU tables and stage breakdown.
- `pipeline_panel.h`: capability-gated motion, chain and latency groups.

CPU interaction checks use the exact matching ImGui 1.92.5 docking sources:

```powershell
python tools/test_shared_panel.py --imgui-source <matching-imgui-source-directory> --output build/ui-check
python tools/test_compatibility_panel_layout.py
```

The test exercises actual mouse events in both architectures, output/model
selection, live strength edits, disabled capability gates, confirmation and
Save callbacks. The dashboard checks all four tabs at 360, 700 and 1100 pixels,
body scrolling, fixed header/footer bounds, settings stability and balanced
style/ID stacks. The baseline inventory retains 136 persisted entries.

Setting `PHOENIX_UI_CAPTURE_DIR` exports the real ImGui vertices and font atlas.
`tools/render_panel_preview.py` rasterizes them on CPU for layout inspection;
these fixture previews contain no game, personal capture or GPU measurements.
They do not establish gameplay, graphics-driver or performance acceptance.
