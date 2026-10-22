# Phoenix Input — optional background automation

The game gets the button presses. Your desktop keeps its mouse. Everybody wins, until the game ignores an input API we have not met yet.

Phoenix Input is a **separate experimental ReShade addon**, originally developed in the companion coverage harness. It is not needed by Phoenix DLSS5 rendering and does not improve image quality or FPS.

- Keyboard and text operations; mouse moves, buttons and wheel.
- Virtual XInput controller state and timed macros.
- Optional virtual focus and isolation of real keyboard/mouse input while the game is in the background.
- Local JSON mailbox with session matching and input-API counters.
- No `SendInput`, desktop cursor movement or foreground-window takeover in its injection path.

## Install / remove

Close the application. Copy **one** file, matching its architecture, beside your full-addon-support ReShade installation:

- `PhoenixInput.addon32` for x86.
- `PhoenixInput.addon64` for x64.

It may live beside `Phoenix.addon32/64`, but it is independent. For a custom ReShade `AddonPath`, use that addon directory. Remove the Input addon while the application is closed to uninstall it; the normal Phoenix addon can remain.

Virtual focus is off by default. Hooks are installed when the companion starts and pass through while virtual focus is off. To request virtual focus from startup, create `PhoenixInput.json` beside this addon:

```json
{"focus_override": true, "virtual_pad": false}
```

## Local mailbox

`PhoenixOutput/input/endpoint.json` appears beside the loaded addon and refreshes about once per second. It contains the current `session`, process, window and hook status. One local controller should own the mailbox at a time.

Write a request to a temporary file, then rename it to `PhoenixOutput/input/request.json`. Copy the **current** session from the endpoint:

```json
{
  "schema": "phoenix-input/1",
  "session": "COPY_CURRENT_ENDPOINT_SESSION",
  "id": "example-1",
  "ops": [
    {"op": "focus", "enabled": true},
    {"op": "key", "key": "space", "action": "tap", "hold_ms": 80, "at_ms": 100},
    {"op": "release_all", "at_ms": 300},
    {"op": "focus", "enabled": false, "at_ms": 350}
  ]
}
```

Read `response.json` and match its `id`. An accepted request means operations were scheduled, not that the game reacted or the macro finished. Use `at_ms` for offsets from request arrival. `status` is a no-input operation. Sessions change across process restarts; reread the endpoint instead of reusing a stale session. Requests are limited to 256 KiB and 4,096 operations.

This is a local file-control interface, not a network service or authenticated remote-control protocol. Use it with applications you intend to automate. ReShade and input hooks may be incompatible with anti-cheat-protected titles.

## Build and validation

From the source checkout, with Python 3 and Visual Studio C++ tools plus the Windows SDK:

```powershell
python tools/build_input_addon.py
```

The builder uses the repository's ReShade, nlohmann/json and MinHook dependencies, builds both architectures in temporary staging, and promotes only the binaries and a compact validation receipt into `artifacts/current/input-addon`. No model or NGX SDK is required.

Published checks cover actual load/unload, mailbox status and stale-session rejection on both architectures. They do **not** establish broad game compatibility, background gameplay or every input API. See [validation scope](VALIDATION_0.15.md) and [third-party notices](../THIRD_PARTY.md).
