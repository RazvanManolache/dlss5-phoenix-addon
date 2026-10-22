# Model switching and resource retirement

The native game-input and in-process backbuffer paths share the same model
replacement transaction. A source, model id or alternate DLL search-folder
change invalidates the old selection. The old executor must be idle before a
candidate is built. Only a successfully constructed candidate is published;
the previous runner and authenticated bundle remain available on failure.
Candidate construction briefly overlaps allocations, but the two generations
never record inference concurrently.

The model displayed in the native panel and the model attached to GPU timing
rows come from the loaded source. A pending selection is not execution evidence.

NVIDIA features and refinement outputs retire after their last frame completes
when no displayed comparison source requests them. Shared runtime modules and
parameter storage remain cached. Driver-owned allocations are not included in
the Phoenix buffer ledger.

Each successful NVAPI initialization has one matching unload after the owning
network's modules and resources retire. Failed constructor setup also balances
its library reference. NVAPI initialization increments a reference counter;
see [NVIDIA's lifecycle documentation](https://docs.nvidia.com/nvapi/group__nvapifunctions.html).

The legacy output-status strings are part of the existing helper completion
protocol. Human-readable display qualifications must remain separate. A recorded
DLSS output texture is not proof that a retail game subsequently displays it.

## Targeted checks

CPU lifetime, rollback and protocol contracts, on x86 and x64:

```powershell
python tools/test_model_switch.py --output artifacts/current/model-switch-fixes
```

Add `--gpu --bundle <native-bundle> --original <original-dll> --lightweight
<lightweight-dll>` for six bounded RTX model/geometry/weight-transfer transitions
at 2560x1080. This checks actual launch counts, finite changed output, exact
Original output after switching back, released buffers and NVAPI reference balance.

For the complete addon, `tools/build_addon.py --test-set model-switch` selects
the focused contracts and software output tests. Its usual SDK, runtime, bundle
and output-name arguments still apply.

`tools/test_model_switch_live.py --help` describes the opt-in Museum D3D12 x64
test. It uses DLSS Everything's control bridge and checks final presented
screenshots. `--nvidia` also checks NVIDIA evaluation and retirement. The fixture
addon and configuration are restored in `finally`; retail installations are not
modified. Model DLLs stay external and are never copied into reports or packages.

Museum presentation tests do not qualify Resident Evil's engine-specific output
chain. A slowdown not reproduced by these bounded tests must remain unresolved
until measured in the affected game.
