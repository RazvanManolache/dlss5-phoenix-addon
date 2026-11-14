"""Run the isolated x86 DX9 to x64 D3D12 GPU texture handoff probe."""

import json
from pathlib import Path
import subprocess
import tempfile

from test_model_selection import ROOT, run, toolchain


def main():
    source = ROOT / "tests/native/d3d9_shared_texture_probe.cpp"
    with tempfile.TemporaryDirectory(prefix="phoenix-d3d9-shared-") as folder:
        work = Path(folder)
        for bits in (32, 64):
            compiler, env = toolchain(bits)
            run(
                [compiler, "/nologo", "/std:c++20", "/EHsc", "/O2", "/MT", source,
                 f"/Fo:{work / f'probe_{bits}.obj'}", f"/Fe:{work / f'probe_{bits}.exe'}",
                 "d3d9.lib", "d3d11.lib", "d3d12.lib", "dxgi.lib", "user32.lib"],
                cwd=work, env=env,
            )
        results = {}
        for mode in ("legacy", "ex"):
            probe = subprocess.run(
                [work / "probe_32.exe", f"--{mode}", work / "probe_64.exe"],
                cwd=work, text=True, capture_output=True, timeout=35, check=False,
            )
            results[mode] = json.loads(probe.stdout)
            if probe.stderr.strip():
                results[mode]["worker_diagnostics"] = probe.stderr.strip()
            if (probe.returncode == 0) != results[mode]["passed"]:
                raise RuntimeError(f"Inconsistent probe result for {mode}")
        print(json.dumps(results, indent=2))
        return 0 if results["ex"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
