"""Regenerate small synthetic runner inputs and jobs; no model extraction or Torch."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "tests/fixtures/runner_regressions.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output, runtime, weight_source, routes, baseline=BASELINE):
    import numpy as np  # Offline fixture generation only; not a runtime dependency.

    reference = json.loads(baseline.read_text(encoding="utf-8"))
    if weight_source.get("kind") != "nvidia_dll":
        raise ValueError("Saved regressions require the pinned NVIDIA DLL weight source")
    model = Path(weight_source["path"]).resolve()
    if digest(model) != reference["model_sha256"]:
        raise ValueError("Model differs from the saved reference; supply matching regression data")
    weight_source = {**weight_source, "path": str(model), "sha256": reference["model_sha256"]}
    known = {job["route"] for job in reference["jobs"]}
    if not routes or set(routes) - known:
        raise ValueError("Select nonempty routes from: " + ", ".join(sorted(known)))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    jobs = output / "jobs"
    jobs.mkdir()
    for specification in reference["jobs"]:
        route = specification["route"]
        if route not in routes:
            continue
        width, height = specification["width"], specification["height"]
        planes = output / specification["fixture"]
        planes.mkdir(exist_ok=True)
        # PCG64 and cast order match the saved regression inputs byte for byte.
        color = (
            np.random.Generator(np.random.PCG64(1059))
            .uniform(0.05, 0.8, (height, width, 3))
            .astype("<f4")
        )
        depth = np.broadcast_to(np.linspace(0.1, 0.9, width, dtype="<f4"), (height, width))
        validity = np.ones((height, width), dtype="<f4")
        validity[: height // 4, : width // 4] = 0
        cases = []
        for case in specification["cases"]:
            fractional = case["name"] == "fractional_motion_mask"
            motion = np.zeros((height, width, 2), dtype="<f4")
            if fractional:
                motion[..., 0], motion[..., 1] = 0.25, -0.125
            values = {"color": color, "motion": motion, "depth": depth, "validity": validity}
            task = {key: case[key] for key in ("name", "metadata", "expected_sha256")}
            task["save_output"] = False
            for plane, expected in case["planes"].items():
                # Share identical planes between cases and routes to avoid disk bloat.
                name = "fractional_motion" if plane == "motion" and fractional else plane
                path = planes / (name + ".bin")
                if not path.exists():
                    values[plane].tofile(path)
                if digest(path) != expected:
                    raise ValueError("Regenerated input differs from reference: " + str(path))
                task[plane] = str(path)
            cases.append(task)
        config = {
            "runtime_root": str(runtime.resolve()),
            "allow_unvalidated": True,
            "execution": "broker",
            "backend": specification["backend"],
            "device": routes[route],
            "weight_source": weight_source,
            "cpu_threads": 0,
            "private_opencl_cpu": "opencl_cpu/intelocl64.dll",
        }
        job = {"width": width, "height": height, "config": config, "cases": cases}
        (jobs / (specification["fixture"] + "_" + route + ".json")).write_text(
            json.dumps(job, indent=2) + "\n", encoding="utf-8"
        )
    print(jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--weight-source", required=True, type=Path)
    parser.add_argument(
        "--routes",
        required=True,
        type=Path,
        help="JSON mapping reference route names to discovered device IDs",
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    args = parser.parse_args()
    prepare(
        args.output,
        args.runtime,
        json.loads(args.weight_source.read_text(encoding="utf-8")),
        json.loads(args.routes.read_text(encoding="utf-8")),
        args.baseline,
    )


if __name__ == "__main__":
    main()
