"""Run nonempty golden-hash jobs and certify only the routes that actually pass."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binary_hashes(runtime):
    suffixes = {".dll", ".exe", ".spv", ".rtl", ".cfg", ".o"}
    return {
        path.relative_to(runtime).as_posix(): digest(path)
        for path in runtime.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    }


def check_job(job):
    cases = job.get("cases", [])
    if not cases or job.get("width", 0) <= 0 or job.get("height", 0) <= 0:
        raise ValueError("Validation requires dimensions and nonempty reference cases")
    for case in cases:
        if not re.fullmatch(r"[a-f0-9]{64}", case.get("expected_sha256", "")):
            raise ValueError("Every case requires a saved expected_sha256")


def check_report(job, report, returncode):
    expected = job["cases"]
    actual = report.get("cases", [])
    if returncode or not report.get("passed") or report.get("python_or_torch_loaded") is not False:
        raise ValueError(
            "Runner failed inference or runtime module audit: " + str(report.get("error"))
        )
    if len(actual) != len(expected) or report.get("device") != job["config"]["device"]:
        raise ValueError("Runner returned incomplete cases or a different device")
    if (report.get("width"), report.get("height")) != (job["width"], job["height"]):
        raise ValueError("Runner returned different dimensions")
    for source, result in zip(expected, actual):
        if (
            result.get("name") != source["name"]
            or result.get("baseline_exact") is not True
            or result.get("output_sha256") != source["expected_sha256"]
        ):
            raise ValueError("Runner output differs from the saved reference")


def validate(runtime, runner, jobs, output, timeout=180):
    runtime, runner, jobs, output = [path.resolve() for path in (runtime, runner, jobs, output)]
    originals = sorted(jobs.glob("*.json"))
    if not originals:
        raise ValueError("No validation jobs found")
    tasks = [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in originals]
    for _, job in tasks:
        check_job(job)
    if (runtime / "validation_manifest.json").exists():
        raise ValueError(
            "Use a fresh package; existing validation must not be silently overwritten"
        )
    output.mkdir(parents=True, exist_ok=False)
    before = binary_hashes(runtime)
    if not before:
        raise ValueError("Runtime package has no binaries")
    runner_hash = digest(runner)
    rows = []
    receipt = {
        "schema": "phoenix-portable-cpp-validation/1",
        "passed": False,
        "inference_validated": False,
        "python_required": False,
        "jobs": rows,
    }
    evidence = output / "portable_validation.json"
    try:
        for name, job in tasks:
            # Validation explicitly opts into execution of an uncertified fresh package.
            job["config"].update(runtime_root=str(runtime), allow_unvalidated=True)
            path = output / (name + ".json")
            path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            started = time.monotonic()
            print("Checking " + name, flush=True)
            with (output / (name + ".log")).open("wb") as log:
                result = subprocess.run(
                    [str(runner), str(path), str(output / name)],
                    cwd=runtime,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            report = json.loads((output / name / "report.json").read_text(encoding="utf-8"))
            check_report(job, report, result.returncode)
            rows.append(
                {
                    "name": name,
                    "device": job["config"]["device"],
                    "resolution": [job["width"], job["height"]],
                    "exact": True,
                    "frames": len(report["cases"]),
                    "seconds": time.monotonic() - started,
                    "job_sha256": digest(path),
                    "report_sha256": digest(output / name / "report.json"),
                }
            )
            evidence.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if binary_hashes(runtime) != before or digest(runner) != runner_hash:
            raise ValueError("Runtime or test runner changed during validation")
        devices = sorted({row["device"] for row in rows})
        config_path = runtime / "phoenix_native_cpp.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(validated_devices=devices, allow_unvalidated=False)
        config.pop("weight_source", None)
        receipt.update(
            passed=True,
            inference_validated=True,
            validated_devices=devices,
            resolutions=sorted({tuple(row["resolution"]) for row in rows}),
            exact_frames=sum(row["frames"] for row in rows),
            binaries=before,
            runner_sha256=runner_hash,
            scope="Exact saved Phoenix regression hashes on the listed devices/resolutions only",
        )
        # Publish only after every requested case has passed. A failure leaves no success manifest.
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        evidence.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        receipt["evidence_sha256"] = digest(evidence)
        (runtime / "validation_manifest.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        receipt.update(passed=False, inference_validated=False, error=str(error))
        evidence.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    print(
        "Validated "
        + str(receipt["exact_frames"])
        + " frames on "
        + str(len(devices))
        + " route(s)"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    validate(args.runtime, args.runner, args.jobs, args.output, args.timeout)


if __name__ == "__main__":
    main()
