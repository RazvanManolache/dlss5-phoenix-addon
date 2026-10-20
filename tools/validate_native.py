"""Validate a native bundle against saved Phoenix synthetic-input regression hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from prepare_native_fixture import prepare

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_single_architecture(manifest):
    """This legacy saved-hash validator has evidence from one device only."""
    if len(manifest.get("architectures", [])) > 1:
        raise ValueError("Single-device validation cannot certify a multi-architecture bundle; "
                         "validate each target separately before packaging")


def bundle_hashes(bundle, manifest):
    hashes = {"native_graph.json": digest(bundle / "native_graph.json")}
    if hashes["native_graph.json"] != manifest["native_graph_sha256"]:
        raise ValueError("Graph differs from build manifest")
    if len(manifest["kernels"]) != 41:
        raise ValueError("Expected 41 selected production kernels")
    for kernel in manifest["kernels"]:
        path = (bundle / kernel["binary"]).resolve()
        if not path.is_relative_to(bundle):
            raise ValueError("Kernel escapes bundle")
        hashes[kernel["binary"]] = digest(path)
        if hashes[kernel["binary"]] != kernel["binary_sha256"]:
            raise ValueError("Kernel differs from build manifest")
    return hashes


def kernel_selection_digest(manifest):
    """Bind validation to the entries and thresholds, as well as module bytes."""
    fields = ("name", "entry", "parameter_bytes", "block", "large_input_entry", "large_input_min_pixels")
    policies = [{key: kernel[key] for key in fields if key in kernel}
                for kernel in sorted(manifest["kernels"], key=lambda kernel: kernel["name"])]
    return hashlib.sha256(json.dumps(policies, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_kernel_selections(manifest, report):
    variants = [kernel for kernel in manifest["kernels"] if "large_input_entry" in kernel]
    if not variants:
        return set()
    selections = report.get("kernel_selections", [])
    actual = {selection["name"]: selection["entry"] for selection in selections}
    if len(actual) != len(manifest["kernels"]) or len(actual) != len(selections):
        raise ValueError("Missing or duplicate native kernel selection evidence")
    pixels = report["width"] * report["height"]
    coverage = set()
    for kernel in manifest["kernels"]:
        large = "large_input_entry" in kernel and pixels >= kernel["large_input_min_pixels"]
        expected = kernel["large_input_entry"] if large else kernel["entry"]
        if actual.get(kernel["name"]) != expected:
            raise ValueError("Native probe selected the wrong resolution kernel")
        if "large_input_entry" in kernel:
            coverage.add((kernel["name"], "large" if large else "default"))
    return coverage


def require_variant_coverage(manifest, coverage):
    for kernel in manifest["kernels"]:
        if "large_input_entry" in kernel and not all((kernel["name"], mode) in coverage
                                                    for mode in ("default", "large")):
            raise ValueError("Validate both default and large-input kernel entries before packaging")


def check_report(reference, report):
    if (
        report.get("passed") is not True
        or report.get("gpu_executed") is not True
        or report.get("output_finite") is not True
        or report.get("debug_errors") != 0
        or report.get("modules_retired") != 41
        or len(report.get("frames", [])) != 128
    ):
        raise ValueError("Native probe did not complete 128 valid GPU frames")
    if (report.get("width"), report.get("height")) != (reference["width"], reference["height"]):
        raise ValueError("Native probe used a different extent")
    if report.get("output_hashes") != reference["output_hashes"]:
        raise ValueError("Native reset/history outputs differ from saved Phoenix reference")


def validate(bundle, probe, weight_source, output, baseline):
    bundle, probe, weight_source, output, baseline = [
        path.resolve() for path in (bundle, probe, weight_source, output, baseline)
    ]
    reference = json.loads(baseline.read_text(encoding="utf-8"))
    cases = reference.get("cases", [])
    if not cases or any(not case.get("output_hashes") for case in cases):
        raise ValueError("Nonempty saved native reference cases required")
    model = json.loads(weight_source.read_text(encoding="utf-8"))
    if (
        model.get("kind") != "nvidia_dll"
        or digest(Path(model["path"])) != reference["model_sha256"]
    ):
        raise ValueError("Native reference requires the pinned model DLL")
    manifest_path = bundle / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("passed") or manifest.get("selection_status") != "compiled":
        raise ValueError("Validate a fresh, successfully compiled native bundle")
    require_single_architecture(manifest)
    before = bundle_hashes(bundle, manifest)
    selection_digest = kernel_selection_digest(manifest)
    variant_coverage = set()
    probe_hash = digest(probe)
    output.mkdir(parents=True, exist_ok=False)
    receipt = {
        "passed": False,
        "inference_validated": False,
        "cases": [],
        "scope": "Saved Phoenix reset/history output hashes; no NVIDIA DLSS parity claim",
    }
    try:
        for case in cases:
            name = f"{case['width']}x{case['height']}"
            prefix = output / name / "input"
            hashes = prepare(prefix, case["width"], case["height"])
            if hashes != case["input_files"]:
                raise ValueError("Generated native inputs differ from reference")
            destination = output / name / "result"
            with (output / (name + ".log")).open("wb") as log:
                subprocess.run(
                    [str(probe), str(bundle), str(prefix), str(destination), str(weight_source)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                    timeout=300,
                )
            report_path = destination / "result.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            check_report(case, report)
            variant_coverage.update(check_kernel_selections(manifest, report))
            receipt["cases"].append(
                {
                    "width": case["width"],
                    "height": case["height"],
                    "exact": True,
                    "frames": len(report["frames"]),
                    "report_sha256": digest(report_path),
                }
            )
        require_variant_coverage(manifest, variant_coverage)
        if (bundle_hashes(bundle, manifest) != before or digest(probe) != probe_hash
                or kernel_selection_digest(json.loads(manifest_path.read_text(encoding="utf-8"))) != selection_digest):
            raise ValueError("Native bundle or probe changed during validation")
        receipt.update(
            passed=True,
            inference_validated=True,
            binaries=before,
            baseline_sha256=digest(baseline),
            probe_sha256=probe_hash,
            kernel_selection_sha256=selection_digest,
            resolution_variant_coverage=[{"name": name, "mode": mode}
                                         for name, mode in sorted(variant_coverage)],
        )
        evidence = output / "native_validation.json"
        evidence.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        manifest.update(
            selection_status="validated",
            validation_pending=False,
            validation=receipt,
            validation_evidence_sha256=digest(evidence),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    except Exception as error:
        receipt.update(error=str(error), passed=False, inference_validated=False)
        (output / "native_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
        raise
    print("Native reference outputs matched at " + str(len(cases)) + " extents")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--weight-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "tests/fixtures/native_regressions.json"
    )
    args = parser.parse_args()
    validate(args.bundle, args.probe, args.weight_source, args.output, args.baseline)


if __name__ == "__main__":
    main()
