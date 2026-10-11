"""Collect validated local C++ runner evidence without publishing test inputs."""

from pathlib import Path
import argparse
import hashlib
import json
import statistics

ROOT = Path(__file__).resolve().parents[1]
ROUTES = [
    ("cpu", "CPU C++", "CPU", "Ryzen 9 9950X3D", "Ordered AVX2/FMA"),
    (
        "cuda",
        "CUDA C++",
        "CUDA · RTX 5080",
        "RTX 5080",
        "FP8-weight Tensor Core GEMMs; ordered FP16 fallback; cached CUDA graph",
    ),
    (
        "vulkan_nvidia",
        "Vulkan C++ · NVIDIA",
        "Vulkan · RTX 5080",
        "RTX 5080",
        "Generic ordered Vulkan",
    ),
    (
        "vulkan_amd",
        "Vulkan C++ · AMD",
        "Vulkan · AMD iGPU",
        "AMD Radeon integrated gfx1036",
        "Generic ordered Vulkan",
    ),
    (
        "hip_amd",
        "HIP C++ · AMD",
        "HIP · AMD iGPU",
        "AMD Radeon integrated gfx1036",
        "Generic ordered tiled HIP",
    ),
    (
        "sycl_cpu",
        "SYCL C++ · CPU",
        "SYCL · Ryzen CPU",
        "Ryzen 9 9950X3D; Intel OpenCL CPU ICD",
        "Ordered 8 by 8 CPU tiles; C++ broker",
    ),
]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/PORTABLE_CPP_PERFORMANCE.json")
    parser.add_argument(
        "--build-report", type=Path, default=ROOT / "build/cpp1059_release/build_report.json"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Previous independently checked checkpoint for byte-exact regression",
    )
    parser.add_argument(
        "--baseline-report", type=Path, help="Preserved measurements before this optimization pass"
    )
    args = parser.parse_args()
    evidence = args.evidence.resolve()
    checks = read(evidence / "numerical_matrix.json")
    regression = read(evidence / "validation.json") if args.baseline else None
    if regression:
        assert len(regression) == 18 and all(row["passed"] and row["exact"] for row in regression)
        assert sum(row["validated_frames"] for row in regression) == 90
    offload = read(evidence / "offload_matrix.json")
    broker_offload = read(evidence / "offload_release_matrix.json")
    bridge = read(evidence / "bridge_host_final/bridge_test.json")
    live_bridge = read(evidence / "bridge_host_live/bridge_test.json")
    contracts = read(evidence / "client_contracts_final.json")
    weights = read(evidence / "weights_final.json")
    assert len(checks) == 18 and all(row["passed"] for row in checks)
    assert len(offload) == 6 and all(row["passed"] for row in offload)
    assert len(broker_offload) == 6 and all(row["passed"] for row in broker_offload)
    assert bridge["passed"] and contracts["passed"] and weights["passed"]
    assert live_bridge["passed"] and live_bridge["stable_device_persisted_after_restart"]
    records, pins = [], {}
    for name, label, short, hardware, engine in ROUTES:
        report_path = evidence / "hd_a" / f"final_{name}" / "report.json"
        report = read(report_path)
        job = read(evidence / "hd_a" / f"final_{name}.json")
        assert report["passed"] and not report["python_or_torch_loaded"]
        indices = job["benchmark_indices"]
        frames = [report["cases"][index] for index in indices]
        average = lambda field: statistics.mean(frame[field] for frame in frames)
        last = report["cases"][-1]
        if last.get("cpu_threads"):
            engine += (
                f"; {last['cpu_threads']} CPU workers ({last['cpu_threads_available']} available)"
            )
        layers = {}
        for frame in frames:
            for layer in frame["layers"]:
                key = (layer["block"], layer["layer"])
                layers[key] = layers.get(key, 0) + layer["ms"] / len(frames)
        stages = dict.fromkeys(
            ["Stem", "Encoder", "Global blocks", "Decoder", "Post blend", "Unattributed"], 0.0
        )
        for (block, _), milliseconds in layers.items():
            stage = (
                "Stem"
                if block == 0
                else "Encoder"
                if 1 <= block <= 30
                else "Global blocks"
                if 31 <= block <= 38
                else "Decoder"
                if 39 <= block <= 69
                else "Post blend"
                if block == 70
                else "Unattributed"
            )
            stages[stage] += milliseconds
        parts = {
            "Preparation / history": average("prepare_ms"),
            "Execution / transfers": average("execute_with_transfers_ms"),
            "Resolve": average("resolve_ms"),
        }
        parts["Other C++ / broker"] = average("call_ms") - sum(parts.values())
        assert parts["Other C++ / broker"] >= 0
        assert abs(sum(stages.values()) - average("operator_ms")) < 0.01
        record = dict(
            id=name,
            label=label,
            short_label=short,
            hardware=hardware,
            engine=engine,
            device=report["device"],
            operator_ms=average("operator_ms"),
            frame_ms=average("call_ms"),
            graph_bytes=last["resident_bytes"],
            host_private_bytes=last["memory"]["private_bytes"],
            host_working_set_bytes=last["memory"]["working_set_bytes"],
            host_peak_working_set_bytes=last["memory"]["peak_working_set_bytes"],
            caller_memory=last["caller_memory"],
            samples=len(frames),
            benchmark_indices=indices,
            frame_min_ms=min(frame_record["call_ms"] for frame_record in frames),
            frame_max_ms=max(frame_record["call_ms"] for frame_record in frames),
            initialization_ms=report["initialization_ms"],
            first_frame_ms=report["cases"][0]["call_ms"],
            operator_clock=last["operator_clock"],
            worker_parts=parts,
            operator_parts=stages,
            command_count=last.get("command_count"),
            timing_intervals=last.get("timing_intervals"),
            cpu_threads=last.get("cpu_threads"),
            cpu_threads_requested=last.get("cpu_threads_requested"),
            cpu_threads_available=last.get("cpu_threads_available"),
            provider_transfer_buffer_bytes=last.get("vulkan_transfer_buffer_bytes"),
            memory_parts={
                "Packed weights": last["resident_weight_bytes"],
                "Scratch / activations": last["resident_scratch_bytes"],
            },
            layers=[
                dict(block=layer_key[0], layer=layer_key[1], ms=layer_time)
                for layer_key, layer_time in sorted(layers.items())
            ],
            execution="broker" if name == "sycl_cpu" else "in_process",
            loaded_modules=report["modules"],
            python_or_torch_loaded=False,
        )
        assert sum(record["memory_parts"].values()) == record["graph_bytes"]
        records.append(record)
        pins[report_path.relative_to(ROOT).as_posix()] = digest(report_path)
    native_path = ROOT / "docs/NATIVE_NVIDIA_MEASUREMENTS.json"
    native = read(native_path)
    references = []
    for key, label in [("native", "DLSS5 Phoenix native RTX"), ("nvidia", "NVIDIA DLSS NR")]:
        route = native["routes"][key]
        references.append(
            dict(
                label=label,
                short_label=label,
                operator_ms=route["gpu_ms"]["mean"],
                host_private_bytes=route["private_mib"]["mean"] * 2**20,
                host_working_set_bytes=route["working_set_mib"]["mean"] * 2**20,
                hardware="RTX 5080",
                samples=route["samples"],
                scope="Fresh captured-input D3D12 GPU interval, excluding upload/readback; different input and boundary from portable complete frames",
            )
        )
    pins[native_path.relative_to(ROOT).as_posix()] = digest(native_path)
    noise = read(evidence / "original_noise_comparison.json")
    validation = dict(
        frames=90,
        max_abs=max(row["max_abs"] for row in checks),
        matched_noise_required=True,
        original_numpy_noise=noise if isinstance(noise, list) else noise["rows"],
        model_tensors=658,
        offload=offload,
        broker_offload=broker_offload,
        bridge_passed=True,
        settings_persisted=bridge["stable_device_persisted_after_restart"],
        contracts=contracts,
        cpp_only_live_configuration_accepted=True,
        scopes=[
            "Graph and temporal equivalence with C++ noise injected into the reference",
            "Separate unmodified NumPy-noise comparison is NOT equivalent",
            "Physical RTX 5080 D3D12 offload; AMD compute tested across devices",
            "Mailbox/settings in owned ReShade host; no live-game visual claim",
        ],
    )
    if regression:
        validation.update(
            method="90 fresh byte-exact regressions against the prior independently checked matched-noise outputs",
            regression_max_abs=0,
            independent_reference_checkpoint=str(args.baseline),
            independent_reference_error_is_inherited=True,
        )
        thread_path = evidence / "thread_comparison.json"
        if thread_path.exists():
            validation.update(
                thread_control_fresh_frames=30,
                unchanged_gpu_frames_inherited_from_same_pass=60,
                method="90 byte-exact regressions across this optimization checkpoint: 30 fresh CPU-thread checks, 60 unchanged GPU checks retained from the same pass",
            )
        pins[(evidence / "validation.json").relative_to(ROOT).as_posix()] = digest(
            evidence / "validation.json"
        )
    for file in [
        "numerical_matrix.json",
        "original_noise_comparison.json",
        "offload_matrix.json",
        "offload_release_matrix.json",
        "weights_final.json",
        "client_contracts_final.json",
        "bridge_host_final/bridge_test.json",
        "bridge_host_live/bridge_test.json",
    ]:
        path = evidence / file
        pins[path.relative_to(ROOT).as_posix()] = digest(path)
    data = dict(
        schema="phoenix-portable-cpp-performance/1",
        passed=True,
        measured_date="2026-10-10",
        backends=records,
        references=references,
        validation=validation,
        fixture=dict(
            width=1920,
            height=1080,
            working_width=1920,
            working_height=1088,
            passes=1,
            repeats=[1] * 8,
            seed=0,
            input="Fixed synthetic RGB32F",
        ),
        hardware=dict(
            cpu="AMD Ryzen 9 9950X3D",
            cores=16,
            threads=32,
            installed_ram_gib=96,
            gpu="NVIDIA GeForce RTX 5080",
            nvidia_driver="616.64",
            amd="AMD Radeon(TM) Graphics gfx1036",
            amd_driver="32.0.21045.5002",
            os="Windows 11 Pro 26200",
            cuda="13.1",
            intel_gpu_tested=False,
            sycl_cpu_icd="Intel OpenCL CPU 2026.6.0.0.0617; process-local forwarding library",
        ),
        model_source_sha256="e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e",
        evidence_sha256=pins,
        alternatives=read(evidence / "alternatives.json"),
        measured_binaries=read(evidence / "measured_binaries.json"),
        source_build=read(args.build_report),
        limitations=[
            "Small warm sample counts; desktop environment, no clock lock or game FPS measurement",
            "Host memory overlaps CPU/iGPU allocations and must not be summed with graph bytes",
            "Host preparation/transfers/resolve remain; no zero-copy claim",
            "CUDA/HIP/SYCL stage-event intervals include gaps between commands; prior per-command operator intervals have a different scope. Complete-frame boundaries are unchanged",
            "NumPy versus C++ noise math produces visible-scale numerical differences in the proxy output",
            "Production stale-result limits can reject slow outputs; probe age limits were relaxed",
        ],
    )
    if args.baseline:
        if args.baseline_report is None:
            parser.error("--baseline requires --baseline-report")
        previous = read(args.baseline_report)
        data["before_after"] = [
            dict(
                id=backend_record["id"],
                before_frame_ms=previous_record["frame_ms"],
                after_frame_ms=backend_record["frame_ms"],
                speedup=previous_record["frame_ms"] / backend_record["frame_ms"],
                before_graph_bytes=previous_record["graph_bytes"],
                after_graph_bytes=backend_record["graph_bytes"],
                before_host_private_bytes=previous_record["host_private_bytes"],
                after_host_private_bytes=backend_record["host_private_bytes"],
            )
            for backend_record in records
            for previous_record in previous["backends"]
            if previous_record["id"] == backend_record["id"]
        ]
    thread_path = evidence / "thread_comparison.json"
    if thread_path.exists():
        data["thread_comparison"] = read(thread_path)
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
