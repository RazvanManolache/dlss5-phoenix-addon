"""Refresh current CUDA/Vulkan README measurements from a verified matched run.

Only aggregate measurements are published. Model paths, local runtime paths and
captured inputs remain in the caller's private evidence directory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import statistics


def median(rows, key):
    return statistics.median(row[key] for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    source = json.loads(args.measurements.read_text(encoding="utf-8"))
    regression = json.loads(args.regression.read_text(encoding="utf-8"))
    if not source.get("passed") or not regression.get("passed") or not regression.get("inference_validated"):
        parser.error("Successful matched measurement and inference regression receipts required")
    if source.get("resolution") != [1920, 1080]:
        parser.error("This chart snapshot requires the 1080p fixture")
    data = copy.deepcopy(json.loads(args.previous.read_text(encoding="utf-8")))
    previous_date = data["date"]
    data["date"] = args.date
    for row in data["records"]:
        row.setdefault("measured_date", previous_date)
        if row["id"] not in ("cuda", "vulkan_nvidia"):
            continue
        samples = [item for item in source["measurements"]
                   if item["route"] == row["id"] and item["variant"] == "batch512"]
        if len(samples) != 3 or len({item["process"] for item in samples}) != 3:
            parser.error("Three independent candidate processes required per updated route")
        if any(item["exact_frames"] != source["frames_per_process"] for item in samples):
            parser.error("Incomplete exact-output run")
        row.update(
            measured_date=args.date, processes=len(samples),
            samples=len(samples) * (source["frames_per_process"] - source["discarded_per_process"]),
            core_ms=median(samples, "model_ms"), wall_ms=median(samples, "frame_ms"),
            process_median_min_ms=min(item["frame_ms"] for item in samples),
            process_median_max_ms=max(item["frame_ms"] for item in samples),
            host_private_bytes=median(samples, "host_private_bytes"),
            host_working_set_bytes=median(samples, "host_working_set_bytes"),
            owned_bytes=median(samples, "resident_bytes"),
            memory_parts={"Packed weights": median(samples, "resident_bytes") - median(samples, "scratch_bytes"),
                          "Scratch / activations": median(samples, "scratch_bytes")},
            memory_sampling="Worker working set/private bytes: median of final-frame observations across processes; not a peak",
            motion_mode=source["motion_mode"], source_commit=args.source_commit,
        )
    data["validation"]["portable_regression_frames"] = regression["exact_frames"]
    data["validation"]["portable_regression_routes"] = len(regression["validated_devices"])
    data["validation"]["portable_temporal_exact_frames_20261012"] = source["exact_frames"]
    data["validation"]["portable_benchmark_frames"] = sum(
        row["processes"] * source["frames_per_process"]
        for row in data["records"] if row["id"] in ("cuda", "vulkan_nvidia")
    )
    data["validation"]["portable_measured_frames"] = sum(
        row["samples"] for row in data["records"] if row["id"] in ("cuda", "vulkan_nvidia")
    )
    data["evidence_sha256"] = {
        "matched_measurements": hashlib.sha256(args.measurements.read_bytes()).hexdigest(),
        "regression_manifest": hashlib.sha256(args.regression.read_bytes()).hexdigest(),
    }
    data["scope"]["portable"] = (
        "CUDA/Vulkan RTX refreshed 12 October: three fresh broker processes per route, "
        "10 frames each; discard the first two. Medians of process medians. Fractional "
        "motion (+0.375,-0.625), eight global blocks, one pass. Full call includes CPU "
        "preparation/history, transfers, inference, resolve and broker overhead; startup "
        "and file I/O excluded. Default grouped GPU timing. Other portable rows retain "
        "11 October measurements, two processes with three measured history frames each. "
        "Native uses a different fixture/boundary. These are not in-game FPS results."
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "measurements.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
