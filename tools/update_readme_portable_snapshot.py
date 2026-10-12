"""Refresh README measurements from verified portable benchmark receipts.

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
    parser.add_argument("--measurements", type=Path, action="append", required=True,
                        help="Repeat for multiple receipts; later receipts replace earlier routes")
    parser.add_argument("--measured-date", help="Measurement date, if different from snapshot date")
    parser.add_argument("--regression", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    regression = json.loads(args.regression.read_text(encoding="utf-8"))
    if not regression.get("passed") or not regression.get("inference_validated"):
        parser.error("Successful matched measurement and inference regression receipts required")
    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.measurements]
    for source in sources:
        if not source.get("passed") or source.get("resolution") != [1920, 1080]:
            parser.error("Successful 1080p measurement receipts required")
    data = copy.deepcopy(json.loads(args.previous.read_text(encoding="utf-8")))
    previous_date = data["date"]
    data["date"] = args.date
    for row in data["records"]:
        row.setdefault("measured_date", previous_date)
    updated = set()
    for source in sources:
        for row in data["records"]:
            samples = [item for item in source["measurements"]
                       if item["route"] == row["id"] and item["variant"] == "batch512"]
            if not samples:
                continue
            required_processes = 3 if row["id"] in ("cuda", "vulkan_nvidia") else 2
            if len(samples) < required_processes or len({item["process"] for item in samples}) != len(samples):
                parser.error("Independent candidate processes required per updated route")
            if any(item["exact_frames"] != source["frames_per_process"] for item in samples):
                parser.error("Incomplete exact-output run")
            row.update(
                measured_date=args.measured_date or args.date, processes=len(samples),
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
                frames_per_process=source["frames_per_process"],
                discarded_per_process=source["discarded_per_process"],
                exact_frames=len(samples) * source["frames_per_process"],
            )
            updated.add(row["id"])
    if not updated:
        parser.error("No candidate route measurements found")
    data["validation"]["portable_regression_frames"] = regression["exact_frames"]
    data["validation"]["portable_regression_routes"] = len(regression["validated_devices"])
    data["validation"]["portable_benchmark_frames"] = sum(
        row["exact_frames"] for row in data["records"] if row["id"] in updated
    )
    data["validation"]["portable_measured_frames"] = sum(
        row["samples"] for row in data["records"] if row["id"] in updated
    )
    data["evidence_sha256"] = {
        "matched_measurements": [hashlib.sha256(path.read_bytes()).hexdigest() for path in args.measurements],
        "regression_manifest": hashlib.sha256(args.regression.read_bytes()).hexdigest(),
    }
    data["scope"]["portable"] = (
        "Per-route process/sample counts and measurement dates are recorded in each row. "
        "Medians of process medians, excluding warmup. Fractional "
        "motion (+0.375,-0.625), eight global blocks, one pass. Full call includes CPU "
        "preparation/history, transfers, inference, resolve and broker overhead; startup "
        "and file I/O excluded. GPU model time uses grouped GPU timing; CPU model time "
        "uses host timing. "
        "Native uses a different fixture/boundary. These are not in-game FPS results."
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "measurements.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
