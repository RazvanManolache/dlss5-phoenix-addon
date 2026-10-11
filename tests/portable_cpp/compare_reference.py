"""Offline numerical comparison only; NumPy is not a runtime dependency."""

from pathlib import Path
import sys, json, numpy as np

folder = Path(sys.argv[1])
run = Path(sys.argv[2])
report = json.loads((run / "report.json").read_text(encoding="utf-8"))
rows = []
count = int(sys.argv[3]) if len(sys.argv) > 3 else len(report["cases"])
assert 0 < count <= len(report["cases"]), "Invalid validation frame count"
for index, case in enumerate(report["cases"][:count]):
    expected = np.fromfile(folder / f"oracle_{index}.bin", dtype="f4")
    actual = np.fromfile(run / f"{index}.bin", dtype="f4")
    assert expected.size == actual.size and expected.size > 0, "Output extent mismatch"
    difference = np.abs(expected - actual)
    row = dict(
        name=case["name"],
        equal=bool(np.array_equal(actual, expected)),
        max_abs=float(difference.max()),
        mean_abs=float(difference.mean()),
        changed=int(np.count_nonzero(difference)),
        finite=bool(np.isfinite(actual).all()),
        history_used=case.get("history_used"),
        ms=case["call_ms"],
    )
    rows.append(row)
    print(row)
result = dict(
    rows=rows,
    max_abs=max(comparison["max_abs"] for comparison in rows),
    passed=all(comparison["finite"] and comparison["max_abs"] < 2e-5 for comparison in rows),
)
(run / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
raise SystemExit(0 if result["passed"] else 1)
