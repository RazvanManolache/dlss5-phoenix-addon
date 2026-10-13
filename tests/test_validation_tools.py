"""Release validation must fail closed on missing, partial or mismatched evidence."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import validate_native
import validate_runners


class ValidationGateTests(unittest.TestCase):
    def setUp(self):
        self.job = {
            "width": 96,
            "height": 64,
            "config": {"device": "cpu"},
            "cases": [{"name": "reset", "expected_sha256": "a" * 64}],
        }
        self.report = {
            "passed": True,
            "python_or_torch_loaded": False,
            "device": "cpu",
            "width": 96,
            "height": 64,
            "cases": [{"name": "reset", "baseline_exact": True, "output_sha256": "a" * 64}],
        }

    def test_no_jobs_cannot_create_validation_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "No validation jobs"):
                validate_runners.validate(root, root / "runner.exe", root, root / "output")
            self.assertFalse((root / "validation_manifest.json").exists())

    def test_empty_or_missing_reference_cannot_pass(self):
        for cases in ([], [{"name": "reset"}], [{"expected_sha256": "short"}]):
            with self.subTest(cases=cases), self.assertRaises(ValueError):
                validate_runners.check_job({**self.job, "cases": cases})

    def test_exact_complete_report_passes(self):
        validate_runners.check_job(self.job)
        validate_runners.check_report(self.job, self.report, 0)

    def test_partial_wrong_device_hash_or_module_audit_fails(self):
        mutations = [
            {"cases": []},
            {"device": "cuda:other"},
            {"width": 1920},
            {"python_or_torch_loaded": True},
            {"passed": False},
            {"cases": [{"name": "reset", "baseline_exact": True, "output_sha256": "b" * 64}]},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_runners.check_report(self.job, {**self.report, **mutation}, 0)
        with self.assertRaises(ValueError):
            validate_runners.check_report(self.job, self.report, 1)

    def test_native_cpu_geometry_only_is_not_inference_validation(self):
        reference = {"width": 1920, "height": 1080, "output_hashes": {"0": "a", "127": "b"}}
        report = {
            "passed": True,
            "gpu_executed": True,
            "output_finite": True,
            "debug_errors": 0,
            "modules_retired": 41,
            "frames": [{}] * 128,
            **reference,
        }
        validate_native.check_report(reference, report)
        for change in (
            {"gpu_executed": False},
            {"frames": []},
            {"debug_errors": 1},
            {"output_hashes": {"0": "a", "127": "wrong"}},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_native.check_report(reference, {**copy.deepcopy(report), **change})

    def test_native_resolution_selection_requires_correct_entries_and_both_modes(self):
        manifest = {"kernels": [{"name": "attention", "entry": "small", "large_input_entry": "large",
                                 "large_input_min_pixels": 18662400}]}
        small = {"width": 1920, "height": 1080,
                 "kernel_selections": [{"name": "attention", "entry": "small"}]}
        large = {"width": 5760, "height": 3240,
                 "kernel_selections": [{"name": "attention", "entry": "large"}]}
        coverage = validate_native.check_kernel_selections(manifest, small)
        with self.assertRaisesRegex(ValueError, "both default and large"):
            validate_native.require_variant_coverage(manifest, coverage)
        coverage.update(validate_native.check_kernel_selections(manifest, large))
        validate_native.require_variant_coverage(manifest, coverage)
        for changed in ({"kernel_selections": []},
                        {"kernel_selections": small["kernel_selections"] * 2},
                        {"kernel_selections": large["kernel_selections"]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_native.check_kernel_selections(manifest, {**small, **changed})
        modified = copy.deepcopy(manifest)
        modified["kernels"][0]["large_input_min_pixels"] += 1
        self.assertNotEqual(validate_native.kernel_selection_digest(manifest),
                            validate_native.kernel_selection_digest(modified))


if __name__ == "__main__":
    unittest.main()
