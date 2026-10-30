import copy
import unittest
from embed_runtime import require_native_packaging
from build_kernel_styles import STYLE_TARGETS

class KernelStyles(unittest.TestCase):
    def manifest(self):
        return dict(passed=True, selection_status='validated', model_weights_included=False,
                    architectures=['sm_120'], kernel_styles={
                        style: dict(architectures=targets, kernels=[{} for _ in range(41)], validation_pending=True)
                        for style, targets in STYLE_TARGETS.items()})

    def test_exact_lower_capability_matrix(self):
        self.assertEqual(STYLE_TARGETS, {'rtx20':['sm_86','sm_89','sm_120'],
                                      'rtx30':['sm_89','sm_120'], 'rtx40':['sm_120']})

    def test_standalone_style_cannot_masquerade_as_native_default(self):
        manifest = self.manifest()
        manifest['kernel_style'] = 'rtx20'
        with self.assertRaisesRegex(ValueError, 'native default'):
            require_native_packaging(manifest, experimental=True)

    def test_compile_evidence_does_not_grant_device_validation(self):
        manifest = self.manifest()
        require_native_packaging(manifest, experimental=True)
        with self.assertRaisesRegex(ValueError, 'pending'):
            require_native_packaging(manifest)

    def test_invalid_targets_and_incomplete_catalogue_refused(self):
        for mutate in (lambda row: row.update(architectures=['sm_75']),
                       lambda row: row.update(architectures=['sm_120','sm_120']),
                       lambda row: row.update(kernels=[])):
            manifest = copy.deepcopy(self.manifest())
            mutate(manifest['kernel_styles']['rtx20'])
            with self.assertRaises(ValueError): require_native_packaging(manifest, experimental=True)
