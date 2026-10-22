"""Keep externally supplied models out of generated x86 bridge packages."""
import json
from pathlib import Path
import tempfile
import unittest
from build_wow64_runtime import validate_native_bundle


class ModelFreePackageTests(unittest.TestCase):
    def test_declared_model_free_bundle_accepts_metadata_and_kernel(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle=Path(temporary)
            (bundle/'native_manifest.json').write_text(json.dumps({'model_weights_included':False,'captured_plans_included':False}))
            (bundle/'kernel.cubin').write_bytes(b'kernel fixture')
            validate_native_bundle(bundle)

    def test_model_files_are_rejected_even_with_a_clean_manifest(self):
        for name in ('weights.bin','model.bin','nvngx_dlssnr.dll'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                bundle=Path(temporary)
                (bundle/'native_manifest.json').write_text(json.dumps({'model_weights_included':False,'captured_plans_included':False}))
                nested=bundle/'nested';nested.mkdir();(nested/name).write_bytes(b'not a distributable input')
                with self.assertRaisesRegex(ValueError,'Model files'):validate_native_bundle(bundle)

    def test_missing_and_model_bearing_declarations_are_rejected(self):
        for declaration in ({},{'model_weights_included':True,'captured_plans_included':False},
            {'model_weights_included':False,'captured_plans_included':True}):
            with self.subTest(declaration=declaration), tempfile.TemporaryDirectory() as temporary:
                bundle=Path(temporary);(bundle/'native_manifest.json').write_text(json.dumps(declaration))
                with self.assertRaisesRegex(ValueError,'model-free'):validate_native_bundle(bundle)


if __name__=='__main__':unittest.main()
