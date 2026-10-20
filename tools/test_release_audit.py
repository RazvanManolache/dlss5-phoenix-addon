"""Keep private files out without omitting maintained model architecture source."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import audit_release
from audit_release import validate_public_content, validate_payload, validate_source_path


class ReleasePrivacyTests(unittest.TestCase):
    def test_rejects_host_paths(self):
        for value in ('C:'+chr(92)+'Users'+chr(92)+'private'+chr(92)+'file',
                      'E:'+chr(92)+'Games'+chr(92)+'private',
                      'D:'+chr(92)+'Kit'+chr(92)+'private',
                      'C:'+'/Users/'+'private/file'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'fixture.txt') as failure:
                    validate_public_content('fixture.txt', value.encode())
                self.assertNotIn(value, str(failure.exception))

    def test_rejects_credentials(self):
        for value in ('gh'+'p_'+'A'*36, 'github_'+'pat_'+'B'*40,
                      '-----BEGIN '+'OPENSSH PRIVATE KEY-----'):
            with self.assertRaises(ValueError):
                validate_public_content('fixture.txt', value.encode())

    def test_rejects_device_identifiers_but_allows_synthetic_parser_examples(self):
        for prefix, suffix in [('cuda:GPU-', 'abcdef12-'+'3456-7890-abcd-ef1234567890'),
                               ('vulkan:', 'abcdef12'*4)]:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                validate_public_content('measurement.json', (prefix+suffix).encode())
        for value in ('cuda:GPU-01234567-89ab-cdef-0123-456789abcdef',
                      'vulkan:0123456789abcdef0123456789abcdef'):
            validate_public_content('parser_test.cpp', value.encode())

    def test_allows_attribution_public_paths_and_urls(self):
        for value in ('Copyright 2026 Razvan Manolache', 'C:/Users/Public/example',
                      'docs/EXECUTION_MECHANISMS.md',
                      'https://github.com/RazvanManolache/dlss5-phoenix-addon'):
            validate_public_content('fixture.txt', value.encode())


class ReleaseSourcePackageTests(unittest.TestCase):
    def test_model_architecture_source_is_publishable(self):
        for name in ('models/README.md', 'native_runtime/models/common/decoder.inc',
                     'native_runtime/models/original/graph.inc',
                     'native_runtime/models/small_fp8/weight_layouts.h',
                     'src/native/model/models/original/definition.h',
                     'src/native/model/models/small_fp8/native_graph.h'):
            with self.subTest(name=name):
                validate_source_path(name)

    def test_source_exception_does_not_allow_weight_payloads(self):
        for prefix in ('models/', 'native_runtime/models/original/',
                       'src/native/model/models/small_fp8/'):
            for name in ('weights.bin', 'weights.safetensors', 'weights.onnx',
                         'weights.json', 'model.pt', 'model.npz', 'nvngx_dlssnr.dll', 'run.log'):
                with self.subTest(name=prefix+name), self.assertRaises(ValueError):
                    validate_source_path(prefix+name)
        for name in ('other/models/definition.h', 'models/notes.md',
                     'native_runtime/models/original/models/weights.h',
                     'native_runtime/Models/original/weights.json'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_source_path(name)

    def test_private_directories_and_unsafe_paths_remain_blocked(self):
        for name in ('.vscode/settings.json', '.VSCODE/settings.json', '.idea/workspace.xml', 'runtime/config.json',
                     'src/native/model/models/original/captures/frame.h',
                     'native_runtime/models/original/archive/weights.h',
                     '../outside.h', '/outside.h', 'C:/outside.h', 'src\\outside.h'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_source_path(name)

    def test_model_source_extension_does_not_bypass_executable_magic(self):
        name = 'native_runtime/models/original/definition.h'
        validate_source_path(name)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'definition.h'
            for raw in (b'MZnot source', b'\x7fELFnot source'):
                path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, 'Binary executable payload'):
                    validate_payload(name, path)
            path.write_bytes(b'// Maintained architecture declarations.\n')
            self.assertEqual(len(validate_payload(name, path)), 64)

    def test_audit_keeps_all_architecture_sources_and_excludes_editor_settings(self):
        names = ['.VSCODE/settings.json', '.idea/workspace.xml',
                 'src/component/.vscode/settings.json', 'models/README.md',
                 'native_runtime/models/common/decoder.inc',
                 'src/native/model/models/original/definition.h']
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in names:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('Maintained source\n', encoding='utf-8')
            # Even a mistakenly tracked editor file must not enter a source ZIP.
            listed = ('\0'.join(names)+'\0').encode()
            with patch.object(audit_release, 'ROOT', root), \
                    patch.object(audit_release.subprocess, 'check_output', return_value=listed), \
                    patch.object(audit_release, 'production_inputs', return_value=({'kernels': []}, None)):
                manifest, found = audit_release.audit()
            expected = set(names[3:])
            self.assertEqual(set(found), expected)
            self.assertEqual(set(manifest['files']), expected)

    def test_model_payload_under_source_directory_still_fails_full_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'native_runtime/models/original/weights.json'
            path.parent.mkdir(parents=True)
            path.write_text('{}', encoding='utf-8')
            with patch.object(audit_release, 'ROOT', root), \
                    patch.object(audit_release, 'candidates', return_value=iter([path])), \
                    patch.object(audit_release, 'production_inputs', return_value=({'kernels': []}, None)), \
                    self.assertRaisesRegex(ValueError, 'Private model payload directory'):
                audit_release.audit()


if __name__ == '__main__':
    unittest.main()
