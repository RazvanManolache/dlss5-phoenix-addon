import unittest
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from prepare_addon_release import ROOT, input_package, safe_payload, sha, package_document_links, VERSION
from validate_addon_release import inspect_payload


class PackageGuards(unittest.TestCase):
    def test_package_documentation_keeps_local_docs_and_pins_missing_assets(self):
        files = {'README.md': b'[Install](docs/INSTALLATION.md) [Source](src/example.cpp)\n',
                 'docs/INSTALLATION.md': b'![UI](features/example.png)\n```md\n[Example](src/root.cpp)\n```\n',
                 'docs/legacy.md': b'[License](../LICENSE) uses 2\xd73 pixels', 'LICENSE': b'License'}
        package_document_links(files)
        self.assertIn(b'(docs/INSTALLATION.md)', files['README.md'])
        self.assertIn(('/blob/' + VERSION + '/src/example.cpp').encode(), files['README.md'])
        self.assertIn(('raw.githubusercontent.com/RazvanManolache/dlss5-phoenix-addon/' + VERSION).encode(), files['docs/INSTALLATION.md'])
        self.assertIn(b'[Example](src/root.cpp)', files['docs/INSTALLATION.md'])
        self.assertIn('2×3', files['docs/legacy.md'].decode('utf-8'))

    def test_nested_core_identity_and_retained_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root/'candidate'; candidate.mkdir()
            (candidate/'phoenix_graphics_host.exe').write_bytes(b'fresh helper')
            (candidate/'Phoenix.addon64').write_bytes(b'public addon with extra helper resources')
            core = b'verified core addon'
            payload = {'compatibility/Phoenix.addon64':core,
                       'compatibility/phoenix_graphics_host.exe':b'fresh helper',
                       'native/native_graph.json':b'unchanged graph'}
            def fake_resources(binary, destination):
                entries = []
                for identity,(name,raw) in enumerate(payload.items(),1100):
                    (destination/f'{identity}.bin').write_bytes(raw)
                    entries.append(dict(path=name,resource=identity,algorithm='raw',bytes=len(raw),sha256=sha(raw)))
                (destination/'1001.bin').write_text(json.dumps(dict(files=entries)))
            baseline = {'native/native_graph.json':sha(b'unchanged graph')}
            with patch('validate_addon_release.resources',side_effect=fake_resources):
                self.assertEqual(inspect_payload(root/'unused',root/'valid',candidate,baseline,sha(core)),baseline)
                with self.assertRaisesRegex(ValueError,'Nested x64 addon'):
                    inspect_payload(root/'unused',root/'badcore',candidate,baseline,sha((candidate/'Phoenix.addon64').read_bytes()))
                del payload['native/native_graph.json']
                with self.assertRaisesRegex(ValueError,'inventory'):
                    inspect_payload(root/'unused',root/'missing',candidate,baseline,sha(core))
                (candidate/'phoenix_graphics_host.exe').write_bytes(b'wrong helper')
                with self.assertRaisesRegex(ValueError,'graphics helper'):
                    inspect_payload(root/'unused',root/'badhelper',candidate,{},sha(core))

    def test_rejects_unsafe_names_and_payloads(self):
        for name in ('../outside', '/outside', 'C:outside', 'a\\outside', 'run.log',
                     'capture.bmp', 'nvngx_dlssnr.dll', 'models/weights.bin'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_payload(name, b'example')

    def test_safe_public_payload(self):
        safe_payload('PhoenixRuntime/wow64/bootstrap.json', b'{"schema":"example"}')
        safe_payload('licenses/WOW64Ext/lgpl-3.0.txt', b'License text')

    def test_legacy_source_location_exception_is_narrow(self):
        separator = bytes([92])
        root = separator.join([b'E:', b'Games', b'Reshade', b'dlss5-phoenix-addon'])+separator
        safe_payload('runtime/phoenix_cpp_devices.exe', b'MZ'+root+b'native_runtime/source.cpp')
        for name, raw in [('diagnostics.json', b'MZ'+root),
                          ('runtime/phoenix_cpp_devices.exe', root),
                          ('runtime/phoenix_cpp_devices.exe', b'MZ'+separator.join([b'C:', b'Users', b'private', b'x']))]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_payload(name, raw)

    def test_input_companion_requires_bound_binaries_sources_and_smoke_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            binaries = {}
            for bits in (32, 64):
                name = f'PhoenixInput.addon{bits}'
                raw = ('public test payload '+str(bits)).encode()
                (folder/name).write_bytes(raw)
                binaries[name] = sha(raw)
            receipt = dict(passed=True, binaries=binaries, source_inputs={
                name: sha((ROOT/name).read_bytes()) for name in (
                    'src/integration/input/phoenix_input_addon.cpp',
                    'tools/build_input_addon.py', 'tests/native/input_addon_load_tests.cpp')},
                cases={str(bits): {key: True for key in ('passed', 'actual_dll_loaded',
                    'balanced_attach_detach', 'mailbox_status', 'stale_session_rejected')}
                    for bits in (32, 64)})

            def check(value):
                (folder/'VALIDATION.json').write_text(json.dumps(value))
                return input_package(folder)

            files, _ = check(receipt)
            self.assertIn('PhoenixInput.addon32', files)
            self.assertIn('PhoenixInput.addon64', files)
            self.assertNotIn('Phoenix.addon64', files)
            for kind in ('failed', 'wrong_binary', 'missing_architecture', 'changed_source', 'missing_smoke'):
                damaged = copy.deepcopy(receipt)
                if kind == 'failed':
                    damaged['passed'] = False
                elif kind == 'wrong_binary':
                    damaged['binaries']['PhoenixInput.addon32'] = '0'*64
                elif kind == 'missing_architecture':
                    del damaged['binaries']['PhoenixInput.addon64']
                elif kind == 'changed_source':
                    damaged['source_inputs']['tools/build_input_addon.py'] = '0'*64
                else:
                    damaged['cases']['32']['stale_session_rejected'] = False
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    check(damaged)


if __name__ == '__main__':
    unittest.main()
