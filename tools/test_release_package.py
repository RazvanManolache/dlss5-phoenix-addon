import unittest
from prepare_addon_release import safe_payload


class PackageGuards(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
