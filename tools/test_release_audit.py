"""Privacy checks must reject secrets/host paths without printing their values."""
import unittest
from audit_release import validate_public_content


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

    def test_allows_attribution_public_paths_and_urls(self):
        for value in ('Copyright 2026 Razvan Manolache', 'C:/Users/Public/example',
                      'docs/EXECUTION_MECHANISMS.md',
                      'https://github.com/RazvanManolache/dlss5-phoenix-addon'):
            validate_public_content('fixture.txt', value.encode())


if __name__ == '__main__':
    unittest.main()
