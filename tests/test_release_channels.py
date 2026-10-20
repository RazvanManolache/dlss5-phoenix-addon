import unittest
from build_compatibility import release_flags


class ReleaseChannelTests(unittest.TestCase):
    def test_experimental_x86_inherits_the_core_marker(self):
        channel, flags = release_flags({'release_channel': 'experimental'})
        self.assertEqual(channel, 'experimental')
        self.assertIn('/DPHOENIX_EXPERIMENTAL_NATIVE=1', flags)

    def test_stable_and_legacy_builds_do_not_enable_experimental_targets(self):
        for report in ({}, {'release_channel': 'stable'}):
            self.assertEqual(release_flags(report), ('stable', ''))

    def test_unknown_channel_fails_closed(self):
        for value in ('', None, 'experimental /DOTHER=1'):
            with self.assertRaises(ValueError):
                release_flags({'release_channel': value})
