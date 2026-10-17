import array
import unittest
from validate_x86_vulkan import stdcall_aliases, frame_comparison, validate_frame_rows, architecture_acceptance


class X86VulkanBuildTests(unittest.TestCase):
    def test_stdcall_aliases_use_only_compiler_undefined_vulkan_symbols(self):
        symbols = ('001 00000000 UNDEF notype () External | _vkCreateInstance@12\n'
                   '002 00000000 UNDEF notype () External | _vkCreateInstance@12\n'
                   '003 00000000 UNDEF notype () External | _vkMapMemory@32\n'
                   '004 00000000 SECT1 notype () External | _vkLocal@4\n'
                   '005 00000000 UNDEF notype () External | _Other@12\n')
        self.assertEqual(stdcall_aliases(symbols), ['_vkCreateInstance@12', '_vkMapMemory@32'])

    def test_x64_functions_do_not_get_x86_aliases(self):
        self.assertEqual(stdcall_aliases('001 UNDEF notype () External | vkCreateInstance\n'), [])

    def test_exact_frame_bytes(self):
        pixels = array.array('f', [0.125, 0.5, 1.]).tobytes()
        result = frame_comparison(pixels, pixels)
        self.assertTrue(result['exact_bytes'])
        self.assertEqual(result['differing_samples'], 0)
        self.assertEqual(result['max_abs_error'], 0)

    def test_numeric_and_bitwise_comparison_are_separate(self):
        result = frame_comparison(array.array('f', [0.]).tobytes(), array.array('f', [-0.]).tobytes())
        self.assertFalse(result['exact_bytes'])
        self.assertEqual(result['differing_samples'], 0)

    def test_mismatched_samples_are_reported(self):
        result = frame_comparison(array.array('f', [0., 1.]).tobytes(), array.array('f', [0., .5]).tobytes())
        self.assertFalse(result['exact_bytes'])
        self.assertEqual(result['differing_samples'], 1)
        self.assertEqual(result['max_abs_error'], .5)
        self.assertEqual(result['mean_abs_error'], .25)
        self.assertGreater(result['max_ulp_error'], 1)

    def test_invalid_extents_are_rejected(self):
        for left, right in ((b'', b''), (b'123', b'123'), (b'1234', b'12345678')):
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                frame_comparison(left, right)

    def test_nonfinite_pixels_are_rejected_even_if_identical(self):
        for value in (float('nan'), float('inf')):
            pixels = array.array('f', [value]).tobytes()
            with self.assertRaises(ValueError):
                frame_comparison(pixels, pixels)

    def frame_rows(self):
        return [dict(width=w, height=h, frame=f, pointer_bits=32, backend='vulkan-cpp',
                     output_changed=True, command_count=1317, history_used=f != 0)
                for w, h in ((64, 64), (129, 97), (1920, 1080)) for f in range(3)]

    def test_complete_native_temporal_frame_evidence(self):
        rows = self.frame_rows()
        byte_count = sum(row['width']*row['height']*12 for row in rows)
        validate_frame_rows(rows, 32, byte_count)
        for bad, size in ((rows[:-1], byte_count), (rows, byte_count-4),
                          ([rows[0]]*9, byte_count)):
            with self.assertRaises(ValueError):
                validate_frame_rows(bad, 32, size)

    def test_wrong_architecture_or_backend_is_not_native_x86_evidence(self):
        for field, value in (('pointer_bits', 64), ('backend', 'helper'),
                             ('output_changed', False), ('history_used', True), ('command_count', 0)):
            rows = self.frame_rows()
            byte_count = sum(row['width']*row['height']*12 for row in rows)
            rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_frame_rows(rows, 32, byte_count)

    def traced_rows(self):
        rows = self.frame_rows()
        for row in rows:
            row['validation_stages'] = [{key: '0'*64 for key in (
                'color', 'conditioning_history', 'conditioning_validity', 'resolve_history',
                'resolve_validity', 'noise', 'network_head', 'next_history', 'resolve_gate', 'display_output')}]
        return rows

    def test_host_rounding_requires_exact_neural_and_history_evidence(self):
        left, right = self.traced_rows(), self.traced_rows()
        right[8]['validation_stages'][0].update(resolve_gate='1'*64, display_output='2'*64)
        comparison = dict(exact_bytes=False, max_ulp_error=1)
        result = architecture_acceptance(comparison, left, right)
        self.assertEqual(result['classification'], 'neural_exact_host_gate_two_ulp')
        self.assertFalse(result['byte_exact_display'])
        for key in ('network_head', 'next_history', 'noise', 'resolve_history'):
            wrong = self.traced_rows()
            wrong[8]['validation_stages'][0][key] = '3'*64
            with self.subTest(key=key), self.assertRaises(ValueError):
                architecture_acceptance(comparison, left, wrong)

    def test_oversized_or_unexplained_display_difference_fails(self):
        left, right = self.traced_rows(), self.traced_rows()
        right[8]['validation_stages'][0]['display_output'] = '2'*64
        with self.assertRaises(ValueError):
            architecture_acceptance(dict(exact_bytes=False, max_ulp_error=1), left, right)
        right[8]['validation_stages'][0]['resolve_gate'] = '1'*64
        with self.assertRaises(ValueError):
            architecture_acceptance(dict(exact_bytes=False, max_ulp_error=3), left, right)
        with self.assertRaises(ValueError):
            architecture_acceptance(dict(exact_bytes=True, max_ulp_error=0), self.frame_rows(), self.frame_rows())

    def test_adjacent_float32_values_are_one_ulp(self):
        left = array.array('I', [0x3f000000, 0xbf000000]).tobytes()
        right = array.array('I', [0x3f000001, 0xbf000001]).tobytes()
        self.assertEqual(frame_comparison(left, right)['max_ulp_error'], 1)


if __name__ == '__main__':
    unittest.main()
