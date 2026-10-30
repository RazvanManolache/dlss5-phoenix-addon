"""Protect proportional baselines and attribution labels used by the review."""
import unittest
from report_native_scaling import excess_ms, group_kernel


class NativeScalingProfile(unittest.TestCase):
    def test_pixel_proportional_growth_has_zero_excess(self):
        self.assertEqual(excess_ms(2., 4.5, 1280*720, 1920*1080), 0.)

    def test_fixed_overhead_does_not_get_called_superlinear(self):
        self.assertLess(excess_ms(2., 3., 100, 200), 0.)

    def test_small_padding_step_is_positive_excess(self):
        self.assertGreater(excess_ms(4., 4.4, 1920*1024, 1920*1032), .3)

    def test_boundaries_are_not_mislabelled_as_local_attention(self):
        self.assertEqual(group_kernel('cc_tinlayout_fused_pre_block_swin_1h_32_1_ds_fp8'), 'boundary')
        self.assertEqual(group_kernel('cc_tinlayout_fused_post_block_swin_1h_32_fp8'), 'boundary')
        self.assertEqual(group_kernel('cc_tinlayout_fused_swin_8h_256_8_chained_fp8'), 'local_128_256')
        self.assertEqual(group_kernel('cc_vit_1d_attention_chained_fp8'), 'global_attention')
        with self.assertRaises(ValueError): group_kernel('unrecognized_new_kernel')


if __name__ == '__main__': unittest.main()
