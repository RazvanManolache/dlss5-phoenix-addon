"""Guard the multi-resolution experiment's input and reconstructed control."""
from pathlib import Path
import hashlib
import tempfile
import unittest

import numpy as np
from benchmark_tuning_resolutions import ROOT, control_source, fixture
from report_tuning_resolutions import cells, fixture_sha, mixed


class TuningResolutions(unittest.TestCase):
    def test_failed_cell_cannot_publish_a_successful_time(self):
        report=dict(decisions=[dict(extent='7680x4320',style='rtx20',model='original',
                                    passed=False,control_ms=123.,candidate_ms=100.,
                                    errors=['Out of memory'])])
        row=cells(report)[0]
        self.assertFalse(row['passed'])
        self.assertNotIn('control_ms',row)
        self.assertNotIn('candidate_ms',row)
        self.assertEqual(row['errors'],['Out of memory'])

    def test_mixed_pair_marker_does_not_hide_individual_regressions(self):
        self.assertTrue(mixed(dict(paired_reductions_percent=[2.,1.,-.1])))
        self.assertFalse(mixed(dict(paired_reductions_percent=[2.,1.,.1])))

    def test_chunked_fixture_matches_original_full_grid_bit_for_bit(self):
        # An odd extent crossing two chunk boundaries catches row offset,
        # broadcasting and precision mistakes without allocating an 8K image.
        width,height=259,257
        y,x=np.mgrid[:height,:width].astype(np.float32)
        expected=np.ones((height,width,4),np.float16)
        expected[:,:,0]=x/width
        expected[:,:,1]=y/height
        expected[:,:,2]=.5+.25*np.sin(x*.17)*np.cos(y*.11)
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as temporary:
            work=Path(temporary)
            fixture(work,width,height)
            self.assertEqual((work/'color.rgba16f').read_bytes(),expected.tobytes())
            self.assertEqual((work/'motion.rg16f').read_bytes(),bytes(width*height*4))
            self.assertEqual(fixture_sha(width,height),hashlib.sha256(expected.tobytes()).hexdigest())

    def test_control_has_scalar_reads_immediate_waits_and_old_register_limit(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'build') as temporary:
            work=Path(temporary)/'production'
            pins=control_source(work)
            self.assertTrue(pins)
            self.assertNotIn('common/fp8_weight_values.cuh',pins)
            math=(work/'common/architecture_math.cuh').read_text()
            self.assertIn('decode_fp8_fragment(b[i], right_low[i], right_high[i]);',math)
            self.assertNotIn('decode_fp8_weight_fragment',math)
            memory=(work/'common/device_memory_scalar.cuh').read_text()
            self.assertNotIn('const uint4 packed',memory)
            copies=(work/'common/compatibility_copy.cuh').read_text()
            self.assertIn('cp.async.commit_group; cp.async.wait_group 0;',copies)
            self.assertNotIn('#if PHOENIX_STYLE_ARCH == 890',copies)
            attention=(work/'optimized/window_attention/channels_64/attention.cu').read_text()
            self.assertIn('__maxnreg__(128)',attention)
            self.assertNotIn('__maxnreg__(160)',attention)


if __name__=='__main__':
    unittest.main()
