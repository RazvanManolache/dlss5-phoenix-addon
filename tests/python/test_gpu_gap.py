"""Model-free checks for the opt-in GPU timestamp investigation."""
from pathlib import Path
import sys
import unittest

# Keep test discovery ahead of tools/test_*.py modules with the same names.
sys.path.append(str(Path(__file__).resolve().parents[2]/'tools'))
from investigate_gpu_gap import STAGES, DEEP_STAGES, validate_segments


class GpuGapTests(unittest.TestCase):
    def fixture(self,names=STAGES):
        return dict(passed=True,process_exit_code=0,gpu_samples_ms=[len(names)*.1],
            gpu_stages={name:{} for name in names},
            gpu_stage_samples_ms={name:[.1] for name in names})

    def test_shallow_and_deep_partitions_conserve_gpu_time(self):
        validate_segments(self.fixture())
        validate_segments(self.fixture(DEEP_STAGES))

    def test_failed_probe_rejected(self):
        case=self.fixture();case['passed']=False
        with self.assertRaisesRegex(ValueError,'failed'):validate_segments(case)

    def test_nonzero_process_exit_rejected(self):
        case=self.fixture();case['process_exit_code']=1
        with self.assertRaisesRegex(ValueError,'failed'):validate_segments(case)

    def test_missing_segment_samples_rejected(self):
        case=self.fixture();case['gpu_stage_samples_ms']['engine']=[]
        with self.assertRaisesRegex(ValueError,'Incomplete'):validate_segments(case)

    def test_negative_and_nonfinite_intervals_rejected(self):
        for invalid in (-.1,float('nan'),float('inf')):
            case=self.fixture();case['gpu_stage_samples_ms']['engine']=[invalid]
            with self.assertRaisesRegex(ValueError,'Invalid'):validate_segments(case)

    def test_nonconserving_partition_rejected(self):
        case=self.fixture();case['gpu_samples_ms']=[.8]
        with self.assertRaisesRegex(ValueError,'accounting'):validate_segments(case)

    def test_unlabelled_extra_interval_rejected(self):
        case=self.fixture();case['gpu_stage_samples_ms']['unknown']=[.1]
        with self.assertRaisesRegex(ValueError,'Incomplete'):validate_segments(case)


if __name__=='__main__':unittest.main()
