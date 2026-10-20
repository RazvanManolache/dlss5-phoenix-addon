"""Model-free guards for the matched old-DLL release comparison."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools'))
from benchmark_release_ab import common_probe, round_order, summarize


class ReleaseComparisonTests(unittest.TestCase):
    def template(self):
        return (Path(__file__).resolve().parents[2]/'tests/native/four_model_probe.cpp').read_text()

    def test_shared_original_probe_keeps_gpu_timer_and_copy_boundaries(self):
        source = common_probe(self.template())
        for unsupported in ('network_schedule.h','NetworkSchedule','source.model_id',
                            'bundle->model_resource.name','phoenix_native::record_progress',
                            'phoenix_native::group_marker'):
            self.assertNotIn(unsupported,source)
        for anchor in ('gpu.command->EndQuery(gpu.queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,0)',
                       'gpu.command->EndQuery(gpu.queries.Get(),D3D12_QUERY_TYPE_TIMESTAMP,1)',
                       'gpu.command->CopyResource(shared.Get(),result)',
                       'Full fixed trained schedule required','Original model only'):
            self.assertIn(anchor,source)
        self.assertEqual(source,common_probe(self.template()))

    def test_changed_template_fails_closed(self):
        source = self.template().replace('#include "../../native_runtime/network_schedule.h"\n','')
        with self.assertRaisesRegex(ValueError,'anchor changed'):
            common_probe(source)

    def test_order_counterbalances_without_changing_cells(self):
        cells = [('released','310.8','phoenix'),('current','310.8','phoenix')]
        self.assertEqual(round_order(cells,0),cells)
        self.assertEqual(round_order(cells,1),list(reversed(cells)))
        self.assertEqual(round_order(cells,2),cells)

    def test_summary_keeps_build_dll_and_resolution_separate(self):
        def case(build,dll,value):
            return dict(build=build,dll=dll,engine='phoenix',extent=[1920,1080],
                        gpu={'median_ms':value},wall={'median_ms':value+1},create_ms=5)
        rows = summarize([case('released','310.8',1),case('released','310.8',3),
                          case('current','310.8',4),case('current','310.9.16',6)])
        self.assertEqual(len(rows),3)
        self.assertEqual(rows[0]['processes'],2)
        self.assertEqual(rows[0]['gpu']['mean_process_median_ms'],2)
        self.assertEqual(rows[0]['gpu']['process_range_ms'],[1,3])
        self.assertEqual(rows[1]['dll'],'310.8')
        self.assertEqual(rows[2]['dll'],'310.9.16')


if __name__ == '__main__':
    unittest.main()
