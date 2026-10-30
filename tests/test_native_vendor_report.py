"""Protect median headlines and the publication boundary for paired GPU studies."""
import copy
import unittest
from benchmark_native_vendor import aggregate
from report_native_vendor import require_complete, value, pixel_axis


class NativeVendorReport(unittest.TestCase):
    def test_resolution_axis_preserves_actual_pixel_ratios(self):
        axis=pixel_axis([(640,360),(1280,720),(1920,1080)])
        self.assertEqual(axis[1]/axis[0],4.)
        self.assertEqual(axis[2]/axis[1],2.25)

    def data(self):
        report=dict(passed=True,state='completed',extents=['1920x1080'],rounds=3,
                    comparisons=[dict(extent='1920x1080',model=model,exact=True) for model in ('original','small-fp8')],
                    records=[dict(route=engine,model=model,extent=[1920,1080])
                             for model in ('original','small-fp8') for engine in ('phoenix','nvidia')],
                    cases=[dict(passed=True,gpu=dict(median_ms=1.),wall=dict(median_ms=2.)) for _ in range(12)])
        styles=dict(passed=True,state='completed',records=[dict(model=model,style=style)
                    for model in ('original','small-fp8') for style in ('rtx20','rtx30','rtx40')])
        return report,styles

    def test_headline_uses_process_median_instead_of_an_outlier_sensitive_mean(self):
        rows=[dict(gpu=dict(median_ms=t),wall=dict(median_ms=t+1),
                   memory=dict(device_local_bytes=128,scope='test'),create_ms=2.) for t in (1.,1.2,100.)]
        result=aggregate(rows)
        self.assertEqual(value(result),1.2)
        self.assertGreater(result['gpu']['mean_process_median_ms'],30.)

    def test_complete_matrix_is_accepted(self):
        require_complete(*self.data())

    def test_missing_or_duplicate_cells_are_rejected(self):
        report,styles=self.data()
        report['records'][0]=copy.deepcopy(report['records'][1])
        with self.assertRaisesRegex(ValueError,'Duplicate or missing'):
            require_complete(report,styles)
        report,styles=self.data();report['cases'].pop()
        with self.assertRaisesRegex(ValueError,'Incomplete native process'):
            require_complete(report,styles)

    def test_failed_parity_or_nonfinite_timings_cannot_be_published(self):
        report,styles=self.data();report['comparisons'][0]['exact']=False
        with self.assertRaises(ValueError):require_complete(report,styles)
        report,styles=self.data();report['cases'][0]['gpu']['median_ms']=float('nan')
        with self.assertRaisesRegex(ValueError,'Invalid native timing'):
            require_complete(report,styles)


if __name__=='__main__':
    unittest.main()
