"""Guard experiment labels, numerical acceptance and matched-round analysis."""
import copy
import unittest
from experiment_cache_registers import with_register_budget
from verify_native_tuning import summarize


class NativeTuning(unittest.TestCase):
    def cases(self):
        rows=[]
        # A drifting GPU makes pooled condition medians misleading. Pair each
        # candidate with its own round: both measured changes are exactly -10%.
        for repeat,(before,after) in enumerate(((10.,9.),(100.,90.))):
            for label,value in (('control',before),('candidate',after)):
                rows.append(dict(round=repeat,label=label,gpu=dict(median_ms=value),
                                 captures=dict(reset_a='same',reset_b='same',history='same'),
                                 reset_repeatable=True,memory={key:1 for key in (
                                     'device_local_bytes','device_nonlocal_bytes','host_private_bytes',
                                     'host_working_set_bytes','host_peak_working_set_bytes')}))
        return rows

    def test_pairs_by_round_even_when_execution_order_changes(self):
        result=summarize(list(reversed(self.cases())),2)
        self.assertTrue(result['exact'])
        self.assertAlmostEqual(result['paired_median_change_percent'],-10.)

    def test_missing_round_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Missing paired'):
            summarize(self.cases()[:-1],2)

    def test_history_or_reset_failure_cannot_pass(self):
        rows=self.cases()
        rows[-1]['captures']=copy.deepcopy(rows[-1]['captures'])
        rows[-1]['captures']['history']='changed'
        self.assertFalse(summarize(rows,2)['exact'])
        rows=self.cases();rows[-1]['reset_repeatable']=False
        self.assertFalse(summarize(rows,2)['exact'])

    def test_register_recipe_changes_only_compatibility_branch(self):
        source='#if PHOENIX_STYLE_ARCH < 890\n__maxnreg__(160)\n#else\n__maxnreg__(128)\n#endif'
        result=with_register_budget(source,224)
        self.assertIn('__maxnreg__(224)',result)
        self.assertIn('#else\n__maxnreg__(128)',result)
        with self.assertRaises(ValueError):
            with_register_budget('__maxnreg__(128) __maxnreg__(128)',160)


if __name__=='__main__':
    unittest.main()
