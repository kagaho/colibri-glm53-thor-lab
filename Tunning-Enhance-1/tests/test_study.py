import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

STUDY=Path(__file__).resolve().parents[1]
ROOT=STUDY.parent
spec=importlib.util.spec_from_file_location('study_summary',STUDY/'scripts/summarize.py')
summary=importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)

class PromotionTests(unittest.TestCase):
    def pairs(self,t,l):
        return [{'throughput_gain_fraction':a,'ttft_reduction_fraction':b} for a,b in zip(t,l)]
    def test_repeatable_gain_with_valid_checks(self):
        pairs=self.pairs([.25,.24,.22],[.12,.08,.10])
        self.assertTrue(summary.assess(pairs,True,True)['promote'])
    def test_large_median_does_not_hide_a_reversed_pair(self):
        pairs=self.pairs([.50,-.01,.40],[0,0,0])
        self.assertFalse(summary.assess(pairs,True,True)['promote'])
    def test_large_gain_does_not_override_failed_quality_or_placement(self):
        pairs=self.pairs([.40,.30,.50],[.10,.10,.10])
        self.assertFalse(summary.assess(pairs,True,False)['promote'])
    def test_large_gain_does_not_override_material_latency_regression(self):
        pairs=self.pairs([.40,.30,.50],[-.10,-.10,-.10])
        self.assertFalse(summary.assess(pairs,True,True)['promote'])
    def test_incomplete_pair_set_cannot_be_promoted(self):
        pairs=self.pairs([.40,.30],[.10,.10])
        self.assertFalse(summary.assess(pairs,False,True)['promote'])
    def test_study_directory_cannot_escape_the_repository(self):
        with tempfile.TemporaryDirectory() as outside:
            result=subprocess.run(['python3','scripts/run_experiment.py','--study-dir',outside],
                cwd=ROOT,capture_output=True,text=True)
        self.assertEqual(result.returncode,2)
        self.assertIn('must stay within',result.stderr)

if __name__=='__main__': unittest.main()
