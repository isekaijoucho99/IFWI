"""Catch wrong default baselines and uniformly confounded dry-run suites."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml
from helpers import ROOT
from run_experiment import load_config


class OriginalSweepEntryTests(unittest.TestCase):
    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, '-X', 'utf8', str(ROOT/'experiments/run_parameter_sweep.py'),
             '--output-dir', str(ROOT/'tmp/unused-original-dry-run'), '--dry-run', *arguments],
            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')

    def test_default_suite_uses_the_users_original_baseline(self):
        result = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        cases = json.loads(result.stdout)
        self.assertEqual(len(cases), 10)
        baseline = cases[0]['config']
        self.assertEqual(baseline['model']['neuron'], [2,128,128,128,128,1])
        self.assertEqual(baseline['model']['omega_0'], 30)
        self.assertEqual(baseline['seed'], 3)
        self.assertEqual(baseline['training']['max_iterations'], 4001)
        self.assertEqual(baseline['training']['log_interval'], 100)
        self.assertIsNone(baseline['training']['shot_batch_size'])
        self.assertEqual(baseline['execution']['protocol'], 'original_baseline')

    def test_dry_run_rejects_a_different_common_learning_rate(self):
        config = copy.deepcopy(load_config(ROOT/'experiments/configs/legacy_random_baseline.yaml'))
        config['optimizer']['learning_rate'] = .002
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'confounded.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            result = self.invoke('--baseline-config', str(path))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('baseline', result.stderr.lower())

    def test_formal_dry_run_rejects_shot_batch_override(self):
        result = self.invoke('--shot-batch-size', '13')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('batch', result.stderr.lower())


if __name__ == '__main__':
    unittest.main()
