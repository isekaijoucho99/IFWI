"""Release regressions for original resumes and portable experiment archives."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from helpers import ROOT
import baseline_experiment
from baseline_protocol import baseline_config
from import_completed_baseline import validate_legacy_config
import experiment_runtime
import run_ablation_suite
import run_parameter_sweep


class OriginalResumeCadenceTests(unittest.TestCase):
    def checkpoint(self, root, previous):
        (root / 'checkpoints').mkdir()
        (root / 'config.json').write_text(json.dumps(previous), encoding='utf8')
        checkpoint = root / 'checkpoints/MarmousiI_random-checkpoint-1.pth'
        torch.save(dict(epoch=1, state_dict={}, optimizer={}, train_loss=[[.1, .1, .2]],
                        best_loss=.1, best_loss_epoch=0, best_loss_model={}), checkpoint)
        return checkpoint

    def test_changed_selection_interval_is_rejected_before_data_and_outputs(self):
        # Otherwise the author trainer inherits a best chosen at extra epochs
        # and the canonical 4001/100 run is incorrectly labeled formal.
        for flat in (False, True):
            with self.subTest(flat=flat), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                root = Path(folder)
                previous = baseline_config()
                previous['training'].update(max_iterations=2, log_interval=1)
                if flat:
                    previous = baseline_experiment._legacy_config(previous, 3)
                checkpoint = self.checkpoint(root, previous)
                output = root / 'continued'
                with patch.object(baseline_experiment, 'prepare_original_data',
                                  side_effect=AssertionError('Data prepared before resume rejection')):
                    with self.assertRaisesRegex(ValueError, 'log_interval'):
                        baseline_experiment.run_config(baseline_config(), output, resume=checkpoint)
                self.assertFalse(output.exists())

    def test_same_selection_interval_allows_an_increased_total_budget(self):
        for flat in (False, True):
            with self.subTest(flat=flat), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                previous = baseline_config()
                previous['training']['max_iterations'] = 101
                if flat:
                    previous = baseline_experiment._legacy_config(previous, 3)
                checkpoint = self.checkpoint(Path(folder), previous)
                self.assertEqual(baseline_experiment.validate_resume(baseline_config(), checkpoint), 1)


class OriginalImportFieldsTests(unittest.TestCase):
    def test_strict_import_accepts_explicit_original_backend_and_no_clip(self):
        config = baseline_config()
        legacy = baseline_experiment._legacy_config(config, 3)
        legacy.update(backend='reference', gradient_clip=None)
        validate_legacy_config(legacy, config, 4001)

    def test_strict_import_rejects_a_changed_backend_or_effective_clip(self):
        config = baseline_config()
        for extra in ({'backend': 'prepared'}, {'gradient_clip': .25}):
            with self.subTest(extra=extra):
                legacy = baseline_experiment._legacy_config(config, 3)
                legacy.update(extra)
                with self.assertRaises(ValueError):
                    validate_legacy_config(legacy, config, 4001)

    def test_strict_import_still_rejects_unknown_settings(self):
        config = baseline_config()
        legacy = baseline_experiment._legacy_config(config, 3)
        legacy['unknown_solver_option'] = False
        with self.assertRaisesRegex(ValueError, 'unknown_solver_option'):
            validate_legacy_config(legacy, config, 4001)


class PortableSuiteArchiveTests(unittest.TestCase):
    def test_completed_baseline_bundle_uses_its_archived_author_sources(self):
        # Only child process dispatch is replaced: validate and stage the real
        # portable bundle in a location with no original repository ancestor.
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            legacy = root / 'portable_baseline'
            (legacy / 'author_sources').mkdir(parents=True)
            (legacy / 'checkpoints').mkdir()
            metadata = baseline_experiment._legacy_config(baseline_config(), 3)
            (legacy / 'config.json').write_text(json.dumps(metadata), encoding='utf8')
            for name in ('initial_velocity.npy', 'true_velocity.npy', 'best_velocity.npy', 'loss.csv'):
                (legacy / name).write_bytes(b'archived input')
            for update in range(1, 4002, 100):
                (legacy / f'checkpoints/MarmousiI_random-checkpoint-{update}.pth').write_bytes(b'archived checkpoint')
            for name in ('ifwi_modules.py', 'rnn_fd.py', 'generator.py'):
                (legacy / 'author_sources' / name).write_bytes((ROOT / name).read_bytes())
            output = root / 'suite'
            argv = ['run_parameter_sweep.py', '--output-dir', str(output),
                    '--completed-baseline', str(legacy), '--device', 'cpu']
            dispatched = subprocess.CompletedProcess(args=[], returncode=0)
            with patch.object(sys, 'argv', argv), \
                 patch.object(run_parameter_sweep.subprocess, 'run', return_value=dispatched), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_parameter_sweep.main(), 0)
            bundle = output / 'completed_legacy_baseline'
            for name in ('ifwi_modules.py', 'rnn_fd.py', 'generator.py'):
                self.assertEqual((bundle / 'author_sources' / name).read_bytes(),
                                 (legacy / 'author_sources' / name).read_bytes())
            manifest = json.loads((output / 'manifest.json').read_text(encoding='utf8'))
            self.assertEqual(manifest['completed_baseline_bundle'], 'completed_legacy_baseline')

    def test_runtime_source_contract_matches_the_ablation_suite_archive(self):
        # A newly nested reference runner belongs to both hash inventories;
        # leaving it out makes summarize_ablations reject every completed run.
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            for name in ('ifwi_modules.py', 'rnn_fd.py', 'generator.py', 'plot_functions.py',
                         'experiments/run_experiment.py', 'experiments/improved_modules/losses.py',
                         'experiments/baseline_reference/ifwi_experiment.py'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('# archived source: ' + name, encoding='utf8')
            config = root / 'experiments/configs/feature_baseline.yaml'
            config.parent.mkdir()
            config.write_text('experiment_name: baseline\n', encoding='utf8')
            output = root / 'suite'
            argv = ['run_ablation_suite.py', '--output-dir', str(output), '--configs', 'baseline',
                    '--iterations', '1', '--log-interval', '1', '--device', 'cpu']
            dispatched = subprocess.CompletedProcess(args=[], returncode=0)
            with patch.object(sys, 'argv', argv), patch.object(run_ablation_suite, 'ROOT', root), \
                 patch.object(run_ablation_suite.subprocess, 'run', return_value=dispatched), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_ablation_suite.main(), 0)
            frozen = json.loads((output / 'source_hashes.json').read_text(encoding='utf8'))
            frozen = {name: digest for name, digest in frozen.items() if name.endswith('.py')}
            with patch.object(experiment_runtime, 'ROOT', root):
                self.assertEqual(experiment_runtime.source_hashes(), frozen)


if __name__ == '__main__':
    unittest.main()
