"""The single-experiment opt-in must keep the author's real training path."""
import contextlib
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

import baseline_experiment as runner
from baseline_protocol import baseline_config
from baseline_reference.ifwi_experiment import EpochLog
from rnn_fd import rnn2D

torch.set_num_threads(1)


def small_observations():
    """Genuine finite-difference data; substitute only the full-size preparation."""
    nz, nx, nt = 10, 12, 24
    xs = torch.tensor([[3, 8]])
    geometry = dict(nz=nz, nx=nx, xs=xs, zs=torch.ones_like(xs),
        xr=torch.arange(nx)[None, None].repeat(1, 2, 1), zr=torch.full((1, 2, nx), 2))
    truth = torch.linspace(2800, 3400, nz)[:, None].expand(nz, nx)[None].contiguous()
    wavelet = torch.zeros(nt)
    wavelet[2:5] = torch.tensor([.5, 1., .5])
    solver = rnn2D(**geometry, dz=15, dt=.0019, npad=15, order=2,
        vmax=truth.max(), log_para=1e-6, freeSurface=True, dtype=torch.float32, device='cpu')
    with torch.no_grad():
        shots = solver(truth, wavelet)[2].clone()
    return dict(vp_true=truth, wavelet=wavelet, shots=shots, geometry=geometry,
                params=dict(dz=15, dt=.0019, nt=nt))


class SingleExperimentRuntimeTests(unittest.TestCase):
    def short_config(self, factor):
        config = baseline_config()
        config['experiment_name'] = 'single_custom_' + factor
        config['training'].update(max_iterations=3, log_interval=10)
        config['evaluation']['save_plots'] = False
        if factor == 'width':
            config['model']['neuron'] = [2, 7, 7, 7, 7, 1]
        elif factor == 'omega':
            config['model']['omega_0'] = 15.5
        elif factor == 'combined':
            config['model']['neuron'] = [2, 7, 7, 7, 7, 1]
            config['model']['omega_0'] = 15.5
        else:
            raise ValueError('Unknown test factor')
        return config

    def assert_nested_equal(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_custom_opt_in_keeps_original_updates_stdout_and_numbered_checkpoints(self):
        # Dropping the opt-in flag, rebuilding a canonical network, or routing
        # through the modern train loop breaks a real three-update comparison.
        data = small_observations()
        for factor in ('width', 'omega', 'combined'):
            with self.subTest(factor=factor), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                root = Path(folder)
                config = self.short_config(factor)
                original = copy.deepcopy(config)
                torch.manual_seed(3)
                reference = runner.build_original_model(config, data, 'cpu')
                (root / 'reference/checkpoints').mkdir(parents=True)
                prefix = str(root / 'reference/checkpoints/MarmousiI_random-')
                reference_stdout = io.StringIO()
                with contextlib.redirect_stdout(reference_stdout):
                    history, _ = reference.train(MaxIter=3, vmodel=None,
                        wavelet=data['wavelet'], shots=data['shots'], alpha=0, option=0,
                        learning_rate=1e-4, log_interval=10, wandb=EpochLog(), save_file_name=prefix)
                    reference.predict(resume_file_name=prefix + 'checkpoint-3.pth', best=True)
                actual_stdout = io.StringIO()
                with patch.object(runner, 'prepare_original_data', return_value=data), \
                     contextlib.redirect_stdout(actual_stdout):
                    result = runner.run_config(config, root / 'output', device='cpu', seed=3,
                                               preliminary=True, allow_custom=True)
                run = Path(result['run_dir'])
                self.assertEqual(len(list((root / 'output').iterdir())), 1)
                self.assertEqual(config, original)
                status = json.loads((run / 'status.json').read_text())
                self.assertEqual(status, {'state': 'completed', 'completed_updates': 3})
                self.assertEqual(result['completed_updates'], 3)
                self.assertEqual(result['executed_updates'], 3)
                self.assertEqual(result['selection_metric'], 'legacy_preupdate_data_loss')
                checkpoint_names = ['MarmousiI_random-checkpoint-1.pth',
                                    'MarmousiI_random-checkpoint-3.pth']
                self.assertEqual(sorted(path.name for path in (run / 'checkpoints').glob('*.pth')),
                                 checkpoint_names)
                for name in checkpoint_names:
                    self.assert_nested_equal(
                        torch.load(run / 'checkpoints' / name, map_location='cpu', weights_only=False),
                        torch.load(root / 'reference/checkpoints' / name, map_location='cpu', weights_only=False))
                np.testing.assert_array_equal(
                    np.loadtxt(run / 'loss.csv', delimiter=',', skiprows=1)[:, 1:], np.asarray(history))
                core_lines = lambda text: [line for line in text.splitlines()
                    if line.startswith(('Completed: ', 'Epoch: ', 'Loading the best loss model'))]
                self.assertEqual(core_lines(actual_stdout.getvalue()), core_lines(reference_stdout.getvalue()))
                self.assertEqual((run / 'progress.log').read_text(), actual_stdout.getvalue())
                if factor == 'width':
                    self.assertEqual(result['parameter_count'], 197)
                saved_config = json.loads((run / 'config.json').read_text())
                self.assertEqual(saved_config['model'], config['model'])

    def test_custom_parameters_without_opt_in_are_rejected_before_output_creation(self):
        # The default suite path must not silently become a custom experiment.
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            for factor in ('width', 'omega'):
                with self.subTest(factor=factor):
                    output = Path(folder) / factor
                    with patch.object(runner, 'prepare_original_data',
                                      side_effect=AssertionError('Prepared data before rejecting a custom parameter')):
                        with self.assertRaises(ValueError):
                            runner.run_config(self.short_config(factor), output, device='cpu',
                                              preliminary=True)
                    self.assertFalse(output.exists())

    def legacy_resume_fixture(self, root, extra=None):
        previous = dict(mode='random', seed=3, mean=3., std=1., dz=15, dt=.0019, nt=1000,
            frequency=8, source_depth_index=1, receiver_depth_index=2, learning_rate=1e-4,
            alpha=0, noise=0, dropout=0, epochs=1, log_interval=100, pretrained=None)
        if extra:
            previous.update(extra)
        (root / 'checkpoints').mkdir()
        (root / 'config.json').write_text(json.dumps(previous))
        checkpoint = root / 'checkpoints/MarmousiI_random-checkpoint-1.pth'
        torch.save(dict(epoch=1, state_dict={}, optimizer={}, train_loss=[[.1, .1, .2]],
                        best_loss=.1, best_loss_epoch=0, best_loss_model={}), checkpoint)
        return checkpoint

    def test_flat_resume_rejects_prepared_backend_and_effective_gradient_clipping(self):
        # Both fields distinguish the later speed/clip experiment from the
        # historical author's exhausted-generator clipping behavior.
        for extra in ({'backend': 'prepared'}, {'gradient_clip': .25}):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory(dir=ROOT) as folder:
                checkpoint = self.legacy_resume_fixture(Path(folder), extra)
                with self.assertRaises(ValueError):
                    runner.validate_resume(baseline_config(), checkpoint)

    def test_flat_resume_accepts_historical_absence_of_backend_and_clip_fields(self):
        # Those fields did not exist in the actual original random run.
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            checkpoint = self.legacy_resume_fixture(Path(folder))
            self.assertEqual(runner.validate_resume(baseline_config(), checkpoint), 1)


if __name__ == '__main__':
    unittest.main()
