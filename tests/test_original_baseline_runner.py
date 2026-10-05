"""The formal runner must preserve the author's actual training behavior."""
import contextlib
import copy
import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from ifwi_modules import IFWI2D

torch.set_num_threads(1)


def tiny_original(iterations=21, interval=10):
    torch.manual_seed(3)
    nz, nx, nt = 10, 12, 24
    xs = torch.tensor([[3, 8]])
    zs = torch.ones_like(xs)
    xr = torch.arange(nx)[None, None].repeat(1, 2, 1)
    zr = torch.full((1, 2, nx), 2)
    model = IFWI2D(mean=3., std=1., neuron=[2, 12, 12, 1], omega_0=30,
        outermost_linear=True, nz=nz, nx=nx, zs=zs, xs=xs, zr=zr, xr=xr,
        dz=15., dt=.001, npad=3, order=2, vmax=4000., freeSurface=True,
        regularization='TV', segment_size=nt, device='cpu', netOpt='IFWI')
    wave = torch.zeros(nt)
    wave[2:5] = torch.tensor([.5, 1., .5])
    truth = torch.linspace(2800, 3400, nz)[:, None].expand(nz, nx)[None].contiguous()
    with torch.no_grad():
        shots = model.rnn(truth, wave)[2].clone()
    data = {'vp_true': truth, 'shots': shots, 'wavelet': wave,
        'geometry': {'nz': nz, 'nx': nx, 'xs': xs, 'zs': zs, 'xr': xr, 'zr': zr},
        'params': {'dz': 15., 'dt': .001, 'nt': nt}}
    config = {'experiment_name': 'original_tiny',
        'model': {'network_type': 'vanilla', 'neuron': [2, 12, 12, 1], 'omega_0': 30},
        'training': {'max_iterations': iterations, 'log_interval': interval,
                     'alpha': 0, 'shot_batch_size': None},
        'optimizer': {'optimizer_type': 'adam', 'learning_rate': 1e-4},
        'data': {'dz': 15., 'dt': .001, 'nt': nt},
        'evaluation': {'depth_threshold': .5, 'corner_size': .25, 'save_plots': False}}
    return model, data, config


class OriginalBaselineRunnerTests(unittest.TestCase):
    def runner(self):
        self.assertTrue((ROOT / 'experiments/baseline_experiment.py').is_file(),
                        'The original baseline training entry has not been implemented')
        return importlib.import_module('baseline_experiment')

    def assert_nested_equal(self, actual, expected):
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, reference in zip(actual, expected):
                self.assert_nested_equal(value, reference)
        else:
            self.assertEqual(actual, expected)

    def direct_train(self, model, data, config, out, resume=None, stop_after=None):
        reference = importlib.import_module('baseline_reference.ifwi_experiment')
        (out / 'checkpoints').mkdir(parents=True)
        total = stop_after or config['training']['max_iterations']
        completed = torch.load(resume, map_location='cpu', weights_only=False)['epoch'] if resume else 0
        prefix = str(out / 'checkpoints/MarmousiI_random-')
        history, _ = model.train(MaxIter=total, vmodel=None, wavelet=data['wavelet'],
            shots=data['shots'], alpha=config['training']['alpha'], option=0,
            learning_rate=1e-4, log_interval=config['training']['log_interval'],
            wandb=reference.EpochLog(completed), resume_file_name=str(resume) if resume else None,
            save_file_name=prefix)
        checkpoint = torch.load(prefix + f'checkpoint-{total}.pth', map_location='cpu', weights_only=False)
        last = copy.deepcopy(model.vel_net.state_dict())
        best, _ = model.predict(resume_file_name=prefix + f'checkpoint-{total}.pth', best=True)
        model.vel_net.load_state_dict(last)
        return np.asarray(history), checkpoint, best.squeeze().cpu().numpy()

    def test_matches_direct_original_stdout_losses_optimizer_and_all_snapshots(self):
        # Replacing the original train with the modern loop changes selection,
        # TV reporting, stdout, checkpoint cadence and Adam/model state.
        runner = self.runner()
        actual, data, config = tiny_original()
        expected = copy.deepcopy(actual)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            expected_stdout, actual_stdout = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(expected_stdout):
                history, final, best = self.direct_train(expected, data, config, root / 'direct')
            batches = []
            hook = actual.rnn.register_forward_hook(lambda module, args, result: batches.append(result[2].shape[1]))
            with contextlib.redirect_stdout(actual_stdout):
                result = runner.train_original(actual, data, config, root / 'runner')
            hook.remove()
            self.assertEqual(actual_stdout.getvalue(), expected_stdout.getvalue())
            self.assertEqual(batches, [2] * 21)
            names = ['MarmousiI_random-checkpoint-1.pth',
                     'MarmousiI_random-checkpoint-11.pth',
                     'MarmousiI_random-checkpoint-21.pth']
            self.assertEqual(sorted(p.name for p in (root / 'runner/checkpoints').glob('*.pth')), names)
            for name in names:
                self.assert_nested_equal(
                    torch.load(root / 'runner/checkpoints' / name, map_location='cpu', weights_only=False),
                    torch.load(root / 'direct/checkpoints' / name, map_location='cpu', weights_only=False))
            self.assert_nested_equal(actual.vel_net.state_dict(), expected.vel_net.state_dict())
            csv = np.loadtxt(root / 'runner/loss.csv', delimiter=',', skiprows=1)
            np.testing.assert_array_equal(csv[:, 1:], history)
            self.assertGreater(csv[0, 3], 0.)
            np.testing.assert_array_equal(np.load(root / 'runner/best_velocity.npy'), best)
            self.assertEqual(result['completed_updates'], 21)
            self.assertEqual(result['best_update'], final['best_loss_epoch'] + 1)
            self.assertEqual(result['best_loss'], final['best_loss'])
            self.assertEqual(result['selection_metric'], 'legacy_preupdate_data_loss')
            metrics = json.loads((root / 'runner/metrics.json').read_text())
            self.assertEqual(set(metrics), {'relative_model_error', 'velocity_rmse_mps', 'final_data_loss'})
            self.assertEqual(metrics['final_data_loss'], history[-1, 1])

    def test_resume_matches_uninterrupted_original_adam_and_model(self):
        # Restoring only weights instead of the author's optimizer/history
        # changes the continued trajectory.
        runner = self.runner()
        uninterrupted, data, config = tiny_original()
        split = copy.deepcopy(uninterrupted)
        resumed = copy.deepcopy(uninterrupted)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            with contextlib.redirect_stdout(io.StringIO()):
                _, expected, _ = self.direct_train(uninterrupted, data, config, root / 'direct')
                runner.train_original(split, data, config, root / 'first', stop_after=11)
                checkpoint = root / 'first/checkpoints/MarmousiI_random-checkpoint-11.pth'
                result = runner.train_original(resumed, data, config, root / 'resumed', resume=checkpoint)
            actual = torch.load(root / 'resumed/checkpoints/MarmousiI_random-checkpoint-21.pth',
                                map_location='cpu', weights_only=False)
            self.assert_nested_equal(actual, expected)
            self.assert_nested_equal(resumed.vel_net.state_dict(), uninterrupted.vel_net.state_dict())
            self.assertEqual(result['resumed_from_updates'], 11)
            self.assertEqual(result['executed_updates'], 10)

    def test_preserves_original_checkpoint_deletion_for_nonstandard_interval(self):
        # The author's deletion condition is intentionally retained, rather
        # than promising snapshots the original core removes.
        runner = self.runner()
        model, data, config = tiny_original(iterations=7, interval=3)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            out = Path(folder) / 'run'
            with contextlib.redirect_stdout(io.StringIO()):
                runner.train_original(model, data, config, out)
            self.assertEqual([p.name for p in (out / 'checkpoints').glob('*.pth')],
                             ['MarmousiI_random-checkpoint-7.pth'])

    def test_rejects_batching_before_creating_training_outputs(self):
        # An unnoticed batching option would turn 'only params differ' into a
        # different training implementation.
        runner = self.runner()
        model, data, config = tiny_original()
        config['training']['shot_batch_size'] = 13
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            out = Path(folder) / 'run'
            with self.assertRaisesRegex(ValueError, 'batch'):
                runner.train_original(model, data, config, out)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
