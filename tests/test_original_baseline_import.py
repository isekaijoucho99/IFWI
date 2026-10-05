"""Import the exact author trajectory without switching to modern training code."""
import contextlib
import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
from baseline_protocol import baseline_config
from baseline_experiment import _legacy_config, _velocity, build_original_model
from import_completed_baseline import main, validate_legacy_config

torch.set_num_threads(1)


class OriginalBaselineImportTests(unittest.TestCase):
    def test_original_marker_rejects_changed_objective_optimizer_and_batching(self):
        # Historically these fields passed validation even though the original
        # checkpoint was trained with a different algorithm.
        config = baseline_config()
        legacy = _legacy_config(config, 3)
        for section, key, value in [
            ('loss', 'data_objective', 'huber'),
            ('optimizer', 'optimizer_type', 'adamw'),
            ('training', 'shot_batch_size', 13),
            ('data', 'num_shots', 25),
            ('model', 'bias', False),
        ]:
            with self.subTest(section=section, key=key):
                wrong = copy.deepcopy(config)
                wrong[section][key] = value
                with self.assertRaises(ValueError):
                    validate_legacy_config(legacy, wrong, 4001)

    def test_original_marker_rejects_changed_author_save_interval(self):
        # Matching the final update count is insufficient: log_interval also
        # changes which preupdate losses select the stored best state.
        config = baseline_config()
        legacy = _legacy_config(config, 3)
        legacy['log_interval'] = 10
        with self.assertRaisesRegex(ValueError, 'log_interval'):
            validate_legacy_config(legacy, config, 4001)

    def test_unmarked_historical_configuration_remains_supported(self):
        config = baseline_config()
        config.pop('execution')
        config['training']['shot_batch_size'] = 13
        validate_legacy_config(_legacy_config(config, 3), config, 4001)

    def test_strict_import_preserves_all_snapshots_and_legacy_metrics_without_training(self):
        # Only the costly 1000-step observation forward pass is replaced.
        # Data preparation, the real original model, checkpoint loading,
        # source validation, and all import outputs remain real.
        config = baseline_config()
        config['evaluation']['save_plots'] = False
        truth = np.array(pd.read_csv(ROOT / 'data/vel_marmousi_376x1151.csv'))[::4, ::4].astype(np.float32)
        vp = torch.from_numpy(truth[None])
        nz, nx = truth.shape
        xs = torch.arange(20, nx - 10, 20)[None]
        ns = xs.shape[1]
        geometry = dict(nz=nz, nx=nx, xs=xs, zs=torch.ones_like(xs),
            xr=torch.arange(nx)[None, None].repeat(1, ns, 1), zr=torch.full((1, ns, nx), 2))
        data = dict(vp_true=vp, geometry=geometry, params=dict(dz=15, dt=.0019, nt=1000))
        torch.manual_seed(3)
        model = build_original_model(config, data, 'cpu')
        initial = _velocity(model)
        best_state = {key: value.detach().clone() for key, value in model.vel_net.state_dict().items()}
        model.vel_net.linear[-1].bias.data.add_(.01)
        last_state = {key: value.detach().clone() for key, value in model.vel_net.state_dict().items()}
        last = _velocity(model)
        legacy_metrics = dict(relative_model_error=float(np.linalg.norm(initial - truth) / np.linalg.norm(truth)),
            velocity_rmse_mps=float(np.sqrt(np.mean((initial - truth) ** 2))), final_data_loss=.002)
        observations = torch.zeros(1, ns, 1000, nx)
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            legacy = root / 'legacy'
            (legacy / 'checkpoints').mkdir(parents=True)
            (legacy / 'author_sources').mkdir()
            for name in ('ifwi_modules.py', 'rnn_fd.py', 'generator.py'):
                (legacy / 'author_sources' / name).write_bytes((ROOT / name).read_bytes())
            (legacy / 'config.json').write_text(json.dumps(_legacy_config(config, 3)))
            (legacy / 'metrics.json').write_text(json.dumps(legacy_metrics))
            np.save(legacy / 'initial_velocity.npy', initial)
            np.save(legacy / 'best_velocity.npy', initial)
            np.save(legacy / 'true_velocity.npy', truth)
            history = [[.002, .002, .3]] * 4001
            np.savetxt(legacy / 'loss.csv', np.column_stack([np.arange(1, 4002), history]),
                delimiter=',', header='completed_updates,total_loss,data_loss,regularization_statistic', comments='')
            for completed in range(1, 4002, 100):
                checkpoint = dict(epoch=completed, best_loss=.002, best_loss_epoch=min(completed - 1, 3900),
                    best_loss_model=best_state, train_loss=history[:completed],
                    state_dict=last_state, optimizer={'state': {}, 'param_groups': []})
                torch.save(checkpoint, legacy / f'checkpoints/MarmousiI_random-checkpoint-{completed}.pth')
            cfg_path = root / 'config.yaml'
            cfg_path.write_text(yaml.safe_dump(config))
            output = root / 'output'
            argv = ['import_completed_baseline.py', '--legacy-run', str(legacy), '--config', str(cfg_path),
                    '--output-dir', str(output), '--device', 'cpu']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(io.StringIO()), \
                 patch('rnn_fd.rnn2D.forward', return_value=(None, None, observations, None)), \
                 patch('run_experiment.ImprovedIFWI', side_effect=AssertionError('Modern model reached')), \
                 patch('import_completed_baseline.evaluate_model_metrics', side_effect=AssertionError('Modern objective reached')), \
                 patch('ifwi_modules.IFWI2D.train', side_effect=AssertionError('Import attempted training')):
                self.assertEqual(main(), 0)
            out = output / 'legacy_baseline_import'
            self.assertEqual(len(list((out / 'checkpoints').glob('*.pth'))), 41)
            for checkpoint in (legacy / 'checkpoints').glob('*.pth'):
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                 hashlib.sha256((out / 'checkpoints' / checkpoint.name).read_bytes()).hexdigest())
            np.testing.assert_array_equal(np.load(out / 'initial_velocity.npy'), initial)
            np.testing.assert_array_equal(np.load(out / 'last_velocity.npy'), last)
            np.testing.assert_array_equal(np.load(out / 'best_velocity.npy'), initial)
            self.assertEqual(json.loads((out / 'metrics.json').read_text()), legacy_metrics)
            final_metrics = json.loads((out / 'final_metrics.json').read_text())
            self.assertIsNone(final_metrics['data_mse'])
            self.assertFalse(final_metrics['data_mse_evaluated'])
            summary = json.loads((out / 'training_summary.json').read_text())
            self.assertEqual(summary['executed_updates'], 0)
            self.assertEqual(summary['completed_updates'], 4001)
            self.assertEqual(summary['best_update'], 3901)
            self.assertEqual(summary['snapshot_updates'], list(range(1, 4002, 100)))
            self.assertFalse((out / 'legacy_loss_history.csv').exists())
            sys.path.insert(0, str(ROOT / 'scripts'))
            from plot_parameter_sweep_losses import collect_cases
            job = dict(name='baseline', factor='baseline', value=0, seed=3, directory='output')
            exported = collect_cases(root, dict(jobs=[job]))
            self.assertEqual(len(exported[0]['rows']), 4001)
            self.assertEqual(Path(exported[0]['source']), out / 'loss.csv')


if __name__ == '__main__':
    unittest.main()
