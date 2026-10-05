import tempfile
from pathlib import Path
import unittest
import torch
from helpers import tiny
from experiment_runtime import train_loop, evaluate_loss


class SpatiotemporalRuntimeTests(unittest.TestCase):
    def config(self):
        return {'spatiotemporal': {'enabled': True, 'schedule_steps': [1, 2],
                'cutoffs_hz': [50., 100., None], 'time_attention': True,
                'spatial_attention': True, 'time_window': 8},
                'training': {'max_iterations': 3, 'log_interval': 1, 'alpha': 0}}

    def test_stage_changes_and_best_uses_raw_mse(self):
        model, data, config = tiny(self.config())
        self.assertTrue(hasattr(model, 'set_training_step'))
        with tempfile.TemporaryDirectory() as directory:
            result = train_loop(model, data, config, directory)
            self.assertEqual(result['selection_metric'], 'data_mse')
            self.assertEqual([row['cutoff_hz'] for row in result['history']], [50., 100., None])
            self.assertEqual(result['best_loss'], min(row['data_mse_after_update'] for row in result['history']))
            saved = torch.load(Path(directory)/'last.pth', weights_only=False)
            self.assertEqual(saved['selection_metric'], 'data_mse')
            self.assertTrue((Path(directory)/'attention_step000003.npz').exists())
            best = torch.load(Path(directory)/'best.pth', weights_only=False)
            row = result['history'][best['completed_updates']-1]
            self.assertEqual(best['selection_score'], row['data_mse_after_update'])
            model.vel_net.load_state_dict(best['model'])
            model.set_training_step(best['objective_step'])
            value, parts = evaluate_loss(model, data, 0)
            self.assertAlmostEqual(value, best['loss_after_update'], places=10)
            self.assertAlmostEqual(parts['data_mse'], best['selection_score'], places=10)

    def test_changed_schedule_rejected_on_resume(self):
        model, data, config = tiny(self.config())
        self.assertTrue(hasattr(model, 'set_training_step'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_loop(model, data, config, root/'first', stop_after=1)
            changed = self.config()
            changed['spatiotemporal']['cutoffs_hz'][0] = 60.
            other, _, cfg = tiny(changed)
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                train_loop(other, data, cfg, root/'resume', resume=root/'first/last.pth')

    def test_resume_across_stage_boundary_matches(self):
        model, data, config = tiny(self.config())
        self.assertTrue(hasattr(model, 'set_training_step'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full = train_loop(model, data, config, root/'full')
            split, _, _ = tiny(self.config())
            train_loop(split, data, config, root/'split', stop_after=1)
            resumed_model, _, _ = tiny(self.config())
            resumed = train_loop(resumed_model, data, config, root/'resume', resume=root/'split/last.pth')
            self.assertEqual(full['history'], resumed['history'])
            for key, value in model.vel_net.state_dict().items():
                torch.testing.assert_close(value, resumed_model.vel_net.state_dict()[key], rtol=0, atol=0)

    def test_identity_fullband_matches_original_gradient(self):
        base, data, _ = tiny()
        active, _, _ = tiny({'spatiotemporal': {'enabled': True, 'schedule_steps': [],
            'cutoffs_hz': [None], 'time_attention': True, 'spatial_attention': True}})
        self.assertTrue(hasattr(active, 'set_training_step'))
        active.set_training_step(0)
        handles = []
        for model in (base, active):
            _, loss, parts, handle = model.objective(data['wavelet'], data['shots'], precondition=True)
            if handle is not None: handles.append(handle)
            loss.backward()
        for a, b in zip(base.vel_net.parameters(), active.vel_net.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        for handle in handles: handle.remove()


if __name__ == '__main__':
    unittest.main()
