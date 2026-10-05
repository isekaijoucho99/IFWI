"""Shared physical metrics must remain comparable across training objectives."""
import unittest
import torch
from helpers import tiny
import run_experiment


class ObjectiveReportingTests(unittest.TestCase):
    def test_robust_objective_does_not_replace_reported_mse(self):
        model, data, config = tiny({'loss': {'data_objective': 'huber', 'huber_beta': 1e-4}})
        config['evaluation'] = {'depth_threshold': .5, 'corner_size': .25}
        self.assertTrue(hasattr(run_experiment, 'evaluate_model_metrics'),
                        'Need a shared evaluator with separate objective and raw MSE')
        metrics = run_experiment.evaluate_model_metrics(model, data, config)
        with torch.no_grad():
            _, _, parts, _ = model.objective(data['wavelet'], data['shots'])
        self.assertAlmostEqual(metrics['data_mse'], float(parts['data_mse']), places=10)
        self.assertAlmostEqual(metrics['data_objective'], float(parts['data_loss']), places=10)
        self.assertGreater(metrics['data_mse'], metrics['data_objective'])

    def test_history_has_common_mse_for_robust_training(self):
        import tempfile
        from experiment_runtime import train_loop
        model, data, config = tiny({'loss': {'data_objective': 'huber', 'huber_beta': 1e-4}})
        config['training'].update(max_iterations=2, log_interval=1)
        with tempfile.TemporaryDirectory() as directory:
            result = train_loop(model, data, config, directory)
        for row in result['history']:
            self.assertIn('data_mse_before_update', row)
            self.assertIn('data_mse_after_update', row)
            self.assertGreater(row['data_mse_before_update'], row['data_loss_before_update'])
            self.assertGreater(row['data_mse_after_update'], row['data_loss_after_update'])
