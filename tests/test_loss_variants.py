"""Numerical loss semantics independent of the wave solver and GPU."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from improved_modules.losses import CombinedLoss, DeepLayerPriorLoss


class LossVariantTests(unittest.TestCase):
    def test_default_mse_reports_raw_mse_and_preserves_gradient(self):
        prediction = torch.tensor([1., -2., 3.], requires_grad=True)
        loss, parts = CombinedLoss(4, 3, {})(prediction, torch.zeros(3), torch.ones(1, 4, 3))
        self.assertIn('data_mse', parts)
        torch.testing.assert_close(loss, prediction.square().mean())
        torch.testing.assert_close(parts['data_mse'], parts['data_loss'])
        loss.backward()
        torch.testing.assert_close(prediction.grad, 2 * prediction.detach() / 3)

    def test_huber_keeps_quadratic_core_and_limits_outlier_gradient(self):
        prediction = torch.tensor([0., .5, 1., -4.], dtype=torch.float64, requires_grad=True)
        objective = CombinedLoss(4, 4, {'data_objective': 'huber', 'huber_beta': 1.})
        loss, parts = objective(prediction, torch.zeros_like(prediction), torch.ones(1, 4, 4))
        torch.testing.assert_close(loss, torch.tensor(8.25 / 4, dtype=torch.float64))
        torch.testing.assert_close(parts['data_mse'], prediction.square().mean())
        self.assertLess(float(parts['data_loss'].detach()), float(parts['data_mse'].detach()))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([0., .25, .5, -.5], dtype=torch.float64))

    def test_huber_rejects_invalid_threshold_and_unknown_objective(self):
        for beta in (0., -1., float('inf'), float('nan')):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                CombinedLoss(4, 3, {'data_objective': 'huber', 'huber_beta': beta})
        with self.assertRaises(ValueError):
            CombinedLoss(4, 3, {'data_objective': 'not_an_objective'})

    def test_normalized_tv_scales_all_components_and_gradients(self):
        physical = torch.tensor([[[5000., 6000., 6500.], [2500., 5000., 1000.],
                                  [1000., 7000., 2000.], [500., 6000., 1000.]]],
                                dtype=torch.float64, requires_grad=True)
        normalized_input = physical.detach().clone().requires_grad_()
        options = dict(deep_start=.5, monotonic_weight=.2, horizontal_weight=.3, range_weight=.4)
        legacy = DeepLayerPriorLoss(4, 3, **options)
        normalized = DeepLayerPriorLoss(4, 3, velocity_scale=1000., **options)
        info = {'deep_velocity_range': [1500., 5500.]}
        before = legacy.components(physical, info)
        after = normalized.components(normalized_input, info)
        for name in before:
            self.assertGreater(float(before[name].detach()), 0.)
            torch.testing.assert_close(after[name], before[name] / 1000.)
        sum(before.values()).backward()
        sum(after.values()).backward()
        torch.testing.assert_close(normalized_input.grad, physical.grad / 1000.)
        torch.testing.assert_close(normalized_input.grad[:, :2], torch.zeros_like(normalized_input.grad[:, :2]))

    def test_charbonnier_is_zero_and_stationary_on_constant_valid_model(self):
        model = torch.full((1, 4, 3), 3000., dtype=torch.float64, requires_grad=True)
        prior = DeepLayerPriorLoss(4, 3, velocity_scale=1000., prior_form='charbonnier',
                                   charbonnier_eps=.02, monotonic_weight=1.)
        loss = prior(model, {'deep_velocity_range': [1500., 5500.]})
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        torch.testing.assert_close(model.grad, torch.zeros_like(model))

    def test_charbonnier_matches_physical_horizontal_gradient_and_deep_region(self):
        model = torch.tensor([[[1., 9.], [5., 20.], [3000., 3500.], [3000., 3500.]]],
                             dtype=torch.float64, requires_grad=True)
        prior = DeepLayerPriorLoss(4, 2, velocity_scale=1000., prior_form='charbonnier',
                                   charbonnier_eps=.1, horizontal_weight=1., range_weight=0.)
        loss = prior(model)
        expected = (.5 ** 2 + .1 ** 2) ** .5 - .1
        self.assertAlmostEqual(float(loss.detach()), expected)
        loss.backward()
        expected_gradient = .5 / (.5 ** 2 + .1 ** 2) ** .5 / 1000. / 2.
        torch.testing.assert_close(model.grad[:, :2], torch.zeros_like(model.grad[:, :2]))
        torch.testing.assert_close(model.grad[:, 2:, 1], torch.full((1, 2), expected_gradient, dtype=torch.float64))
        torch.testing.assert_close(model.grad[:, 2:, 0], torch.full((1, 2), -expected_gradient, dtype=torch.float64))

    def test_charbonnier_changes_horizontal_only_keeps_linear_hinges(self):
        model = torch.tensor([[[3000., 3000.], [3000., 3000.], [1000., 6500.], [500., 6000.]]])
        options = dict(velocity_scale=1000., monotonic_weight=.2, range_weight=.3)
        info = {'deep_velocity_range': [1500., 5500.]}
        tv = DeepLayerPriorLoss(4, 2, **options).components(model, info)
        smooth = DeepLayerPriorLoss(4, 2, prior_form='charbonnier', **options).components(model, info)
        for name in ('prior_monotonic', 'prior_range'):
            self.assertGreater(float(smooth[name]), 0.)
            torch.testing.assert_close(smooth[name], tv[name], rtol=0, atol=0)
        self.assertLess(float(smooth['prior_horizontal']), float(tv['prior_horizontal']))

    def test_prior_rejects_invalid_normalization_and_kernel(self):
        for name in ('velocity_scale', 'charbonnier_eps'):
            for value in (0., -1., float('inf'), float('nan')):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    DeepLayerPriorLoss(4, 3, **{name: value})
        with self.assertRaises(ValueError):
            DeepLayerPriorLoss(4, 3, prior_form='unknown')

    def test_disabled_prior_does_not_construct_or_add_penalties(self):
        objective = CombinedLoss(4, 3, {'use_prior': False, 'prior_params': {'velocity_scale': 0.}})
        loss, parts = objective(torch.ones(3), torch.zeros(3), torch.ones(1, 4, 3) * 9000)
        self.assertEqual(float(loss), 1.)
        for name in ('prior_loss', 'prior_horizontal', 'prior_monotonic', 'prior_range'):
            self.assertEqual(float(parts[name]), 0.)


if __name__ == '__main__':
    unittest.main()
