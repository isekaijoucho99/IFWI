"""CPU checks for observed-data time weighting and gradient-response weighting."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
from improved_modules.spatiotemporal import SpatiotemporalController


class SpatiotemporalTests(unittest.TestCase):
    def controller(self, **config):
        settings = {'schedule_steps': [], 'cutoffs_hz': [None],
                    'time_attention': True, 'spatial_attention': True}
        settings.update(config)
        return SpatiotemporalController(settings, .001)

    def test_absolute_schedule_boundaries_and_identity_final_filter(self):
        controller = self.controller(schedule_steps=[30, 70], cutoffs_hz=[5., 8., None])
        value = torch.randn(1, 2, 65, 3)
        for step, cutoff in [(0, 5.), (29, 5.), (30, 8.), (69, 8.), (70, None), (123, None)]:
            controller.set_step(step)
            self.assertEqual(controller.current_cutoff_hz, cutoff)
        self.assertIs(controller.filter_data(value), value)

    def test_filter_passes_low_frequency_and_rejects_high_frequency(self):
        controller = self.controller(cutoffs_hz=[5.])
        t = torch.arange(4096, dtype=torch.float64) * .001
        low = torch.sin(2 * torch.pi * 2 * t)
        high = torch.sin(2 * torch.pi * 30 * t)
        low_out = controller.filter_data(low[None, None, :, None])[0, 0, :, 0]
        high_out = controller.filter_data(high[None, None, :, None])[0, 0, :, 0]
        interior = slice(1000, -1000)
        low_gain = (low_out[interior].square().mean() / low[interior].square().mean()).sqrt()
        high_gain = (high_out[interior].square().mean() / high[interior].square().mean()).sqrt()
        self.assertGreater(float(low_gain), .98)
        self.assertLess(float(high_gain), .005)

    def test_filter_does_not_mix_shots_receivers_or_batch(self):
        controller = self.controller(cutoffs_hz=[8.])
        value = torch.zeros(2, 3, 128, 4, dtype=torch.float64)
        value[1, 2, 30, 3] = 1.
        output = controller.filter_data(value)
        self.assertEqual(output.shape, value.shape)
        self.assertGreater(float(output[1, 2, :, 3].abs().sum()), 0.)
        output[1, 2, :, 3] = 0.
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    def test_filtered_objective_gradient_matches_finite_difference(self):
        controller = self.controller(cutoffs_hz=[8.], time_attention=False)
        generator = torch.Generator().manual_seed(71)
        basis = torch.randn(1, 2, 128, 3, dtype=torch.float64, generator=generator)
        observed = torch.randn(1, 2, 128, 3, dtype=torch.float64, generator=generator)
        parameter = torch.tensor(.7, dtype=torch.float64, requires_grad=True)
        controller.data_loss(parameter * basis, observed).backward()
        epsilon = 1e-6
        plus = controller.data_loss((parameter.detach() + epsilon) * basis, observed)
        minus = controller.data_loss((parameter.detach() - epsilon) * basis, observed)
        numerical = (plus - minus) / (2 * epsilon)
        self.assertGreater(abs(float(parameter.grad)), 1e-7)
        torch.testing.assert_close(parameter.grad, numerical, rtol=1e-7, atol=1e-9)

    def test_step_zero_and_disabled_attention_are_exact_unweighted_mse(self):
        prediction, observed = torch.randn(2, 3, 129, 4), torch.randn(2, 3, 129, 4)
        for controller in (self.controller(), self.controller(time_attention=False)):
            if not controller.time_attention:
                controller.set_step(20)
            actual = controller.data_loss(prediction, observed)
            torch.testing.assert_close(actual, (prediction - observed).square().mean(), rtol=0, atol=0)
            torch.testing.assert_close(controller.last_time_weights, torch.ones_like(prediction), rtol=0, atol=0)

    def test_time_weights_bounded_mean_one_detached_and_gradient_correct(self):
        controller = self.controller(time_strength=.4, time_window=32)
        controller.set_step(10)
        observed = torch.rand(2, 3, 129, 4, dtype=torch.float64)
        prediction = torch.randn_like(observed).requires_grad_()
        loss = controller.data_loss(prediction, observed, update_state=True)
        weights = controller.last_time_weights
        self.assertEqual(weights.shape, prediction.shape)
        self.assertFalse(weights.requires_grad)
        self.assertIsNone(weights.grad_fn)
        self.assertGreaterEqual(float(weights.min()), .6 - 1e-12)
        self.assertLessEqual(float(weights.max()), 1.4 + 1e-12)
        torch.testing.assert_close(weights.mean(dim=2), torch.ones_like(weights.mean(dim=2)), rtol=0, atol=1e-12)
        loss.backward()
        torch.testing.assert_close(prediction.grad, 2 * (prediction.detach() - observed) * weights / prediction.numel())

    def test_silent_observed_windows_never_receive_positive_attention(self):
        controller = self.controller(time_window=32)
        controller.set_step(10)
        observed = torch.zeros(1, 2, 96, 3, dtype=torch.float64)
        observed[:, :, 32:64] = 1.
        prediction = torch.full_like(observed, 100.)
        prediction[:, :, 32:64] = 1.1
        controller.data_loss(prediction, observed)
        weights = controller.last_time_weights
        self.assertTrue(bool((weights[:, :, :32] <= 1.).all()))
        self.assertTrue(bool((weights[:, :, 64:] <= 1.).all()))
        controller.data_loss(prediction, torch.zeros_like(observed))
        torch.testing.assert_close(controller.last_time_weights, torch.ones_like(observed), rtol=0, atol=0)

    def test_time_weights_are_trace_local_and_stateless_without_truth(self):
        controller = self.controller(time_window=16)
        controller.set_step(11)
        observed = torch.rand(1, 2, 80, 3, dtype=torch.float64)
        prediction = observed + torch.randn_like(observed) * .2
        first = controller.data_loss(prediction, observed)
        weights = controller.last_time_weights.clone()
        changed = prediction.clone()
        changed[:, 1, :, 2] += 20
        controller.data_loss(changed, observed, update_state=True)
        torch.testing.assert_close(controller.last_time_weights[:, 0], weights[:, 0], rtol=0, atol=0)
        torch.testing.assert_close(controller.last_time_weights[:, 1, :, :2], weights[:, 1, :, :2], rtol=0, atol=0)
        repeated = controller.data_loss(prediction, observed, update_state=False)
        torch.testing.assert_close(repeated, first, rtol=0, atol=0)
        torch.testing.assert_close(controller.last_time_weights, weights, rtol=0, atol=0)

    def test_time_ramp_controls_maximum_deviation(self):
        controller = self.controller(time_strength=.4, ramp_steps=10, time_window=16)
        controller.set_step(5)
        controller.data_loss(torch.randn(1, 1, 100, 2), torch.rand(1, 1, 100, 2))
        self.assertLessEqual(float((controller.last_time_weights - 1).abs().max()), .200001)

    def test_spatial_identity_zero_finiteness_and_positive_direction(self):
        controller = self.controller(spatial_strength=.4)
        gradient = torch.randn(2, 13, 17, dtype=torch.float64, requires_grad=True)
        torch.testing.assert_close(controller.spatial_gradient(gradient), gradient, rtol=0, atol=0)
        controller.set_step(10)
        zero = torch.zeros_like(gradient)
        torch.testing.assert_close(controller.spatial_gradient(zero), zero, rtol=0, atol=0)
        torch.testing.assert_close(controller.last_spatial_weights, torch.ones_like(zero), rtol=0, atol=0)
        result = controller.spatial_gradient(gradient)
        weights = controller.last_spatial_weights
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(result.shape, gradient.shape)
        self.assertFalse(weights.requires_grad)
        self.assertIsNone(weights.grad_fn)
        self.assertGreaterEqual(float(weights.min()), .6 - 1e-12)
        self.assertLessEqual(float(weights.max()), 1.4 + 1e-12)
        torch.testing.assert_close(weights.mean(dim=(1, 2)), torch.ones(2, dtype=torch.float64), rtol=0, atol=1e-12)
        self.assertTrue(bool((result * gradient >= 0).all()))

    def test_disabled_spatial_attention_records_exact_ones_after_ramp(self):
        controller = self.controller(spatial_attention=False)
        controller.set_step(70)
        gradient = torch.randn(2, 9, 11)
        self.assertIs(controller.spatial_gradient(gradient), gradient)
        torch.testing.assert_close(controller.last_spatial_weights, torch.ones_like(gradient), rtol=0, atol=0)

    def test_spatial_zero_response_not_amplified_and_depth_bias_fades(self):
        controller = self.controller(schedule_steps=[30, 70], cutoffs_hz=[5., 8., None])
        controller.set_step(10)
        gradient = torch.zeros(1, 15, 15, dtype=torch.float64)
        gradient[:, 7, 7] = 1
        controller.spatial_gradient(gradient)
        self.assertLessEqual(float(controller.last_spatial_weights[0, 0, 0]), 1.)
        uniform = torch.ones_like(gradient)
        controller.spatial_gradient(uniform)
        self.assertGreater(float(controller.last_spatial_weights[0, -1].mean()),
                           float(controller.last_spatial_weights[0, 0].mean()))
        controller.set_step(70)
        controller.spatial_gradient(uniform)
        torch.testing.assert_close(controller.last_spatial_weights, uniform, rtol=0, atol=1e-12)

    def test_depth_preference_decays_in_magnitude_before_full_band(self):
        controller = self.controller(schedule_steps=[30, 70], cutoffs_hz=[5., 8., None],
                                     spatial_strength=.4, spatial_depth_bias=.25)
        uniform = torch.ones(2, 15, 17, dtype=torch.float64)
        deviations = []
        for step in (10, 30, 70):
            controller.set_step(step)
            controller.spatial_gradient(uniform)
            deviations.append(controller.last_spatial_weights - 1.)
        self.assertGreater(float(deviations[0].abs().max()), 0.)
        torch.testing.assert_close(deviations[1], .5 * deviations[0], rtol=0, atol=1e-12)
        torch.testing.assert_close(deviations[2], torch.zeros_like(uniform), rtol=0, atol=1e-12)
        self.assertAlmostEqual(float(deviations[0].abs().max()), .4 * .25, places=12)

    def test_spatial_zero_response_stays_unamplified_in_every_stage(self):
        controller = self.controller(schedule_steps=[30, 70], cutoffs_hz=[5., 8., None])
        gradient = torch.zeros(1, 15, 15, dtype=torch.float64)
        gradient[:, 7, 7] = 1.
        for step in (10, 30, 70):
            controller.set_step(step)
            controller.spatial_gradient(gradient)
            weights = controller.last_spatial_weights
            self.assertLessEqual(float(weights[0, 0, 0]), 1.)
            self.assertLessEqual(float(weights[0, -1, -1]), 1.)
            torch.testing.assert_close(weights.mean(), torch.tensor(1., dtype=torch.float64), rtol=0, atol=1e-12)

    def test_invalid_configuration_shapes_and_steps_fail(self):
        invalid = [dict(schedule_steps=[30, 20], cutoffs_hz=[5, 8, None]),
                   dict(schedule_steps=[1.5], cutoffs_hz=[5, None]),
                   dict(schedule_steps=[0], cutoffs_hz=[5, None]),
                   dict(schedule_steps=[30], cutoffs_hz=[None]),
                   dict(cutoffs_hz=[600]), dict(cutoffs_hz=[0]), dict(cutoffs_hz=[float('nan')]),
                   dict(time_window=0), dict(time_window=2.5), dict(ramp_steps=0),
                   dict(time_strength=.6), dict(spatial_strength=-.1), dict(spatial_depth_bias=float('nan')),
                   dict(residual_clip=0), dict(energy_floor=0), dict(time_attention='true')]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.controller(**config)
        for dt in (0, -1, float('nan'), float('inf')):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                SpatiotemporalController({}, dt)
        controller = self.controller()
        for step in (-1, 1.5, True):
            with self.subTest(step=step), self.assertRaises(ValueError):
                controller.set_step(step)
        with self.assertRaises(ValueError):
            controller.data_loss(torch.ones(3, 4), torch.ones(3, 4))
        with self.assertRaises(ValueError):
            controller.data_loss(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 1))
        with self.assertRaises(ValueError):
            controller.spatial_gradient(torch.ones(2, 3))


if __name__ == '__main__':
    unittest.main()
