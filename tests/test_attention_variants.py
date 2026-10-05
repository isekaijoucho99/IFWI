"""Behavioral controls for identity-initialized, coordinate-local attention."""
import unittest

import torch

from helpers import ROOT  # Adds the repository and experiment import paths.
from ifwi_modules import IRN
from improved_modules.networks import create_improved_network


class AttentionVariantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        self.coords = torch.rand(1, 5, 7, 2)
        self.coords[..., 1] *= 1.4

    def make_network(self, kind, **kwargs):
        config = dict(network_type='attention', attention_type=kind,
                      neuron=[2, 12, 12, 1], attention_hidden=8,
                      depth_min=0., depth_max=1.4, outermost_linear=True)
        config.update(kwargs)
        return create_improved_network(config)

    def test_identity_initialization_preserves_author_output_and_backbone_gradient(self):
        # Any nonidentity gate or modulation after output bias breaks this control.
        for kind in ('depth_residual', 'feature_film'):
            with self.subTest(kind=kind):
                reference = IRN(neuron=[2, 12, 12, 1], outermost_linear=True)
                network = self.make_network(kind)
                network.linear.load_state_dict(reference.linear.state_dict())
                expected = reference(self.coords)[0]
                actual = network(self.coords)[0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                target = torch.linspace(-1., 1., actual.numel()).reshape_as(actual)
                ((actual - target) ** 2).sum().backward()
                ((expected - target) ** 2).sum().backward()
                for got, want in zip(network.linear.parameters(), reference.linear.parameters()):
                    torch.testing.assert_close(got.grad, want.grad, rtol=0, atol=0)

    def test_attention_parameters_receive_gradient_and_change_predictions(self):
        # Zeroing the entire gate or detaching it would make the new variants inert.
        for kind in ('depth_residual', 'feature_film'):
            with self.subTest(kind=kind):
                network = self.make_network(kind)
                for parameter in network.linear.parameters():
                    parameter.requires_grad_(False)
                attention_parameters = list(network.depth_attention.parameters())
                optimizer = torch.optim.SGD(attention_parameters, lr=.05)
                initial = network(self.coords)[0].detach().clone()
                for _ in range(2):
                    optimizer.zero_grad()
                    (network(self.coords)[0] - 1.).square().mean().backward()
                    self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                        for p in attention_parameters))
                    self.assertGreater(sum(float(p.grad.abs().sum())
                                           for p in attention_parameters), 0.)
                    optimizer.step()
                self.assertGreater(float(attention_parameters[0].grad.abs().sum()), 0.)
                self.assertFalse(torch.equal(initial, network(self.coords)[0]))

    def test_trained_gate_is_coordinate_subset_invariant(self):
        # Batch-relative depth normalization would make cropped evaluations change.
        for kind in ('legacy_output', 'depth_residual', 'feature_film'):
            with self.subTest(kind=kind):
                network = self.make_network(kind)
                optimizer = torch.optim.SGD(network.depth_attention.parameters(), lr=.1)
                (network(self.coords)[0] - 1.).square().mean().backward()
                optimizer.step()
                full = network(self.coords)[0]
                part = network(self.coords[:, 1:3, 2:5])[0]
                torch.testing.assert_close(part, full[:, 1:3, 2:5], rtol=1e-5, atol=1e-7)

    def test_residual_and_film_gain_remain_positive_and_bounded(self):
        # Removing tanh or confusing gain with raw logits violates stable gating.
        for kind in ('depth_residual', 'feature_film'):
            with self.subTest(kind=kind):
                network = self.make_network(kind, attention_strength=.4,
                                            attention_shift_strength=.08)
                with torch.no_grad():
                    for parameter in network.depth_attention.parameters():
                        parameter.fill_(100.)
                depths = torch.tensor([[[-100.], [0.], [1.4], [100.]]])
                affine_layers = [layer for layer in network.depth_attention.modules()
                                 if isinstance(layer, torch.nn.Linear)]
                for sign in (-1., 1.):
                    with torch.no_grad():
                        affine_layers[-1].weight.fill_(100.*sign)
                        affine_layers[-1].bias.fill_(100.*sign)
                    if kind == 'feature_film':
                        gain, shift = network.depth_attention.modulation(depths)
                        self.assertTrue(torch.isfinite(shift).all())
                        self.assertLessEqual(float(shift.detach().abs().max()), .08000001)
                    else:
                        gain = network.depth_attention(depths)
                    self.assertTrue(torch.isfinite(gain).all())
                    self.assertGreaterEqual(float(gain.detach().min()), .5999999)
                    self.assertLessEqual(float(gain.detach().max()), 1.4000001)

    def test_film_modulates_hidden_features_before_output_bias(self):
        # Output gating would rescale the final bias and cannot match this example.
        network = self.make_network('feature_film', neuron=[2, 2, 1], activation='relu')
        with torch.no_grad():
            network.linear[0].weight.copy_(torch.eye(2))
            network.linear[0].bias.zero_()
            network.linear[1].weight.copy_(torch.tensor([[2., 3.]]))
            network.linear[1].bias.fill_(5.)
            affine_layers = [layer for layer in network.depth_attention.modules()
                             if isinstance(layer, torch.nn.Linear)]
            affine_layers[-1].weight.zero_()
            # scale=[1.2,.8], shift=[.05,-.02], with bounds .5 and .1.
            affine_layers[-1].bias.copy_(torch.atanh(torch.tensor([.4, -.4, .5, -.2])))
        actual = network(torch.tensor([[[[1., .5]]]]))[0]
        torch.testing.assert_close(actual, torch.tensor([[[[8.64]]]]))

    def test_invalid_attention_configuration_is_rejected(self):
        # Typos must not silently choose an unrelated attention experiment.
        with self.assertRaises(ValueError):
            self.make_network('misspelled_kind')
        for kind in ('depth_residual', 'feature_film'):
            for strength in (0., -1., 1., float('nan'), float('inf')):
                with self.subTest(kind=kind, strength=strength), self.assertRaises(ValueError):
                    self.make_network(kind, attention_strength=strength)
            with self.assertRaises(ValueError):
                self.make_network(kind, depth_min=1., depth_max=1.)
            with self.assertRaises(ValueError):
                self.make_network(kind, attention_hidden=1)
        for shift in (0., -1., float('nan'), float('inf')):
            with self.subTest(shift=shift), self.assertRaises(ValueError):
                self.make_network('feature_film', attention_shift_strength=shift)

    def test_default_attention_preserves_explicit_legacy_initialization(self):
        # Adding a new default must not silently reinterpret existing YAML files.
        config = dict(network_type='attention', neuron=[2, 12, 12, 1],
                      attention_hidden=8, depth_max=1.4)
        torch.manual_seed(53)
        default = create_improved_network(config)
        torch.manual_seed(53)
        explicit = create_improved_network(dict(config, attention_type='legacy_output'))
        self.assertEqual(set(default.state_dict()), set(explicit.state_dict()))
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(default(self.coords)[0], explicit(self.coords)[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
