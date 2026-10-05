import copy
import unittest
import numpy as np
import torch
from helpers import ROOT, tiny
from parameter_sweep import make_cases, validate_case, source_positions
from run_experiment import load_config
from ifwi_modules import IRN


class ParameterSweepTests(unittest.TestCase):
    def setUp(self):
        self.base = load_config(ROOT / 'experiments/configs/baseline.yaml')

    def test_nested_shots_preserve_original_aperture(self):
        original = np.arange(20, 278, 20)
        np.testing.assert_array_equal(source_positions(288, {}), original)
        for count in (13, 25, 49):
            positions = source_positions(288, {'num_shots': count})
            self.assertEqual(len(positions), count)
            self.assertEqual(len(set(positions)), count)
            self.assertTrue(set(original).issubset(positions))
            self.assertEqual((positions[0], positions[-1]), (20, 260))
        with self.assertRaises(ValueError):
            source_positions(288, {'num_shots': 300})

    def test_each_case_changes_only_named_factor(self):
        cases = make_cases(self.base)
        self.assertEqual(len(cases), 10)
        baseline = cases[0]['config']
        for case in cases:
            validate_case(baseline, case)
            if case['factor'] == 'depth':
                self.assertEqual(set(case['config']['model']['neuron'][1:-1]), {128})
            if case['factor'] == 'width':
                self.assertEqual(len(case['config']['model']['neuron']) - 2, 4)

    def test_original_suite_uses_full_shots_and_rejects_batch_override(self):
        cases = make_cases(self.base)
        self.assertEqual({c['config']['training']['shot_batch_size'] for c in cases}, {None})
        for case in cases:
            validate_case(cases[0]['config'], case)
        for invalid in (0, -1, True, 1.5, 13):
            with self.assertRaises(ValueError):
                make_cases(self.base, shot_batch_size=invalid)

    def test_all_registered_network_shapes_have_finite_forward_and_gradient(self):
        for case in make_cases(self.base):
            with self.subTest(case=case['name']):
                mc = case['config']['model']
                torch.manual_seed(42)
                network = IRN(neuron=mc['neuron'], omega_0=mc['omega_0'], outermost_linear=True)
                pred, _ = network(torch.rand(1, 8, 9, 2))
                self.assertEqual(pred.shape, (1, 8, 9, 1))
                self.assertEqual(network.omega_0, mc['omega_0'])
                self.assertTrue(torch.isfinite(pred).all())
                pred.square().mean().backward()
                for parameter in network.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_reject_confounded_configuration(self):
        cases = make_cases(self.base)
        bad = copy.deepcopy(cases[1])
        bad['config']['optimizer']['learning_rate'] *= 2
        with self.assertRaises(ValueError):
            validate_case(cases[0]['config'], bad)
        bad = copy.deepcopy(next(c for c in cases if c['factor'] == 'depth'))
        bad['config']['model']['neuron'][1] = 512
        with self.assertRaises(ValueError):
            validate_case(cases[0]['config'], bad)

    def test_original_protocol_rejects_a_baseline_disguised_as_a_treatment(self):
        baseline = make_cases(self.base)[0]['config']
        case = dict(name='width_128', factor='width', value=128, config=copy.deepcopy(baseline))
        with self.assertRaises(ValueError):
            validate_case(baseline, case)

    def test_shot_accumulation_matches_full_batch_with_unequal_chunks(self):
        model, data, _ = tiny({'training': {'clip_grad': None, 'shot_batch_size': 2}})
        for key in ('xs', 'zs', 'xr', 'zr'):
            value = getattr(model.rnn, key)
            setattr(model.rnn, key, torch.cat([value, value[:, -1:]], dim=1))
        shots = torch.cat([data['shots'], data['shots'][:, -1:]], dim=1)
        reference = copy.deepcopy(model)
        reference.shot_batch_size = None
        batches = []
        hook = model.rnn.register_forward_hook(lambda module, args, output: batches.append(output[2].shape[1]))
        opt = torch.optim.Adam(model.vel_net.parameters(), lr=1e-4)
        refopt = torch.optim.Adam(reference.vel_net.parameters(), lr=1e-4)
        _, actual = model.train_one_epoch(opt, wavelet=data['wavelet'], shots=shots)
        _, expected = reference.train_one_epoch(refopt, wavelet=data['wavelet'], shots=shots)
        hook.remove()
        self.assertEqual(batches, [2, 1])
        self.assertEqual(model.rnn.xs.shape[1], 3)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        for p, q in zip(model.vel_net.parameters(), reference.vel_net.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-7)
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
