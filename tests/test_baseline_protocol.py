"""Reject drift from the verified original random IFWI experiment."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT.parent / 'IFWI 1' / '11479806'


def verified_config():
    # Independent fixture derived from the archived random config and runner.
    return {
        'experiment_name': 'legacy_random_baseline',
        'description': 'Verified original random IFWI',
        'seed': 3,
        'execution': {'protocol': 'original_baseline'},
        'model': {'network_type': 'vanilla', 'neuron': [2, 128, 128, 128, 128, 1],
                  'omega_0': 30, 'activation': 'sine', 'dropout': False,
                  'prob': .2, 'outermost_linear': True, 'bias': True},
        'loss': {'type': 'standard_mse', 'use_prior': False},
        'optimizer': {'optimizer_type': 'adam', 'learning_rate': .0001,
                      'use_scheduler': False},
        'training': {'max_iterations': 4001, 'log_interval': 100, 'alpha': 0,
                     'clip_grad': None, 'shot_batch_size': None},
        'data': {'model_file': 'vel_marmousi_376x1151.csv', 'downsample': 4,
                 'noise_level': 0., 'dz': 15, 'dt': .0019, 'nt': 1000,
                 'num_shots': 13},
        'evaluation': {'depth_threshold': .5, 'corner_size': .25, 'save_plots': True},
        'gradient_preconditioner': {'enabled': False},
    }


class BaselineProtocolTests(unittest.TestCase):
    def module(self):
        path = ROOT / 'experiments' / 'baseline_protocol.py'
        self.assertTrue(path.is_file(), 'The original-baseline guard is missing')
        spec = importlib.util.spec_from_file_location('baseline_protocol_guard_tests', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_canonical_profile_matches_the_verified_runner(self):
        guard = self.module()
        config = guard.baseline_config()
        expected = verified_config()
        for profile in (config, expected):
            profile.pop('experiment_name', None)
            profile.pop('description', None)
        self.assertEqual(config, expected)
        guard.validate_baseline_protocol(verified_config())

    def test_retrieving_the_profile_cannot_mutate_future_validation(self):
        guard = self.module()
        changed = guard.baseline_config()
        changed['model']['neuron'][1] = 512
        with self.assertRaises(ValueError):
            guard.validate_baseline_protocol(changed)
        guard.validate_baseline_protocol(verified_config())

    def test_common_parameter_drift_is_rejected(self):
        guard = self.module()
        changes = [
            ('seed', None, 42),
            ('execution', 'protocol', 'checked_v2'),
            ('model', 'network_type', 'attention'),
            ('model', 'neuron', [2, 256, 256, 256, 256, 1]),
            ('model', 'omega_0', 20),
            ('model', 'activation', 'relu'),
            ('model', 'bias', False),
            ('model', 'outermost_linear', False),
            ('model', 'dropout', True),
            ('model', 'prob', .4),
            ('loss', 'type', 'other'),
            ('loss', 'use_prior', True),
            ('optimizer', 'optimizer_type', 'adamw'),
            ('optimizer', 'learning_rate', .0002),
            ('optimizer', 'use_scheduler', True),
            ('training', 'max_iterations', 4000),
            ('training', 'log_interval', 99),
            ('training', 'alpha', .1),
            ('training', 'clip_grad', .25),
            ('training', 'shot_batch_size', 13),
            ('data', 'model_file', 'other.csv'),
            ('data', 'downsample', 2),
            ('data', 'noise_level', .1),
            ('data', 'dz', 20),
            ('data', 'dt', .001),
            ('data', 'nt', 999),
            ('data', 'num_shots', 25),
            ('evaluation', 'depth_threshold', .4),
            ('evaluation', 'corner_size', .2),
            ('gradient_preconditioner', 'enabled', True),
        ]
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                config = verified_config()
                if key is None:
                    config[section] = value
                else:
                    config[section][key] = value
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config)

    def test_unknown_settings_cannot_enable_an_unchecked_path(self):
        guard = self.module()
        for section, key, value in [
            (None, 'spatiotemporal', {'enabled': True}),
            (None, 'pretrained', 'smooth.pth'),
            ('loss', 'data_objective', 'huber'),
            ('loss', 'use_multiscale', True),
            ('optimizer', 'weight_decay', .01),
            ('optimizer', 'use_grad_modifier', True),
            ('model', 'attention', {'enabled': True}),
            ('training', 'segment_size', 100),
            ('data', 'frequency', 12),
            ('evaluation', 'choose_best_by_truth', True),
        ]:
            with self.subTest(section=section, key=key):
                config = verified_config()
                (config if section is None else config[section])[key] = value
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True)

    def test_required_sections_and_execution_marker_cannot_be_omitted(self):
        guard = self.module()
        for key in ('seed', 'execution', 'model', 'loss', 'optimizer', 'training',
                    'data', 'evaluation', 'gradient_preconditioner'):
            with self.subTest(key=key):
                config = verified_config()
                config.pop(key)
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config)

    def test_descriptions_and_plot_output_do_not_change_training(self):
        guard = self.module()
        config = verified_config()
        config['experiment_name'] = 'shots_comparison'
        config.pop('description')
        config['evaluation']['save_plots'] = False
        original = copy.deepcopy(config)
        guard.validate_baseline_protocol(config)
        self.assertEqual(config, original)

    def test_only_registered_single_factor_treatments_are_allowed(self):
        guard = self.module()
        treatments = [
            ('data', 'num_shots', 25), ('data', 'num_shots', 49),
            ('model', 'neuron', [2] + [128] * 6 + [1]),
            ('model', 'neuron', [2] + [128] * 8 + [1]),
            ('model', 'neuron', [2] + [256] * 4 + [1]),
            ('model', 'neuron', [2] + [512] * 4 + [1]),
            ('model', 'omega_0', 10), ('model', 'omega_0', 20),
            ('model', 'omega_0', 50),
        ]
        for section, key, value in treatments:
            with self.subTest(section=section, key=key, value=value):
                config = verified_config()
                config[section][key] = value
                guard.validate_baseline_protocol(config, allow_treatment=True)
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config)

    def test_multiple_factors_and_unregistered_values_are_rejected(self):
        guard = self.module()
        changes = [
            {'data': {'num_shots': 25}, 'model': {'omega_0': 20}},
            {'model': {'neuron': [2] + [256] * 6 + [1]}},
            {'model': {'neuron': [2, 128, 256, 128, 128, 1]}},
            {'model': {'neuron': [2] + [384] * 4 + [1]}},
            {'model': {'omega_0': 40}},
            {'data': {'num_shots': 26}},
        ]
        for update in changes:
            with self.subTest(update=update):
                config = verified_config()
                for section, values in update.items():
                    config[section].update(values)
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True)

    def test_preliminary_runs_relax_only_positive_budgets_and_log_intervals(self):
        guard = self.module()
        config = verified_config()
        config['training'].update(max_iterations=2, log_interval=1)
        guard.validate_baseline_protocol(config, preliminary=True)
        with self.assertRaises(ValueError):
            guard.validate_baseline_protocol(config)
        for key in ('max_iterations', 'log_interval'):
            for value in (0, -1, True, 1.5, None):
                with self.subTest(key=key, value=value):
                    invalid = copy.deepcopy(config)
                    invalid['training'][key] = value
                    with self.assertRaises(ValueError):
                        guard.validate_baseline_protocol(invalid, preliminary=True)
        config['optimizer']['learning_rate'] = .0002
        with self.assertRaises(ValueError):
            guard.validate_baseline_protocol(config, preliminary=True)

    def test_bool_values_cannot_impersonate_numeric_parameters(self):
        guard = self.module()
        for section, key, value in [('training', 'alpha', False),
                                    ('model', 'neuron', [2, 128, 128, 128, 128, True]),
                                    ('data', 'noise_level', False)]:
            with self.subTest(section=section, key=key):
                config = verified_config()
                config[section][key] = value
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True)

    def test_integer_parameters_cannot_be_replaced_by_float_values(self):
        guard = self.module()
        for section, key, value in [('seed', None, 3.),
                                    ('data', 'num_shots', 25.),
                                    ('data', 'downsample', 4.),
                                    ('model', 'omega_0', 20.),
                                    ('model', 'neuron', [2, 128, 128, 128, 128, 1.])]:
            with self.subTest(section=section, key=key):
                config = verified_config()
                if key is None:
                    config[section] = value
                else:
                    config[section][key] = value
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True)

    def test_custom_single_factor_values_require_both_explicit_flags(self):
        guard = self.module()
        config = verified_config()
        config['data']['num_shots'] = 17
        for treatment, custom in ((False, False), (True, False), (False, True)):
            with self.subTest(treatment=treatment, custom=custom):
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=treatment,
                                                     allow_custom=custom)
        guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True)

    def test_custom_values_preserve_fixed_aperture_and_network_factor(self):
        guard = self.module()
        values = [
            ('data', 'num_shots', 2), ('data', 'num_shots', 17), ('data', 'num_shots', 241),
            ('model', 'neuron', [2, 128, 1]),
            ('model', 'neuron', [2] + [128] * 3 + [1]),
            ('model', 'neuron', [2] + [128] * 9 + [1]),
            ('model', 'neuron', [2] + [1] * 4 + [1]),
            ('model', 'neuron', [2] + [192] * 4 + [1]),
            ('model', 'neuron', [2] + [384] * 4 + [1]),
            ('model', 'omega_0', .5), ('model', 'omega_0', 31.25),
            ('model', 'omega_0', 30.),
        ]
        for section, key, value in values:
            with self.subTest(section=section, key=key, value=value):
                config = verified_config()
                config[section][key] = value
                original = copy.deepcopy(config)
                guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True)
                self.assertEqual(config, original)
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True)

    def test_custom_values_reject_invalid_ranges_types_and_network_shapes(self):
        guard = self.module()
        values = [
            ('data', 'num_shots', value) for value in (1, 242, 0, -1, 17., True, None)
        ] + [
            ('model', 'neuron', value) for value in (
                [2, 1], [2, 0, 0, 0, 0, 1], [2, -1, -1, -1, -1, 1],
                [2, 128., 128., 128., 128., 1], [2, 128, 128, 128, 128, True],
                [2, 128, 192, 128, 128, 1],
                [3, 128, 1], [2, 128, 2], None,
            )
        ] + [
            ('model', 'omega_0', value)
            for value in (0, -1, True, float('nan'), float('inf'), float('-inf'), '31', None)
        ]
        for section, key, value in values:
            with self.subTest(section=section, key=key, value=value):
                config = verified_config()
                config[section][key] = value
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True)

    def test_custom_mode_accepts_combined_parameters(self):
        guard = self.module()
        updates = [
            {'data': {'num_shots': 17}, 'model': {'omega_0': 31.25}},
            {'model': {'neuron': [2] + [192] * 3 + [1]}},
            {'model': {'neuron': [2] + [128] * 3 + [1], 'omega_0': 31.25}},
            {'data': {'num_shots': 17}, 'model': {'neuron': [2] + [192] * 3 + [1], 'omega_0': 31.25}},
            {'data': {'num_shots': 241}, 'model': {'neuron': [2, 1, 1], 'omega_0': .5}},
        ]
        for update in updates:
            with self.subTest(update=update):
                config = verified_config()
                for section, values in update.items():
                    config[section].update(values)
                original = copy.deepcopy(config)
                guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True)
                self.assertEqual(config, original)
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True)

    def test_custom_mode_rejects_common_parameter_drift(self):
        guard = self.module()
        updates = [
            {'data': {'num_shots': 17}, 'optimizer': {'learning_rate': .0002}},
            {'data': {'num_shots': 17}, 'training': {'shot_batch_size': 13}},
            {'data': {'num_shots': 17, 'noise_level': .1}},
            {'data': {'num_shots': 17, 'dz': 20}},
            {'data': {'num_shots': 17, 'dt': .001}},
            {'data': {'num_shots': 17, 'nt': 999}},
            {'model': {'omega_0': 31.25, 'activation': 'relu'}},
            {'model': {'omega_0': 31.25}, 'loss': {'data_objective': 'huber'}},
        ]
        for update in updates:
            with self.subTest(update=update):
                config = verified_config()
                for section, values in update.items():
                    config[section].update(values)
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True)

    def test_custom_mode_keeps_preliminary_budget_permission_separate(self):
        guard = self.module()
        config = verified_config()
        config['model']['omega_0'] = 31.25
        config['training'].update(max_iterations=2, log_interval=1)
        with self.assertRaises(ValueError):
            guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True)
        guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True,
                                         preliminary=True)
        config['seed'] = 42
        with self.assertRaises(ValueError):
            guard.validate_baseline_protocol(config, allow_treatment=True, allow_custom=True,
                                             preliminary=True)

    def test_invalid_shapes_raise_a_protocol_error(self):
        guard = self.module()
        for config in (None, [], {'model': None}, {'training': 'unvalidated'}):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    guard.validate_baseline_protocol(config)

    def source_fixture(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in ('ifwi_modules.py', 'rnn_fd.py', 'generator.py', 'plot_functions.py'):
            shutil.copy2(ROOT / name, root / name)
        (root / 'data').mkdir()
        shutil.copy2(ROOT / 'data/vel_marmousi_376x1151.csv',
                     root / 'data/vel_marmousi_376x1151.csv')
        reference = root / 'experiments/baseline_reference/ifwi_experiment.py'
        reference.parent.mkdir(parents=True)
        source = ROOT / 'experiments/baseline_reference/ifwi_experiment.py'
        if not source.exists():
            source = ORIGINAL / 'ifwi_experiment.py'
        shutil.copy2(source, reference)
        return root

    def test_source_guard_accepts_verified_bytes_and_reports_their_hashes(self):
        guard = self.module()
        root = self.source_fixture()
        actual = guard.validate_original_sources(root)
        self.assertEqual(set(actual), {
            'ifwi_modules.py', 'rnn_fd.py', 'generator.py', 'plot_functions.py',
            'data/vel_marmousi_376x1151.csv',
            'experiments/baseline_reference/ifwi_experiment.py',
        })
        self.assertEqual(actual['ifwi_modules.py'],
                         '900adab18822769946a64109fa77314d71d396403e70179b815094cd952bb043')

    def test_source_guard_rejects_changed_missing_and_caller_supplied_identity(self):
        guard = self.module()
        for name in ('ifwi_modules.py', 'rnn_fd.py', 'generator.py', 'plot_functions.py',
                     'data/vel_marmousi_376x1151.csv',
                     'experiments/baseline_reference/ifwi_experiment.py'):
            with self.subTest(name=name):
                root = self.source_fixture()
                path = root / name
                path.write_bytes(path.read_bytes() + b'\n# drift\n')
                # A freshly computed caller manifest cannot redefine the author source.
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                (root / 'source_hashes.json').write_text(json.dumps({name: digest}))
                with self.assertRaisesRegex(ValueError, name.replace('.', r'\.')):
                    guard.validate_original_sources(root)
                path.unlink()
                with self.assertRaisesRegex(ValueError, name.replace('.', r'\.')):
                    guard.validate_original_sources(root)

    def test_source_identity_table_cannot_be_mutated_by_callers(self):
        guard = self.module()
        root = self.source_fixture()
        first = guard.validate_original_sources(root)
        first['ifwi_modules.py'] = 'untrusted'
        self.assertNotEqual(guard.validate_original_sources(root)['ifwi_modules.py'], 'untrusted')

    def test_guard_import_does_not_load_the_training_stack(self):
        before = set(sys.modules)
        self.module()
        self.assertFalse({'torch', 'numpy', 'run_experiment', 'ifwi_modules'} &
                         (set(sys.modules) - before))


if __name__ == '__main__':
    unittest.main()
