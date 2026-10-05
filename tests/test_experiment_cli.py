"""The public command runs one original IFWI experiment with chosen parameters."""
import importlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from helpers import ROOT
from baseline_protocol import baseline_config


class ExperimentCliTests(unittest.TestCase):
    def cli(self):
        self.assertTrue((ROOT / 'experiment.py').is_file(), 'Simple experiment.py entry is missing')
        return importlib.import_module('experiment')

    def test_no_arguments_select_one_baseline(self):
        args, config = self.cli().build_run([])
        reference = baseline_config()
        for key in ('model', 'data', 'optimizer', 'loss', 'training', 'seed', 'execution'):
            self.assertEqual(config[key], reference[key])
        self.assertEqual(config['experiment_name'], 'baseline')
        self.assertFalse(args.preliminary)

    def test_each_preset_matches_its_single_parameter_override(self):
        cli = self.cli()
        self.assertEqual(len(cli.PRESETS), 10)
        for name, overrides in cli.PRESETS.items():
            with self.subTest(preset=name):
                _, preset = cli.build_run(['--preset', name])
                arguments = [part for factor, value in overrides.items() for part in ('-' + factor, str(value))]
                _, explicit = cli.build_run(arguments)
                self.assertEqual(preset, explicit)

    def test_custom_single_factor_values_preserve_other_settings(self):
        cli = self.cli()
        baseline = baseline_config()
        for flag, value, section, key, expected in [
            ('-shots', '37', 'data', 'num_shots', 37),
            ('--depth', '5', 'model', 'neuron', [2] + [128] * 5 + [1]),
            ('-width', '192', 'model', 'neuron', [2] + [192] * 4 + [1]),
            ('--omega', '15.5', 'model', 'omega_0', 15.5),
        ]:
            with self.subTest(flag=flag):
                _, config = cli.build_run([flag, value])
                self.assertEqual(config[section][key], expected)
                config[section][key] = baseline[section][key]
                for field in ('model', 'data', 'optimizer', 'loss', 'training', 'seed', 'execution'):
                    self.assertEqual(config[field], baseline[field])

    def test_multiple_changes_and_preset_overrides_are_allowed(self):
        cli = self.cli()
        _, config = cli.build_run(['-shots', '25', '-omega', '20'])
        self.assertEqual(config['data']['num_shots'], 25)
        self.assertEqual(config['model']['omega_0'], 20)
        self.assertEqual(config['experiment_name'], 'shots_25_omega_20')
        _, config = cli.build_run(['-depth', '6', '-width', '256'])
        self.assertEqual(config['model']['neuron'], [2] + [256] * 6 + [1])
        _, config = cli.build_run(['--preset', 'width_256', '-omega', '15'])
        self.assertEqual(config['model']['neuron'], [2] + [256] * 4 + [1])
        self.assertEqual(config['model']['omega_0'], 15)

    def test_explicit_unchanged_values_and_same_factor_override_are_allowed(self):
        cli = self.cli()
        _, config = cli.build_run(['-depth', '4', '-width', '192', '-omega', '30'])
        self.assertEqual(config['experiment_name'], 'width_192')
        _, overridden = cli.build_run(['--preset', 'shots_25', '-shots', '37'])
        self.assertEqual(overridden['experiment_name'], 'shots_37')

    def test_invalid_values_and_protocol_changes_are_not_accepted(self):
        cli = self.cli()
        for arguments in (['-shots', '1'], ['-shots', '242'], ['-depth', '0'],
                          ['-width', '-2'], ['-omega', 'nan'], ['-omega', 'inf'],
                          ['--seed', '42'], ['--shot-batch-size', '13']):
            with self.subTest(arguments=arguments):
                with self.assertRaises(SystemExit):
                    cli.build_run(arguments)

    def test_changed_budget_or_log_interval_is_marked_preliminary(self):
        cli = self.cli()
        args, config = cli.build_run(['-width', '192', '--epochs', '3'])
        self.assertTrue(args.preliminary)
        self.assertEqual(config['training']['max_iterations'], 3)
        args, _ = cli.build_run(['--log-interval', '10'])
        self.assertTrue(args.preliminary)

    def test_dry_run_creates_no_output_and_does_not_call_trainer(self):
        cli = self.cli()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'unused'
            printed = io.StringIO()
            with redirect_stdout(printed), patch('baseline_experiment.run_config', side_effect=AssertionError('Training reached')):
                self.assertEqual(cli.main(['-shots', '37', '--dry-run', '--output-dir', str(out)]), 0)
            payload = json.loads(printed.getvalue())
            self.assertEqual(payload['config']['data']['num_shots'], 37)
            self.assertFalse(payload['preliminary'])
            self.assertFalse(out.exists())

    def test_list_presets_does_not_load_training_stack_in_fresh_process(self):
        import subprocess
        result = subprocess.run([sys.executable, '-X', 'utf8', '-c',
            "import experiment, sys; experiment.main(['--list-presets']); assert 'torch' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True, encoding='utf8')
        self.assertEqual(result.returncode, 0, result.stderr)
        for preset in self.cli().PRESETS:
            self.assertIn(preset, result.stdout)


if __name__ == '__main__':
    unittest.main()
