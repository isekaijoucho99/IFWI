import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SweepLossExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.suite = Path(self.temporary.name)

    def module(self):
        path = ROOT / 'scripts' / 'plot_parameter_sweep_losses.py'
        self.assertTrue(path.is_file(), 'Loss-history export is not implemented yet')
        spec = importlib.util.spec_from_file_location('sweep_loss_export', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def history(self, name, fields, rows):
        path = self.suite / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)
        return path

    def test_legacy_update_numbers_and_unused_regularization_are_preserved(self):
        module = self.module()
        path = self.history('legacy_loss_history.csv',
                            ['completed_updates', 'total_loss', 'data_loss', 'regularization_statistic'],
                            [['1.000e+00', .04, .04, .2], ['4.001e+03', .00006, .00006, 5.7]])
        rows = module.read_loss_history(path)
        self.assertEqual([r['completed_updates'] for r in rows], [1, 4001])
        self.assertEqual(rows[-1]['total_loss'], .00006)
        self.assertEqual(rows[-1]['regularization_statistic'], 5.7)
        self.assertIsNone(rows[-1]['prior_loss'])

    def test_active_csv_partial_write_does_not_invent_an_update(self):
        module = self.module()
        path = self.history('loss_history.csv',
                            ['completed_updates', 'loss_before_update', 'data_loss_before_update'],
                            [[1, .04, .04], [2, .03, .03]])
        with path.open('a', encoding='utf-8') as stream:
            stream.write('3,0.02,')
        rows = module.read_loss_history(path)
        self.assertEqual([r['completed_updates'] for r in rows], [1, 2])

    def test_duplicate_updates_are_rejected_instead_of_drawing_a_false_trajectory(self):
        module = self.module()
        path = self.history('loss_history.csv', ['completed_updates', 'loss_before_update'],
                            [[1, .04], [1, .02]])
        with self.assertRaisesRegex(ValueError, 'increasing'):
            module.read_loss_history(path)

    def test_original_runner_loss_csv_is_available_to_grouped_exports(self):
        module = self.module()
        job = dict(name='baseline', factor='baseline', value=0, seed=3,
                   directory='runs/baseline_seed3')
        run = self.suite / job['directory'] / 'original_run'
        run.mkdir(parents=True)
        (run / 'status.json').write_text(json.dumps(dict(state='completed')))
        self.history(str(run.relative_to(self.suite) / 'loss.csv'),
                     ['completed_updates', 'total_loss', 'data_loss', 'regularization_statistic'],
                     [[1, .04, .04, .2], [101, .003, .003, .8]])
        cases = module.collect_cases(self.suite, dict(jobs=[job]))
        self.assertEqual([row['completed_updates'] for row in cases[0]['rows']], [1, 101])
        self.assertEqual(Path(cases[0]['source']).name, 'loss.csv')

    def test_only_registered_current_runs_are_used_and_pending_has_no_zero_loss(self):
        module = self.module()
        jobs = [dict(name=name, factor=factor, value=value, seed=3,
                     directory=f'runs/{name}_seed3')
                for name, factor, value in [('baseline', 'baseline', 0), ('shots_25', 'shots', 25),
                                             ('shots_49', 'shots', 49)]]
        manifest = dict(iterations=4001, seeds=[3], jobs=jobs)
        (self.suite / 'manifest.json').write_text(json.dumps(manifest))
        base = self.suite / jobs[0]['directory'] / 'legacy_baseline_import'
        base.mkdir(parents=True)
        (base / 'status.json').write_text(json.dumps(dict(state='completed', completed_updates=4001)))
        self.history(str(base.relative_to(self.suite) / 'legacy_loss_history.csv'),
                     ['completed_updates', 'total_loss', 'data_loss'], [[1, .04, .04], [4001, .00006, .00006]])
        run = self.suite / jobs[1]['directory'] / 'resumed_run'
        run.mkdir(parents=True)
        (run / 'status.json').write_text(json.dumps(dict(state='running', completed_updates=2)))
        self.history(str(run.relative_to(self.suite) / 'loss_history.csv'),
                     ['completed_updates', 'loss_before_update'], [[1, .04], [2, .03], [3, .02]])
        self.history('archive/shots_25_seed3/old_run/loss_history.csv',
                     ['completed_updates', 'loss_before_update'], [[3, 99.]])
        cases = module.collect_cases(self.suite, manifest)
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[1]['rows'][-1]['total_loss'], .02)
        self.assertEqual(cases[1]['rows'][-1]['completed_updates'], 3)
        self.assertEqual(cases[2]['state'], 'pending')
        self.assertEqual(cases[2]['rows'], [])


if __name__ == '__main__':
    unittest.main()
