import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from helpers import ROOT
sys.path.insert(0, str(ROOT/'scripts'))
from resume_parameter_sweep import plan_job, command_for_job, archive_interrupted_run, verify_suite
from baseline_protocol import baseline_config, ORIGINAL_PROTOCOL


class SweepRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.suite = Path(self.temporary.name)
        self.job = dict(name='shots_25', seed=3, directory='runs/shots_25_seed3',
                       config={'training': {'max_iterations': 4001}})

    def partial(self, state='running', updates=900, checkpoint=True):
        run = self.suite/self.job['directory']/'old_run'
        run.mkdir(parents=True)
        (run/'status.json').write_text(json.dumps(dict(state=state, completed_updates=updates)))
        (run/'config.json').write_text(json.dumps(dict(self.job['config'], seed=3)))
        if checkpoint:
            (run/'last.pth').write_bytes(b'preserved-checkpoint')
        return run

    def test_new_case_starts_without_resume(self):
        plan = plan_job(self.suite, self.job)
        self.assertEqual(plan['action'], 'start')
        self.assertNotIn('--resume', command_for_job(self.suite, self.job, plan, 'python', 'cuda:0'))

    def test_interrupted_case_uses_strict_resume(self):
        run = self.partial()
        plan = plan_job(self.suite, self.job)
        self.assertEqual(plan['action'], 'resume')
        self.assertEqual(plan['checkpoint'], run/'last.pth')
        command = command_for_job(self.suite, self.job, plan, 'python', 'cuda:0')
        self.assertIn('--resume', command)
        self.assertNotIn('--allow-execution-change', command)
        self.assertNotIn('--iterations', command)

    def test_completed_case_is_skipped_and_budget_is_checked(self):
        self.partial('completed', 4001)
        self.assertEqual(plan_job(self.suite, self.job)['action'], 'skip')
        self.job['config']['training']['max_iterations'] = 4000
        with self.assertRaises(ValueError):
            plan_job(self.suite, self.job)

    def test_missing_checkpoint_does_not_silently_restart(self):
        self.partial(checkpoint=False)
        with self.assertRaises(ValueError):
            plan_job(self.suite, self.job)

    def test_changed_config_is_rejected_before_archiving(self):
        self.partial()
        self.job['config']['training']['max_iterations'] = 2000
        with self.assertRaises(ValueError):
            plan_job(self.suite, self.job)

    def test_archive_preserves_checkpoint_and_avoids_ambiguous_summary(self):
        old = self.partial()
        plan = plan_job(self.suite, self.job)
        archived = archive_interrupted_run(self.suite, self.job, plan)
        self.assertEqual(archived.read_bytes(), b'preserved-checkpoint')
        self.assertFalse(old.exists())
        self.assertEqual(list((self.suite/self.job['directory']).glob('*/status.json')), [])

    def test_path_outside_suite_is_rejected(self):
        self.job['directory'] = '../outside'
        with self.assertRaises(ValueError):
            plan_job(self.suite, self.job)

    def test_verifies_frozen_code_and_registered_yaml(self):
        import yaml
        from parameter_sweep import make_cases
        from run_experiment import load_config
        base = load_config(ROOT/'experiments/configs/legacy_random_baseline.yaml')
        # This fixture checks the historical schema2 recovery path.
        base.pop('execution', None)
        jobs = [dict(case, seed=3, directory=f"runs/{case['name']}_seed3") for case in make_cases(base)]
        (self.suite/'manifest.json').write_text(json.dumps(dict(iterations=4001, seeds=[3], jobs=jobs)))
        (self.suite/'source').mkdir()
        (self.suite/'source/core.py').write_text('unchanged')
        digest = hashlib.sha256(b'unchanged').hexdigest()
        (self.suite/'source_hashes.json').write_text(json.dumps({'core.py': digest}))
        (self.suite/'configs').mkdir()
        for job in jobs:
            (self.suite/'configs'/f"{job['name']}.yaml").write_text(yaml.safe_dump(job['config']))
        self.assertEqual(len(verify_suite(self.suite)['jobs']), 10)
        (self.suite/'source/core.py').write_text('changed')
        with self.assertRaises(ValueError):
            verify_suite(self.suite)
        (self.suite/'source/core.py').write_text('unchanged')
        changed = copy.deepcopy(jobs[1]['config'])
        changed['optimizer']['learning_rate'] = .002
        (self.suite/'configs/shots_25.yaml').write_text(yaml.safe_dump(changed))
        with self.assertRaises(ValueError):
            verify_suite(self.suite)

    def test_original_resume_preserves_numbered_checkpoint_and_registered_seed(self):
        config = baseline_config()
        config['data']['num_shots'] = 25
        self.job['config'] = config
        run = self.partial(updates=150, checkpoint=False)
        (run/'checkpoints').mkdir()
        for update in (1,101):
            (run/'checkpoints'/f'MarmousiI_random-checkpoint-{update}.pth').write_bytes(b'original checkpoint')
        plan = plan_job(self.suite, self.job)
        self.assertEqual(plan['action'], 'resume')
        self.assertEqual(plan['saved_updates'], 101)
        self.assertEqual(plan['checkpoint'], run/'checkpoints/MarmousiI_random-checkpoint-101.pth')
        archived = archive_interrupted_run(self.suite, self.job, plan)
        self.assertEqual(archived.name, 'MarmousiI_random-checkpoint-101.pth')
        self.assertEqual(archived.read_bytes(), b'original checkpoint')
        self.assertEqual(read_json(archived.parent.parent/'config.json'), config)
        self.assertFalse(run.exists())

    def original_suite(self, preliminary=False):
        import yaml
        from parameter_sweep import make_cases
        config = baseline_config()
        if preliminary:
            config['training'].update(max_iterations=2, log_interval=1)
        jobs = [dict(case, seed=3, directory=f"runs/{case['name']}_seed3")
                for case in make_cases(config, preliminary=preliminary)]
        manifest = dict(iterations=config['training']['max_iterations'], seeds=[3],
                        jobs=jobs, protocol=ORIGINAL_PROTOCOL, preliminary=preliminary)
        (self.suite/'manifest.json').write_text(json.dumps(manifest))
        source = self.suite/'source'
        source.mkdir()
        names = ('ifwi_modules.py', 'rnn_fd.py', 'generator.py', 'plot_functions.py',
                 'data/vel_marmousi_376x1151.csv',
                 'experiments/baseline_reference/ifwi_experiment.py')
        hashes = {}
        for name in names:
            target = source/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
            hashes[name] = hashlib.sha256(target.read_bytes()).hexdigest()
        (self.suite/'source_hashes.json').write_text(json.dumps(hashes))
        (self.suite/'configs').mkdir()
        for job in jobs:
            (self.suite/'configs'/f"{job['name']}.yaml").write_text(yaml.safe_dump(job['config']))
        return manifest

    def test_original_suite_uses_pinned_sources_even_when_manifest_hashes_are_replaced(self):
        self.original_suite()
        self.assertEqual(len(verify_suite(self.suite)['jobs']), 10)
        source = self.suite/'source/ifwi_modules.py'
        source.write_bytes(source.read_bytes() + b'\n# drift\n')
        hashes = read_json(self.suite/'source_hashes.json')
        hashes['ifwi_modules.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
        (self.suite/'source_hashes.json').write_text(json.dumps(hashes))
        with self.assertRaises(ValueError):
            verify_suite(self.suite)

    def test_original_manifest_cannot_omit_marker_or_change_registered_seed(self):
        import yaml
        manifest = self.original_suite()
        job = manifest['jobs'][1]
        for seed, marker in ((42, ORIGINAL_PROTOCOL), (3, None)):
            with self.subTest(seed=seed, marker=marker):
                changed = copy.deepcopy(manifest)
                modified = changed['jobs'][1]
                modified['seed'] = seed
                modified['directory'] = f"runs/{modified['name']}_seed{seed}"
                modified['config']['seed'] = seed
                if marker is None:
                    modified['config'].pop('execution')
                (self.suite/'manifest.json').write_text(json.dumps(changed))
                (self.suite/'configs'/f"{job['name']}.yaml").write_text(yaml.safe_dump(modified['config']))
                with self.assertRaises(ValueError):
                    verify_suite(self.suite)

    def test_original_preliminary_suite_recovers_with_explicit_preliminary_command(self):
        manifest = self.original_suite(preliminary=True)
        self.assertTrue(verify_suite(self.suite)['preliminary'])
        job = manifest['jobs'][1]
        command = command_for_job(self.suite, job, {'action': 'start'}, 'python', 'cpu')
        self.assertIn('--preliminary', command)
        self.assertNotIn('--allow-execution-change', command)

    def test_original_completed_run_matches_config_with_seed_inside_both_profiles(self):
        config = baseline_config()
        config['data']['num_shots'] = 25
        self.job['config'] = config
        self.partial('completed', updates=4001, checkpoint=False)
        self.assertEqual(plan_job(self.suite, self.job)['action'], 'skip')

    def test_original_resume_uses_numeric_checkpoint_order_even_if_status_is_stale(self):
        self.job['config'] = baseline_config()
        self.job['config']['data']['num_shots'] = 25
        run = self.partial(updates=1, checkpoint=False)
        (run/'checkpoints').mkdir()
        for update in (1, 101, 901, 1001):
            (run/'checkpoints'/f'MarmousiI_random-checkpoint-{update}.pth').write_bytes(b'checkpoint')
        plan = plan_job(self.suite, self.job)
        self.assertEqual(plan['saved_updates'], 1001)
        self.assertEqual(plan['checkpoint'].name, 'MarmousiI_random-checkpoint-1001.pth')

    def test_original_resume_rejects_schema2_last_checkpoint(self):
        self.job['config'] = baseline_config()
        self.job['config']['data']['num_shots'] = 25
        self.partial()
        with self.assertRaisesRegex(ValueError, 'original|Original|numbered'):
            plan_job(self.suite, self.job)

    def test_pending_imported_baseline_uses_importer_without_training(self):
        manifest = self.original_suite()
        manifest['completed_baseline_bundle'] = 'completed_legacy_baseline'
        (self.suite/'manifest.json').write_text(json.dumps(manifest))
        bundle = self.suite/'completed_legacy_baseline'
        bundle.mkdir()
        baseline = manifest['jobs'][0]
        plan = plan_job(self.suite, baseline)
        command = command_for_job(self.suite, baseline, plan, 'python', 'cpu')
        self.assertEqual(Path(command[4]).name, 'import_completed_baseline.py')
        self.assertEqual(Path(command[command.index('--legacy-run') + 1]), bundle)
        self.assertNotIn('--resume', command)

    def test_pending_imported_baseline_from_historical_bundle_uses_importer(self):
        self.job['name'] = 'baseline'
        self.job['directory'] = 'runs/baseline_seed3'
        (self.suite/'completed_legacy_baseline').mkdir()
        (self.suite/'completed_baseline_reference.json').write_text('{}')
        command = command_for_job(self.suite, self.job, {'action': 'start'}, 'python', 'cpu')
        self.assertEqual(Path(command[4]).name, 'import_completed_baseline.py')

    def test_missing_or_external_import_bundle_is_rejected(self):
        manifest = self.original_suite()
        baseline = manifest['jobs'][0]
        for bundle in ('missing_legacy_bundle', '../outside'):
            with self.subTest(bundle=bundle):
                manifest['completed_baseline_bundle'] = bundle
                (self.suite/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    command_for_job(self.suite, baseline, {'action': 'start'}, 'python', 'cpu')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
