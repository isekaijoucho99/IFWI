import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from helpers import tiny
from experiment_runtime import train_loop, contract, validate_resume_contract


class ExecutionResumeTests(unittest.TestCase):
    def test_only_execution_metadata_may_change(self):
        model, data, cfg = tiny()
        cfg['training'].update(max_iterations=4, log_interval=1)
        original = contract(cfg, data)
        updated = copy.deepcopy(original)
        updated['config']['training']['shot_batch_size'] = 13
        updated['config']['data'] = original['config'].get('data', {})
        original['config']['data'] = copy.deepcopy(updated['config']['data'])
        updated['sources']['experiments/run_experiment.py'] = 'new CLI'
        validate_resume_contract(original, updated, actual_shots=2, allow_execution_change=True)
        with self.assertRaises(ValueError):
            validate_resume_contract(original, updated, actual_shots=2)
        confounded = copy.deepcopy(updated)
        confounded['config']['optimizer']['learning_rate'] *= 2
        with self.assertRaises(ValueError):
            validate_resume_contract(original, confounded, actual_shots=2, allow_execution_change=True)
        changed_solver = copy.deepcopy(updated)
        changed_solver['sources']['rnn_fd.py'] = 'changed physics'
        with self.assertRaises(ValueError):
            validate_resume_contract(original, changed_solver, actual_shots=2, allow_execution_change=True)

    def test_resume_preserves_original_initial_velocity_and_adam_state(self):
        model, data, cfg = tiny()
        cfg['training'].update(max_iterations=4, log_interval=1)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_loop(model, data, cfg, root/'first', stop_after=2)
            resumed, data2, cfg2 = tiny()
            cfg2['training'].update(max_iterations=4, log_interval=1, shot_batch_size=1)
            resumed.shot_batch_size = 1
            result = train_loop(resumed, data2, cfg2, root/'resumed', resume=root/'first/last.pth',
                                allow_execution_change=True)
            reference, data3, cfg3 = tiny()
            cfg3['training'].update(max_iterations=4, log_interval=1)
            train_loop(reference, data3, cfg3, root/'continuous')
            self.assertEqual(result['resumed_from_updates'], 2)
            self.assertEqual(result['executed_updates'], 2)
            np.testing.assert_array_equal(np.load(root/'first/initial_velocity.npy'),
                                          np.load(root/'resumed/initial_velocity.npy'))
            np.testing.assert_allclose(np.load(root/'resumed/last_velocity.npy'),
                                       np.load(root/'continuous/last_velocity.npy'), rtol=1e-5, atol=1e-3)


if __name__ == '__main__':
    unittest.main()
