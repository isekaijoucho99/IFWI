import copy
import unittest
from helpers import ROOT
from run_experiment import load_config
from parameter_sweep import make_cases, validate_case
from import_completed_baseline import validate_legacy_config


class LegacyBaselineTests(unittest.TestCase):
    def test_old_baseline_preserves_width_when_changing_depth(self):
        base = load_config(ROOT/'experiments/configs/baseline.yaml')
        base['model']['neuron'] = [2,128,128,128,128,1]
        cases = make_cases(base)
        self.assertEqual(len(cases), 10)
        self.assertEqual([c['value'] for c in cases if c['factor']=='width'], [256,512])
        for case in cases:
            validate_case(cases[0]['config'], case)
            if case['factor']=='depth':
                self.assertEqual(set(case['config']['model']['neuron'][1:-1]), {128})

    def test_legacy_config_requires_original_protocol(self):
        config = load_config(ROOT/'experiments/configs/baseline.yaml')
        config['model']['neuron'] = [2,128,128,128,128,1]
        config['training']['max_iterations'] = 4001
        config['seed'] = 3
        legacy = dict(mode='random',seed=3,mean=3.,std=1.,dz=15,dt=.0019,nt=1000,
            frequency=8,source_depth_index=1,receiver_depth_index=2,learning_rate=1e-4,
            alpha=0,noise=0,dropout=0,epochs=4001,log_interval=100,pretrained=None)
        validate_legacy_config(legacy, config, 4001)
        for key, value in [('seed',42),('noise',2),('pretrained','smooth.pth')]:
            wrong=copy.deepcopy(legacy); wrong[key]=value
            with self.assertRaises(ValueError): validate_legacy_config(wrong,config,4001)
        with self.assertRaises(ValueError): validate_legacy_config(legacy,config,4000)


if __name__ == '__main__': unittest.main()
