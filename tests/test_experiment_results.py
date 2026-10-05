import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from helpers import ROOT
from improved_modules.evaluate_deep import evaluate_deep_layers
from compare_results import load_completed_results

class ResultTests(unittest.TestCase):
    def test_evaluation_regions_part_of_contract(self):
        import copy
        from helpers import tiny
        from experiment_runtime import comparison_contract
        _,data,c=tiny(); c['training']['max_iterations']=3
        c['evaluation']={'depth_threshold':.5,'corner_size':.25}
        changed=copy.deepcopy(c); changed['evaluation']['depth_threshold']=.8
        self.assertNotEqual(comparison_contract(c,data),comparison_contract(changed,data))

    def test_constant_model_metrics_defined_or_null(self):
        truth=np.ones((12,12))*3000
        m=evaluate_deep_layers(truth,truth)
        self.assertEqual(m['deep_rmse'],0)
        self.assertIsNone(m['deep_ssim'])
        self.assertIsNone(m['deep_gradient_fidelity'])
        json.dumps(m,allow_nan=False)

    def test_invalid_fields_rejected(self):
        for shape,threshold in [((2,2),.99),((10,10),1.)]:
            with self.assertRaises(ValueError): evaluate_deep_layers(np.ones(shape),np.ones(shape),threshold)
        v=np.ones((12,12)); v[0,0]=np.nan
        with self.assertRaises(ValueError): evaluate_deep_layers(v,np.ones_like(v))

    def test_failed_ignored_duplicates_kept_and_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name,state in [('a','completed'),('b','completed'),('failed','failed')]:
                p=root/name; p.mkdir()
                (p/'status.json').write_text(json.dumps({'state':state}))
                (p/'comparison_contract.json').write_text(json.dumps({'seed':42,'observed':'same','budget':1}))
                (p/'config.json').write_text(json.dumps({'experiment_name':'baseline','seed':42}))
                (p/'metrics.json').write_text(json.dumps({'deep_rmse':1.}))
                (p/'loss_history.csv').write_text('completed_updates,loss_before_update\n1,2\n')
            rows=load_completed_results(root)
            self.assertEqual(len(rows),2)
            self.assertNotEqual(rows[0]['run_id'],rows[1]['run_id'])
            (root/'a/final_metrics.json').write_text(json.dumps({'data_mse':7.,'deep_rmse':9.}))
            fixed=next(row for row in load_completed_results(root) if row['run_id']=='a')
            self.assertEqual(fixed['final_data_mse'],7.)
            self.assertEqual(fixed['final_deep_rmse'],9.)
            self.assertEqual(fixed['deep_rmse'],1.)
            (root/'b/comparison_contract.json').write_text(json.dumps({'seed':42,'observed':'different','budget':1}))
            with self.assertRaisesRegex(ValueError,'Incompatible'):
                load_completed_results(root)
