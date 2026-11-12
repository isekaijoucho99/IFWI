"""CPU/mock scheduling tests. No FD solver or GPU is executed here."""
import importlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

try:
    suite=importlib.import_module('experiments.batched_encoding_suite')
except ModuleNotFoundError:
    suite=None


ARMS=(dict(name='full13',grouping='identity',fields=13,progressive=False),
      dict(name='single',grouping='global',fields=1,progressive=False),
      dict(name='global2',grouping='global',fields=2,progressive=False),
      dict(name='interleaved2',grouping='interleaved',fields=2,progressive=False),
      dict(name='interleaved3',grouping='interleaved',fields=3,progressive=False),
      dict(name='interleaved2_progressive',grouping='interleaved',fields=2,progressive=True))


class UncertainOptimizerError(RuntimeError):
    """Mock the helper contract without importing Torch or running Adam."""
    def __init__(self,attempted_update):
        self.attempted_update=attempted_update
        super().__init__('mock Adam.step started but did not return')


class FakeModel:
    def __init__(self,background,parameters=50049):self.background=background;self.completed=0;self.count=parameters
    def __call__(self):return self.background[None]+self.completed/1000
    def set_progress(self,n):self.completed=int(n)
    def parameters(self):return (SimpleNamespace(requires_grad=True,numel=lambda:self.count),)


class FakeDependencies:
    arms=ARMS
    def __init__(self):
        self.models=[];self.train_calls=[];self.evaluations=[];self.saved=[];self.loaded=[]
        self.fail_correctness=False;self.fail_step=None;self.fail_error=None;self.stop_after=None
        self.sources={'physics.py':'fixed-source'}
        self.assets=dict(truth=np.full((2,3),3,dtype=np.float32),
            background=np.full((2,3),3,dtype=np.float32),
            clean=np.zeros((1,13,2,3),np.float32),noisy=np.ones((1,13,2,3),np.float32),
            wavelet=np.ones(2,np.float32),config={'fixed':True},manifest={'input':'fixed-data'},
            measured_snr_db=-12.04,canonical_checkpoint=None)
    def configure_runtime(self,device):return {'device':device,'threads':1,'dtype':'float32'}
    def load_assets(self,*args):return self.assets
    def source_fingerprints(self):return dict(self.sources)
    def make_model(self,assets,arm,device):
        model=FakeModel(assets['background'],100098 if arm['progressive']else 50049);self.models.append((arm['name'],model))
        return model,SimpleNamespace(completed=0)
    def build_problem(self,assets,condition,device):return dict(condition=condition)
    def make_encoder(self,problem,arm):return dict(problem=problem,arm=arm)
    def run_correctness(self,problem,velocity,*,record_cost,update):
        for key in ['forward_fields_attempt','forward_fields_complete','backward_fields_attempt','backward_fields_complete']:
            record_cost('mock_correctness',**{key:48})
        return {'passed':not self.fail_correctness,'status':'PASS'if not self.fail_correctness else'FAIL',
            'cost_totals':dict(forward_fields_attempt=48,forward_fields_complete=48,
                backward_fields_attempt=48,backward_fields_complete=48)}
    def reset_peak(self,device):return None
    def memory_peak(self,device):return dict(peak_allocated_bytes=123,peak_reserved_bytes=456)
    def cleanup(self,device):return None
    def train_step(self,model,optimizer,encoder,completed_updates,*,record_cost,check_stop,phase,arm_name):
        check_stop()
        if self.fail_step==len(self.train_calls):raise self.fail_error or RuntimeError('mock step failure')
        assert model.completed==optimizer.completed==completed_updates
        arm=encoder['arm'];n=arm['fields']
        for key in ['forward_fields_attempt','forward_fields_complete','backward_fields_attempt','backward_fields_complete']:
            record_cost('mock_step',phase=phase,arm=arm_name,update=completed_updates+1,**{key:n})
        model.completed=optimizer.completed=completed_updates+1
        self.train_calls.append((encoder['problem']['condition'],arm_name,model.completed,phase))
        if self.stop_after==len(self.train_calls):
            control=check_stop.__self__
            (control.output/'STOP_REQUEST.json').write_text(json.dumps({'run_id':control.run_id,'action':'stop'}))
        return dict(completed_updates=model.completed,loss=1.,wall_seconds=.01,cuda_ms=10.,velocity_min=3.,velocity_max=4.)
    def evaluate(self,model,problem,assets,*,record_cost,check_stop,label):
        check_stop();self.evaluations.append((label,problem['condition']))
        record_cost('mock_eval',forward_fields_attempt=13)
        record_cost('mock_eval',forward_fields_complete=13)
        return dict(clean_mse_fp64=1.,noisy_mse_fp64=2.,rmse_mps=3.),np.zeros((1,13,2,3),np.float32),model().copy()
    def save_checkpoint(self,model,opt,path,completed,metadata):
        path=Path(path)
        with path.open('x')as f:json.dump(dict(completed_updates=completed,model=model.completed,optimizer=opt.completed,metadata=metadata),f)
        self.saved.append((str(path),completed));return path
    def load_checkpoint(self,model,opt,path,metadata):
        data=json.loads(Path(path).read_text());assert data['metadata']==metadata
        model.completed=data['model'];opt.completed=data['optimizer']
        self.loaded.append((str(path),data['completed_updates']));return data['completed_updates']


class SuiteTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(suite,'batched_encoding_suite.py not implemented')
        root=Path(__file__).resolve().parents[1]
        scratch=root/'verification'/'_suite_unit_scratch';scratch.mkdir(exist_ok=True)
        self.assertTrue(scratch.resolve().is_relative_to(root.resolve()))
        self.temp=tempfile.TemporaryDirectory(dir=scratch);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.deps=FakeDependencies()
    def run_stage(self,stage,**kwargs):
        options=dict(stage=stage,output=self.root/stage,dataset_index=self.root/'index.json',
            reference_config=self.root/'config.json',initial_checkpoint=self.root/'initial.pt',
            code_commit='mock-no-git',preflight=[],device='cpu',run_id='run-'+stage,
            dependencies=self.deps)
        options.update(kwargs);return suite.run_suite(**options)
    def test_correctness_uses_one_fixed_problem_and_records_exact_cost(self):
        result=self.run_stage('correctness')
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(result['cost_totals']['forward_fields_complete'],48)
        self.assertEqual(result['cost_totals']['backward_fields_complete'],48)
        report=json.loads((self.root/'correctness'/'correctness.json').read_text())
        self.assertTrue(report['passed'])
        self.assertEqual(len(report['initial_velocity_equal_arms']),6)
        receipt=json.loads((self.root/'correctness'/'control'/'EXIT.json').read_text())
        self.assertEqual(receipt['exit_code'],0)
        self.assertFalse((self.root/'ENCODING_SUITE_GPU.lock').exists())
        self.assertFalse(self.deps.train_calls)
    def test_train_without_preflight_is_rejected_before_any_scientific_step(self):
        with self.assertRaisesRegex(ValueError,'preflight'):
            self.run_stage('train')
        self.assertFalse(self.deps.train_calls)
        self.assertFalse(self.deps.evaluations)
    def test_initial_checkpoint_is_mandatory(self):
        with self.assertRaisesRegex(ValueError,'initial.checkpoint'):
            self.run_stage('correctness',initial_checkpoint=None)
        self.assertFalse(self.deps.train_calls)
    def test_failed_correctness_has_failure_exit_and_does_not_run_later_stage(self):
        self.deps.fail_correctness=True
        result=self.run_stage('correctness')
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(result['exit_code'],1)
        self.assertFalse(self.deps.train_calls)
        self.assertTrue((self.root/'correctness'/'failure.json').exists())
        self.assertFalse((self.root/'ENCODING_SUITE_GPU.lock').exists())

    def test_timing_fresh_arms_have_two_warmups_ten_samples_and_actual276_fields(self):
        result=self.run_stage('timing')
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(result['cost_totals']['forward_fields_complete'],276)
        self.assertEqual(result['cost_totals']['backward_fields_complete'],276)
        timing=json.loads((self.root/'timing/timing.json').read_text())
        self.assertEqual(len(timing['cases']),6)
        for case in timing['cases']:
            self.assertEqual(case['warmup_updates'],2)
            self.assertEqual(len(case['samples']),10)
            self.assertEqual(case['completed_updates'],12)
            self.assertEqual(case['peak_allocated_bytes'],123)
            self.assertEqual(case['trainable_parameters'],100098 if case['arm'].endswith('progressive')else 50049)
        self.assertEqual([k for _,k in self.deps.saved],[12]*6)

    def preflights(self):
        self.run_stage('correctness');self.run_stage('timing')
        return [self.root/'correctness',self.root/'timing']

    def test_training_round_robin_restores_all_cases_and_evaluates_fixed_milestones(self):
        old=(suite.MILESTONES,suite.CHECKPOINT_INTERVAL,suite.STATUS_INTERVAL)
        suite.MILESTONES=(2,3,4);suite.CHECKPOINT_INTERVAL=2;suite.STATUS_INTERVAL=2
        self.addCleanup(lambda:setattr(suite,'MILESTONES',old[0]))
        self.addCleanup(lambda:setattr(suite,'CHECKPOINT_INTERVAL',old[1]))
        self.addCleanup(lambda:setattr(suite,'STATUS_INTERVAL',old[2]))
        preflight=self.preflights();self.deps.train_calls.clear();self.deps.evaluations.clear();self.deps.loaded.clear()
        result=self.run_stage('train',preflight=preflight)
        self.assertEqual(result['status'],'COMPLETED')
        self.assertEqual(result['cost_totals']['forward_fields_complete'],184+481)
        self.assertEqual(result['cost_totals']['backward_fields_complete'],184)
        self.assertEqual(len(self.deps.train_calls),12*4)
        self.assertEqual(len(self.deps.evaluations),1+12*3)
        self.assertEqual([r[2]for r in self.deps.train_calls[:4]],[1,2,1,2])
        self.assertEqual({k for _,k in self.deps.loaded},{2,3})
        report=json.loads((self.root/'train/train.json').read_text())
        self.assertEqual(len(report['cases']),12)
        self.assertTrue(all(c['completed_updates']==4 for c in report['cases']))
        self.assertTrue((self.root/'train/initial_shared/prediction.npy').exists())
        self.assertFalse((self.root/'ENCODING_SUITE_GPU.lock').exists())

    def test_training_rejects_changed_sources_before_any_training_or_evaluation(self):
        receipts=self.preflights();self.deps.sources['physics.py']='changed'
        self.deps.train_calls.clear();self.deps.evaluations.clear()
        result=self.run_stage('train',preflight=receipts)
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(self.deps.train_calls);self.assertFalse(self.deps.evaluations)

    def test_user_stop_saves_returned_update_and_acknowledges_without_next_case(self):
        receipts=self.preflights();self.deps.train_calls.clear();self.deps.evaluations.clear()
        self.deps.stop_after=1
        result=self.run_stage('train',preflight=receipts)
        self.assertEqual(result['status'],'STOPPED_BY_USER');self.assertEqual(result['exit_code'],130)
        self.assertEqual(len(self.deps.train_calls),1)
        saved=json.loads(Path(result['last_safe_checkpoint']).read_text())
        self.assertEqual(saved['completed_updates'],1)
        ack=json.loads((self.root/'train/control/STOP_ACK.json').read_text())
        self.assertEqual(ack['status'],'STOPPED_BY_USER')
        self.assertFalse((self.root/'ENCODING_SUITE_GPU.lock').exists())

    def test_exception_never_saves_current_failed_model_as_safe_or_runs_next_case(self):
        receipts=self.preflights();self.deps.train_calls.clear();self.deps.saved.clear()
        self.deps.fail_step=1
        result=self.run_stage('train',preflight=receipts)
        self.assertEqual(result['status'],'FAILED')
        self.assertEqual(len(self.deps.train_calls),1)
        self.assertEqual(self.deps.saved,[])
        self.assertIsNone(result['last_safe_checkpoint'])

    def test_protocol_excludes_git_label_but_binds_actual_physics_and_data(self):
        a=suite.build_protocol(self.deps.assets,{'device':'cpu'},self.deps,'uncommitted')
        b=suite.build_protocol(self.deps.assets,{'device':'cpu'},self.deps,'committed')
        self.assertEqual(a['protocol_signature'],b['protocol_signature'])
        self.deps.assets['manifest']['input']='changed'
        c=suite.build_protocol(self.deps.assets,{'device':'cpu'},self.deps,'committed')
        self.assertNotEqual(a['protocol_signature'],c['protocol_signature'])

    def test_incomplete_timing_receipt_cannot_authorize_long_training(self):
        receipts=self.preflights()
        path=self.root/'timing/timing.json'
        report=json.loads(path.read_text());report['cases'][0]['samples'].pop()
        path.write_text(json.dumps(report))
        self.deps.train_calls.clear();self.deps.evaluations.clear()
        self.deps.fail_step=0
        result=self.run_stage('train',preflight=receipts)
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(self.deps.train_calls);self.assertFalse(self.deps.evaluations)

    def test_missing_run_identity_cannot_authorize_long_training(self):
        receipts=self.preflights()
        for relative in ('summary.json','correctness.json','control/EXIT.json'):
            path=self.root/'correctness'/relative
            report=json.loads(path.read_text());report.pop('run_id')
            path.write_text(json.dumps(report))
        self.deps.train_calls.clear();self.deps.evaluations.clear()
        self.deps.fail_step=0
        result=self.run_stage('train',preflight=receipts)
        self.assertEqual(result['status'],'FAILED')
        self.assertFalse(self.deps.train_calls);self.assertFalse(self.deps.evaluations)

    def test_stop_before_attempt_does_not_charge_or_append_attempt(self):
        class StoppedControl:
            def check_stop(self):raise suite.StopRequested('test stop')
            def record(self,*args,**kwargs):raise AssertionError('must not append')
        costs=suite.Costs(StoppedControl())
        with self.assertRaises(suite.StopRequested):
            costs.record('attempt',forward_fields_attempt=13)
        self.assertEqual(costs.totals,{key:0 for key in suite.COUNTERS})

    def test_mid_adam_failure_is_explicitly_uncertain_and_never_saved(self):
        receipts=self.preflights();self.deps.train_calls.clear();self.deps.saved.clear()
        self.deps.fail_step=0;self.deps.fail_error=UncertainOptimizerError(1)
        result=self.run_stage('train',preflight=receipts)
        self.assertEqual(result['status'],'FAILED')
        self.assertTrue(result['optimizer_step_outcome_uncertain'])
        self.assertEqual(result['attempted_optimizer_update'],1)
        self.assertFalse(self.deps.saved)


if __name__=='__main__':unittest.main(verbosity=2)
