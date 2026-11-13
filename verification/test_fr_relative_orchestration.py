"""CPU orchestration regression; no CUDA context or wave-equation execution.

The tiny model uses real Adam. Physical solves, CUDA timing and metadata
publication are replaced; suite-control atomic publication has its own tests.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import fr_relative_suite as suite
from experiments.encoding_training_step import CommittedUpdateError, UncertainOptimizerError


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.value = torch.nn.Parameter(torch.zeros(()))

    def forward(self):
        return self.value.reshape(1, 1, 1)


def fake_execution(monkeypatch, tmp_path, failure=None):
    """Replace physical/GPU work, retain actual loop decisions and CPU Adam."""
    state = SimpleNamespace(controls=[], saved=[], loaded=[], made=[], steps=[], evaluations=[])
    lock = tmp_path/'GPU.lock'
    state.data = tmp_path/'dense'
    state.data.mkdir()
    manifest = dict(data_key='test-physical', clean_sha256='test-clean', noisy_sha256='test-noisy')
    (state.data/'data_manifest.json').write_text(json.dumps(manifest))
    input_manifest = dict(dataset_index=dict(path='test-index',sha256='same-input-index'))
    plan = dict(signature='test-protocol', config={}, dense_data_manifest=manifest,
                input_manifest=copy.deepcopy(input_manifest))
    arrays = {name: np.zeros((1, 52, 1, 1), np.float32) for name in ('clean', 'noisy')}
    assets = dict(truth=np.zeros((1, 1), np.float32), background=np.zeros((1, 1), np.float32),
                  manifest=input_manifest,
                  wavelet=np.zeros(1, np.float32))

    class Control:
        def __init__(self, output, run_id, lock_path):
            self.output, self.run_id, self.lock_path = Path(output), run_id, lock_path
            self.events=[]; self.terminal=None
            state.controls.append(self)

        def acquire(self):
            if self.lock_path.exists(): raise FileExistsError(self.lock_path)
            self.lock_path.write_text(self.run_id)
            self.write_json('control/LAUNCH.json', dict(run_id=self.run_id))

        def write_json(self, relative, payload):
            target=self.output/relative; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, allow_nan=False), encoding='utf-8')
            return target

        def record(self, event, **details):
            row=dict(event=event, **details); self.events.append(row); return row

        def check_stop(self): pass

        def finish(self, status, code, **details):
            self.terminal=dict(status=status, exit_code=code, run_id=self.run_id, **details)
            self.write_json('control/EXIT.json', self.terminal)

        def release(self):
            assert self.terminal is not None
            assert self.lock_path.read_text()==self.run_id
            self.lock_path.unlink()

    class Acquisition:
        def __init__(self, problem, batch_size=13, **kwargs):
            self.fields=kwargs.get('fields', 52)

    def model(*args):
        m=TinyVelocity(); opt=torch.optim.Adam(m.parameters(),lr=.001)
        state.made.append((m,opt)); return m,opt

    def step(m, opt, acq, completed, *, record_cost, check_stop, phase, arm_name,
             illumination_preconditioning):
        state.steps.append(dict(model=id(m), completed=completed, phase=phase, arm=arm_name,
                                preconditioned=illumination_preconditioning))
        check_stop()
        for counter in ('forward_fields_attempt','forward_fields_complete',
                        'backward_fields_attempt','backward_fields_complete'):
            record_cost('fake_'+counter, phase=phase, **{counter:acq.fields})
        if failure=='before_optimizer': raise ValueError('before optimizer')
        opt.zero_grad(); (m().sum()+1).square().backward()
        if failure=='uncertain':
            raise UncertainOptimizerError(completed+1, RuntimeError('simulated Adam failure'))
        opt.step()
        if failure=='committed':
            raise CommittedUpdateError(completed+1, ValueError('simulated CFL postcheck'))
        assert all(int(s['step'])==completed+1 for s in opt.state.values())
        return dict(completed_updates=completed+1, wall_seconds=99. if phase=='warmup' else 1.,
                    cuda_ms=1000., loss=1., illumination_preconditioning=illumination_preconditioning)

    def evaluate(m, acq, assets, *, record_cost, check_stop, label):
        check_stop(); state.evaluations.append(label)
        for counter in ('forward_fields_attempt','forward_fields_complete'):
            record_cost('fake_'+counter, phase='evaluation', **{counter:52})
        with torch.no_grad(): velocity=m()[0].numpy().copy()
        metrics=dict(rmse_mps=1., ssim=.5, clean_mse=2., noisy_mse=3., evaluation_wall_seconds=0.)
        return metrics, np.zeros((1,52,1,1),np.float32), velocity

    snapshots={}
    def save(m,opt,path,completed,meta):
        assert all(int(s['step'])==completed for s in opt.state.values())
        state.saved.append(dict(path=Path(path),completed=completed,model=id(m)))
        snapshots[str(path)]=(copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict()),completed,meta)
        Path(path).write_text(str(completed)); return path

    def load(m,opt,path,meta):
        weights,optimizer,completed,saved_meta=snapshots[str(path)]
        assert meta==saved_meta
        m.load_state_dict(weights); opt.load_state_dict(optimizer)
        state.loaded.append(dict(path=path,completed=completed)); return completed

    monkeypatch.setattr(suite,'SuiteControl',Control)
    monkeypatch.setattr(suite,'verify_plan',lambda p: None)
    monkeypatch.setattr(suite,'verify_previous_stage',lambda *args: dict(status='COMPLETED',target_updates=500))
    monkeypatch.setattr(suite,'release_guard',lambda: lock)
    monkeypatch.setattr(suite,'configure_runtime',lambda *args: dict(device='mocked_CPU_only'))
    monkeypatch.setattr(suite,'_dense_data',lambda *args: arrays)
    monkeypatch.setattr(suite,'build_dense_problem',lambda *args,**kwargs: object())
    monkeypatch.setattr(suite,'DenseAcquisition',Acquisition)
    monkeypatch.setattr(suite,'acquisition',lambda problem,arm: Acquisition(problem,fields=arm['fields']))
    monkeypatch.setattr(suite,'make_model',model)
    monkeypatch.setattr(suite,'train_dense_step',step)
    monkeypatch.setattr(suite,'evaluate_dense',evaluate)
    monkeypatch.setattr(suite,'velocity_metrics',lambda *args: dict(rmse_mps=1.,ssim=.5))
    monkeypatch.setattr(suite,'save_dense_checkpoint',save)
    monkeypatch.setattr(suite,'load_dense_checkpoint',load)
    monkeypatch.setattr(suite.gc,'collect',lambda: None)
    for name in ('reset_peak_memory_stats','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda *args,**kwargs: None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda *args,**kwargs: 0)
    return assets,plan,state


@pytest.mark.parametrize('stage', ('timing','formal'))
def test_fixed_stage_budget_clocks_reset_evaluations_and_no_expansion(monkeypatch,tmp_path,stage):
    assets,plan,state=fake_execution(monkeypatch,tmp_path)
    out=tmp_path/'new'
    assert suite.execute(stage,out,assets,plan,state.data,preflight='verified',timing='verified')==0
    result=json.loads((out/'result.json').read_text())
    n=6 if stage=='timing' else 12
    assert len(result['cases'])==n
    assert result['optimizer_updates']==dict(warmup=2*n,training=(10 if stage=='timing' else 500)*n)
    assert all(c['completed_updates']==(12 if stage=='timing' else 500) for c in result['cases'])
    assert all(c['measured_updates']==(10 if stage=='timing' else 500) for c in result['cases'])
    assert all(c['wall_seconds_median']==1. for c in result['cases'])
    assert all(c['training_seconds']==(10. if stage=='timing' else 500.) for c in result['cases'])
    expected_F=1380 if stage=='timing' else 118632
    expected_B=1380 if stage=='timing' else 115460
    assert result['cost_totals']==dict(forward_fields_attempt=expected_F,forward_fields_complete=expected_F,
        backward_fields_attempt=expected_B,backward_fields_complete=expected_B)
    assert len(state.evaluations)==(0 if stage=='timing' else 61)
    if stage=='timing':
        assert len(state.saved)==6 and len(state.loaded)==6
        assert all(s['completed']==12 for s in state.saved)
        assert len(state.made)==12  # original six + independent restored clones
    else:
        assert len(state.saved)==60 and not state.loaded
        assert set(s['completed'] for s in state.saved)=={100,200,300,400,500}
        assert len(state.made)==24  # warmup models discarded before each fresh formal model
        for condition in ('clean','noisy'):
            for arm in suite.ARMS:
                case=out/condition/arm['name']
                assert (case/'metrics_0000.json').exists()
                assert (case/'prediction_0000.npy').exists()
                assert (case/'metrics_0500.json').exists()
                assert not (case/'metrics_1000.json').exists()
        assert all(s['completed']<500 for s in state.steps if s['phase']=='training')
        reused=[e for e in state.controls[0].events if e['event']=='initial_waveform_reused']
        assert len(reused)==11
        initial=np.load(out/'shared_initial/prediction.npy',allow_pickle=False)
        for condition in ('clean','noisy'):
            for arm in suite.ARMS:
                assert np.array_equal(np.load(out/condition/arm['name']/'prediction_0000.npy',allow_pickle=False),initial)
    assert state.controls[0].terminal['status']=='COMPLETED'
    assert not (tmp_path/'GPU.lock').exists()


@pytest.mark.parametrize('failure', ('before_optimizer','committed','uncertain'))
def test_failure_has_no_retry_or_unsafe_checkpoint(monkeypatch,tmp_path,failure):
    assets,plan,state=fake_execution(monkeypatch,tmp_path,failure)
    out=tmp_path/'failed'
    assert suite.execute('formal',out,assets,plan,state.data,preflight='verified',timing='verified')==1
    assert len(state.steps)==1 and not state.saved
    receipt=json.loads((out/'control/EXIT.json').read_text())
    detail=json.loads((out/'failure.json').read_text())
    assert receipt['status']=='FAILED' and receipt['exit_code']==1
    assert detail['automatic_restart'] is False
    assert detail['cost_totals']==dict(forward_fields_attempt=52,forward_fields_complete=52,
        backward_fields_attempt=52,backward_fields_complete=52)
    assert detail['last_safe_checkpoint'] is None
    if failure=='committed':
        assert detail['optimizer_commit_state']=='KNOWN_COMMITTED_POSTCHECK_FAILED'
        assert detail['optimizer_updates_confirmed']==dict(warmup=1,training=0)
        assert detail['known_committed_update_at_failure']==1
        assert detail['optimizer_updates_uncertain']==0
        assert receipt['optimizer_updates']==dict(warmup=1,training=0)
    elif failure=='uncertain':
        assert detail['optimizer_commit_state']=='UNKNOWN'
        assert detail['optimizer_updates_confirmed']==dict(warmup=0,training=0)
        assert detail['optimizer_updates_uncertain']==1
        assert detail['attempted_update']==1
        assert receipt['optimizer_updates_uncertain']==1
    else:
        assert detail['optimizer_updates_confirmed']==dict(warmup=0,training=0)
        assert detail['optimizer_updates_uncertain']==0
    with pytest.raises(FileExistsError):
        suite.execute('formal',out,assets,plan,state.data,preflight='verified',timing='verified')
    assert len(state.steps)==1


def write_stage(path,stage,*,identity='one',signature='frozen',cost=None):
    (path/'control').mkdir(parents=True)
    totals=cost or dict(forward_fields_attempt=34,forward_fields_complete=34,
                       backward_fields_attempt=34,backward_fields_complete=34)
    report=dict(status='COMPLETED',stage=stage,signature=signature,cost_totals=totals)
    launch=dict(run_id=identity)
    end=dict(run_id=identity,status='COMPLETED',exit_code=0,cost_totals=totals)
    for name,payload in [('result.json',report),('control/LAUNCH.json',launch),('control/EXIT.json',end)]:
        (path/name).write_text(json.dumps(payload))


@pytest.mark.parametrize('tamper', ('identity','signature','failure','wrong_cost'))
def test_prerequisite_receipts_refuse_inconsistent_proof(tmp_path,tamper):
    path=tmp_path/'prior';write_stage(path,'preflight')
    if tamper in ('identity','failure'):
        p=path/'control/EXIT.json';payload=json.loads(p.read_text())
        payload.update(run_id='other') if tamper=='identity' else payload.update(status='FAILED',exit_code=1)
        p.write_text(json.dumps(payload))
    elif tamper=='signature':
        p=path/'result.json';payload=json.loads(p.read_text());payload['signature']='other';p.write_text(json.dumps(payload))
    else:
        for name in ('result.json','control/EXIT.json'):
            p=path/name;payload=json.loads(p.read_text());payload['cost_totals']['forward_fields_complete']=33
            p.write_text(json.dumps(payload))
    with pytest.raises(ValueError): suite.verify_previous_stage(path,'preflight',dict(signature='frozen'))


def test_prerequisite_receipts_accept_matching_complete_cost(tmp_path):
    path=tmp_path/'prior';write_stage(path,'preflight')
    assert suite.verify_previous_stage(path,'preflight',dict(signature='frozen'))['stage']=='preflight'


def test_frozen_dense_manifest_refuses_changed_data_before_physical_work(monkeypatch,tmp_path):
    assets,plan,state=fake_execution(monkeypatch,tmp_path)
    plan['dense_data_manifest']={**plan['dense_data_manifest'],'clean_sha256':'different'}
    out=tmp_path/'changed_data'
    assert suite.execute('formal',out,assets,plan,state.data,preflight='verified',timing='verified')==1
    assert not state.steps and not state.evaluations and not state.saved
    failure=json.loads((out/'failure.json').read_text())
    assert 'manifest differs' in failure['error']
    assert all(value==0 for value in failure['cost_totals'].values())


def test_explicit_1000_requires_complete_source_before_output_or_training(monkeypatch,tmp_path):
    assets,plan,state=fake_execution(monkeypatch,tmp_path)
    out=tmp_path/'unapproved_expansion'
    with pytest.raises(ValueError,match='1000 requires'):
        suite.execute('formal',out,assets,plan,state.data,target=1000,
                      preflight='verified',timing='verified')
    assert not out.exists() and not state.controls and not state.steps


def test_busy_release_guard_prevents_new_output_and_work(monkeypatch,tmp_path):
    assets,plan,state=fake_execution(monkeypatch,tmp_path)
    def busy(): raise RuntimeError('GPU suite lock exists')
    monkeypatch.setattr(suite,'release_guard',busy)
    out=tmp_path/'busy'
    with pytest.raises(RuntimeError,match='lock'):
        suite.execute('timing',out,assets,plan,state.data,preflight='verified')
    assert not out.exists() and not state.controls and not state.steps


def test_frozen_input_manifest_binds_loaded_assets_before_gpu_or_output(monkeypatch,tmp_path):
    assets,plan,state=fake_execution(monkeypatch,tmp_path)
    assets['manifest']['dataset_index']['sha256']='different-input-index'
    out=tmp_path/'changed_input'
    with pytest.raises(ValueError,match='(?i)(input|asset).*manifest'):
        suite.execute('formal',out,assets,plan,state.data,preflight='verified',timing='verified')
    assert not out.exists() and not state.controls and not state.steps
