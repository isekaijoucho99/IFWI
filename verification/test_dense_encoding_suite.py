import copy,json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.residual_ifwi_experiment import default_config
try:
    from experiments import dense_encoding_suite as suite
except ImportError:suite=None


def assets():
    cfg=default_config()
    return dict(config=cfg,truth=np.full((8,288),2400,np.float32),
        background=np.full((8,288),2300,np.float32),wavelet=np.ones(1000,np.float32),
        manifest={},canonical_checkpoint=None)


def test_dense_plan_is_legal_and_does_not_reuse13shot_observations():
    assert suite is not None
    for n in (48,52):
        p=suite.build_plan(assets(),n,13)
        xs=p['config']['acquisition']['source_x_indices']
        assert len(xs)==n and min(xs)>=10 and max(xs)<=277
        assert np.diff(xs).tolist()==[5]*(n-1)
        assert p['status']=='PREPARED_NOT_RUN' and p['quick_updates']==200
        assert p['full_batch_fields']==([13]*3+[n-39])
        assert p['dense_observations_required']is True


def test_small_memory_cap_is_declared_not_claimed_measured():
    assert suite is not None
    p=suite.build_plan(assets(),48,12)
    assert p['full_batch_fields']==[12]*4
    assert p['batch_limit_evidence']=='unmeasured configurable limit; GPU measurement required'


def test_plain_static_and_gated_have_equal_parameter_count_and_initial_velocity():
    assert suite is not None
    a=assets()
    for arm in suite.ARMS:
        model,opt=suite.make_dense_model(a,arm,'cpu')
        assert sum(p.numel()for p in model.parameters())==50049
        assert torch.equal(model(),torch.from_numpy(a['background'])[None])


def test_gpu_guard_refuses_busy_or_unproven_release(tmp_path):
    assert suite is not None
    active=tmp_path/'old_run';(active/'control').mkdir(parents=True)
    with pytest.raises(RuntimeError):suite.gpu_execution_guard(active)
    launch=dict(run_id='known');(active/'control/LAUNCH.json').write_text(json.dumps(launch))
    (active/'control/EXIT.json').write_text(json.dumps(dict(run_id='known',status='COMPLETED',exit_code=0)))
    (tmp_path/'ENCODING_SUITE_GPU.lock').write_text('{}')
    with pytest.raises(RuntimeError):suite.gpu_execution_guard(active)


def test_gpu_guard_accepts_only_matching_explicit_terminal_receipt(tmp_path):
    assert suite is not None
    active=tmp_path/'old_run';(active/'control').mkdir(parents=True)
    (active/'control/LAUNCH.json').write_text(json.dumps(dict(run_id='known')))
    path=active/'control/EXIT.json'
    path.write_text(json.dumps(dict(run_id='other',status='COMPLETED',exit_code=0)))
    with pytest.raises(RuntimeError):suite.gpu_execution_guard(active)
    path.write_text(json.dumps(dict(run_id='known',status='COMPLETED',exit_code=0)))
    assert suite.gpu_execution_guard(active)==tmp_path/'ENCODING_SUITE_GPU.lock'


def test_dry_run_and_execution_budget_do_not_allow_1000_or_launch_gpu(tmp_path,monkeypatch):
    assert suite is not None
    monkeypatch.setattr(suite,'load_assets',lambda *args:assets())
    monkeypatch.setattr(suite,'execute',lambda **kwargs:pytest.fail('unexpected GPU execution'))
    out=tmp_path/'prepared'
    assert suite.main(['--stage','timing','--output',str(out),'--dataset-index','unused',
        '--reference-config','unused','--initial-checkpoint','unused'])==0
    assert json.loads((out/'plan.json').read_text())['status']=='PREPARED_NOT_RUN'
    with pytest.raises(SystemExit):suite.main(['--stage','train1000'])


def fake_execution(monkeypatch,tmp_path):
    """Only GPU orchestration is replaced; use real CPU model/Adam/control IO."""
    a=assets();plan=suite.build_plan(a,48,12)
    active=tmp_path/'old_run';(active/'control').mkdir(parents=True)
    for name,payload in [('LAUNCH',dict(run_id='old')),('EXIT',dict(run_id='old',status='COMPLETED',exit_code=0))]:
        (active/'control'/(name+'.json')).write_text(json.dumps(payload))
    monkeypatch.setattr(suite,'configure_runtime',lambda device:dict(device='cpu_orchestration_test'))
    for name in('reset_peak_memory_stats','empty_cache'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:None)
    for name in('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:0)
    data={name:np.zeros((1,48,1,1),np.float32)for name in('clean','noisy')}
    monkeypatch.setattr(suite,'_dense_data',lambda *args:data)
    monkeypatch.setattr(suite,'_problem',lambda *args:object())
    class Acquisition:
        def __init__(self,problem,batch_size,grouping,fields,seed):self.fields=48 if grouping=='identity'else fields
    monkeypatch.setattr(suite,'DenseAcquisition',Acquisition)
    def model(*args):
        torch.manual_seed(3);m=torch.nn.Linear(1,1)
        return m,torch.optim.Adam(m.parameters(),lr=.001)
    monkeypatch.setattr(suite,'make_dense_model',model)
    def step(m,opt,acq,k,*,record_cost,check_stop,phase,arm_name):
        check_stop()
        for counter in suite.COUNTERS:record_cost('fake_'+counter,phase=phase,**{counter:acq.fields})
        opt.zero_grad();m(torch.ones(1,1)).square().sum().backward();opt.step()
        return dict(completed_updates=k+1,wall_seconds=99. if phase=='warmup'else 1.,cuda_ms=None)
    monkeypatch.setattr(suite,'train_dense_step',step)
    def evaluate(m,acq,assets,*,record_cost,check_stop,label):
        check_stop()
        for counter in('forward_fields_attempt','forward_fields_complete'):
            record_cost('fake_'+counter,phase='evaluation',**{counter:48})
        return dict(clean_mse=0.,noisy_mse=0.),np.zeros((1,48,1,1)),np.zeros((8,288))
    monkeypatch.setattr(suite,'evaluate_dense',evaluate)
    saved=[]
    def save(m,opt,path,completed,metadata):
        assert all(int(state['step'])==completed for state in opt.state.values())
        saved.append(completed);path.write_text(str(completed));return path
    monkeypatch.setattr(suite,'save_dense_checkpoint',save)
    return a,plan,active,saved


@pytest.mark.parametrize('stage',('timing','quick'))
def test_fixed_budgets_warmup_exclusion_costs_and_checkpoint_clock(tmp_path,monkeypatch,stage):
    assert suite is not None
    a,p,active,saved=fake_execution(monkeypatch,tmp_path);out=tmp_path/'new'
    assert suite.execute(stage=stage,output=out,assets=a,plan=p,dense_data='fake',active_run=active)==0
    result=json.loads((out/'result.json').read_text());n=7 if stage=='timing'else 14
    assert len(result['cases'])==n
    assert all(row['completed_updates']==(12 if stage=='timing'else 200)for row in result['cases'])
    assert result['optimizer_updates']==dict(warmup=2*n,training=(10 if stage=='timing'else 200)*n)
    fields=(48+11)*(12 if stage=='timing'else 404)
    evaluations=0 if stage=='timing'else 28*48
    assert result['cost_totals']['forward_fields_complete']==fields+evaluations
    assert result['cost_totals']['backward_fields_complete']==fields
    assert all(row['wall_seconds_median']==1. for row in result['cases'])
    assert all(len(row['samples'])==(10 if stage=='timing'else 200)for row in result['cases'])
    assert all(sample['wall_seconds']==1. for row in result['cases']for sample in row['samples'])
    assert len(saved)==(7 if stage=='timing'else 14*8)


def test_cpu_cli_refuses_real_live_gpu_before_loading_assets(tmp_path,monkeypatch):
    assert suite is not None
    active=tmp_path/'old_run';active.mkdir();(tmp_path/'ENCODING_SUITE_GPU.lock').write_text('{}')
    monkeypatch.setattr(suite,'load_assets',lambda *args:pytest.fail('loaded before busy guard'))
    with pytest.raises(RuntimeError,match='lock'):
        suite.main(['--execute','--stage','timing','--active-run',str(active),'--output',str(tmp_path/'new'),
            '--dataset-index','unused','--reference-config','unused','--initial-checkpoint','unused'])


@pytest.mark.parametrize('bad',('none','nan','shape','overflow'))
def test_synthesis_casts_truth_and_rejects_invalid_records_before_publication(tmp_path,monkeypatch,bad):
    assert suite is not None
    a=assets();a['truth']=a['truth'].astype(np.float64);a['config']['forward']['nt']=4;a['wavelet']=np.ones(4,np.float32)
    p=suite.build_plan(a,48,12);shape=(1,48,4,288)
    class Solver:
        def __call__(self,velocity,wavelet):
            assert velocity.dtype==torch.float32
            prediction=torch.ones(shape)
            if bad=='nan':prediction.fill_(float('nan'))
            if bad=='shape':prediction=prediction[:,:,:,:1]
            if bad=='overflow':prediction.fill_(3e38)
            return None,None,prediction
    class Acq:
        def __init__(self,*args):pass
        def batches(self,*args):yield dict(observed=torch.zeros(shape),propagators=[dict(solver=Solver())])
    class Control:
        output=tmp_path
        def check_stop(self):pass
        def write_json(self,name,payload):(tmp_path/name).write_text(json.dumps(payload))
    class Costs:
        def record(self,*args,**kwargs):pass
    monkeypatch.setattr(suite,'DenseAcquisition',Acq)
    monkeypatch.setattr(suite,'_problem',lambda *args:dict(wavelet=torch.ones(4)))
    if bad=='none':
        suite._synthesize(Control(),Costs(),a,p,'cpu')
        assert np.load(tmp_path/'clean.npy').dtype==np.float32
    else:
        with pytest.raises(ValueError,match='generated (noisy )?records'):
            suite._synthesize(Control(),Costs(),a,p,'cpu')
        assert not(tmp_path/'clean.npy').exists()
