import copy
import importlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments import residual_ifwi_experiment as old

torch.set_num_threads(1)

def api():
    assert importlib.util.find_spec('experiments.natural_gradient_experiment'), 'integrated runner missing'
    return importlib.import_module('experiments.natural_gradient_experiment')

def config(tmp_path, kind='natural_gradient', epochs=2):
    z,x=np.indices((9,11))
    a=(2300+70*z+25*x+200*(x>=6)).astype(np.float32)
    path=tmp_path/'truth.csv'; np.savetxt(path,a,delimiter=',')
    b=old.default_config()
    b['data'].update(model_file=str(path),csv_header='none',downsample=1)
    b['initialization']['sigma']=1.
    b['model']['neurons']=[2,8,8,1]
    b['acquisition'].update(num_shots=3,source_x_indices=[1,5,9])
    b['forward'].update(dt_s=.001,nt=24,frequency_hz=60.,npad=3)
    b['training'].update(epochs=epochs,checkpoint_interval=1,shot_batch_size=2,shots_per_update=2)
    return dict(base=b,optimizer=dict(kind=kind,damping=.1,step_size=.01,cg_max_iter=3,cg_rtol=.001),evaluation_interval=1)

def test_real_fd_runner_outputs_and_counts(tmp_path):
    cfg=config(tmp_path)
    out=api().run_experiment(cfg,tmp_path/'runs')
    result=json.loads((out/'metrics.json').read_text())
    assert result['completed_updates']==2
    assert result['counts']['forward_shots_completed']==16 # 3obs+3initial+4train+6eval
    assert result['counts']['backward_shots_completed']==4
    assert result['counts']['updates_completed']==2
    assert result['counts']['inner_pde_calls']==0
    assert result['counts']['network_matvecs']>0
    assert np.isfinite(np.load(out/'final_prediction.npy')).all()
    observed=np.load(out/'observed.npy').astype('float64')
    pred=np.load(out/'final_prediction.npy').astype('float64')
    assert result['final']['data_mse_fp64']==pytest.approx(np.mean((pred-observed)**2))
    assert json.loads((out/'status.json').read_text())['state']=='completed'

@pytest.mark.parametrize('kind',['adam','natural_gradient'])
def test_resume_exact_model_and_sampling(tmp_path,kind):
    cfg=config(tmp_path,kind,epochs=3)
    full=api().run_experiment(cfg,tmp_path/'full')
    partial=copy.deepcopy(cfg);partial['base']['training']['epochs']=1
    first=api().run_experiment(partial,tmp_path/'first')
    resumed=api().run_experiment(cfg,tmp_path/'resumed',resume=first/'checkpoint.pt')
    a=torch.load(full/'checkpoint.pt',weights_only=False)
    b=torch.load(resumed/'checkpoint.pt',weights_only=False)
    for n in a['model']: assert torch.equal(a['model'][n],b['model'][n]), n
    assert [r['shot_indices'] for r in a['history']]==[r['shot_indices'] for r in b['history']]
    assert torch.equal(a['rng']['torch'],b['rng']['torch'])
    assert b['counts']['updates_completed']==3
    assert b['counts']['forward_shots_completed']>a['counts']['forward_shots_completed'] # regeneration/eval not hidden
    changed=copy.deepcopy(cfg); changed['optimizer']['damping']=.2
    with pytest.raises(ValueError,match='configuration'):
        api().run_experiment(changed,tmp_path/'bad',resume=first/'checkpoint.pt')

def test_old_adam_checkpoint_can_branch_without_resetting_model(tmp_path):
    cfg=config(tmp_path,epochs=2);b=copy.deepcopy(cfg['base']);b['training']['epochs']=1
    prev=old.run_experiment(b,tmp_path/'old',save_figures=False)
    new=api().run_experiment(cfg,tmp_path/'new',initialize_from=prev/'checkpoint.pt')
    np.testing.assert_array_equal(np.load(new/'run_start_velocity.npy'),np.load(prev/'final_velocity.npy'))
    result=json.loads((new/'metrics.json').read_text())
    assert result['starting_update']==1
    assert result['counts']['updates_completed']==1 # new branch actual work only

def test_fail_after_pde_keeps_counts_and_checkpoint(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    mod=api()
    def fail(*args,**kwargs): raise RuntimeError('after physical forward')
    monkeypatch.setattr(mod.ng,'cg',fail)
    with pytest.raises(RuntimeError,match='after physical'):
        mod.run_experiment(cfg,tmp_path/'fail')
    out=next((tmp_path/'fail').iterdir())
    status=json.loads((out/'status.json').read_text())
    assert status['state']=='failed' and status['completed_updates']==0
    assert status['counts']['backward_shots_completed']==2
    assert status['counts']['forward_shots_completed']==8
    assert (out/'interrupted.pt').exists()

def test_rejects_invalid_before_work(tmp_path):
    cfg=config(tmp_path);cfg['optimizer']['damping']=0
    with pytest.raises(ValueError):api().run_experiment(cfg,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()

def test_ng_to_adam_rejection_does_not_create_checkpoint(tmp_path):
    cfg=config(tmp_path,epochs=1)
    out=api().run_experiment(cfg,tmp_path/'source')
    nextcfg=copy.deepcopy(cfg);nextcfg['optimizer']['kind']='adam';nextcfg['base']['training']['epochs']=2
    with pytest.raises(ValueError,match='reset Adam'):
        api().run_experiment(nextcfg,tmp_path/'rejected',initialize_from=out/'checkpoint.pt')
    assert not (tmp_path/'rejected').exists()

def test_finalization_only_resume_does_not_add_updates(tmp_path):
    cfg=config(tmp_path,epochs=1)
    out=api().run_experiment(cfg,tmp_path/'source')
    resumed=api().run_experiment(cfg,tmp_path/'finalize',resume=out/'checkpoint.pt')
    metrics=json.loads((resumed/'metrics.json').read_text())
    assert metrics['run_added_counts']['updates_completed']==0
    assert metrics['run_added_counts']['backward_shots_completed']==0
    assert metrics['completed_updates']==1
    np.testing.assert_array_equal(np.load(out/'final_velocity.npy'),np.load(resumed/'final_velocity.npy'))
