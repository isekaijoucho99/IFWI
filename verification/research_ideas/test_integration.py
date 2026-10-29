"""Original-RNN-FD checks, not an independently substituted toy solver."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch

torch.set_num_threads(1)


def tiny_spec(tmp_path):
    from experiments.residual_ifwi_experiment import default_config
    cfg=default_config()
    truth=np.linspace(2300,3000,9,dtype=np.float32)[:,None]+np.zeros((9,11),np.float32)
    truth[5:7,4:8]+=250
    source=tmp_path/'truth.csv'; np.savetxt(source,truth,delimiter=',')
    cfg['data'].update(model_file=str(source),downsample=1,csv_header='none')
    cfg['model']['neurons']=[2,8,8,1]
    cfg['initialization']['sigma']=1.
    cfg['acquisition'].update(num_shots=3,source_x_indices=[2,5,8])
    cfg['forward'].update(dt_s=.001,nt=48,frequency_hz=60.,npad=3)
    cfg['training'].update(epochs=3,checkpoint_interval=1,shot_batch_size=1)
    return dict(name='tiny',experiment=cfg,control=dict(kind='zero_output'),
                robustness=dict(condition='tiny',noise_sigma=0.,noise_seed=3))


def test_original_physics_value_gradient_and_counters(tmp_path):
    from research_ideas.physics import OriginalPhysics
    from experiments import residual_ifwi_experiment as e
    cfg=tiny_spec(tmp_path)['experiment']; truth=np.loadtxt(cfg['data']['model_file'],delimiter=',').astype(np.float32)
    problem=e.build_forward_problem(truth,cfg,'cpu')
    adapter=OriginalPhysics(problem)
    v=torch.full((1,9,11),2500.,requires_grad=True)
    report=adapter.value_and_gradient(v)
    direct=0.; gd=torch.zeros_like(v)
    for item in problem['propagators']:
        pred=e._simulate(item['solver'],v,problem['wavelet'])
        loss=e.waveform_mse(pred,problem['observed'][:,item['first']:item['last']],3)
        gd+=torch.autograd.grad(loss,v)[0]; direct+=float(loss.detach())
    assert report['loss']==pytest.approx(direct,rel=1e-6)
    torch.testing.assert_close(report['gradient'],gd,rtol=1e-5,atol=1e-9)
    torch.testing.assert_close(report['groups'][0]+report['groups'][1],gd,rtol=1e-5,atol=1e-9)
    assert adapter.counts['forward_shots']==3 and adapter.counts['adjoint_shots']==3


def test_adapter_parameter_pullback_matches_original_adam(tmp_path):
    from research_ideas.physics import OriginalPhysics
    from research_ideas.safety import parameter_vjp,assign_gradients
    from experiments import residual_ifwi_experiment as e
    cfg=tiny_spec(tmp_path)['experiment']; truth=np.loadtxt(cfg['data']['model_file'],delimiter=',').astype(np.float32)
    problem=e.build_forward_problem(truth,cfg,'cpu'); adapter=OriginalPhysics(problem)
    torch.manual_seed(3)
    a=e.model_from_config(np.full((9,11),2500.,np.float32),cfg); b=copy.deepcopy(a)
    oa=torch.optim.Adam(a.parameters(),lr=1e-4); ob=torch.optim.Adam(b.parameters(),lr=1e-4)
    e.train_update(a,problem,oa)
    v=b(); report=adapter.value_and_gradient(v)
    gp=parameter_vjp(v,b,report['gradient']); assign_gradients(b,gp); ob.step()
    torch.testing.assert_close(a(),b(),rtol=1e-5,atol=5e-3)


@pytest.mark.parametrize('method',['baseline','routing','wavelet','interfaces','gradient_attention'])
def test_each_method_runs_original_fd_and_records_cost(tmp_path,method):
    from research_ideas.runner import run
    spec=tiny_spec(tmp_path)
    options={'interfaces':dict(interfaces=2,width_m=30.),
             'routing':dict(top_k=1,samples=1,refresh_every=2),
             'gradient_attention':dict(warmup=0)}.get(method,{})
    recipe=dict(method=method,options=options,steps=2,eval_every=1,safeguard=False)
    result=run(spec,recipe,tmp_path/method,device='cpu')
    metrics=json.loads((result/'metrics.json').read_text())
    assert np.isfinite(metrics['velocity']['rmse_mps'])
    assert metrics['costs']['forward_shots']>0 and metrics['costs']['adjoint_shots']>0
    assert (result/'checkpoint.pt').exists()


def test_runner_rejects_unsupported_configuration(tmp_path):
    from research_ideas.runner import run
    spec=tiny_spec(tmp_path); spec['speedup']={'source_encoding':{'supershots':1,'seed':3}}
    with pytest.raises(ValueError): run(spec,dict(method='baseline'),tmp_path/'bad',device='cpu')
    with pytest.raises(ValueError): run(tiny_spec(tmp_path),dict(method='wavelet',typo=True),tmp_path/'bad2',device='cpu')


def test_checkpoint_replay_matches_continuous_run(tmp_path):
    from research_ideas.runner import run
    spec=tiny_spec(tmp_path)
    config=dict(method='routing',steps=3,eval_every=1,options=dict(top_k=1,samples=1,refresh_every=2))
    continuous=run(spec,config,tmp_path/'continuous',device='cpu')
    partial=run(spec,{**config,'steps':1},tmp_path/'partial',device='cpu')
    replay=run(spec,config,tmp_path/'replay',device='cpu',resume=partial/'checkpoint.pt')
    np.testing.assert_array_equal(np.load(continuous/'final_velocity.npy'),np.load(replay/'final_velocity.npy'))


def test_dfm_train_smoke_checkpoint_and_formal_refusal(tmp_path):
    from research_ideas.train_dfm import train
    from research_ideas.dfm import FrozenDFMPrior
    a=tmp_path/'a.npy'; b=tmp_path/'b.npy'
    np.save(a,np.linspace(1800,3500,16,dtype=np.float32)[:,None]+np.zeros((16,16),np.float32))
    np.save(b,np.linspace(1900,3600,16,dtype=np.float32)[:,None]+np.zeros((16,16),np.float32))
    ckpt=train([a],[b],tmp_path/'prior',steps=2,patch_size=8,width=8,group_size=2,groups=2,smoke_only=True)
    with pytest.raises(ValueError): FrozenDFMPrior(ckpt,'cpu',15.)
    prior=FrozenDFMPrior(ckpt,'cpu',15.,allow_smoke=True)
    g=prior.gradient(torch.full((1,9,11),2500.),1)
    assert g.shape==(1,9,11) and torch.isfinite(g).all()
    with pytest.raises(ValueError): train([a],[a],tmp_path/'leak',steps=1,patch_size=8)
    from research_ideas.runner import run
    recipe=dict(method='dfm',steps=2,eval_every=1,options=dict(checkpoint=str(ckpt),warmup=0,every=1,allow_smoke=True))
    result=run(tiny_spec(tmp_path),recipe,tmp_path/'dfm_run',device='cpu',smoke=True)
    assert json.loads((result/'metrics.json').read_text())['prior_calls']>0
