"""Actual CPU FD update checks: original-shot denominator also reaches VJP.

The expected objective is constructed separately before train_update. Hooks
capture raw parameter gradients before clipping or Adam can conceal scaling.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
torch.set_num_threads(1)

from experiments import residual_ifwi_experiment as experiment
from core_speed import SourceEncoder,normalized_encoding_loss


@pytest.mark.parametrize('grouping,count',[('global',2),('interleaved',2),('interleaved',3),('identity',13)])
@pytest.mark.parametrize('max_hz',[None,20.])
def test_train_update_matches_original_shot_mean_loss_and_raw_gradient(grouping,count,max_hz):
    cfg=experiment.default_config()
    cfg['model']['neurons']=[2,8,8,1]
    cfg['acquisition'].update(num_shots=13,source_x_indices=list(range(1,27,2)))
    cfg['forward'].update(dt_s=.001,nt=24,frequency_hz=60.,npad=3)
    cfg['training'].update(epochs=1,checkpoint_interval=1)
    truth=np.linspace(2300,2800,8,dtype=np.float32)[:,None]+np.zeros((8,29),np.float32)
    base=experiment.build_forward_problem(truth,cfg,'cpu')
    problem=SourceEncoder(base,count,3,grouping=grouping).problem(7)
    problem['frequency_scales']={20.:1.}
    torch.manual_seed(3)
    model=experiment.VelocityParameterization(truth*.96,15.,[2,8,8,1])
    velocity=model()
    prediction=problem['propagators'][0]['solver'](velocity,problem['wavelet'])[2]
    expected_data=normalized_encoding_loss(prediction,problem['observed'],13)
    expected_objective=experiment.training_objective(prediction,problem['observed'],13,max_hz,problem)
    params=tuple(model.parameters())
    expected_gradients=torch.autograd.grad(expected_objective,params)
    observed_gradients={}
    handles=[]
    for index,parameter in enumerate(params):
        def observe(gradient,index=index):
            observed_gradients[index]=gradient.detach().clone()
        handles.append(parameter.register_hook(observe))
    try:
        report=experiment.train_update(model,problem,torch.optim.Adam(model.parameters(),lr=1e-4),max_hz=max_hz)
    finally:
        for handle in handles:handle.remove()
    assert report['data_mse']==pytest.approx(float(expected_data.detach()),rel=2e-6)
    assert len(observed_gradients)==len(params)
    for index,expected in enumerate(expected_gradients):
        difference=torch.linalg.vector_norm((observed_gradients[index]-expected).double())
        reference=torch.linalg.vector_norm(expected.double()).clamp_min(1e-30)
        assert float(difference/reference)<1e-5,(grouping,count,max_hz,index,float(difference/reference))
