"""Runs ONLY after installing this additive package in the actual IFWI checkout.

A missing repository is reported as skipped, not as a successful FD integration.
Once original files exist, import/numerical errors are real test failures.
"""
from pathlib import Path
import copy
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
pytestmark=pytest.mark.skipif(not (ROOT/'core_runner.py').exists(),reason='Original IFWI checkout not present; real RNN-FD integration UNVERIFIED')


def problem():
    from experiments import residual_ifwi_experiment as e
    cfg=e.default_config(); cfg['model']['neurons']=[2,8,8,1]
    cfg['forward'].update(nt=48,dt_s=.001,frequency_hz=30.,npad=3)
    cfg['data']['grid_spacing_m']=20.
    cfg['acquisition'].update(num_shots=3,source_x_indices=[3,8,13],receiver_stride=1)
    cfg['training']['shot_batch_size']=1
    initial=np.full((10,18),2000,dtype=np.float32); truth=initial.copy(); truth[4:7,5:12]+=150
    raw=e.build_forward_problem(truth,cfg,'cpu')
    from ifwi_research.repository import RepositoryObjective
    return cfg,initial,raw,RepositoryObjective(raw)


def test_original_fd_code_basis_equivalence():
    from ifwi_research.routing import ContrastRouter
    cfg,initial,raw,obj=problem(); v=torch.from_numpy(initial)[None]
    exact=obj.evaluate(v,gradient=True)
    coded=obj.evaluate(v,plan=ContrastRouter(3).plan(full=True),gradient=True)
    assert coded.value==pytest.approx(exact.value,rel=2e-4,abs=1e-9)
    torch.testing.assert_close(exact.gradient,coded.gradient,rtol=5e-3,atol=1e-8)


def test_original_fd_baseline_update_parity():
    from experiments import residual_ifwi_experiment as e
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    cfg,initial,raw,obj=problem(); torch.manual_seed(3)
    base=e.model_from_config(initial,cfg); control=copy.deepcopy(base)
    opt=torch.optim.Adam(control.parameters(),lr=1e-4)
    engine=ResearchEngine(base,torch.from_numpy(initial)[None],obj,20.,ResearchConfig())
    e.train_update(control,raw,opt); engine.step()
    torch.testing.assert_close(control(),engine.model(),rtol=1e-5,atol=.01)


@pytest.mark.parametrize('method',['routing','wavelet','interface','gradient_attention'])
def test_original_fd_new_path_connected(method):
    from experiments import residual_ifwi_experiment as e
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    cfg,initial,raw,obj=problem(); torch.manual_seed(3)
    engine=ResearchEngine(e.model_from_config(initial,cfg),torch.from_numpy(initial)[None],obj,20.,
        ResearchConfig(method=method,attention_block=2))
    engine.step(); engine.step(); engine.step()
    assert torch.isfinite(engine.model()).all()
    assert not torch.equal(engine.model().detach(),torch.from_numpy(initial)[None])
