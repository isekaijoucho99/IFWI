"""Small CPU physical oracles; these tests never request a CUDA device."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.residual_ifwi_experiment import default_config
from experiments.encoding_batch_correctness import CorrectnessError
from experiments import fr_relative_preflight as preflight


def inputs():
    cfg = default_config()
    cfg['model']['neurons'] = [2, 8, 8, 1]
    cfg['acquisition'].update(num_shots=52, source_x_indices=list(range(16,272,5)),
                              source_spacing_m=75., source_margin_cells=10)
    cfg['forward'].update(dt_s=.0001, nt=6, frequency_hz=60., npad=2)
    truth = np.linspace(2300, 2600, 5, dtype=np.float32)[:,None]+np.zeros((5,288),np.float32)
    assets = dict(truth=truth, background=truth*.96,
                  wavelet=np.asarray([1.,.3,-.2,.1,0.,0.],np.float32),
                  noisy=np.random.RandomState(2).normal(0,.01,(1,52,6,288)).astype(np.float32))
    return assets, dict(config=cfg)


class Control:
    def __init__(self):
        self.rows = []
        self.stop_calls = 0
    def record(self, event, **details):
        self.rows.append(dict(event=event, **details))
    def check_stop(self):
        self.stop_calls += 1


class Costs:
    def __init__(self): self.rows = []
    def record(self,event,**details): self.rows.append(dict(event=event,**details))


def test_actual_tiny_cpu_preflight_has_exact_scoped_budget_and_normalization():
    assets, plan = inputs()
    control, costs = Control(), Costs()
    report = preflight.run_preflight(control,costs,assets,plan,'cpu')
    assert report['passed'] and report['status']=='PASS'
    assert report['Adam_updates']==0
    assert report['cost_totals']==dict(forward_fields_attempt=34,forward_fields_complete=34,
                                       backward_fields_attempt=34,backward_fields_complete=34)
    assert [case['code_channels'] for case in report['cases']]==[13,4]
    assert report['cases'][0]['scope']=='physical_shots_0_to_12_of_52'
    assert report['reference_shots']==52
    for case in report['cases']:
        assert case['passed']
        assert set(case['metrics'])==set(preflight.TOLERANCES)
        assert all(item['passed'] for item in case['metrics'].values())
    recorded = {name:sum(row.get(name,0) for row in costs.rows)
                for name in report['cost_totals']}
    assert recorded==report['cost_totals']
    assert control.stop_calls>=18
    assert report['tolerances']==preflight.TOLERANCES


def test_failure_retains_real_cost_and_does_not_execute_remaining_case(monkeypatch):
    assets, plan = inputs()
    original = preflight._solve
    def corrupt(*args,**kwargs):
        result = original(*args,**kwargs)
        if kwargs['implementation']=='batch13': result['gradient']*=1.1
        return result
    monkeypatch.setattr(preflight,'_solve',corrupt)
    with pytest.raises(CorrectnessError) as failure:
        preflight.run_preflight(Control(),Costs(),assets,plan,'cpu')
    report=failure.value.report
    assert report['status']=='FAIL' and not report['passed']
    assert report['cost_totals']['forward_fields_complete']==26
    assert len(report['cases'])==1
    assert not report['cases'][0]['metrics']['gradient']['passed']


def test_invalid_shot_count_is_rejected_before_any_physical_cost():
    assets,plan=inputs()
    plan['config']['acquisition']['num_shots']=13
    costs=Costs()
    with pytest.raises(ValueError,match='52'):
        preflight.run_preflight(Control(),costs,assets,plan,'cpu')
    assert not costs.rows


def test_relative_fr_chain_rule_and_raw_parameter_vjp_without_physics():
    from experiments.relative_velocity import RelativeVelocityParameterization
    torch.manual_seed(13)
    background = np.linspace(1700,3500,20,dtype=np.float32).reshape(4,5)
    # Same FR mechanism with small valid bases for a cheap CPU derivative test.
    model=RelativeVelocityParameterization(background,15.,[2,8,8,1],fourier=dict(
        high_freq_num=8,low_freq_num=8,phi_num=4,alpha=.01,lambda_init='fr_inr'))
    with torch.no_grad(): model.net.linear[-1].weight.fill_(.001)
    raw=torch.linspace(-.03,.04,20).reshape(1,4,5)
    report=preflight.parameter_vjp_checks(model,raw)
    assert report['passed']
    assert report['parameter_count']==sum(p.numel() for p in model.parameters())
    assert all(item['passed'] for item in report['raw_vjp'].values())
    assert all(item['passed'] for item in report['relative_chain_rule'].values())
    assert report['physical_forward_fields']==report['physical_backward_fields']==0
    assert report['Adam_updates']==0
