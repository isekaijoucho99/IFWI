from types import SimpleNamespace
import numpy as np
import pytest
import torch
import experiments.rom.adaptive as adaptive
from experiments.rom.adaptive import RefreshController


def controller():
    c=object.__new__(RefreshController)
    c.options=dict(transport_tolerance=.01,primal_rank=2,probe_growth_limit=2.,source_consistency_tolerance=.01)
    c.initial_probe=dict(primal=.1,adjoint=.2);c.cache=object()
    return c


@pytest.mark.parametrize('which',['primal','adjoint'])
def test_failed_transport_indicator_rejects_before_final_rom_construction(which,monkeypatch):
    c=controller();calls=[]
    def transport(label):
        calls.append(label)
        return np.zeros((2,2,1)),dict(aggregate_relative_residual=.02 if label==which else .001)
    c.cache=SimpleNamespace(transport_primal=lambda *args:transport('primal'),
        transport_adjoint=lambda *args:transport('adjoint'))
    c.observed=torch.zeros(1,1,2,1)
    c.make_rom=lambda q:object();c.prediction=lambda *args:np.zeros((1,2,1))
    monkeypatch.setattr(adaptive,'primal_space',lambda *args:np.eye(2))
    c._finish=lambda *args:pytest.fail('Rejected transport must not produce a final solver')
    basis,rom,meta=c.corrected(torch.ones(1,1,1),1)
    assert basis is None and rom is None and meta['reason']==which+'_transport_indicator'
    assert calls==(['primal'] if which=='primal' else ['primal','adjoint'])


@pytest.mark.parametrize('reason',['probe','source','consistency_residual','correction_failure'])
def test_bad_candidate_reanchors_and_retains_attempt_logs(reason):
    c=controller();new_rom=object()
    meta=dict(construction=dict(probe=dict(primal=.1,adjoint=.2),adjoint_source_discrepancy_after=0,events=[]))
    if reason=='probe':meta['construction']['probe']['adjoint']=.5
    if reason=='source':meta['construction']['adjoint_source_discrepancy_after']=.02
    if reason=='consistency_residual':meta['construction']['events']=[dict(aggregate_relative_residual=.02)]
    def corrected(*args):
        if reason=='correction_failure':raise FloatingPointError('bad candidate')
        return np.eye(2),object(),meta
    c.corrected=corrected
    c.fresh=lambda *args:(np.eye(2),new_rom,dict(probe=dict(primal=.1,adjoint=.2),adjoint_source_discrepancy_after=0))
    basis,rom,events=c.prepare(torch.ones(1,1,1),'correction1')
    assert rom is new_rom and len(events)>=2


@pytest.mark.parametrize('error_type',[np.linalg.LinAlgError,MemoryError])
def test_failed_fresh_cache_keeps_previous_anchor_and_falls_back_to_native(error_type):
    c=controller();anchor=c.cache
    c.corrected=lambda *args:(None,None,dict(reason='primal_transport_indicator',seconds=.123))
    def failed(*args):raise error_type('failed refresh')
    c.fresh=failed
    basis,rom,events=c.prepare(torch.ones(1,1,1),'correction2')
    assert c.cache is anchor and basis is None and rom is None
    assert events[0]['seconds']==.123 and events[-1]['event']=='native_fallback'


def test_fresh_space_probe_failure_uses_native_fallback():
    c=controller()
    c.fresh=lambda *args:(np.eye(2),object(),dict(probe=dict(primal=.3,adjoint=.2),adjoint_source_discrepancy_after=0))
    basis,rom,events=c.prepare(torch.ones(1,1,1),'fresh')
    assert basis is None and rom is None and events[-1]['event']=='native_fallback'
