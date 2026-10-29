import copy
from pathlib import Path
import numpy as np
import pytest
import torch


def datasets(tmp_path):
    train=tmp_path/'train'; val=tmp_path/'val'; train.mkdir(); val.mkdir()
    z,x=np.mgrid[:12,:14]
    np.save(train/'a.npy',(1800+20*z+10*np.sin(x)).astype('float32'))
    np.save(train/'b.npy',(2000+12*z+5*np.cos(x)).astype('float32'))
    np.save(val/'c.npy',(1900+25*z+4*np.sin(x)).astype('float32'))
    return train,val


def test_model_level_split_and_duplicate_rejection(tmp_path):
    from ifwi_research.train_dfm import model_inventory, validate_split
    train,val=datasets(tmp_path)
    a,b=model_inventory(train,8),model_inventory(val,8)
    assert len(a)==2 and len(b)==1
    validate_split(a,b)
    np.save(val/'duplicate.npy',np.load(train/'a.npy').astype('float64'))
    with pytest.raises(ValueError,match='overlap'): validate_split(a,model_inventory(val,8))


def test_dfm_tiny_training_saves_usable_checkpoint(tmp_path):
    from ifwi_research.train_dfm import train
    from ifwi_research.dfm import FrozenDFMPrior
    train_dir,val_dir=datasets(tmp_path)
    out=tmp_path/'run'
    train(train_dir,val_dir,out,spacing_m=20.,patch_size=8,width=8,group_size=4,
          groups_per_step=1,steps=2,validate_every=1,seed=5,device='cpu')
    payload=torch.load(out/'last.pt',weights_only=True)
    assert payload['meta']['trained_steps']==2
    assert len(payload['meta']['train_model_hashes'])==2
    assert payload['meta']['val_model_hashes']
    prior=FrozenDFMPrior(out/'best.pt',20.)
    assert prior.meta['trained_steps']>=1
    assert (out/'history.jsonl').exists()


def test_checkpoint_safe_rng_serialization(tmp_path):
    from ifwi_research.safety import rng_state,restore_rng
    path=tmp_path/'rng.pt'; state=rng_state(); torch.save(state,path)
    loaded=torch.load(path,weights_only=True)
    restore_rng(loaded)
    expected=torch.randn(5)
    restore_rng(state)
    assert torch.equal(expected,torch.randn(5))


def test_dfm_engine_uses_real_candidate_guard(tmp_path):
    from ifwi_research.train_dfm import train
    from ifwi_research.engine import ResearchEngine,ResearchConfig
    from ifwi_research.toy_wave import TinyINR,ToyWaveObjective
    from ifwi_research.dfm import FrozenDFMPrior
    td,vd=datasets(tmp_path); out=tmp_path/'trained'
    train(td,vd,out,spacing_m=20.,patch_size=8,width=8,group_size=4,
          groups_per_step=1,steps=1,validate_every=1,seed=5,device='cpu')
    prior=FrozenDFMPrior(out/'last.pt',20.)
    initial=torch.full((1,10,14),2000.); true=initial.clone(); true[:,5:8,4:10]+=200
    objective=ToyWaveObjective(true,shots=3,nt=45)
    engine=ResearchEngine(TinyINR(initial,8),initial,objective,20.,
                          ResearchConfig(method='dfm',prior_start=1,prior_every=1),prior=prior)
    result=engine.step()
    assert result['prior_calls']>0 and result['prior'] is not None
    assert result['step']['physical_after']<=result['step']['physical_before']
    assert objective.counters['forward_solves']>=6


def test_runner_fails_closed_outside_repository(tmp_path):
    from ifwi_research.repository import require_repository
    with pytest.raises(RuntimeError,match='IFWI checkout'): require_repository(tmp_path)
