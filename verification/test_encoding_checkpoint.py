"""CPU-only full-state resume and no-overwrite publication checks."""
import copy
import importlib.util
import os
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
torch.set_num_threads(1)
TARGET=ROOT/'experiments/encoding_checkpoint.py'
if TARGET.exists():
    spec=importlib.util.spec_from_file_location('encoding_checkpoint_target',TARGET)
    checkpoint=importlib.util.module_from_spec(spec);spec.loader.exec_module(checkpoint)
else:
    checkpoint=None


@pytest.fixture(autouse=True)
def require_module():
    assert checkpoint is not None,'encoding_checkpoint.py not implemented'


class SmallProgressive(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(3,2,dtype=torch.float64)
        self.register_buffer('completed_updates',torch.tensor(0,dtype=torch.int64))

    def forward(self,x):
        return self.linear(x)*(1+self.completed_updates.double()/10)


def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value)


def step(model,optimizer):
    x=torch.randn(4,3,dtype=torch.float64)+random.random()
    y=torch.from_numpy(np.random.randn(4,2)).double()
    optimizer.zero_grad(set_to_none=True)
    loss=(model(x)-y).square().mean();loss.backward();optimizer.step()
    model.completed_updates.add_(1)


def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict)and set(a)==set(b)and all(exact(a[k],b[k])for k in a)
    if isinstance(a,(tuple,list)):return type(a)is type(b)and len(a)==len(b)and all(exact(x,y)for x,y in zip(a,b))
    return type(a)is type(b)and a==b


def meta():return {'schema':'test-protocol-v1','noise':'clean','seed':3,'code':{'grouping':'global','B':2},'data_hash':'abc'}


def initialized():
    seed(123);model=SmallProgressive();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    step(model,optimizer)
    return model,optimizer


def test_save_load_then_second_update_matches_uninterrupted_model_adam_and_rng(tmp_path):
    model,optimizer=initialized();path=tmp_path/'checkpoint-1.pt'
    assert checkpoint.save_checkpoint(model,optimizer,path,1,meta())==path
    step(model,optimizer)
    expected_model=copy.deepcopy(model.state_dict());expected_optimizer=copy.deepcopy(optimizer.state_dict())
    expected_draws=(random.random(),np.random.randn(3),torch.randn(3))
    seed(999);resumed=SmallProgressive();resumed_optimizer=torch.optim.Adam(resumed.parameters(),lr=1e-3)
    assert checkpoint.load_checkpoint(resumed,resumed_optimizer,path,meta())==1
    assert int(resumed.completed_updates)==1
    step(resumed,resumed_optimizer)
    assert exact(resumed.state_dict(),expected_model)
    assert exact(resumed_optimizer.state_dict(),expected_optimizer)
    assert random.random()==expected_draws[0]
    assert np.array_equal(np.random.randn(3),expected_draws[1])
    assert torch.equal(torch.randn(3),expected_draws[2])


def test_new_checkpoint_is_weights_only_readable_and_save_does_not_consume_rng(tmp_path):
    model,optimizer=initialized()
    python_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.get_rng_state().clone()
    path=checkpoint.save_checkpoint(model,optimizer,tmp_path/'snapshot.pt',1,meta())
    value=torch.load(path,map_location='cpu',weights_only=True)
    assert value['completed']==1
    assert value['metadata']==meta()
    assert isinstance(value['rng']['numpy']['keys'],list)
    assert all(type(x)is int for x in value['rng']['numpy']['keys'])
    assert random.getstate()==python_state
    after=np.random.get_state();assert after[0]==numpy_state[0]and np.array_equal(after[1],numpy_state[1])and after[2:]==numpy_state[2:]
    assert torch.equal(torch.get_rng_state(),torch_state)


def test_initial_zero_checkpoint_accepts_empty_adam_state(tmp_path):
    seed(2);model=SmallProgressive();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    path=checkpoint.save_checkpoint(model,optimizer,tmp_path/'zero.pt',0,meta())
    step(model,optimizer)
    assert checkpoint.load_checkpoint(model,optimizer,path,meta())==0
    assert optimizer.state_dict()['state']=={}
    assert int(model.completed_updates)==0


def test_save_rejects_mismatched_completed_buffer_or_adam_step(tmp_path):
    model,optimizer=initialized()
    with pytest.raises(ValueError):checkpoint.save_checkpoint(model,optimizer,tmp_path/'bad-progress.pt',2,meta())
    model.completed_updates.fill_(2)
    with pytest.raises(ValueError):checkpoint.save_checkpoint(model,optimizer,tmp_path/'bad-adam.pt',2,meta())
    assert not list(tmp_path.glob('*.pt'))


def test_load_rejects_wrong_metadata_before_mutation(tmp_path):
    model,optimizer=initialized();path=checkpoint.save_checkpoint(model,optimizer,tmp_path/'state.pt',1,meta())
    before_model=copy.deepcopy(model.state_dict());before_optimizer=copy.deepcopy(optimizer.state_dict());before_rng=torch.get_rng_state().clone()
    for wrong in (dict(meta(),noise='sigma4'),dict(meta(),seed=3.),dict(meta(),extra=True)):
        with pytest.raises(ValueError):checkpoint.load_checkpoint(model,optimizer,path,wrong)
        assert exact(model.state_dict(),before_model)and exact(optimizer.state_dict(),before_optimizer)
        assert torch.equal(torch.get_rng_state(),before_rng)


@pytest.mark.parametrize('damage',['completed','adam','model_nan','optimizer_nan','rng'])
def test_corrupt_saved_state_is_rejected_before_mutation(tmp_path,damage):
    model,optimizer=initialized();path=checkpoint.save_checkpoint(model,optimizer,tmp_path/'good.pt',1,meta())
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if damage=='completed':payload['completed']=2
    elif damage=='adam':next(iter(payload['optimizer']['state'].values()))['step'].fill_(2)
    elif damage=='model_nan':next(value for value in payload['model'].values()if value.is_floating_point()).view(-1)[0]=float('nan')
    elif damage=='optimizer_nan':next(iter(payload['optimizer']['state'].values()))['exp_avg'].view(-1)[0]=float('nan')
    else:payload['rng']['torch_cpu']=torch.zeros(3,dtype=torch.uint8)
    bad=tmp_path/(damage+'.pt');torch.save(payload,bad)
    before_model=copy.deepcopy(model.state_dict());before_optimizer=copy.deepcopy(optimizer.state_dict());before_rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):checkpoint.load_checkpoint(model,optimizer,bad,meta())
    assert exact(model.state_dict(),before_model)and exact(optimizer.state_dict(),before_optimizer)
    assert torch.equal(torch.get_rng_state(),before_rng)


def test_existing_target_is_never_overwritten(tmp_path):
    model,optimizer=initialized();target=tmp_path/'user.pt';target.write_bytes(b'user-evidence')
    with pytest.raises(FileExistsError):checkpoint.save_checkpoint(model,optimizer,target,1,meta())
    assert target.read_bytes()==b'user-evidence'


def test_publication_retry_serializes_once_and_uses_same_paths(tmp_path,monkeypatch):
    model,optimizer=initialized();real_link=checkpoint.os.link;real_save=checkpoint.torch.save
    calls=[];saved=[]
    def link(source,target):
        calls.append((source,target))
        if len(calls)<3:
            error=PermissionError('synthetic transient publication');error.winerror=5;raise error
        return real_link(source,target)
    def save(*args,**kwargs):saved.append(True);return real_save(*args,**kwargs)
    monkeypatch.setattr(checkpoint.os,'link',link);monkeypatch.setattr(checkpoint.torch,'save',save)
    monkeypatch.setattr(checkpoint.time,'sleep',lambda seconds:None)
    target=checkpoint.save_checkpoint(model,optimizer,tmp_path/'retry.pt',1,meta())
    assert target.exists()and len(calls)==3 and len(set(calls))==1 and len(saved)==1


def test_persistent_publication_failure_is_explicit_after_six_attempts(tmp_path,monkeypatch):
    model,optimizer=initialized();calls=[]
    def link(source,target):
        calls.append((source,target));error=PermissionError('synthetic persistent publication');error.winerror=32;raise error
    monkeypatch.setattr(checkpoint.os,'link',link);monkeypatch.setattr(checkpoint.time,'sleep',lambda seconds:None)
    with pytest.raises(checkpoint.CheckpointPublishError)as result:
        checkpoint.save_checkpoint(model,optimizer,tmp_path/'blocked.pt',1,meta())
    assert len(calls)==6 and len(set(calls))==1 and result.value.winerror==32
    assert not(tmp_path/'blocked.pt').exists()
    assert Path(calls[0][0]).is_file()


def test_other_publication_error_is_not_retried(tmp_path,monkeypatch):
    model,optimizer=initialized();calls=[]
    def link(source,target):calls.append(True);raise OSError('nonsharing error')
    monkeypatch.setattr(checkpoint.os,'link',link)
    with pytest.raises(OSError):checkpoint.save_checkpoint(model,optimizer,tmp_path/'error.pt',1,meta())
    assert len(calls)==1


def test_racing_target_creator_is_not_overwritten(tmp_path,monkeypatch):
    model,optimizer=initialized();real_link=checkpoint.os.link
    def link(source,target):Path(target).write_bytes(b'other-writer');return real_link(source,target)
    monkeypatch.setattr(checkpoint.os,'link',link)
    with pytest.raises(FileExistsError):checkpoint.save_checkpoint(model,optimizer,tmp_path/'race.pt',1,meta())
    assert(tmp_path/'race.pt').read_bytes()==b'other-writer'


def test_non_json_metadata_is_rejected(tmp_path):
    model,optimizer=initialized()
    for invalid in ({'bad':np.int64(3)},{'bad':(1,2)},{'bad':float('nan')}):
        with pytest.raises(ValueError):checkpoint.save_checkpoint(model,optimizer,tmp_path/'invalid.pt',1,invalid)


@pytest.mark.parametrize('damage',['moment_shape','moment_dtype','parameter_names','group_length'])
def test_malformed_optimizer_structure_is_rejected_before_model_mutation(tmp_path,damage):
    # Construct a standalone malformed input; no publication or hard-link is needed.
    source,source_optimizer=initialized();n=np.random.get_state()
    payload={'schema':checkpoint.SCHEMA,'completed':1,'metadata':meta(),
        'model':copy.deepcopy(source.state_dict()),'optimizer':copy.deepcopy(source_optimizer.state_dict()),
        'optimizer_parameter_names':[['linear.weight','linear.bias']],
        'rng':{'python':random.getstate(),'numpy':{'bit_generator':n[0],'keys':[int(v)for v in n[1]],
            'pos':int(n[2]),'has_gauss':int(n[3]),'cached_gaussian':float(n[4])},
            'torch_cpu':torch.get_rng_state().clone(),'cuda_initialized':False,'torch_cuda':[]}}
    state=next(iter(payload['optimizer']['state'].values()))
    if damage=='moment_shape':state['exp_avg']=torch.zeros(7,dtype=torch.float64)
    elif damage=='moment_dtype':state['exp_avg_sq']=state['exp_avg_sq'].float()
    elif damage=='parameter_names':payload['optimizer_parameter_names']=[['linear.weight']]
    else:payload['optimizer']['param_groups'][0]['params'].pop()
    path=tmp_path/'malformed.pt';torch.save(payload,path)
    seed(900);model=SmallProgressive();optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    before=copy.deepcopy(model.state_dict());before_optimizer=copy.deepcopy(optimizer.state_dict())
    with pytest.raises(ValueError):checkpoint.load_checkpoint(model,optimizer,path,meta())
    assert exact(model.state_dict(),before)and exact(optimizer.state_dict(),before_optimizer)
