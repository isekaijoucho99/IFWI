import math
import os
import sys
from pathlib import Path
import pytest
import torch
from torch.func import jacrev

# Tests first: implementation must expose the requested contract.
import importlib.util

def test_implementation_exists():
    assert importlib.util.find_spec('experiments.natural_gradient') is not None, 'natural_gradient implementation missing'

def api():
    from experiments import natural_gradient as ng
    return ng

D=torch.float64

def test_matrix_free_matches_dense_and_chain_rule():
    ng=api(); torch.manual_seed(3)
    theta=torch.randn(7,dtype=D)
    W=torch.randn(5,7,dtype=D)
    q=lambda t: torch.sin(W@t)
    gm=torch.randn(5,dtype=D); scale=1000.
    rhs,A=ng.linear_system(q,theta,scale*gm,0.2)
    J=jacrev(q)(theta)
    t=theta.clone().requires_grad_()
    direct=torch.autograd.grad(((2500+scale*q(t))*gm).sum(),t)[0]
    torch.testing.assert_close(rhs,direct)
    p=torch.randn_like(theta)
    torch.testing.assert_close(A(p),(J.T@J+0.2*torch.eye(7,dtype=D))@p)
    x,info=ng.cg(A,rhs,max_iter=30,rtol=1e-10)
    torch.testing.assert_close(x,torch.linalg.solve(J.T@J+0.2*torch.eye(7,dtype=D),rhs))
    assert info['converged']
    assert torch.dot(rhs,x)>0

def test_rank_deficiency_zero_rhs_and_bad_input():
    ng=api(); theta=torch.ones(4,dtype=D)
    q=lambda t:t[:2]
    rhs,A=ng.linear_system(q,theta,torch.ones(2,dtype=D),.1)
    x,_=ng.cg(A,rhs,max_iter=10)
    assert torch.isfinite(x).all()
    z,info=ng.cg(A,torch.zeros_like(rhs)); assert torch.equal(z,torch.zeros_like(z)); assert info['iterations']==0
    with pytest.raises(ValueError): ng.linear_system(q,theta,torch.ones(2,dtype=D),0)
    with pytest.raises(ValueError): ng.linear_system(q,theta,torch.ones(2,dtype=D),float('nan'))
    with pytest.raises(ValueError): ng.cg(A,rhs,max_iter=0)
    with pytest.raises(FloatingPointError): ng.cg(lambda p:-p,rhs)

class ToyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.weight=torch.nn.Parameter(torch.tensor([.2,-.1,.4],dtype=D))
        self.register_buffer('base',torch.tensor([2500.,2600.,2700.],dtype=D))
    def forward(self): return self.base+1000*torch.sin(self.weight)

def test_physical_calls_once_per_batch_and_accepted_update():
    ng=api(); model=ToyVelocity(); original=model.weight.detach().clone(); calls=[]
    target=torch.tensor([2600.,2700.,2800.],dtype=D)
    def loss(m):
        calls.append(1); return ((m-target)/1000).square().sum()/2
    report=ng.step(model,[loss,loss],scale=1000.,offset=model.base,damping=.1,step_size=.05,cg_max_iter=10)
    assert len(calls)==2 # No simulator callback in CG or candidate checks.
    assert report['physical_batches']==2
    assert report['direction_dot_rhs']>0
    assert not torch.equal(model.weight,original)
    assert model.weight.grad is None
    assert report['linearization_relative_error']>=0

def test_rejected_candidate_rolls_back_without_trial_pde():
    ng=api(); model=ToyVelocity(); old={k:v.clone() for k,v in model.state_dict().items()}; calls=[]
    def loss(m): calls.append(1); return m.square().mean()
    def reject(m): raise ValueError('CFL rejected')
    with pytest.raises(ValueError,match='CFL'): ng.step(model,[loss],scale=1000.,damping=.1,step_size=.1,validate_candidate=reject)
    assert len(calls)==1
    for k,v in model.state_dict().items(): torch.testing.assert_close(v,old[k],rtol=0,atol=0)

def test_nonfinite_physical_gradient_rejected():
    ng=api(); model=ToyVelocity(); old=model.weight.clone()
    with pytest.raises(FloatingPointError): ng.step(model,[lambda m:(m*float('nan')).sum()],scale=1000.,damping=.1,step_size=.1)
    torch.testing.assert_close(model.weight,old)

@pytest.mark.parametrize("parameterization", ["residual", "absolute"])
def test_cached_siren_forward_and_gradient_equivalence(parameterization):
    ng=api()
    repo=Path(os.environ.get('IFWI_REPO_ROOT',Path(__file__).resolve().parents[2]))
    if not (repo/'experiments/residual_ifwi_experiment.py').exists():
        pytest.skip('Set IFWI_REPO_ROOT to the inspected compatible IFWI checkout')
    sys.path.insert(0,str(repo))
    from experiments.residual_ifwi_experiment import VelocityParameterization
    torch.manual_seed(4)
    original=VelocityParameterization(torch.full((3,4),2500.),15,[2,5,5,1], parameterization=parameterization).double()
    with torch.no_grad():
        for parameter in original.parameters(): parameter.add_(.01*torch.randn_like(parameter))
    adapter=ng.ResidualSirenView(original)
    torch.testing.assert_close(adapter(),original(),rtol=0,atol=0)
    theta,q=ng.functional_coordinates(adapter,offset=original.fixed_init_mps,scale=1000.)
    gm=torch.randn_like(original())
    rhs,_=ng.linear_system(q,theta,1000*gm,.1)
    direct=torch.autograd.grad((original()*gm).sum(),tuple(original.parameters()))
    torch.testing.assert_close(rhs,torch.cat([v.reshape(-1) for v in direct]),rtol=1e-9,atol=1e-9)

def test_integration_adapter_preserves_batch_normalization():
    ng=api()
    assert hasattr(ng,'train_update_ng'), 'explicit integration adapter missing'
    model=ToyVelocity(); calls=[]; totals=[]
    obs=torch.zeros(1,3,1,3,dtype=D)
    problem={'propagators':[{'solver':1,'first':0,'last':1},{'solver':2,'first':1,'last':3}],
             'observed':obs,'wavelet':None}
    def simulate(n,m,w): calls.append(n); return m[None,None,None,:].expand(1,n,1,3)/1000
    def objective(pred,obs,total):
        totals.append(total)
        return (pred-obs).square().sum()/total/3
    rep=ng.train_update_ng(model,problem,simulate=simulate,objective=objective,
       scale=1000.,damping=.1,step_size=.01)
    assert calls==[1,2]
    assert totals==[3,3]
    assert rep['physical_batches']==2
    full=ToyVelocity()
    full_problem={**problem,'propagators':[{'solver':3,'first':0,'last':3}]}
    full_rep=ng.train_update_ng(full,full_problem,simulate=simulate,objective=objective,
       scale=1000.,damping=.1,step_size=.01)
    torch.testing.assert_close(full.weight,model.weight)
    assert full_rep['loss']==pytest.approx(rep['loss'])

def test_tiny_float32_rhs_not_mistaken_for_zero():
    ng=api(); rhs=torch.tensor([1e-24],dtype=torch.float32)
    x,info=ng.cg(lambda p:p,rhs)
    assert info['iterations']>0
    torch.testing.assert_close(x,rhs,rtol=1e-6,atol=0)

def test_parameter_overflow_is_not_hidden_by_saturation():
    ng=api()
    class Saturating(torch.nn.Module):
        def __init__(self): super().__init__(); self.w=torch.nn.Parameter(torch.zeros(1,dtype=D))
        def forward(self): return self.w.tanh()
    model=Saturating()
    with pytest.raises(FloatingPointError):
        ng.step(model,[lambda m:(10*m).sum()],scale=1.,damping=.1,step_size=1e308)
    assert model.w.item()==0

def test_low_precision_rejected_explicitly():
    ng=api()
    with pytest.raises(ValueError,match='float32'):
        ng.cg(lambda p:p,torch.tensor([1e-4],dtype=torch.float16))
    with pytest.raises(ValueError,match='float32'):
        ng.functional_coordinates(ToyVelocity().half())

def test_partial_commit_exception_restores_all_parameters(monkeypatch):
    ng=api()
    class Two(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.a=torch.nn.Parameter(torch.tensor([.2],dtype=D)); self.b=torch.nn.Parameter(torch.tensor([.4],dtype=D))
        def forward(self): return self.a+2*self.b
    model=Two(); old=[p.detach().clone() for p in model.parameters()]
    original=torch.Tensor.copy_; count=[0]
    def fail_second(self,other,*args,**kwargs):
        count[0]+=1
        if count[0]==2: raise RuntimeError('injected commit interruption')
        return original(self,other,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'copy_',fail_second)
    with pytest.raises(RuntimeError,match='commit interruption'):
        ng.step(model,[lambda m:m.square().sum()],scale=1.,damping=.1,step_size=.01)
    for p,q in zip(model.parameters(),old): torch.testing.assert_close(p,q,rtol=0,atol=0)
