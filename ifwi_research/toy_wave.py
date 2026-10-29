"""Tiny acoustic finite-difference fixture, NOT the repository's RNN-FD solver.

Only for arithmetic, gradient, routing and recovery tests. Its small domain,
sponge boundary and synthetic observations cannot establish geological accuracy.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .engine import Evaluation
from .routing import common_contrast_basis


class TinyINR(nn.Module):
    def __init__(self,background: torch.Tensor,width: int=16):
        super().__init__()
        self.register_buffer('background',background.detach().clone())
        h,w=background.shape[-2:]
        z,x=torch.meshgrid(torch.linspace(-1,1,h),torch.linspace(-1,1,w),indexing='ij')
        self.register_buffer('coordinates',torch.stack((x,z),-1).to(background))
        self.hidden=nn.Linear(2,width).to(background)
        self.output=nn.Linear(width,1).to(background)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self):
        return self.background+1000*self.output(torch.sin(5*self.hidden(self.coordinates))).squeeze(-1)[None]


class ToyWaveObjective:
    def __init__(self,truth: torch.Tensor,shots: int=3,nt: int=60,
                 spacing_m: float=20.,dt_s: float=.002):
        if truth.ndim!=3 or truth.shape[0]!=1: raise ValueError('Expected [1,H,W] model')
        if shots<2 or nt<4 or min(truth.shape[-2:])<5: raise ValueError('Toy domain too small')
        self.shots,self.nt,self.spacing_m,self.dt_s=shots,nt,spacing_m,dt_s
        if not self.valid(truth): raise ValueError('Unstable toy model')
        h,w=truth.shape[-2:]; self.shape=(h,w)
        self.source_x=torch.linspace(2,w-3,shots).round().long()
        self.source_z=2; self.receiver_z=2
        t=torch.arange(nt)*dt_s; a=math.pi*25*(t-.04)
        self.wavelet=(1-2*a.square())*torch.exp(-a.square())
        z,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
        edge=torch.minimum(torch.minimum(z,h-1-z),torch.minimum(x,w-1-x)).float()
        self.damping=torch.exp(-.10*(2-edge).clamp_min(0).square())
        with torch.no_grad():
            identity=torch.eye(shots,device=truth.device,dtype=truth.dtype)
            self.observed=torch.stack([self._simulate(truth,identity[i]) for i in range(shots)]).detach()
        self.counters={'forward_solves':0,'backward_solves':0,'forward_time_steps':0}

    def valid(self,v):
        return bool(torch.isfinite(v).all() and v.min()>0 and v.max()*self.dt_s/self.spacing_m<1/math.sqrt(2))

    def _simulate(self,v,weights):
        h,w=self.shape; device,dtype=v.device,v.dtype
        current=torch.zeros(1,1,h,w,device=device,dtype=dtype); previous=torch.zeros_like(current)
        kernel=v.new_tensor([[0,1,0],[1,-4,1],[0,1,0]])[None,None]
        inject=torch.zeros(h,w,device=device,dtype=dtype)
        inject.index_put_((torch.full((self.shots,),self.source_z,device=device,dtype=torch.long),
                           self.source_x.to(device)),weights.to(v),accumulate=True)
        inject=inject[None,None]; damping=self.damping.to(v)[None,None]
        coefficient=(v[:,None]*self.dt_s/self.spacing_m).square()
        records=[]
        for source in self.wavelet.to(v):
            following=(2*current-previous+coefficient*F.conv2d(current,kernel,padding=1)+source*inject)*damping
            records.append(following[0,0,self.receiver_z,:])
            previous,current=current,following
        return torch.stack(records)

    def _filter(self,traces,cutoff):
        if cutoff is None: return traces
        f=torch.fft.rfftfreq(2*self.nt,self.dt_s,device=traces.device,dtype=traces.dtype)
        response=1/(1+(f/cutoff)**12)
        return torch.fft.irfft(torch.fft.rfft(traces,n=2*self.nt,dim=-2)*response[:,None],
                               n=2*self.nt,dim=-2)[...,:self.nt,:]

    def evaluate(self,velocity,plan=None,cutoff=None,gradient=True):
        if not self.valid(velocity): raise FloatingPointError('Invalid trial velocity')
        if cutoff is not None and (not math.isfinite(cutoff) or cutoff<=0): raise ValueError('Invalid cutoff')
        observed=self.observed.to(velocity)
        if plan is None:
            codes=torch.eye(self.shots,device=velocity.device,dtype=velocity.dtype)
            ids=torch.arange(self.shots); weights=torch.full((self.shots,),1/self.shots)
        else:
            codes=common_contrast_basis(self.shots,device=velocity.device,dtype=velocity.dtype)
            ids,weights=plan.indices,plan.weights
        filtered=self._filter(observed,cutoff)
        scale=float(observed.double().square().sum()/filtered.double().square().sum()) if cutoff is not None else 1.
        total=0.; grad=torch.zeros_like(velocity) if gradient else None; energies=[]
        for index,weight in zip(ids.tolist(),weights.tolist()):
            with torch.enable_grad() if gradient else torch.no_grad():
                v=velocity.detach().requires_grad_(gradient)
                predicted=self._simulate(v,codes[index])
                target=torch.einsum('s,str->tr',codes[index],observed)
                loss=(self._filter(predicted,cutoff)-self._filter(target,cutoff)).square().mean()*scale
                value=float(loss.detach()); energies.append(value); total+=weight*value
                self.counters['forward_solves']+=1; self.counters['forward_time_steps']+=self.nt
                if gradient:
                    grad.add_(torch.autograd.grad(loss,v)[0],alpha=weight)
                    self.counters['backward_solves']+=1
        return Evaluation(total,grad,ids.clone(),torch.tensor(energies,dtype=torch.float64))
