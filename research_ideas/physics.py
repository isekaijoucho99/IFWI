"""A read-only adapter to the repository's ORIGINAL RNN finite-difference code."""
from __future__ import annotations
import time
import torch
from experiments import residual_ifwi_experiment as e


class OriginalPhysics:
    def __init__(self,problem):
        if any(item['last']-item['first']!=1 for item in problem['propagators']):
            raise ValueError('Build with shot_batch_size=1 for exact routed weights and bounded memory')
        self.problem=problem; self.shots=problem['observed'].shape[1]
        self.counts=dict(forward_shots=0,adjoint_shots=0,observation_forward_shots=0)
        self.seconds=0.

    def _weights(self,weights):
        if weights is None: return torch.full((self.shots,),1/self.shots,dtype=torch.float64)
        w=torch.as_tensor(weights,dtype=torch.float64).detach().cpu()
        if w.shape!=(self.shots,) or not torch.isfinite(w).all() or (w<0).any() or not (w>0).any():
            raise ValueError('Invalid shot estimator weights')
        return w

    def _sync(self):
        if self.problem['observed'].is_cuda: torch.cuda.synchronize(self.problem['observed'].device)

    def _loss(self,velocity,item,weight,cutoff):
        self.counts['forward_shots']+=1
        predicted=e._simulate(item['solver'],velocity,self.problem['wavelet'])
        observed=self.problem['observed'][:,item['first']:item['last']]
        loss=e.training_objective(predicted,observed,self.shots,cutoff,self.problem)*(weight*self.shots)
        e.require_finite(loss,'weighted physical objective')
        return loss,predicted

    def value_and_gradient(self,velocity,weights=None,cutoff=None):
        self._sync(); started=time.perf_counter()
        e.check_velocity(velocity,self.problem['config']); weights=self._weights(weights)
        v=velocity.detach().requires_grad_(True); total=torch.zeros_like(v)
        groups=[torch.zeros_like(v),torch.zeros_like(v)]; value=0.
        try:
            for item in self.problem['propagators']:
                index=item['first']; w=float(weights[index])
                if w==0: continue
                loss,predicted=self._loss(v,item,w,cutoff)
                self.counts['adjoint_shots']+=1
                g=torch.autograd.grad(loss,v)[0].detach()
                e.require_finite(g,'velocity gradient')
                total+=g; groups[index%2]+=g; value+=float(loss.detach())
                del loss,predicted,g
            return dict(loss=value,gradient=total,groups=groups)
        finally:
            self._sync(); self.seconds+=time.perf_counter()-started

    @torch.no_grad()
    def value(self,velocity,weights=None,cutoff=None,return_records=False):
        self._sync(); started=time.perf_counter()
        e.check_velocity(velocity,self.problem['config']); weights=self._weights(weights)
        if return_records and (weights==0).any(): raise ValueError('Full records require all shots')
        value=0.; records=[]
        try:
            for item in self.problem['propagators']:
                w=float(weights[item['first']])
                if w==0: continue
                loss,predicted=self._loss(velocity,item,w,cutoff); value+=float(loss)
                if return_records: records.append(predicted.detach())
            return (value,torch.cat(records,dim=1)) if return_records else value
        finally:
            self._sync(); self.seconds+=time.perf_counter()-started
