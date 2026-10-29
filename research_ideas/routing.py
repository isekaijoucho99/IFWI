"""MiTA-inspired compress-and-route shot budgeting, NOT the MiTA architecture.

Top-k shots are exact; the complement uses an importance-corrected estimator.
Scores are detached. Full-shot scoring costs belong in the physics counter.
"""
from __future__ import annotations
import copy
import math
import torch
from torch.nn import functional as F


def compress_traces(traces,bins=(8,4)):
    if traces.ndim!=4 or traces.shape[0]!=1 or not torch.isfinite(traces).all():
        raise ValueError('Expected finite [1,S,T,R] records')
    return F.adaptive_avg_pool2d(traces.detach().transpose(0,1).square(),bins).sqrt().flatten(1).cpu()


class CompressedShotRouter:
    def __init__(self,descriptors,top_k=3,samples=2,exploration=.2,diversity=.2,seed=3):
        features=torch.as_tensor(descriptors,dtype=torch.float32).detach().cpu()
        if features.ndim!=2 or len(features)<1 or not torch.isfinite(features).all():
            raise ValueError('Finite per-shot descriptors are required')
        n=len(features)
        if type(top_k) is not int or not 0<=top_k<=n or type(samples) is not int or samples<0:
            raise ValueError('Invalid shot budgets')
        if top_k<n and samples<1: raise ValueError('Partial deterministic top-k is biased; sample its complement')
        if not 0<exploration<=1 or not math.isfinite(diversity) or diversity<0:
            raise ValueError('Require positive exploration and nonnegative diversity')
        self.features=F.normalize(features,dim=1,eps=1e-8); self.n=n
        self.k=top_k; self.m=samples; self.exploration=exploration; self.diversity=diversity
        self.scores=torch.ones(n); self.rng=torch.Generator().manual_seed(seed)

    def refresh(self,observed,predicted):
        if observed.shape!=predicted.shape or observed.shape[1]!=self.n:
            raise ValueError('Scoring geometry changed')
        residual=compress_traces(predicted-observed)
        energy=compress_traces(observed).square().mean(1).clamp_min(1e-12)
        self.scores=(residual.square().mean(1)/energy).sqrt().clamp_min(1e-8)

    def distribution(self):
        if not torch.isfinite(self.scores).all() or (self.scores<0).any():
            raise ValueError('Scores must be finite and nonnegative')
        remaining=list(range(self.n)); top=[]
        for _ in range(self.k):
            priority=self.scores.clone()
            if top:
                similarity=self.features@self.features[top].T
                novelty=(1-similarity.max(1).values).clamp(0,2)
                priority=priority*(1+self.diversity*novelty)
            index=max(remaining,key=lambda j:(float(priority[j]),-j))
            top.append(index); remaining.remove(index)
        rest=torch.tensor(remaining,dtype=torch.long)
        score=self.scores[rest]
        if len(rest):
            normalized=score/score.sum() if float(score.sum())>0 else torch.full_like(score,1/len(score))
            p=(1-self.exploration)*normalized+self.exploration/len(rest)
        else: p=torch.empty(0)
        return torch.tensor(top,dtype=torch.long),rest,p

    def weights_for(self,top,rest,probabilities,draws):
        weights=torch.zeros(self.n); weights[top]=1/self.n
        if len(rest):
            if not draws: raise ValueError('Complement must be sampled')
            inverse={int(j):float(probabilities[i]) for i,j in enumerate(rest)}
            for j in draws:
                if j not in inverse or inverse[j]<=0: raise ValueError('Invalid complement draw')
                weights[j]+=1/(self.n*len(draws)*inverse[j])
        return weights

    def sample(self):
        top,rest,p=self.distribution()
        draws=rest[torch.multinomial(p,self.m,replacement=True,generator=self.rng)].tolist() if len(rest) else []
        return self.weights_for(top,rest,p,draws)

    def state_dict(self):
        return dict(scores=self.scores.clone(),rng=self.rng.get_state().clone())

    def load_state_dict(self,state):
        if state['scores'].shape!=(self.n,): raise ValueError('Router shot count changed')
        self.scores=state['scores'].detach().cpu().clone(); self.rng.set_state(state['rng'].cpu())
