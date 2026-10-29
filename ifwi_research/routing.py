"""MiTA-inspired common-source compression with unbiased contrast routing.

No Transformer is required. Source linearity and a common receiver layout
are prerequisites. The objective is preserved IN EXPECTATION, not per step.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn


def common_contrast_basis(shots: int, *, dtype=torch.float32, device=None) -> torch.Tensor:
    """Helmert orthonormal rows; row 0 is the normalized common source."""
    if type(shots) is not int or shots<2: raise ValueError('At least two shots are required')
    q=torch.zeros(shots,shots,dtype=dtype,device=device)
    q[0]=1/math.sqrt(shots)
    for j in range(1,shots):
        q[j,:j]=1/math.sqrt(j*(j+1)); q[j,j]=-j/math.sqrt(j*(j+1))
    return q


def weighted_code_loss(encoded_residual: torch.Tensor, sampled_indices: torch.Tensor,
                       probabilities: torch.Tensor) -> torch.Tensor:
    """Proof/testing helper. Real execution must simulate ONLY chosen codes.

    probabilities is for contrast rows 1..S-1; draws are WITH replacement.
    """
    s=encoded_residual.shape[0]
    p=probabilities.detach().to(encoded_residual)
    ids=sampled_indices.to(device=encoded_residual.device,dtype=torch.long)
    if p.shape!=(s-1,) or not torch.isfinite(p).all() or (p<=0).any():
        raise ValueError('Strictly positive contrast probabilities required')
    if not torch.isclose(p.sum(),p.new_tensor(1.)) or ids.numel()==0 or (ids<1).any() or (ids>=s).any():
        raise ValueError('Invalid probability mass or contrast draws')
    losses=encoded_residual.reshape(s,-1).square().mean(-1)
    return (losses[0]+(losses[ids]/p[ids-1]).mean())/s


@dataclass(frozen=True)
class CodePlan:
    indices: torch.Tensor
    weights: torch.Tensor
    draw_count: int
    probabilities: torch.Tensor


class ContrastRouter(nn.Module):
    """Lagged residual-energy EMA, top-k exploitation and positive exploration.

    Probability selection is detached and happens BEFORE current solves.
    Duplicate samples are merged using their exact multiplicities.
    """
    def __init__(self, shots: int, samples: int=2, topk: int=4,
                 exploration: float=.2, seed: int=3, decay: float=.9):
        super().__init__()
        if type(shots) is not int or shots<2 or type(samples) is not int or samples<1:
            raise ValueError('shots>=2 and samples>=1 must be integers')
        if type(topk) is not int or topk<1 or not 0<exploration<=1 or not 0<=decay<1:
            raise ValueError('Invalid topk, exploration or EMA decay')
        self.shots,self.samples,self.topk=shots,samples,min(topk,shots-1)
        self.exploration,self.decay=float(exploration),float(decay)
        self.register_buffer('scores',torch.ones(shots-1,dtype=torch.float64))
        self.register_buffer('rng_state',torch.Generator().manual_seed(seed).get_state())

    @torch.no_grad()
    def probabilities(self) -> torch.Tensor:
        scores=self.scores.detach().cpu().clamp_min(0)
        best=torch.topk(scores,self.topk).indices
        exploit=torch.zeros_like(scores)
        selected=scores[best].sqrt()
        if selected.sum()==0: selected=torch.ones_like(selected)
        exploit[best]=selected/selected.sum()
        return (1-self.exploration)*exploit+self.exploration/len(scores)

    @torch.no_grad()
    def plan(self, *, full: bool=False) -> CodePlan:
        p=self.probabilities()
        if full:
            return CodePlan(torch.arange(self.shots),torch.full((self.shots,),1/self.shots,dtype=p.dtype),0,p)
        rng=torch.Generator(); rng.set_state(self.rng_state.cpu())
        draws=torch.multinomial(p,self.samples,replacement=True,generator=rng)+1
        self.rng_state.copy_(rng.get_state().to(self.rng_state.device))
        ids,counts=torch.unique(draws,sorted=True,return_counts=True)
        weights=counts.to(p.dtype)/(self.shots*self.samples*p[ids-1])
        return CodePlan(torch.cat((torch.zeros(1,dtype=torch.long),ids)),
                        torch.cat((p.new_tensor([1/self.shots]),weights)),self.samples,p)

    @torch.no_grad()
    def update_scores(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        ids=indices.detach().to(self.scores.device,dtype=torch.long)
        values=losses.detach().to(self.scores)
        if ids.shape!=values.shape or not torch.isfinite(values).all() or (values<0).any():
            raise ValueError('Nonnegative finite, one-per-code residual energies required')
        if (ids<0).any() or (ids>=self.shots).any() or len(torch.unique(ids))!=len(ids):
            raise ValueError('Score update requires unique valid code indices')
        keep=ids>0; ids,values=ids[keep]-1,values[keep]
        self.scores[ids]=self.decay*self.scores[ids]+(1-self.decay)*values
