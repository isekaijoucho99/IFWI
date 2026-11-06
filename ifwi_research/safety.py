"""One-sided gradient protection and transactional optimizer trial steps.

The gradient inequality is first-order. Only evaluating the actual candidate
with the same deterministic physical objective can certify a finite step.
"""
from __future__ import annotations
import copy
import math
import random
from collections.abc import Callable, Sequence
import numpy as np
import torch
from torch import nn


class InvalidProposalError(ValueError):
    """Generated values failed a numerical/physical check, not an API contract.

    Only this explicitly typed failure may opt into physical-only recovery.
    Configuration, context, shape and source identity errors remain fail-closed.
    """
    def __init__(self, message: str, *, reason: str):
        super().__init__(message)
        self.reason=reason


def protect_physical_gradient(physical: Sequence[torch.Tensor], proposal: Sequence[torch.Tensor],
                              ratio: float=.025) -> tuple[list[torch.Tensor],dict]:
    """Project ONLY the proposal, then cap its norm without amplifying it.

    Norms describe gradient contributions, not the eventual Adam parameter step.
    The reported ratio is the actual composed contribution / physical norm.
    """
    if len(physical)!=len(proposal) or not physical or not math.isfinite(ratio) or not 0<=ratio<1:
        raise ValueError('Matching nonempty gradient lists and ratio in [0,1) required')
    for a,b in zip(physical,proposal):
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Gradient shapes/finiteness disagree')
    gp=[x.detach() for x in physical]; gs=[x.detach() for x in proposal]
    norm2=sum(x.double().square().sum() for x in gp)
    raw_norm=sum(x.double().square().sum() for x in gs).sqrt()
    dot=sum((a.double()*b.double()).sum() for a,b in zip(gp,gs))
    coefficient=min(0.,float(dot/norm2)) if norm2>0 and ratio>0 else 0.
    orth=[b-coefficient*a for a,b in zip(gp,gs)]
    pn=sum(x.double().square().sum() for x in orth).sqrt()
    scale=min(1.,float(ratio*norm2.sqrt()/pn)) if pn>0 else 0.
    mixed=[a+scale*b for a,b in zip(gp,orth)]
    applied=sum((m.double()-a.double()).square().sum() for m,a in zip(mixed,gp)).sqrt()
    return mixed,{'projected':coefficient<0,'ratio':float(applied/norm2.sqrt()) if norm2>0 else 0.,
                  'requested_ratio':float(ratio),'scale':scale,'physical_norm':float(norm2.sqrt()),
                  'raw_proposal_norm':float(raw_norm),'projected_proposal_norm':float(pn),
                  'applied_proposal_norm':float(applied),'proposal_norm':float(pn)}


def rng_state() -> dict:
    state=np.random.get_state()
    numpy_state=(state[0],state[1].tolist(),int(state[2]),int(state[3]),float(state[4]))
    return {'torch':torch.get_rng_state().clone(),'numpy':numpy_state,
            'python':random.getstate(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state['torch'].cpu())
    ns=state['numpy']; np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),ns[2],ns[3],ns[4]))
    random.setstate(state['python'])
    if state['cuda'] is not None: torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def guarded_optimizer_step(model: nn.Module, optimizer: torch.optim.Optimizer,
                           physical_value: Callable[[],torch.Tensor | float], reference_loss: float,
                           *, max_trials: int=5, shrink: float=.5,
                           valid_candidate: Callable[[],bool] | None=None,
                           fallback_gradients: Sequence[torch.Tensor | None] | None=None) -> dict:
    """Backtrack optimizer trials, then a gradient-descent recovery if needed.

    Every trial starts from the SAME parameters, buffers, optimizer and RNG.
    Up to max_trials optimizer trials are followed by up to max_trials raw
    gradient trials, all subject to the same strict physical decrease check.
    fallback_gradients, when supplied, must follow model.parameters() and can
    provide the physical gradient instead of a composed proposal gradient.
    Only accepted recovery resets optimizer history for updated parameters.
    Total rejection or any exception restores the entire original transaction.
    Original learning rates are restored on success. The caller must keep data
    selection, frequency and model context fixed across all physical evaluations.
    """
    if (not math.isfinite(reference_loss) or type(max_trials) is not int
            or max_trials<1 or not 0<shrink<1):
        raise ValueError('Invalid line-search settings/reference loss')
    parameters=list(model.parameters())
    gradients=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
    fallback=list(gradients if fallback_gradients is None else fallback_gradients)
    if len(fallback)!=len(parameters):
        raise ValueError('Fallback gradients must follow model.parameters()')
    for values in (gradients,fallback):
        for p,g in zip(parameters,values):
            if g is not None and (g.shape!=p.shape or g.device!=p.device or not torch.isfinite(g).all()):
                raise ValueError('Matching finite gradients on the parameter device required')
    fallback={id(p):None if g is None else g.detach().clone() for p,g in zip(parameters,fallback)}
    if any(id(p) not in fallback for group in optimizer.param_groups for p in group['params']):
        raise ValueError('Optimizer parameters must belong to the guarded model')
    lrs=[g['lr'] for g in optimizer.param_groups]
    if any(not math.isfinite(float(lr)) or float(lr)<0 for lr in lrs):
        raise ValueError('Finite nonnegative learning rates required')
    snapshot=copy.deepcopy(model.state_dict()); old_opt=copy.deepcopy(optimizer.state_dict()); old_rng=rng_state()
    def restore():
        model.load_state_dict(snapshot); optimizer.load_state_dict(copy.deepcopy(old_opt)); restore_rng(old_rng)
        for p,g in zip(parameters,gradients): p.grad=None if g is None else g.clone()
    def finite_model():
        return all(not isinstance(v,torch.Tensor) or torch.isfinite(v).all()
                   for v in model.state_dict().values())
    forward_calls=0; optimizer_trials=0; recovery_trials=0
    def report(accepted,value,factor,recovery):
        return {'accepted':accepted,'trials':optimizer_trials+recovery_trials,
                'optimizer_trials':optimizer_trials,'recovery_trials':recovery_trials,
                'forward_calls':forward_calls,'step_scale':factor,'recovery':recovery,
                'physical_before':reference_loss,'physical_after':value}
    try:
        for recover in (False,True):
            for trial in range(max_trials):
                restore(); factor=shrink**trial
                if recover:
                    recovery_trials+=1
                    with torch.no_grad():
                        for group,lr in zip(optimizer.param_groups,lrs):
                            for p in group['params']:
                                g=fallback[id(p)]
                                if g is not None: p.add_(g,alpha=-float(lr)*factor)
                else:
                    optimizer_trials+=1
                    for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr*factor
                    optimizer.step()
                with torch.no_grad():
                    valid=finite_model() and (valid_candidate is None or bool(valid_candidate()))
                    value=float('inf')
                    if valid:
                        forward_calls+=1
                        try: value=float(physical_value())
                        except FloatingPointError: value=float('inf')
                if math.isfinite(value) and value<reference_loss and finite_model():
                    if recover:
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if fallback[id(p)] is not None: optimizer.state.pop(p,None)
                    for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr
                    return report(True,value,factor,'gradient_descent' if recover else None)
        restore()
        return report(False,reference_loss,0.,None)
    except BaseException:
        restore()
        raise
