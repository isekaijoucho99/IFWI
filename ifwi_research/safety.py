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


def protect_physical_gradient(physical: Sequence[torch.Tensor], proposal: Sequence[torch.Tensor],
                              ratio: float=.025) -> tuple[list[torch.Tensor],dict]:
    """Project ONLY the proposal; physical gradient is never projected away."""
    if len(physical)!=len(proposal) or not physical or not math.isfinite(ratio) or not 0<=ratio<1:
        raise ValueError('Matching nonempty gradient lists and ratio in [0,1) required')
    for a,b in zip(physical,proposal):
        if a.shape!=b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Gradient shapes/finiteness disagree')
    gp=[x.detach() for x in physical]; gs=[x.detach() for x in proposal]
    norm2=sum(x.double().square().sum() for x in gp)
    dot=sum((a.double()*b.double()).sum() for a,b in zip(gp,gs))
    if norm2==0 or ratio==0:
        return [x.clone() for x in gp],{'projected':False,'ratio':0.,'physical_norm':float(norm2.sqrt())}
    coefficient=min(0.,float(dot/norm2))
    orth=[b-coefficient*a for a,b in zip(gp,gs)]
    pn=sum(x.double().square().sum() for x in orth).sqrt()
    scale=float(ratio*norm2.sqrt()/pn) if pn>0 else 0.
    mixed=[a+scale*b for a,b in zip(gp,orth)]
    return mixed,{'projected':coefficient<0,'ratio':float(ratio) if pn>0 else 0.,
                  'physical_norm':float(norm2.sqrt()),'proposal_norm':float(pn)}


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
                           valid_candidate: Callable[[],bool] | None=None) -> dict:
    """Retry an optimizer step from the SAME parameters, moments, buffers, RNG.

    A rejected trial does not advance Adam's clock. A partially scaled accepted
    trial is rerun using the scaled learning rate, not a post-hoc parameter mix.
    The original configured learning rates are restored after an accepted trial.
    The caller must keep data selection, frequency and model context fixed.
    """
    if not math.isfinite(reference_loss) or max_trials<1 or not 0<shrink<1:
        raise ValueError('Invalid line-search settings/reference loss')
    parameters=list(model.parameters())
    snapshot=copy.deepcopy(model.state_dict()); old_opt=copy.deepcopy(optimizer.state_dict()); old_rng=rng_state()
    gradients=[None if p.grad is None else p.grad.detach().clone() for p in parameters]
    lrs=[g['lr'] for g in optimizer.param_groups]
    def restore():
        model.load_state_dict(snapshot); optimizer.load_state_dict(copy.deepcopy(old_opt)); restore_rng(old_rng)
        for p,g in zip(parameters,gradients): p.grad=None if g is None else g.clone()
    forward_calls=0
    try:
        for trial in range(max_trials):
            restore(); factor=shrink**trial
            for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr*factor
            optimizer.step()
            finite=all(torch.isfinite(p).all() for p in parameters)
            with torch.no_grad():
                valid=bool(finite) and (valid_candidate is None or bool(valid_candidate()))
                value=float('inf')
                if valid:
                    forward_calls+=1
                    try: value=float(physical_value())
                    except FloatingPointError: value=float('inf')
            if math.isfinite(value) and value<reference_loss:
                for group,lr in zip(optimizer.param_groups,lrs): group['lr']=lr
                return {'accepted':True,'trials':trial+1,'forward_calls':forward_calls,
                        'step_scale':factor,'physical_before':reference_loss,'physical_after':value}
        restore()
        return {'accepted':False,'trials':max_trials,'forward_calls':forward_calls,
                'step_scale':0.,'physical_before':reference_loss,'physical_after':reference_loss}
    except BaseException:
        restore()
        raise
