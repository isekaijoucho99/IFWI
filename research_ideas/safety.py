"""Parameter-space guidance protection and actual-displacement line search.

A descent *gradient* does not imply an Adam descent *step*. Acceptance is
therefore based on the proposed parameter displacement and a real data loss.
"""
from __future__ import annotations
import copy
import math
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def _check(grads):
    if not grads or any(not torch.isfinite(g).all() for g in grads):
        raise ValueError('Gradient list must be nonempty and finite')


def dot(a, b):
    return sum((x.double()*y.double()).sum() for x,y in zip(a,b))


def combine_gradients(physical, auxiliary, ratio=.1):
    """Remove the auxiliary component opposing g_data; cap ||g_aux||/||g_data||.

    Both lists are PARAMETER gradients of matching shapes, not velocity fields.
    Zero physical gradient yields zero guidance. This is a direction property,
    not a claim of Adam monotonicity or recovery of the true velocity.
    """
    _check(physical); _check(auxiliary)
    if len(physical)!=len(auxiliary) or any(a.shape!=b.shape for a,b in zip(physical,auxiliary)):
        raise ValueError('Gradient structures differ')
    if not math.isfinite(ratio) or not 0<=ratio<=1:
        raise ValueError('ratio must be in [0,1]')
    p=[g.detach() for g in physical]; a=[g.detach() for g in auxiliary]
    pp=dot(p,p); pa=dot(p,a); conflict=bool(pa<0)
    if float(pp)==0 or ratio==0:
        return [g.clone() for g in p],dict(conflict=conflict,aux_ratio=0.)
    if conflict:
        a=[x-(pa/pp).to(x.dtype)*y for x,y in zip(a,p)]
    aa=dot(a,a)
    scale=min(1.,ratio*math.sqrt(float(pp)/float(aa))) if float(aa)>0 else 0.
    out=[x+scale*y for x,y in zip(p,a)]
    return out,dict(conflict=conflict,aux_ratio=scale*math.sqrt(float(aa/pp)))


def parameter_vjp(velocity, model, velocity_gradient, *, retain_graph=False):
    params=tuple(p for p in model.parameters() if p.requires_grad)
    grads=torch.autograd.grad(velocity,params,grad_outputs=velocity_gradient.detach(),
                              allow_unused=True,retain_graph=retain_graph)
    return [torch.zeros_like(p) if g is None else g.detach() for p,g in zip(params,grads)]


def assign_gradients(model, gradients):
    params=[p for p in model.parameters() if p.requires_grad]
    if len(params)!=len(gradients): raise ValueError('Parameter/gradient count differs')
    _check(gradients)
    for p,g in zip(params,gradients):
        if p.shape!=g.shape: raise ValueError('Gradient shape differs')
        p.grad=g.detach().clone()


def safeguarded_step(model, optimizer, physical, proposal, closure, *, before_loss=None,
                     max_backtracks=6, armijo=1e-4):
    """Trial one adaptive step, then backtrack its actual displacement.

    Rejection restores model parameters/buffers, optimizer, RNG and parameter
    gradients. An accepted scaled displacement retains the trial Adam moments;
    alpha is an explicit effective step multiplier, NOT ordinary fixed-lr Adam.
    closure must evaluate the SAME fixed data objective and must not mutate an
    external router or prior. Its forward-solve cost must be recorded by caller.
    """
    if type(max_backtracks) is not int or max_backtracks<0 or not 0<armijo<1:
        raise ValueError('Invalid line-search settings')
    _check(physical); _check(proposal)
    params=[p for p in model.parameters() if p.requires_grad]
    if len(params)!=len(physical) or len(params)!=len(proposal):
        raise ValueError('Gradient structure differs from model')
    state=copy.deepcopy(model.state_dict()); opt_state=copy.deepcopy(optimizer.state_dict())
    rng=rng_state(); old=[p.detach().clone() for p in params]
    old_grads=[None if p.grad is None else p.grad.detach().clone() for p in params]
    evaluations=0
    def reset():
        model.load_state_dict(state); optimizer.load_state_dict(copy.deepcopy(opt_state)); restore_rng(rng)
        for p,g in zip(params,old_grads): p.grad=None if g is None else g.clone()
    try:
        if before_loss is None:
            with torch.no_grad(): before_loss=float(closure())
            evaluations+=1
        if not math.isfinite(before_loss): raise ValueError('Initial loss is not finite')
        assign_gradients(model,proposal); optimizer.step()
        delta=[p.detach()-q for p,q in zip(params,old)]
        slope=float(dot(physical,delta))
        if not math.isfinite(slope) or slope>=0:
            reset(); return dict(accepted=False,alpha=0.,reason='non_descent_adam_step',evaluations=evaluations)
        for attempt in range(max_backtracks+1):
            alpha=2.**(-attempt)
            # Keep the trial moments; restore all buffers before each candidate.
            with torch.no_grad():
                model.load_state_dict(state)
                for p,q,d in zip(params,old,delta): p.copy_(q+alpha*d)
                restore_rng(rng)
                try: value=float(closure())
                except (FloatingPointError,ValueError): value=float('inf')
            evaluations+=1
            if math.isfinite(value) and value <= before_loss+armijo*alpha*slope:
                return dict(accepted=True,alpha=alpha,reason='armijo',after_loss=value,
                            slope=slope,evaluations=evaluations)
        reset(); return dict(accepted=False,alpha=0.,reason='backtracking_exhausted',evaluations=evaluations)
    except BaseException:
        reset(); raise
