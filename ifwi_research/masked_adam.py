"""Adam with per-element clocks and exact freezing, including coupled decay.

Only real dense floating parameters and ordinary Adam are supported. Missing
gradients skip updates. Explicit masks must cover every optimizer parameter;
omitting masks activates all elements. Locked moments are preserved, not reset.
"""
import math

import torch


class MaskedAdam(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(.9,.999), eps=1e-8, weight_decay=0.):
        super().__init__(params,dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay))
        for group in self.param_groups:
            for key in ['lr','eps','weight_decay']:
                value = group[key]
                if not isinstance(value,(float,int)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f'{key} must be finite and nonnegative')
            if len(group['betas']) != 2 or any(not 0 <= b < 1 for b in group['betas']):
                raise ValueError('betas must be in [0,1)')
            for key in ['amsgrad','maximize','capturable','differentiable','fused','foreach','decoupled_weight_decay']:
                if group.get(key,False):
                    raise ValueError(f'{key} is not supported')
            for p in group['params']:
                if not p.is_floating_point() or p.layout != torch.strided:
                    raise ValueError('only dense real floating parameters are supported')

    def load_state_dict(self, state_dict):
        # Optimizer normally casts non-step tensors to parameter dtype, which
        # would silently lose large integer ages. Restore from the original.
        super().load_state_dict(state_dict)
        for saved,group in zip(state_dict['param_groups'],self.param_groups):
            for identifier,p in zip(saved['params'],group['params']):
                if 'age' in state_dict['state'].get(identifier,{}):
                    self.state[p]['age'] = state_dict['state'][identifier]['age'].to(device=p.device,dtype=torch.int64).clone()

    @torch.no_grad()
    def step(self, *, masks=None):
        # Validate the entire update before mutating any parameter.
        params = [p for group in self.param_groups for p in group['params']]
        if masks is not None and set(masks) != set(params):
            raise ValueError('masks must cover exactly the optimizer parameters')
        for p in params:
            if masks is not None:
                mask = masks[p]
                if not isinstance(mask,torch.Tensor) or mask.dtype != torch.bool or mask.shape != p.shape or mask.device != p.device:
                    raise ValueError('each mask must be bool with parameter shape and device')
            if p.grad is not None and p.grad.layout != torch.strided:
                raise ValueError('sparse gradients are not supported')
        for group in self.param_groups:
            beta1,beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                mask = torch.ones_like(p,dtype=torch.bool) if masks is None else masks[p]
                state = self.state[p]
                if not state:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['age'] = torch.zeros_like(p,dtype=torch.int64)
                if not mask.any():
                    continue
                values = p[mask]
                grad = p.grad[mask]
                if group['weight_decay']:
                    grad = grad.add(values,alpha=group['weight_decay'])
                moment = state['exp_avg'][mask].lerp_(grad,1-beta1)
                variance = state['exp_avg_sq'][mask].mul_(beta2).addcmul_(grad,grad,value=1-beta2)
                age = state['age'][mask]+1
                # Float64 correction avoids precision loss at long local ages.
                correction1 = (1-torch.pow(beta1,age.to(torch.float64))).to(p.dtype)
                correction2 = (1-torch.pow(beta2,age.to(torch.float64))).to(p.dtype)
                denom = variance.sqrt()/correction2.sqrt()+group['eps']
                p[mask] = values - (group['lr']/correction1)*moment/denom
                state['exp_avg'][mask] = moment
                state['exp_avg_sq'][mask] = variance
                state['age'][mask] = age
