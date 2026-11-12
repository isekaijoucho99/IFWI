"""Single author-IRN SIREN with hard first-activation channel progression.

This project adaptation is not a strict spatial band limit. The first fixed
frequencies are 4..30; later sine layers retain the author's omega=30.
"""
from numbers import Integral

import torch

from experiments.residual_ifwi_experiment import VelocityParameterization


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= torch.iinfo(torch.int64).max:
        raise ValueError(f'{name} must be an integer >= {minimum} fitting int64')
    return int(value)


class GatedSirenVelocity(VelocityParameterization):
    def __init__(self, initial_mps, dx_m, neurons=(2,128,128,128,128,1),
                 mean_kmps=3., std_kmps=1., warmup_updates=200, initial_channels=32):
        if list(neurons) != [2,128,128,128,128,1]:
            raise ValueError('Gated SIREN requires neurons=[2,128,128,128,128,1]')
        warmup_updates = _integer(warmup_updates,'warmup_updates',1)
        initial_channels = _integer(initial_channels,'initial_channels',1)
        if initial_channels > 128:
            raise ValueError('initial_channels must not exceed 128')
        super().__init__(initial_mps, dx_m, neurons, omega_0=30.,
                         mean_kmps=mean_kmps, std_kmps=std_kmps,
                         parameterization='residual', zero_residual_head=True)
        self.register_buffer('first_omega',torch.linspace(4.,30.,128))
        self.register_buffer('completed_updates',torch.tensor(0,dtype=torch.int64))
        self.register_buffer('_warmup_updates',torch.tensor(warmup_updates,dtype=torch.int64))
        self.register_buffer('_initial_channels',torch.tensor(initial_channels,dtype=torch.int64))

    @property
    def warmup_updates(self):
        return int(self._warmup_updates.item())

    @property
    def initial_channels(self):
        return int(self._initial_channels.item())

    @property
    def active_channels(self):
        k = min(int(self.completed_updates.item()), self.warmup_updates)
        return self.initial_channels + (128-self.initial_channels)*k//self.warmup_updates

    @property
    def gate(self):
        return (torch.arange(128,device=self.first_omega.device) < self.active_channels).to(self.first_omega.dtype)

    def set_progress(self, completed_updates):
        self.completed_updates.fill_(_integer(completed_updates,'completed_updates'))

    def raw_output(self):
        feature = torch.sin(self.net.linear[0](self.coords)*self.first_omega)*self.gate
        for layer in self.net.linear[1:-1]:
            feature = torch.sin(30.*layer(feature))
        return self.net.linear[-1](feature).squeeze(-1)

    def optimizer_masks(self):
        masks = {p:torch.ones_like(p,dtype=torch.bool) for p in self.parameters()}
        gate = self.gate.bool()
        first,second = self.net.linear[:2]
        masks[first.weight] = gate[:,None].expand_as(first.weight)
        masks[first.bias] = gate
        masks[second.weight] = gate[None,:].expand_as(second.weight)
        return masks
