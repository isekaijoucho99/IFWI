"""Existing discrete pressure-energy observer and shared illumination direction.

PressureEnergy is copied unchanged from the project's frozen 2026-11-04
encoded_illumination_probe.py (SHA256 fcc47496a7d67bc3ca4ee1bccbae9ce382d4c1419c25e597837361df27ec54f8).
The water-level formula matches DeepWave-KAUST/ifwi-pub _precondition,
with the project's existing RMS-norm matching, not a new pseudo-Hessian.
All methods use the same summed source/weighted-adjoint energy convention.
"""
import torch


class PressureEnergy:
    """Observe the existing solver without modifying its state updates or gradients."""
    def __init__(self,nz,nx,pad,free_surface,need_receiver=True):
        self.window=(slice(0 if free_surface else pad,(0 if free_surface else pad)+nz), slice(pad,pad+nx))
        self.source=self.receiver=None
        self.forward_samples=self.adjoint_samples=0
        self.need_receiver=need_receiver

    def _sum(self,v):
        return v.detach()[...,self.window[0],self.window[1]].double().square().sum(dim=(0,1))

    def observe(self,module,inputs,output):
        pressure=output[1]
        value=self._sum(pressure)
        if self.source is None:self.source=value
        else:self.source.add_(value)
        self.forward_samples+=1
        if self.need_receiver:
            def adjoint(g):
                val=self._sum(g)
                if self.receiver is None:self.receiver=val
                else:self.receiver.add_(val)
                self.adjoint_samples+=1
                # Returning None leaves the native adjoint untouched.
            pressure.register_hook(adjoint)


def illumination_direction(gradient,source,receiver,water_level=.1):
    """Existing relative water level and RMS matching, with no EMA or gate.

    Loss normalization is already present in the adjoint. Never silently divide
    energy by the number of code channels. A zero gradient is a valid no-op.
    """
    if (gradient.ndim!=3 or gradient.shape[0]!=1 or source.shape!=gradient.shape[1:]
            or receiver.shape!=source.shape or not 0<water_level<=1):
        raise ValueError('Shared [1,nz,nx] gradient and nonnegative energy maps required')
    if (not torch.isfinite(gradient).all() or not torch.isfinite(source).all()
            or not torch.isfinite(receiver).all() or source.min()<0 or receiver.min()<0):
        raise ValueError('Nonfinite or negative illumination/gradient')
    h=(source.double()*receiver.double()).sqrt()
    peak=h.max()
    # Matches the existing author helper: grad/(H/(max(H)+1e-30)+water).
    weighted=gradient.double()/(h/(peak+1e-30)+water_level)
    multiplier=gradient.double().norm()/weighted.norm().clamp_min(1e-30)
    result=(weighted*multiplier).to(gradient.dtype)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite illumination direction')
    return result,dict(water_level=water_level,source_max=float(source.max()),
        adjoint_max=float(receiver.max()),illumination_max=float(peak),
        raw_gradient_norm=float(gradient.double().norm()),RMS_multiplier=float(multiplier))
