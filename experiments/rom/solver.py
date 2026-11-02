"""Differentiable, fixed-basis, discrete-frequency pressure ROM.

All spatial/PML coefficients are taken from the original FDTD. No original
time step or waveform loss is changed. Finite-contour, bandwidth and projection
errors are independent approximations and must be measured.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from torch.nn import functional as F


class PressureROM(torch.nn.Module):
    def __init__(self,system,basis,source_indices,receiver_indices,nt,*,nfft=None,
                 pad_factor=4,alias_tolerance=1e-12,max_hz=None,dtype=torch.float64,
                 frequency_batch_size=128,preprojected=None):
        super().__init__()
        basis=np.asarray(basis)
        if basis.ndim!=2 or basis.shape[0]!=system.dimension or np.iscomplexobj(basis) or not np.isfinite(basis).all():
            raise ValueError('Require a finite real full-grid pressure basis')
        sources=np.asarray(source_indices,dtype=np.int64);receivers=np.asarray(receiver_indices,dtype=np.int64)
        if receivers.ndim==1:receivers=np.broadcast_to(receivers,(len(sources),len(receivers))).copy()
        if sources.ndim!=1 or receivers.ndim!=2 or receivers.shape[0]!=len(sources):
            raise ValueError('Receiver rows must match physical sources')
        if np.any(sources<0) or np.any(sources>=system.dimension) or np.any(receivers<0) or np.any(receivers>=system.dimension):
            raise ValueError('Source/receiver index outside padded domain')
        if nt<2 or not 0<alias_tolerance<1 or frequency_batch_size<1:
            raise ValueError('Invalid time grid, contour tolerance or frequency batch')
        self.nt=int(nt);self.nfft=int(nfft or 2**math.ceil(math.log2(pad_factor*nt)))
        if self.nfft<2*nt:raise ValueError('Contour requires at least twice the record length')
        if max_hz is not None and not 0<max_hz<=.5/system.dt:raise ValueError('Invalid frequency cutoff')
        self.shape=system.shape;self.physical_shape=system.physical_shape
        self.npad=system.npad;self.free_surface=system.free_surface;self.dt=system.dt
        self.max_hz=max_hz;self.alias_tolerance=alias_tolerance
        self.frequency_batch_size=int(frequency_batch_size)
        self.reference_vmax=system.reference_vmax
        tensor=lambda value:torch.as_tensor(np.asarray(value).copy(),dtype=dtype)
        self.register_buffer('basis',tensor(basis))
        self.register_buffer('delta',tensor(system.delta))
        self.register_buffer('source_indices',torch.tensor(sources,dtype=torch.long))
        self.register_buffer('receiver_indices',torch.tensor(receivers,dtype=torch.long))
        self.register_buffer('receiver_basis',tensor(basis[receivers]))
        if preprojected is None:
            stiffness=tensor(basis.T@(system.stiffness@basis))
            projected=tensor(np.stack([basis.T@(term@basis) for term in system.pml_terms]))
        else:
            stiffness,projected=[torch.as_tensor(value,dtype=dtype).detach().clone() for value in preprojected]
            rank=basis.shape[1]
            if stiffness.shape!=(rank,rank) or projected.shape!=(len(system.pml_terms),rank,rank):
                raise ValueError('Preprojected operators must match basis and original PML profiles')
            if not torch.isfinite(stiffness).all() or not torch.isfinite(projected).all():
                raise ValueError('Preprojected operators must be finite')
        self.register_buffer('stiffness_r',stiffness)
        self.register_buffer('pml_r',projected)
        eta=-np.log(alias_tolerance)/self.nfft
        indices=np.arange(self.nfft//2+1)
        if max_hz is not None:indices=indices[indices/(self.nfft*self.dt)<=max_hz]
        log_z=eta+2j*np.pi*indices/self.nfft
        zm1=np.expm1(log_z);z=1+zm1
        complex_dtype=torch.complex128 if dtype==torch.float64 else torch.complex64
        self.register_buffer('active',torch.tensor(indices,dtype=torch.long))
        self.register_buffer('z',torch.tensor(z,dtype=complex_dtype))
        self.register_buffer('a0',torch.tensor(zm1*zm1/z,dtype=complex_dtype))
        self.register_buffer('a1',torch.tensor(self.dt*zm1,dtype=complex_dtype))
        factors=z[:,None]/(zm1[:,None]+(1-system.alphas)[None])
        self.register_buffer('pml_factors',torch.tensor(factors,dtype=complex_dtype))
        self.register_buffer('damping',tensor(np.exp(-eta*np.arange(self.nt))))

    def projected_model(self,velocity):
        if velocity.ndim!=3 or velocity.shape!=(1,*self.physical_shape):
            raise ValueError('ROM accepts one shared physical [1,nz,nx] velocity')
        top=0 if self.free_surface else self.npad
        padded=F.pad(velocity[:,None].to(self.basis.dtype),(self.npad,self.npad,top,self.npad),mode='replicate').reshape(-1)
        mass=padded.reciprocal().square()
        q=self.basis
        mr=(q.T*mass[None])@q
        md=(q.T*(mass*self.delta)[None])@q
        # This velocity dependence MUST NOT be frozen, although the original
        # unscaled point injection is independent of velocity.
        amplitudes=mass[self.source_indices]*(1+self.dt*self.delta[self.source_indices])
        br=q[self.source_indices].T*amplitudes[None]
        return mr,md,br

    def coefficients(self,velocity,wavelet):
        if wavelet.ndim!=1 or len(wavelet)!=self.nt:
            raise ValueError('ROM requires the complete configured wavelet')
        mr,md,br=self.projected_model(velocity)
        transform=torch.fft.rfft(wavelet.to(self.basis.dtype)*self.damping,n=self.nfft)[self.active]
        matrices=[]
        pml=self.pml_r.to(self.z.dtype)
        for first in range(0,len(self.active),self.frequency_batch_size):
            last=min(first+self.frequency_batch_size,len(self.active));part=slice(first,last)
            lhs=self.a0[part,None,None]*mr[None]+self.a1[part,None,None]*md[None]+self.stiffness_r[None]
            lhs=lhs+torch.einsum('fk,kij->fij',self.pml_factors[part],pml)
            rhs=transform[part,None,None]*br[None]
            solution=torch.linalg.solve(lhs,rhs)
            matrices.append(self.z[part,None,None]*solution)
        solved=torch.cat(matrices,dim=0)
        spectrum=torch.zeros((self.nfft//2+1,self.basis.shape[1],len(self.source_indices)),dtype=self.z.dtype,device=self.basis.device)
        spectrum=spectrum.index_copy(0,self.active,solved)
        result=torch.fft.irfft(spectrum,n=self.nfft,dim=0)[:self.nt]/self.damping[:,None,None]
        return result

    def forward(self,vmodel,segment_wavelet,prev_state=None,curr_state=None,option=0):
        if prev_state is not None or curr_state is not None or option!=0:
            raise ValueError('Fixed-record ROM does not support segmented continuation')
        coefficients=self.coefficients(vmodel,segment_wavelet)
        records=torch.einsum('sor,trs->sto',self.receiver_basis,coefficients)[None]
        current=(self.basis@coefficients[-1]).T.reshape(1,len(self.source_indices),*self.shape)
        previous=(self.basis@coefficients[-2]).T.reshape_as(current)
        return previous,current,records,vmodel.new_zeros((1,1))

    def wavefields(self,velocity,wavelet,indices):
        coefficients=self.coefficients(velocity,wavelet)[indices]
        return torch.einsum('nr,trs->tsn',self.basis,coefficients).reshape(len(indices),len(self.source_indices),*self.shape)

    def specification(self):
        return dict(rank=self.basis.shape[1],full_pressure_dimension=self.basis.shape[0],
            padded_shape=self.shape,physical_shape=self.physical_shape,nt=self.nt,nfft=self.nfft,
            active_frequencies=len(self.active),max_hz=self.max_hz,alias_tolerance=self.alias_tolerance,
            reference_vmax=self.reference_vmax,dtype=str(self.basis.dtype),
            frequency_batch_size=self.frequency_batch_size,
            stability='PML reduced stability not certified by the elastic FEM theorem')
