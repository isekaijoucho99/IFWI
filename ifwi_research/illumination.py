"""Bounded CPU receiver-Jacobian sensitivity experiment, not uncertainty.

The API deliberately accepts no observed records, truth, or repository problem.
Callers must supply independent acquisition metadata, wavelet, context velocity,
and a declared physical PML velocity bound. Provenance of those inputs remains
caller's responsibility. This does not infer masks from measured data.
The estimator's operator_identity identifies the base native operator. Each
result's operator_identity additionally binds any nontrivial data-domain mask;
an all-ones mask is canonicalized to the unmasked identity. Cutoff must be bound
separately by the consuming stage/cache contract.
"""
from __future__ import annotations
import math
import hashlib
import json
import torch


def native_operator_identity(*, geometry, wavelet, dt_s, spacing_m, npad,
                             free_surface, pml_vmax_mps, pml_reflection=1e-6):
    """Pure SHA256 of the native numerical operator, excluding shot grouping.

    No solver is constructed and no data/truth inputs are accepted. The schema
    fixes float32/order-2 RNN-FD, zero initial fields and full-record option=0.
    Geometry tensors are canonical little-endian int64; wavelet is float32.
    """
    if set(geometry)!={'nz','nx','xs','zs','xr','zr'}:
        raise ValueError('Explicit nz,nx,xs,zs,xr,zr geometry required')
    if any(type(geometry[k]) is not int for k in ('nz','nx')):
        raise ValueError('Integer grid dimensions required')
    if type(npad) is not int or type(free_surface) is not bool:
        raise ValueError('Integer npad and boolean free_surface required')
    settings=dict(schema='native_rnnfd_operator_v1',nz=geometry['nz'],nx=geometry['nx'],
                  dt_s=float(dt_s),spacing_m=float(spacing_m),npad=npad,
                  free_surface=free_surface,pml_vmax_mps=float(pml_vmax_mps),
                  pml_reflection=float(pml_reflection),order=2,dtype='float32',option=0)
    digest=hashlib.sha256(json.dumps(settings,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
    def add_tensor(name,value,dtype,numpy_dtype):
        if not isinstance(value,torch.Tensor) or value.device.type!='cpu':
            raise ValueError('CPU tensors required for native operator identity')
        array=value.detach().to(dtype=dtype).contiguous().numpy().astype(numpy_dtype,copy=False)
        digest.update(json.dumps([name,list(array.shape)],separators=(',',':')).encode())
        digest.update(array.tobytes(order='C'))
    for name in ('xs','zs','xr','zr'):
        if not isinstance(geometry[name],torch.Tensor) or geometry[name].dtype not in (torch.int32,torch.int64):
            raise ValueError('Integer acquisition coordinates required')
        add_tensor(name,geometry[name],torch.int64,'<i8')
    if not isinstance(wavelet,torch.Tensor): raise ValueError('Tensor wavelet required')
    if wavelet.ndim==2 and wavelet.shape[0]==1: wavelet=wavelet[0]
    if wavelet.ndim!=1: raise ValueError('Wavelet must have shape [nt] or [1,nt]')
    add_tensor('wavelet',wavelet,torch.float32,'<f4')
    return digest.hexdigest()


def _squared_vjp(prediction, field, probes, generator):
    """Mean squared receiver-space Rademacher VJP; no residual or loss."""
    total=torch.zeros_like(field,dtype=torch.float64)
    for k in range(probes):
        z=torch.randint(0,2,prediction.shape,generator=generator,dtype=torch.int64).to(prediction)*2-1
        g=torch.autograd.grad(prediction,field,grad_outputs=z,
                              retain_graph=k+1<probes,create_graph=False)[0]
        if not torch.isfinite(g).all(): raise FloatingPointError('Nonfinite sensitivity VJP')
        total.add_(g.detach().double().square())
    return total/probes


class NativeSensitivity:
    """Native RNN-FD, fixed independent boundary coefficients and CPU-only caps.

    All receiver/time/shot entries have independent random signs. Finite-probe
    estimates may differ with shot batching, though their expectations agree.
    Raw normalization is full B*S*T*R even when an explicit mask is used.
    Resource caps are conservative prototype checks, not memory guarantees.
    """
    def __init__(self, *, geometry, wavelet, dt_s, spacing_m, npad,
                 free_surface, pml_vmax_mps, pml_reflection=1e-6, shot_batch_size=1):
        if set(geometry)!={'nz','nx','xs','zs','xr','zr'}:
            raise ValueError('Explicit nz,nx,xs,zs,xr,zr geometry required')
        nz,nx=geometry['nz'],geometry['nx']
        if any(type(n) is not int or n<3 for n in (nz,nx)) or nz*nx>4096:
            raise ValueError('CPU prototype requires grid cells <=4096 and dimensions >=3')
        if type(npad) is not int or not 1<=npad<=32 or type(free_surface) is not bool:
            raise ValueError('Require integer PML padding 1..32 and boolean free_surface')
        if type(shot_batch_size) is not int or not 1<=shot_batch_size<=16:
            raise ValueError('Shot batch size must be 1..16')
        if any(isinstance(x,bool) or not math.isfinite(x) or x<=0 for x in (dt_s,spacing_m,pml_vmax_mps)):
            raise ValueError('Finite positive sampling and physical PML bound required')
        if pml_vmax_mps*dt_s/spacing_m>=1/math.sqrt(2):
            raise ValueError('Declared physical PML bound violates CFL')
        if not math.isfinite(pml_reflection) or not 0<pml_reflection<1:
            raise ValueError('PML reflection must lie between zero and one')
        geom={}
        for name in ('xs','zs','xr','zr'):
            v=geometry[name]
            if not isinstance(v,torch.Tensor) or v.device.type!='cpu' or v.dtype not in (torch.int32,torch.int64):
                raise ValueError('CPU integer acquisition index tensors required')
            geom[name]=v.detach().clone().long()
        xs,zs,xr,zr=(geom[n] for n in ('xs','zs','xr','zr'))
        if xs.ndim!=2 or xs.shape[0]!=1 or zs.shape!=xs.shape or not 1<=xs.shape[1]<=16:
            raise ValueError('Source coordinates must have shape [1,S], S<=16')
        if xr.ndim!=3 or xr.shape[:2]!=xs.shape or zr.shape!=xr.shape or not 1<=xr.shape[2]<=4096:
            raise ValueError('Receiver coordinates must have shape [1,S,R]')
        for name,v in geom.items():
            bound=nx if name.startswith('x') else nz
            if (v<0).any() or (v>=bound).any(): raise ValueError('Acquisition index outside grid')
        if not isinstance(wavelet,torch.Tensor) or wavelet.device.type!='cpu':
            raise ValueError('CPU wavelet required')
        if wavelet.ndim==2 and wavelet.shape[0]==1: wavelet=wavelet[0]
        if wavelet.ndim!=1 or not 1<=len(wavelet)<=256 or not torch.isfinite(wavelet).all():
            raise ValueError('Finite wavelet with 1..256 samples required')
        self.wavelet=wavelet.detach().clone().float()
        self.nz,self.nx,self.shots,self.receivers=nz,nx,xs.shape[1],xr.shape[2]
        self.dt_s,self.pml_vmax_mps=dt_s,float(pml_vmax_mps)
        self.nt=len(wavelet)
        padded_cells=(nz+npad*(1 if free_surface else 2))*(nx+2*npad)
        self.recorded_samples=self.shots*self.nt*self.receivers
        self.max_batch_entries=min(shot_batch_size,self.shots)*self.nt*padded_cells
        if self.recorded_samples>1_000_000:
            raise ValueError('CPU recorded samples cap exceeded (1000000)')
        if self.max_batch_entries>2_000_000:
            raise ValueError('CPU batch graph entries cap exceeded (2000000)')
        self.work_base=self.shots*self.nt*padded_cells
        if 3*self.work_base>100_000_000: raise ValueError('CPU work cap exceeded')
        self.operator_identity=native_operator_identity(geometry=geometry,wavelet=self.wavelet,
            dt_s=dt_s,spacing_m=spacing_m,npad=npad,free_surface=free_surface,
            pml_vmax_mps=pml_vmax_mps,pml_reflection=pml_reflection)
        from rnn_fd import rnn2D
        self.batches=[]
        for first in range(0,self.shots,shot_batch_size):
            last=min(first+shot_batch_size,self.shots)
            solver=rnn2D(nz=nz,nx=nx,**{k:v[:,first:last] for k,v in geom.items()},
                         dz=spacing_m,dt=dt_s,npad=npad,order=2,vmax=pml_vmax_mps,
                         log_para=pml_reflection,freeSurface=free_surface,dtype=torch.float32,device='cpu')
            self.batches.append((first,last,solver))

    def estimate(self, context, *, probes=2, seed=3, cutoff=None, data_mask=None):
        """Return raw diag(J'J)/M, bounded illumination, status and solve costs.

        J differentiates mask * lowpass(native_prediction), at a detached context.
        A zero map is explicitly no_sensitivity; never interpret it as evidence
        authorizing a full-strength prior. No observed-energy rescaling is used.
        """
        if type(probes) is not int or not 2<=probes<=8 or type(seed) is not int or not 0<=seed<2**63:
            raise ValueError('Require probes 2..8 and integer seed in [0,2**63)')
        work=(probes+1)*self.work_base
        if work>100_000_000: raise ValueError('CPU work cap exceeded')
        if (not isinstance(context,torch.Tensor) or context.device.type!='cpu'
                or context.shape!=(1,self.nz,self.nx) or not torch.isfinite(context).all()
                or (context<=0).any() or context.max()>self.pml_vmax_mps):
            raise ValueError('Finite positive CPU context [1,nz,nx] within physical bound required')
        if cutoff is not None and (isinstance(cutoff,bool) or not math.isfinite(cutoff) or not 0<cutoff<.5/self.dt_s):
            raise ValueError('Cutoff must lie strictly between zero and Nyquist')
        shape=(1,self.shots,self.nt,self.receivers)
        if data_mask is not None:
            if (not isinstance(data_mask,torch.Tensor) or data_mask.device.type!='cpu' or data_mask.shape!=shape
                    or not torch.isfinite(data_mask).all() or (data_mask<0).any() or (data_mask>1).any()):
                raise ValueError('Explicit CPU mask [1,S,T,R] must be finite in [0,1]')
            data_mask=data_mask.detach().clone().float()
        result_identity=self.operator_identity
        if data_mask is not None and not torch.all(data_mask==1):
            digest=hashlib.sha256()
            digest.update(b'native_masked_operator_v1:')
            digest.update(self.operator_identity.encode('ascii'))
            digest.update(json.dumps(list(data_mask.shape),separators=(',',':')).encode())
            digest.update(data_mask.contiguous().numpy().astype('<f4',copy=False).tobytes(order='C'))
            result_identity=digest.hexdigest()
        from experiments.frequency_continuation import lowpass
        rng=torch.Generator().manual_seed(seed)
        diagonal=torch.zeros_like(context,dtype=torch.float64)
        for first,last,solver in self.batches:
            try:
                with torch.enable_grad():
                    field=context.detach().clone().float().requires_grad_(True)
                    prediction=solver(vmodel=field,segment_wavelet=self.wavelet,option=0)[2]
                    prediction=lowpass(prediction,cutoff,self.dt_s)
                    if data_mask is not None: prediction=prediction*data_mask[:,first:last]
                    if not torch.isfinite(prediction).all(): raise FloatingPointError('Nonfinite native prediction')
                    diagonal.add_(_squared_vjp(prediction,field,probes,rng))
            finally:
                # The original FD stores its last padded velocity on the module.
                if hasattr(solver.fd,'velocity'): solver.fd.velocity=solver.fd.velocity.detach()
            del prediction,field
        count=math.prod(shape)
        diagonal/=count
        if not torch.isfinite(diagonal).all(): raise FloatingPointError('Nonfinite sensitivity diagonal')
        peak=diagonal.max()
        status='ok' if peak>0 else 'no_sensitivity'
        if peak>0:
            scaled=diagonal/peak
            illumination=scaled/(scaled+scaled.mean())
        else: illumination=torch.zeros_like(diagonal)
        return dict(operator_identity=result_identity,raw_diagonal=diagonal.detach(),illumination=illumination.detach(),status=status,
                    diagnostics=dict(source_forward_solves=self.shots,source_backward_solves=probes*self.shots,
                        forward_time_steps=self.shots*self.nt,probes=probes,normalization_count=count,
                        work_units=work,pml_vmax_mps=self.pml_vmax_mps,cutoff_hz=cutoff,
                        masked=data_mask is not None,seed=seed,
                        recorded_samples=self.recorded_samples,max_recorded_samples=1_000_000,
                        max_batch_time_grid_entries=self.max_batch_entries,
                        max_batch_time_grid_entries_limit=2_000_000))
