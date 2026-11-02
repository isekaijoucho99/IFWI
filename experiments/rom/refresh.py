"""Fixed-PML snapshot transport using cached reference sparse factorizations.

Residuals are indicators, not certified gradient bounds. A cache is immutable;
reanchoring constructs a new complete cache before replacing the old one.
"""
import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg import solve_triangular
from experiments.rom.paper import weighted_pod


class ReferenceCache:
    def __init__(self,system,velocity,sources,receivers,wavelet,residual,*,
                 training_frequencies=64,max_hz=32.,nfft=4096,alias_tolerance=1e-12):
        start=time.perf_counter();self.system=system;self.velocity=np.asarray(velocity).copy()
        self.sources=np.asarray(sources,dtype=int);self.receivers=np.asarray(receivers,dtype=int)
        self.wavelet=np.asarray(wavelet,dtype=float);self.nfft=int(nfft)
        if self.receivers.ndim!=2 or len(self.receivers)!=len(self.sources):raise ValueError('Source/receiver shape mismatch')
        if nfft<2*len(wavelet) or not 0<alias_tolerance<1:raise ValueError('Invalid FFT contour')
        self.eta=-np.log(alias_tolerance)/nfft
        self.mass=system.pad_numpy(velocity).ravel()**-2
        self.metric=system.stiffness+diags(self.mass)
        active=np.arange(nfft//2+1)
        if max_hz is not None:active=active[active/(nfft*system.dt)<=max_hz]
        if not 2<=training_frequencies<=len(active):raise ValueError('Invalid number of training frequencies')
        self.selected=active[np.linspace(0,len(active)-1,training_frequencies).round().astype(int)]
        self.z=np.exp(self.eta+2j*np.pi*self.selected/nfft)
        zm1=np.expm1(self.eta+2j*np.pi*self.selected/nfft)
        self.a0=zm1*zm1/self.z;self.a1=system.dt*zm1
        self.wave_spectrum=np.fft.rfft(self.wavelet*np.exp(-self.eta*np.arange(len(wavelet))),nfft)
        self.rhs=np.stack([system.source_rhs(velocity,self.sources)*self.wave_spectrum[k] for k in self.selected])
        self.dual_rhs=None if residual is None else self.adjoint_rhs(residual)
        factors=[];primal=[];adjoint=[];factor_seconds=0.;solve_seconds=0.
        for k,z in enumerate(self.z):
            stamp=time.perf_counter();factor=splu(system.pressure_matrix(velocity,z).tocsc())
            factor_seconds+=time.perf_counter()-stamp;stamp=time.perf_counter()
            primal.append(factor.solve(self.rhs[k]))
            if self.dual_rhs is not None:adjoint.append(factor.solve(self.dual_rhs[k],trans='H'))
            solve_seconds+=time.perf_counter()-stamp;factors.append(factor)
        self.factors=factors;self.primal=np.stack(primal);self.adjoint=np.stack(adjoint) if adjoint else None
        self.info=dict(training_frequencies=len(factors),factorization_seconds=factor_seconds,
            snapshot_solve_seconds=solve_seconds,total_seconds=time.perf_counter()-start,
            full_pressure_dimension=system.dimension,selected_bins=self.selected.tolist(),
            lu_reported_nnz=sum(factor.nnz for factor in factors),
            lu_storage_estimate_bytes=sum(factor.nnz*20+16*(system.dimension+1) for factor in factors),
            lu_storage_estimate_limit='Complex128 values plus estimated int32 sparse indices/permutations; excludes native SuperLU workspaces; process RSS/peak also reported',
            snapshot_storage_bytes=self.primal.nbytes+self.rhs.nbytes+(0 if self.adjoint is None else self.adjoint.nbytes+self.dual_rhs.nbytes),
            fft_adjoint_weighting='Exact real-irfft bin multiplicity, 1/Nfft, and waveform-MSE 2/Ndata')

    def complete_adjoint(self,residual):
        """Complete an unpublished primal-only cache exactly once, atomically."""
        if self.adjoint is not None:raise ValueError('Published reference adjoint cannot be overwritten')
        stamp=time.perf_counter();rhs=self.adjoint_rhs(residual)
        values=np.stack([factor.solve(value,trans='H') for factor,value in zip(self.factors,rhs)])
        self.dual_rhs=rhs;self.adjoint=values
        self.info['adjoint_completion_seconds']=time.perf_counter()-stamp
        self.info['snapshot_storage_bytes']=self.primal.nbytes+self.rhs.nbytes+self.adjoint.nbytes+self.dual_rhs.nbytes

    def adjoint_rhs(self,residual,selected=None):
        residual=np.asarray(residual,dtype=float)
        if residual.shape!=(len(self.sources),len(self.wavelet),self.receivers.shape[1]) or not np.isfinite(residual).all():
            raise ValueError('Residual must be finite source,time,receiver data')
        transform=np.fft.rfft((2/residual.size)*residual*
            np.exp(self.eta*np.arange(len(self.wavelet)))[None,:,None],self.nfft,axis=1)
        bins=self.selected if selected is None else np.asarray(selected,dtype=int)
        z=self.z if selected is None else np.exp(self.eta+2j*np.pi*bins/self.nfft)
        weights=np.full(len(bins),2/self.nfft)
        weights[(bins==0)|(bins==self.nfft//2)]=1/self.nfft
        result=np.zeros((len(bins),self.system.dimension,len(self.sources)),dtype=complex)
        for j,k in enumerate(bins):
            for shot in range(len(self.sources)):
                np.add.at(result[j,:,shot],self.receivers[shot],np.conj(z[j])*weights[j]*transform[shot,k])
        return result

    def _transport(self,velocity,rhs,reference,reference_rhs,corrections,adjoint):
        if corrections not in (1,2):raise ValueError('Small validation supports one or two corrections')
        start=time.perf_counter();difference=self.system.pad_numpy(velocity).ravel()**-2-self.mass
        fields=[];residuals=[];individual=[]
        for k,factor in enumerate(self.factors):
            diagonal=(self.a0[k]+self.a1[k]*self.system.delta)*difference
            if adjoint:diagonal=diagonal.conj()
            defect=rhs[k]-reference_rhs[k]-diagonal[:,None]*reference[k]
            value=reference[k].copy()
            for _ in range(corrections):
                correction=factor.solve(defect,trans='H' if adjoint else 'N')
                value+=correction;defect=-diagonal[:,None]*correction
            fields.append(value);residuals.append(defect)
            individual.append(float(np.linalg.norm(defect)/max(np.linalg.norm(rhs[k]),1e-300)))
        fields=np.stack(fields);residuals=np.stack(residuals)
        relative=float(np.linalg.norm(residuals)/max(np.linalg.norm(rhs),1e-300))
        if not np.isfinite(fields).all() or not np.isfinite(relative) or not np.isfinite(individual).all():
            raise FloatingPointError('Nonfinite transported fields or indicators')
        return fields,dict(correction_order=corrections,solve_count=len(self.factors)*corrections,
            aggregate_relative_residual=relative,max_frequency_relative_residual=max(individual),
            seconds=time.perf_counter()-start,residual_vectors=residuals)

    def transport_primal(self,velocity,corrections):
        base=self.system.source_rhs(velocity,self.sources)
        rhs=np.stack([base*self.wave_spectrum[k] for k in self.selected])
        return self._transport(velocity,rhs,self.primal,self.rhs,corrections,False)

    def transport_adjoint(self,velocity,residual,corrections):
        if self.adjoint is None:raise ValueError('Complete reference adjoint before transport')
        return self._transport(velocity,self.adjoint_rhs(residual),self.adjoint,self.dual_rhs,corrections,True)


def _real_columns(fields):
    return np.concatenate([a for field in fields for a in (field.real,field.imag)],axis=1)


def _reorthogonalize(basis,metric):
    gram=basis.T@(metric@basis);lower=np.linalg.cholesky((gram+gram.T)/2)
    return solve_triangular(lower,basis.T,lower=True).T


def primal_space(cache,fields,rank):
    points=np.zeros((cache.system.dimension,len(cache.sources)))
    points[cache.sources,np.arange(len(cache.sources))]=1
    fixed,_=weighted_pod(points,cache.metric,len(cache.sources));columns=_real_columns(fields)
    for _ in range(2):columns-=fixed@(fixed.T@(cache.metric@columns))
    remainder,_=weighted_pod(columns,cache.metric,rank-len(cache.sources))
    return _reorthogonalize(np.concatenate([fixed,remainder],axis=1),cache.metric)


def enriched_space(cache,primal,fields,rank):
    columns=_real_columns(fields)
    for _ in range(2):columns-=primal@(primal.T@(cache.metric@columns))
    remainder,_=weighted_pod(columns,cache.metric,rank-primal.shape[1])
    return _reorthogonalize(np.concatenate([primal,remainder],axis=1),cache.metric)


def scalar_stats(stats):
    return {k:v for k,v in stats.items() if k!='residual_vectors'}


def apply_pressure(system,velocity,z,fields,adjoint=False):
    mass=system.pad_numpy(velocity).ravel()**-2;zm1=z-1
    diagonal=(zm1*zm1/z+system.dt*zm1*system.delta)*mass
    if adjoint:
        value=diagonal.conj()[:,None]*fields+system.stiffness.conj().T@fields
        for alpha,term in zip(system.alphas,system.pml_terms):value+=np.conj(z/(z-alpha))*(term.conj().T@fields)
    else:
        value=diagonal[:,None]*fields+system.stiffness@fields
        for alpha,term in zip(system.alphas,system.pml_terms):value+=(z/(z-alpha))*(term@fields)
    return value


def galerkin_probe(cache,rom,basis,velocity,residual,count=8):
    """Final-Galerkin primal/dual residuals at held-out retained frequency bins.

    No FDTD call; no certified beta. The forcing uses this final ROM's residual.
    """
    import torch
    active=rom.active.detach().cpu().numpy()
    candidates=np.setdiff1d(active,cache.selected)
    if not len(candidates):candidates=active
    bins=candidates[np.linspace(0,len(candidates)-1,min(count,len(candidates))).round().astype(int)]
    mass,damped,source=[a.detach().cpu().numpy() for a in rom.projected_model(velocity)]
    kr=rom.stiffness_r.detach().cpu().numpy();pr=rom.pml_r.detach().cpu().numpy()
    z=np.exp(cache.eta+2j*np.pi*bins/cache.nfft);dual_rhs=cache.adjoint_rhs(residual,bins)
    physical=velocity.detach().cpu().numpy()[0]
    base=cache.system.source_rhs(physical,cache.sources)
    rp=[];ra=[];bp=[];ba=[]
    with torch.no_grad():
        for j,(k,point) in enumerate(zip(bins,z)):
            zm1=point-1
            lhs=(zm1*zm1/point)*mass+cache.system.dt*zm1*damped+kr
            lhs=lhs+np.einsum('k,kij->ij',point/(point-cache.system.alphas),pr)
            rhs=source*cache.wave_spectrum[k]
            u=basis@np.linalg.solve(lhs,rhs)
            dual=basis@np.linalg.solve(lhs.conj().T,basis.T@dual_rhs[j])
            full_rhs=base*cache.wave_spectrum[k]
            rp.append(full_rhs-apply_pressure(cache.system,physical,point,u))
            ra.append(dual_rhs[j]-apply_pressure(cache.system,physical,point,dual,True))
            bp.append(full_rhs);ba.append(dual_rhs[j])
    def relative(r,b):return float(np.linalg.norm(r)/max(np.linalg.norm(b),1e-300))
    result=dict(selected_bins=bins.tolist(),primal=relative(rp,bp),adjoint=relative(ra,ba),
        interpretation='Held-out final-space residual indicators, not certified gradient bounds')
    if not np.isfinite(result['primal']) or not np.isfinite(result['adjoint']):raise FloatingPointError('Nonfinite Galerkin indicators')
    return result
