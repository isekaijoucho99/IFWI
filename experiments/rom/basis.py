"""Offline real rational snapshots/POD for the discrete pressure operator."""
from __future__ import annotations
import time
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg import solve_triangular
from experiments.rom.paper import weighted_pod


def build_pressure_basis(system,velocity,source_indices,wavelet,rank,*,
                         training_frequencies=32,max_hz=32.,nfft=None,
                         alias_tolerance=1e-12,preserve_sources=True,progress=None):
    started=time.perf_counter();wavelet=np.asarray(wavelet,dtype=float)
    nfft=int(nfft or 2**int(np.ceil(np.log2(4*len(wavelet)))))
    if nfft<2*len(wavelet) or not 0<alias_tolerance<1:
        raise ValueError('Invalid snapshot contour')
    indices=np.arange(nfft//2+1)
    if max_hz is not None:indices=indices[indices/(nfft*system.dt)<=max_hz]
    if training_frequencies<2 or training_frequencies>len(indices):raise ValueError('Insufficient contour frequencies')
    selected=indices[np.linspace(0,len(indices)-1,training_frequencies).round().astype(int)]
    eta=-np.log(alias_tolerance)/nfft
    spectrum=np.fft.rfft(wavelet*np.exp(-eta*np.arange(len(wavelet))),nfft)
    source_indices=np.asarray(source_indices,dtype=int)
    rhs=system.source_rhs(velocity,source_indices)
    snapshots=[];factor_seconds=0.;solve_seconds=0.
    for number,k in enumerate(selected):
        z=np.exp(eta+2j*np.pi*k/nfft);start=time.perf_counter()
        factor=splu(system.pressure_matrix(velocity,z).tocsc());factor_seconds+=time.perf_counter()-start
        start=time.perf_counter();field=factor.solve(rhs*spectrum[k]);solve_seconds+=time.perf_counter()-start
        snapshots.extend([field.real,field.imag])
        if progress is not None:progress(dict(stage='offline_snapshots',frequency=number+1,total=len(selected)))
    snapshots=np.concatenate(snapshots,axis=1)
    metric=system.stiffness+diags(system.pad_numpy(velocity).ravel()**-2)
    source_rank=len(source_indices) if preserve_sources else 0
    if not source_rank<rank<=min(snapshots.shape):raise ValueError('Rank must exceed source constraints and fit snapshots')
    if preserve_sources:
        points=np.zeros((system.dimension,source_rank));points[source_indices,np.arange(source_rank)]=1.
        source_basis,_=weighted_pod(points,metric,source_rank)
        for _ in range(2):snapshots-=source_basis@(source_basis.T@(metric@snapshots))
        remainder,singular=weighted_pod(snapshots,metric,rank-source_rank)
        basis=np.concatenate([source_basis,remainder],axis=1)
    else:basis,singular=weighted_pod(snapshots,metric,rank)
    gram=basis.T@(metric@basis);cholesky=np.linalg.cholesky((gram+gram.T)/2)
    basis=solve_triangular(cholesky,basis.T,lower=True).T
    info=dict(rank=rank,source_count=len(source_indices),source_constraints=source_rank,
        full_pressure_dimension=system.dimension,snapshot_count=snapshots.shape[1],
        selected_frequency_bins=selected.tolist(),frequencies_hz=(selected/(nfft*system.dt)).tolist(),
        nfft=nfft,max_hz=max_hz,alias_tolerance=alias_tolerance,
        factorization_seconds=factor_seconds,snapshot_solve_seconds=solve_seconds,
        total_offline_seconds=time.perf_counter()-started,
        weighted_orthogonality_fro=float(np.linalg.norm(basis.T@(metric@basis)-np.eye(rank))),
        retained_last_singular=float(singular[rank-source_rank-1]),
        metric='diag(v_background_padded^-2) - original_L_dt',
        material_models_used=1,truth_snapshots_used=False)
    return basis,info
