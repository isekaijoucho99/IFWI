"""Offline real rational snapshots/POD for the discrete pressure operator."""
from __future__ import annotations
import time
import hashlib
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.linalg import solve_triangular
from experiments.rom.paper import weighted_pod


def build_pressure_basis(system,velocity,source_indices,wavelet,rank,*,
                         training_frequencies=32,max_hz=32.,nfft=None,
                         alias_tolerance=1e-12,preserve_sources=True,progress=None,
                         receiver_indices=None,adjoint_residual=None):
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
    snapshots=[];dual_snapshots=[];factor_seconds=0.;solve_seconds=0.;dual_seconds=0.
    if adjoint_residual is not None:
        receiver_indices=np.asarray(receiver_indices,dtype=int)
        adjoint_residual=np.asarray(adjoint_residual,dtype=float)
        if receiver_indices.ndim!=2 or receiver_indices.shape[0]!=len(source_indices) or adjoint_residual.shape!=(len(source_indices),len(wavelet),receiver_indices.shape[1]):
            raise ValueError('Adjoint calibration requires source,time,receiver residuals')
        if np.any(receiver_indices<0) or np.any(receiver_indices>=system.dimension) or not np.isfinite(adjoint_residual).all():
            raise ValueError('Invalid adjoint receiver indices or residual values')
        # Adjoint of the finite cropped/de-damped time reconstruction. Factors
        # 2/Nfft at conjugate bins do not affect individual snapshot spans.
        dual_transform=np.fft.rfft(adjoint_residual*np.exp(eta*np.arange(len(wavelet)))[None,:,None],nfft,axis=1)
    for number,k in enumerate(selected):
        z=np.exp(eta+2j*np.pi*k/nfft);start=time.perf_counter()
        factor=splu(system.pressure_matrix(velocity,z).tocsc());factor_seconds+=time.perf_counter()-start
        start=time.perf_counter();field=factor.solve(rhs*spectrum[k]);solve_seconds+=time.perf_counter()-start
        snapshots.extend([field.real,field.imag])
        if adjoint_residual is not None:
            dual_rhs=np.zeros_like(rhs,dtype=complex)
            for shot in range(len(source_indices)):
                np.add.at(dual_rhs[:,shot],receiver_indices[shot],np.conj(z)*dual_transform[shot,k])
            start=time.perf_counter();dual=factor.solve(dual_rhs,trans='H');dual_seconds+=time.perf_counter()-start
            dual_snapshots.extend([dual.real,dual.imag])
        if progress is not None:progress(dict(stage='offline_snapshots',frequency=number+1,total=len(selected)))
    snapshots=np.concatenate(snapshots,axis=1)
    dual_snapshots=np.concatenate(dual_snapshots,axis=1) if dual_snapshots else None
    metric=system.stiffness+diags(system.pad_numpy(velocity).ravel()**-2)
    source_rank=len(source_indices) if preserve_sources else 0
    if not source_rank<rank<=system.dimension:raise ValueError('Rank must exceed source constraints and fit the pressure space')
    if preserve_sources:
        points=np.zeros((system.dimension,source_rank));points[source_indices,np.arange(source_rank)]=1.
        source_basis,_=weighted_pod(points,metric,source_rank)
        for _ in range(2):snapshots-=source_basis@(source_basis.T@(metric@snapshots))
        if dual_snapshots is not None:
            for _ in range(2):dual_snapshots-=source_basis@(source_basis.T@(metric@dual_snapshots))
    if dual_snapshots is not None:
        primal_energy=float(np.sum(snapshots*(metric@snapshots)))
        dual_energy=float(np.sum(dual_snapshots*(metric@dual_snapshots)))
        if primal_energy<=0 or dual_energy<=0:raise ValueError('Adjoint calibration has no finite energy')
        dual_scale=np.sqrt(primal_energy/dual_energy)
        snapshots=np.concatenate([snapshots,dual_scale*dual_snapshots],axis=1)
    if preserve_sources:
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
        adjoint_solve_seconds=dual_seconds,adjoint_snapshot_count=0 if dual_snapshots is None else dual_snapshots.shape[1],
        data_conditioned_adjoint_calibration=dual_snapshots is not None,
        calibration_residual_sha256=None if dual_snapshots is None else hashlib.sha256(adjoint_residual.tobytes()).hexdigest(),
        joint_energy_balance=None if dual_snapshots is None else 'Equal primal and dual reference-metric energies after source projection',
        total_offline_seconds=time.perf_counter()-started,
        weighted_orthogonality_fro=float(np.linalg.norm(basis.T@(metric@basis)-np.eye(rank))),
        retained_last_singular=float(singular[rank-source_rank-1]),
        metric='diag(v_background_padded^-2) - original_L_dt',
        material_models_used=1,truth_snapshots_used=False)
    return basis,info
