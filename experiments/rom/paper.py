"""Core algebra of the SIAM2025 seismic ROM, independently testable on CPU.

Weeks FFT/Clenshaw ordering follows the BSD3-Clause Zenodo15033159 v_2
implementation by Hawkins, Khalid, Schlottbom and Smetana. See LICENSE_SIAM_MOR.
The SPD residual bound is for M,K>0 and is NOT a PML stability certificate.
"""
from __future__ import annotations
import numpy as np
from scipy.special import erfcx


def weighted_pod(snapshots,metric,rank):
    """Return X-orthonormal POD modes and sqrt eigenvalues of S*XS/Ns."""
    snapshots=np.asarray(snapshots)
    if snapshots.ndim!=2 or not 1<=rank<=min(snapshots.shape):
        raise ValueError('POD rank must fit the snapshot matrix')
    gram=snapshots.conj().T@(metric@snapshots)/snapshots.shape[1]
    values,vectors=np.linalg.eigh((gram+gram.conj().T)/2)
    ids=np.argsort(values)[::-1];values=values[ids];vectors=vectors[:,ids]
    tolerance=np.finfo(float).eps*max(snapshots.shape)*max(float(values[0]),1e-300)
    if values[rank-1]<=tolerance:raise ValueError('Requested POD space is numerically rank deficient')
    basis=(snapshots@vectors[:,:rank])/np.sqrt(snapshots.shape[1]*values[:rank])[None]
    return basis,np.sqrt(np.maximum(values,0))


def galerkin_solve(mass,stiffness,source,basis,s,amplitude=1.):
    """Elastic FEM reduced solve; basis is fixed and may be complex."""
    v=np.asarray(basis)
    mr=v.conj().T@(mass@v);kr=v.conj().T@(stiffness@v)
    br=v.conj().T@np.asarray(source)
    return np.linalg.solve(s*s*mr+kr,amplitude*br)


def weeks_nodes(count,real_shift,scale):
    if count<2 or real_shift<=0 or scale<=0:raise ValueError('Invalid Weeks contour')
    theta=(np.arange(-count,count)+.5)*np.pi/count
    c=np.exp(1j*theta)
    return real_shift-scale*(c+1)/(c-1)


def weeks_inverse(transforms,times,real_shift,scale):
    """Inverse using all 2*N Weeks contour values; first axis is frequency.

    Output has [time,...] shape. Physical real-valued signals use real Laguerre
    coefficients, as in the original code. No frequency truncation is hidden.
    """
    f=np.asarray(transforms);t=np.asarray(times,dtype=float)
    if f.ndim<1 or f.shape[0]%2 or f.shape[0]<4 or t.ndim!=1:
        raise ValueError('Weeks needs an even contour and a vector of times')
    n=f.shape[0]//2;weeks_nodes(n,real_shift,scale)
    shape=f.shape[1:];flat=f.reshape(2*n,-1)
    theta=(np.arange(-n,n)+.5)*np.pi/n
    samples=2*scale/(1-np.exp(1j*theta))[:,None]*flat
    transformed=np.fft.fftshift(np.fft.fft(np.fft.fftshift(samples,axes=0),axis=0),axes=0)/(2*n)
    coefficients=(transformed[n:]*np.exp(-1j*np.arange(n)*np.pi/(2*n))[:,None]).real
    previous=np.broadcast_to(coefficients[-1],(len(t),coefficients.shape[1])).copy()
    older=np.zeros_like(previous);x=2*scale*t[:,None]
    for k in range(n-1,0,-1):
        current=((2*k-1-x)/k)*previous-(k/(k+1))*older+coefficients[k-1]
        older,previous=previous,current
    result=np.exp((real_shift-scale)*t[:,None])*previous
    return result.reshape((len(t),)+shape)


def ricker_laplace(s,alpha,t0,amplitude=1.):
    """Causal [0,infinity) transform of the paper's Ricker definition."""
    s=np.asarray(s,dtype=complex)
    if alpha<=0 or t0<0:raise ValueError('Invalid Ricker width or delay')
    z=-alpha*t0/2+s/alpha
    return amplitude*np.exp(-alpha**2*t0**2/4)/alpha**3*(
        alpha**3*t0+2*s*alpha-2*s*s*np.sqrt(np.pi)*erfcx(z))


def spd_stability_lower_bound(s,tau0):
    """Distance from zero to the numerical-range segment [s²(1-tau0)+tau0,1].

    tau0=lambda_min(K,M)/(1+lambda_min(K,M)). Valid for SPD M,K in
    metric X=M+K; it is not a bound for an arbitrary complex PML matrix.
    """
    if np.real(s)<=0 or not 0<=tau0<=1:raise ValueError('Require Re(s)>0 and 0<=tau0<=1')
    z=s*s;direction=1-z
    tau=tau0 if abs(direction)==0 else np.clip(-np.real(np.conj(direction)*z)/abs(direction)**2,tau0,1.)
    return float(abs(z+tau*direction))


def residual_error_bound(residual,metric,stability_lower_bound):
    if stability_lower_bound<=0:raise ValueError('A positive certified lower bound is required')
    from scipy.sparse import issparse
    from scipy.sparse.linalg import spsolve
    residual=np.asarray(residual)
    riesz=spsolve(metric,residual) if issparse(metric) else np.linalg.solve(metric,residual)
    norm=float(np.sqrt(max(0.,np.real(np.vdot(residual,riesz)))))
    return norm/stability_lower_bound
