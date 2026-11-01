import numpy as np
import pytest
from scipy.linalg import fractional_matrix_power
from scipy.integrate import quad
from experiments.rom.paper import (weighted_pod,galerkin_solve,weeks_nodes,weeks_inverse,
    ricker_laplace,spd_stability_lower_bound,residual_error_bound)


def test_weighted_pod_metric_orthonormal_and_full_space_interpolation():
    rng=np.random.default_rng(3);a=rng.normal(size=(8,8));metric=a.T@a+np.eye(8)
    snapshots=rng.normal(size=(8,12))+1j*rng.normal(size=(8,12))
    basis,singular=weighted_pod(snapshots,metric,8)
    np.testing.assert_allclose(basis.conj().T@metric@basis,np.eye(8),atol=1e-12)
    assert np.all(np.diff(singular)<=0)
    mass=np.diag(np.arange(1,9));stiffness=metric+np.eye(8)
    b=rng.normal(size=8);s=.3+2j
    actual=basis@galerkin_solve(mass,stiffness,b,basis,s)
    np.testing.assert_allclose(actual,np.linalg.solve(s*s*mass+stiffness,b),rtol=1e-11,atol=1e-12)


def test_weighted_pod_rejects_rank_deficient_requested_space():
    with pytest.raises(ValueError):weighted_pod(np.ones((5,4)),np.eye(5),3)


def test_weeks_inversion_matches_known_exponential_and_sine():
    nodes=weeks_nodes(256,.2,3.)
    transforms=np.stack([1/(nodes+1.3),2/(nodes*nodes+4)],axis=1)
    times=np.linspace(0,8,71)
    actual=weeks_inverse(transforms,times,.2,3.)
    expected=np.stack([np.exp(-1.3*times),np.sin(2*times)],axis=1)
    np.testing.assert_allclose(actual,expected,atol=1e-10,rtol=1e-9)


def test_ricker_laplace_agrees_with_numerical_causal_transform():
    alpha=3*np.pi;t0=4*np.pi/alpha;s=.3+2j
    def integrand(t):return (1-.5*alpha**2*(t-t0)**2)*np.exp(-alpha**2*(t-t0)**2/4-s*t)
    expected=quad(lambda t:integrand(t).real,0,8,epsabs=1e-12)[0]+1j*quad(lambda t:integrand(t).imag,0,8,epsabs=1e-12)[0]
    np.testing.assert_allclose(ricker_laplace(s,alpha,t0),expected,rtol=1e-11,atol=1e-12)


@pytest.mark.parametrize('s',[.1+.2j,.1+3j,.5+8j,.5+0j,1+0j])
def test_spd_bound_is_below_exact_discrete_inf_sup(s):
    mass=np.diag([1.,2.,3.]);stiffness=np.array([[3.,1,0],[1,4,1],[0,1,5]])
    metric=mass+stiffness;xi=fractional_matrix_power(metric,-.5)
    lam=np.linalg.eigvalsh(np.diag(1/np.sqrt(np.diag(mass)))@stiffness@np.diag(1/np.sqrt(np.diag(mass))))[0]
    beta=spd_stability_lower_bound(s,lam/(1+lam))
    assert 0<beta<=np.linalg.svd(xi@(s*s*mass+stiffness)@xi,compute_uv=False)[-1]+1e-12
    b=np.array([1.,2,3.]);v=np.eye(3)[:,:1]
    reduced=v@galerkin_solve(mass,stiffness,b,v,s)
    exact=np.linalg.solve(s*s*mass+stiffness,b)
    residual=b-(s*s*mass+stiffness)@reduced
    actual=np.sqrt(np.real((exact-reduced).conj()@metric@(exact-reduced)))
    assert actual<=residual_error_bound(residual,metric,beta)+1e-12
