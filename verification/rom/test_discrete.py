import numpy as np
import pytest
import torch
from rnn_fd import rnn2D
from experiments.rom.discrete import DiscretePMLSystem


def reference(free_surface=True):
    nz,nx,npad=4,5,2
    solver=rnn2D(nz,nx,zs=torch.tensor([[1]]),xs=torch.tensor([[2]]),
        zr=torch.tensor([[[1,2]]]),xr=torch.tensor([[[2,3]]]),dz=20.,dt=.001,
        npad=npad,order=2,vmax=3200.,freeSurface=free_surface,dtype=torch.float64)
    v=torch.linspace(2200,3000,nz)[:,None].expand(nz,nx).clone()[None].double()
    return solver,v


def black_box_state_matrix(solver,v):
    n=solver.nz_pad*solver.nx_pad;basis=torch.eye(4*n,dtype=torch.float64)
    u,prev,vx,vz=[basis[j*n:(j+1)*n].T.reshape(1,4*n,solver.nz_pad,solver.nx_pad) for j in range(4)]
    coords=torch.zeros((1,4*n),dtype=torch.long)
    output=solver.fd(v,[torch.zeros(1,dtype=torch.float64),coords,coords,torch.zeros((1,4*n,1),dtype=torch.long),torch.zeros((1,4*n,1),dtype=torch.long)],prev,u,vx,vz)
    previous,current,records,reg,vxnew,vznew=output
    return torch.cat([current.reshape(4*n,n).T,previous.reshape(4*n,n).T,
        vxnew.reshape(4*n,n).T,vznew.reshape(4*n,n).T],dim=0).numpy()


@pytest.mark.parametrize('free_surface',[True,False])
def test_pressure_schur_is_exact_for_original_black_box_pml_step(free_surface):
    solver,v=reference(free_surface);system=DiscretePMLSystem.from_solver(solver)
    a=black_box_state_matrix(solver,v);n=system.dimension
    z=np.exp(.04+.3j);indices=np.array([0,n//2])
    b=np.eye(4*n)[:,indices]
    exact=np.linalg.solve(z*np.eye(4*n)-a,b)[:n]
    actual=np.linalg.solve(system.pressure_matrix(v.numpy()[0],z).toarray(),system.source_rhs(v.numpy()[0],indices))
    np.testing.assert_allclose(actual,exact,rtol=2e-11,atol=2e-12)
    np.testing.assert_allclose(system.pad_numpy(v.numpy()[0]),solver.fd.velocity.numpy()[0,0],rtol=0,atol=0)


def test_affine_pml_profiles_reconstruct_direct_matrix():
    solver,v=reference();system=DiscretePMLSystem.from_solver(solver)
    z=np.exp(.02+.1j);vp=system.pad_numpy(v.numpy()[0]).ravel()
    m=vp**-2;delta=system.delta
    expected=np.diag(((z-1)**2/z+system.dt*(z-1)*delta)*m)+system.stiffness.toarray()
    for alpha,operator in zip(system.alphas,system.pml_terms):expected+=z/(z-alpha)*operator.toarray()
    np.testing.assert_allclose(expected,system.pressure_matrix(v.numpy()[0],z).toarray(),atol=1e-18)


def test_geometry_maps_free_surface_and_padding_and_rejects_outside():
    solver,v=reference(False);system=DiscretePMLSystem.from_solver(solver)
    index=system.physical_indices([[0,0]])[0]
    assert index==system.npad*system.shape[1]+system.npad
    with pytest.raises(ValueError):system.physical_indices([[-1,0]])
