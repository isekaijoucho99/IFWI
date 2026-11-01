import io
import numpy as np
import pytest
import torch
from rnn_fd import rnn2D
from experiments.rom.discrete import DiscretePMLSystem
from experiments.rom.solver import PressureROM

torch.set_num_threads(1)


def case(free=True,shots=1):
    nz,nx=4,5;zs=torch.ones((1,shots),dtype=torch.long)
    xs=torch.tensor([[1,3][:shots]])
    zr=zs[:,:,None].repeat(1,1,nx);xr=torch.arange(nx)[None,None].repeat(1,shots,1)
    fd=rnn2D(nz,nx,zs,xs,zr,xr,20.,.001,npad=2,order=2,vmax=3200.,freeSurface=free,dtype=torch.float64)
    v=torch.linspace(2200,3000,nz,dtype=torch.float64)[:,None].expand(nz,nx).clone()[None]
    system=DiscretePMLSystem.from_solver(fd)
    sources=system.physical_indices(np.stack([zs[0].numpy(),xs[0].numpy()],axis=-1))
    receivers=system.physical_indices(np.stack([zr[0].numpy(),xr[0].numpy()],axis=-1).reshape(-1,2)).reshape(shots,nx)
    return fd,system,sources,receivers,v


@pytest.mark.parametrize('free,shots',[(True,1),(False,1),(True,2)])
def test_full_basis_reproduces_original_fd_impulse_and_velocity_gradient(free,shots):
    fd,system,sources,receivers,v=case(free,shots)
    nt=32;wave=torch.zeros(nt,dtype=torch.float64);wave[0]=1.;wave[3]=-.4
    rom=PressureROM(system,np.eye(system.dimension),sources,receivers,nt,alias_tolerance=1e-13)
    a=v.clone().requires_grad_(True);b=v.clone().requires_grad_(True)
    original=fd(a,wave)[2];reduced=rom(b,wave)[2]
    np.testing.assert_allclose(reduced.detach().numpy(),original.detach().numpy(),rtol=3e-7,atol=3e-9)
    # Same source and receiver depth: post-injection first sample is exactly q[0].
    for k,x in enumerate([1,3][:shots]):assert abs(reduced[0,k,0,x].item()-1)<1e-9
    target=torch.linspace(-.1,.1,original.numel(),dtype=torch.float64).reshape_as(original)
    ga=torch.autograd.grad((original-target).square().mean(),a)[0]
    gb=torch.autograd.grad((reduced-target).square().mean(),b)[0]
    assert float((ga-gb).norm()/ga.norm())<2e-6
    assert torch.dot(ga.ravel(),gb.ravel())/(ga.norm()*gb.norm())>1-1e-9


def test_fixed_basis_finite_difference_and_checkpoint_buffers():
    fd,system,sources,receivers,v=case()
    wave=torch.sin(torch.arange(24,dtype=torch.float64)*.4)
    rom=PressureROM(system,np.eye(system.dimension),sources,receivers,24,alias_tolerance=1e-13)
    assert not list(rom.parameters())
    assert 'basis' in dict(rom.named_buffers())
    value=v.clone().requires_grad_(True)
    loss=rom(value,wave)[2].square().mean();gradient=torch.autograd.grad(loss,value)[0]
    generator=torch.Generator().manual_seed(3)
    for _ in range(3):
        direction=torch.randn(v.shape,generator=generator,dtype=torch.float64);direction/=direction.norm()
        h=.1
        finite=(rom(v+h*direction,wave)[2].square().mean()-rom(v-h*direction,wave)[2].square().mean())/(2*h)
        np.testing.assert_allclose(finite.item(),(gradient*direction).sum().item(),rtol=2e-5,atol=1e-10)
    buffer=io.BytesIO();torch.save(rom.state_dict(),buffer);buffer.seek(0)
    other=PressureROM(system,np.eye(system.dimension),sources,receivers,24,alias_tolerance=1e-13)
    other.load_state_dict(torch.load(buffer,weights_only=True))
    assert torch.equal(other(v,wave)[2],rom(v,wave)[2])
    with pytest.raises(ValueError):rom(v,wave,prev_state=torch.zeros(1))
