import numpy as np
import torch
from verification.rom.test_solver import case
from experiments.rom.solver import PressureROM
from experiments.rom.stability import projected_recurrence
from verification.rom.test_discrete import black_box_state_matrix


def test_reduced_recurrence_matches_frequency_transfer_and_original_impulse():
    fd,system,sources,receivers,v=case()
    rom=PressureROM(system,np.eye(system.dimension),sources,receivers,24)
    mr,md,br=[x.detach().numpy() for x in rom.projected_model(v)]
    a,b=projected_recurrence(mr,md,rom.stiffness_r.numpy(),rom.pml_r.numpy(),system.alphas,system.dt,br)
    n=system.dimension;z=np.exp(.03+.2j)
    state=np.linalg.solve(z*np.eye(len(a))-a,b)
    expected=z*np.linalg.solve(system.pressure_matrix(v.numpy()[0],z).toarray(),system.source_rhs(v.numpy()[0],sources))
    np.testing.assert_allclose(z*state[:n],expected,rtol=1e-11,atol=1e-11)
    x=np.zeros((len(a),1));fields=[]
    for k in range(24):
        x=a@x+(b if k==0 else 0);fields.append(x[:n,0])
    wave=torch.zeros(24,dtype=torch.float64);wave[0]=1
    original=fd(v,wave)[2].detach().numpy()[0,0]
    np.testing.assert_allclose(np.stack(fields)[:,receivers[0]],original,rtol=1e-10,atol=1e-11)
    radius=float(np.max(np.abs(np.linalg.eigvals(a))))
    original_radius=float(np.max(np.abs(np.linalg.eigvals(black_box_state_matrix(fd,v)))))
    # This deliberately tiny, strongly damped original PML is itself unstable;
    # CFL alone must not be misrepresented as a proof for the full PML step.
    assert abs(radius-original_radius)<1e-8
    assert original_radius>1.01
