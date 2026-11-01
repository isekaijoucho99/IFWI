import numpy as np
from experiments.rom.solver import PressureROM
from experiments.diagnose_rom_benchmark import pole_probe
from verification.rom.test_solver import case
from verification.rom.test_discrete import black_box_state_matrix


def test_matrix_free_pole_probe_detects_original_toy_pml_instability():
    fd,system,sources,receivers,v=case()
    rom=PressureROM(system,np.eye(system.dimension),sources,receivers,24)
    actual=pole_probe(rom,v,system)
    original_radius=float(np.max(abs(np.linalg.eigvals(black_box_state_matrix(fd,v)))))
    assert abs(actual['largest_observed_pole_modulus']-original_radius)<1e-6
    assert actual['observed_poles_outside_unit_disk']>0
    # An unstable recurrence may still admit a causal inverse on a wider circle.
    assert actual['observed_poles_outside_contour']==0
    assert max(actual['ritz_absolute_residuals'])<1e-6
