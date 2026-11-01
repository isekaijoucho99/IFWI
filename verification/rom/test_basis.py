import numpy as np
from scipy.sparse import diags
from verification.rom.test_solver import case
from experiments.rom.basis import build_pressure_basis


def test_real_rational_basis_is_weighted_orthonormal_and_preserves_point_sources():
    fd,system,sources,receivers,v=case(shots=2)
    wave=np.sin(np.arange(64)*.3)*np.exp(-np.arange(64)/20)
    basis,info=build_pressure_basis(system,v.numpy()[0],sources,wave,rank=10,
        training_frequencies=8,max_hz=160.,nfft=256,alias_tolerance=1e-12)
    metric=system.stiffness+diags(system.pad_numpy(v.numpy()[0]).ravel()**-2)
    np.testing.assert_allclose(basis.T@(metric@basis),np.eye(10),rtol=0,atol=1e-7)
    point=np.eye(system.dimension)[:,sources]
    np.testing.assert_allclose(basis@(basis.T@(metric@point)),point,atol=1e-9)
    assert not np.iscomplexobj(basis)
    assert info['source_count']==2 and info['snapshot_count']==32


def test_joint_basis_includes_declared_adjoint_data_and_keeps_source_constraints():
    fd,system,sources,receivers,v=case(shots=2)
    wave=np.sin(np.arange(64)*.3)*np.exp(-np.arange(64)/20)
    residual=np.random.default_rng(11).normal(size=(2,64,5))
    basis,info=build_pressure_basis(system,v.numpy()[0],sources,wave,rank=10,
        training_frequencies=8,max_hz=160.,nfft=256,alias_tolerance=1e-12,
        receiver_indices=receivers,adjoint_residual=residual)
    metric=system.stiffness+diags(system.pad_numpy(v.numpy()[0]).ravel()**-2)
    np.testing.assert_allclose(basis.T@(metric@basis),np.eye(10),atol=1e-7)
    assert info['adjoint_snapshot_count']==32 and info['snapshot_count']==64
    assert info['data_conditioned_adjoint_calibration'] is True
    assert info['truth_snapshots_used'] is False
