import numpy as np
import torch
from verification.rom.test_solver import case
from experiments.rom.refresh import ReferenceCache,primal_space,enriched_space,galerkin_probe,apply_pressure
from experiments.rom.projector import SparseProjector
from experiments.rom.solver import PressureROM


def cache_case():
    fd,system,sources,receivers,v=case(shots=2)
    wave=np.sin(np.arange(64)*.3)*np.exp(-np.arange(64)/20)
    residual=np.random.default_rng(11).normal(size=(2,64,5))
    cache=ReferenceCache(system,v.numpy()[0],sources,receivers,wave,residual,
        training_frequencies=8,max_hz=160.,nfft=256,alias_tolerance=1e-12)
    return fd,cache,v,wave,residual


def test_primal_and_conjugate_adjoint_transport_have_the_exact_residual_and_improve_order():
    fd,cache,v,wave,residual=cache_case();velocity=v.numpy()[0]*1.003
    first,one=cache.transport_primal(velocity,1);second,two=cache.transport_primal(velocity,2)
    adjoint,dual=cache.transport_adjoint(velocity,residual*.97,2)
    e1=[];e2=[]
    for k,z in enumerate(cache.z):
        matrix=cache.system.pressure_matrix(velocity,z)
        rhs=cache.system.source_rhs(velocity,cache.sources)*cache.wave_spectrum[cache.selected[k]]
        exact=np.linalg.solve(matrix.toarray(),rhs)
        e1.append(np.linalg.norm(first[k]-exact));e2.append(np.linalg.norm(second[k]-exact))
        np.testing.assert_allclose(rhs-matrix@first[k],one['residual_vectors'][k],rtol=1e-7,atol=1e-16)
        np.testing.assert_allclose(rhs-matrix@second[k],two['residual_vectors'][k],rtol=1e-7,atol=1e-16)
        force=cache.adjoint_rhs(residual*.97)[k]
        np.testing.assert_allclose(force-matrix.conj().T@adjoint[k],dual['residual_vectors'][k],rtol=1e-7,atol=1e-14)
    assert np.linalg.norm(e2)<.1*np.linalg.norm(e1)
    exact,meta=cache.transport_primal(v.numpy()[0],1)
    np.testing.assert_allclose(exact,cache.primal,atol=1e-15)
    assert meta['aggregate_relative_residual']==0


def test_finite_fft_adjoint_forcing_matches_autograd_including_dc_and_mse_scale():
    fd,cache,v,wave,residual=cache_case()
    rng=np.random.default_rng(3)
    u=torch.tensor(rng.normal(size=cache.primal.shape)+1j*rng.normal(size=cache.primal.shape),requires_grad=True)
    rec=u[:,cache.receivers,np.arange(len(cache.sources))[:,None]]
    # rec has [frequency,shot,receiver]. Original records include z phase.
    full=torch.zeros((cache.nfft//2+1,*rec.shape[1:]),dtype=torch.complex128)
    full=full.index_copy(0,torch.tensor(cache.selected),torch.tensor(cache.z)[:,None,None]*rec)
    predicted=torch.fft.irfft(full,n=cache.nfft,dim=0)[:len(wave)]
    predicted=predicted/torch.tensor(np.exp(-cache.eta*np.arange(len(wave))))[:,None,None]
    target=torch.tensor(residual.transpose(1,0,2))
    difference=predicted-target;loss=difference.square().mean()
    expected=torch.autograd.grad(loss,u)[0].numpy()
    actual=cache.adjoint_rhs(difference.detach().numpy().transpose(1,0,2))
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)


def test_corrected_spaces_preserve_sources_and_primal_span_without_global_numpy_randomness():
    before=np.random.get_state();fd,cache,v,wave,residual=cache_case();after=np.random.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])
    fields,_=cache.transport_primal(v.numpy()[0]*1.001,1)
    dual,_=cache.transport_adjoint(v.numpy()[0]*1.001,residual,1)
    qp=primal_space(cache,fields,6);q=enriched_space(cache,qp,dual,10)
    metric=cache.metric
    np.testing.assert_allclose(q.T@(metric@q),np.eye(10),atol=1e-8)
    np.testing.assert_allclose(q@(q.T@(metric@qp)),qp,atol=1e-8)
    points=np.eye(cache.system.dimension)[:,cache.sources]
    np.testing.assert_allclose(q@(q.T@(metric@points)),points,atol=1e-8)


def test_sparse_preprojection_preserves_rom_outputs_gradients_and_fixed_buffers():
    fd,cache,v,wave,residual=cache_case();q=primal_space(cache,cache.primal,12)
    pre=SparseProjector(cache.system,'cpu').project(q)
    a=PressureROM(cache.system,q,cache.sources,cache.receivers,len(wave),nfft=256,max_hz=160.)
    b=PressureROM(cache.system,q,cache.sources,cache.receivers,len(wave),nfft=256,max_hz=160.,preprojected=pre)
    for key in a.state_dict():torch.testing.assert_close(a.state_dict()[key],b.state_dict()[key],rtol=1e-10,atol=1e-12)
    assert not list(b.parameters())
    x=v.clone().requires_grad_(True);y=v.clone().requires_grad_(True);w=torch.tensor(wave)
    pa=a(x,w)[2];pb=b(y,w)[2]
    torch.testing.assert_close(pa,pb,rtol=1e-8,atol=1e-10)
    ga=torch.autograd.grad(pa.square().mean(),x)[0];gb=torch.autograd.grad(pb.square().mean(),y)[0]
    torch.testing.assert_close(ga,gb,rtol=1e-7,atol=1e-10)


def test_heldout_probe_and_sparse_operator_match_complete_native_pressure_space():
    fd,cache,v,wave,residual=cache_case();q=np.eye(cache.system.dimension)
    rom=PressureROM(cache.system,q,cache.sources,cache.receivers,len(wave),nfft=256,max_hz=160.)
    probe=galerkin_probe(cache,rom,q,v,residual,count=4)
    assert probe['primal']<1e-10 and probe['adjoint']<1e-10
    assert not set(probe['selected_bins'])&set(cache.selected)
    z=cache.z[3];field=cache.primal[3]
    matrix=cache.system.pressure_matrix(v.numpy()[0],z)
    np.testing.assert_allclose(apply_pressure(cache.system,v.numpy()[0],z,field),matrix@field,atol=1e-15)
    np.testing.assert_allclose(apply_pressure(cache.system,v.numpy()[0],z,field,True),matrix.conj().T@field,atol=1e-15)


def test_third_correction_improves_the_exact_discrete_residual():
    fd,cache,v,wave,residual=cache_case();velocity=v.numpy()[0]*1.003
    second,a=cache.transport_primal(velocity,2);third,b=cache.transport_primal(velocity,3)
    assert b['aggregate_relative_residual']<.1*a['aggregate_relative_residual']
    for k,z in enumerate(cache.z):
        rhs=cache.system.source_rhs(velocity,cache.sources)*cache.wave_spectrum[cache.selected[k]]
        np.testing.assert_allclose(rhs-cache.system.pressure_matrix(velocity,z)@third[k],b['residual_vectors'][k],atol=1e-16)


def test_independent_frequency_parallelism_preserves_cached_fields_and_corrections():
    fd,serial,v,wave,residual=cache_case()
    parallel=ReferenceCache(serial.system,v.numpy()[0],serial.sources,serial.receivers,wave,residual,
        training_frequencies=8,max_hz=160.,nfft=256,alias_tolerance=1e-12,workers=2)
    assert all(item['num_threads']==1 for item in parallel.info['worker_blas_context'] if item['user_api']=='blas')
    np.testing.assert_allclose(parallel.primal,serial.primal,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(parallel.adjoint,serial.adjoint,rtol=1e-12,atol=1e-12)
    for order in [1,2,3]:
        for method in ['transport_primal','transport_adjoint']:
            arguments=(v.numpy()[0]*1.003,order) if method=='transport_primal' else (v.numpy()[0]*1.003,residual*.97,order)
            a,sa=getattr(serial,method)(*arguments);b,sb=getattr(parallel,method)(*arguments)
            np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(sa['residual_vectors'],sb['residual_vectors'],rtol=1e-12,atol=1e-12)
