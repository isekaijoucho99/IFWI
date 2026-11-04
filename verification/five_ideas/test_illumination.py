"""No observed records or target velocity enter these sensitivity fixtures."""
import pytest
import torch


def native(**overrides):
    from ifwi_research.illumination import NativeSensitivity
    geometry=dict(nz=6,nx=8,xs=torch.tensor([[2,5,6]]),zs=torch.tensor([[1,1,1]]),
                  xr=torch.tensor([[[1,3,5,7]]]*3).reshape(1,3,4),zr=torch.ones(1,3,4,dtype=torch.long))
    kw=dict(geometry=geometry,wavelet=torch.sin(torch.arange(12)*.4),dt_s=.001,
            spacing_m=20.,npad=2,free_surface=True,pml_vmax_mps=2500.,shot_batch_size=2)
    kw.update(overrides)
    return NativeSensitivity(**kw)


def test_diagonal_linear_single_output_exact():
    # Squaring a VJP must give squared column norms, not the loss gradient.
    from ifwi_research.illumination import _squared_vjp
    x=torch.tensor([2.,7.,3.],requires_grad=True)
    y=(x*torch.tensor([2.,-3.,0.])).sum().reshape(1)
    d=_squared_vjp(y,x,4,torch.Generator().manual_seed(3))
    torch.testing.assert_close(d,torch.tensor([4.,9.,0.],dtype=torch.float64))
    assert x.grad is None


def test_native_repeatability_rng_accounting_and_bounded_map():
    estimator=native(); v=torch.full((1,6,8),2000.,requires_grad=True)
    state=torch.random.get_rng_state().clone()
    a=estimator.estimate(v,probes=2,seed=7)
    b=estimator.estimate(v,probes=2,seed=7)
    assert a['status']=='ok'
    assert torch.isfinite(a['raw_diagonal']).all() and a['raw_diagonal'].max()>0
    torch.testing.assert_close(a['raw_diagonal'],b['raw_diagonal'],rtol=0,atol=0)
    assert ((a['illumination']>=0)&(a['illumination']<=1)).all()
    assert not a['raw_diagonal'].requires_grad and v.grad is None
    assert torch.equal(state,torch.random.get_rng_state())
    assert a['diagnostics']['source_forward_solves']==3
    assert a['diagnostics']['source_backward_solves']==6
    assert a['diagnostics']['normalization_count']==144


def test_mask_zero_does_not_claim_weak_illumination():
    a=native().estimate(torch.full((1,6,8),2000.),data_mask=torch.zeros(1,3,12,4))
    assert a['status']=='no_sensitivity'
    assert a['raw_diagonal'].count_nonzero()==0


def test_mask_scaling_and_frequency_filter():
    e=native(); v=torch.full((1,6,8),2000.)
    a=e.estimate(v,cutoff=50.,seed=9)
    b=e.estimate(v,cutoff=50.,seed=9,data_mask=torch.full((1,3,12,4),.5))
    torch.testing.assert_close(b['raw_diagonal'],a['raw_diagonal']*.25)
    assert not torch.equal(a['raw_diagonal'],e.estimate(v,seed=9)['raw_diagonal'])


@pytest.mark.parametrize('kw',[{'probes':0},{'probes':9},{'cutoff':0.},{'cutoff':float('nan')},
                               {'data_mask':torch.ones(1,1,12,4)}])
def test_estimate_rejects_invalid_requests(kw):
    with pytest.raises(ValueError): native().estimate(torch.full((1,6,8),2000.),**kw)


@pytest.mark.parametrize('kw',[{'pml_vmax_mps':100000.},{'wavelet':torch.ones(257)},
                               {'shot_batch_size':0}])
def test_preflight_bounds(kw):
    with pytest.raises(ValueError): native(**kw)


def test_context_cannot_exceed_declared_physical_bound():
    with pytest.raises(ValueError): native().estimate(torch.full((1,6,8),2600.))


def test_dense_linear_vjp_matches_explicit_rademacher_average():
    # Dense rows catch accidental summation before squaring or shared signs.
    from ifwi_research.illumination import _squared_vjp
    a=torch.tensor([[1.,2.,0.],[-3.,4.,0.],[2.,-1.,0.]])
    x=torch.tensor([3.,-2.,5.],requires_grad=True)
    seed=23; probes=8
    rng=torch.Generator().manual_seed(seed)
    signs=torch.stack([torch.randint(0,2,(3,),generator=rng)*2-1 for _ in range(probes)]).float()
    expected=(signs@a).double().square().mean(0)
    actual=_squared_vjp(a@x,x,probes,torch.Generator().manual_seed(seed))
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_rnnfd_vjp_matches_small_explicit_jacobian():
    from ifwi_research.illumination import _squared_vjp
    from rnn_fd import rnn2D
    solver=rnn2D(nz=4,nx=4,zs=torch.tensor([[1]]),xs=torch.tensor([[1]]),
        zr=torch.tensor([[[1,2]]]),xr=torch.tensor([[[2,2]]]),dz=20.,dt=.001,
        npad=1,vmax=2500.,device='cpu')
    v=torch.full((1,4,4),2000.,requires_grad=True)
    wavelet=torch.tensor([1.,.5,0.])
    def prediction(field): return solver(field,wavelet)[2]
    jac=torch.autograd.functional.jacobian(prediction,v).reshape(6,16)
    rng=torch.Generator().manual_seed(12)
    signs=torch.stack([torch.randint(0,2,(1,1,3,2),generator=rng).flatten()*2-1 for _ in range(4)]).float()
    expected=(signs@jac).double().square().mean(0).reshape_as(v)
    actual=_squared_vjp(prediction(v),v,4,torch.Generator().manual_seed(12))
    torch.testing.assert_close(actual,expected,rtol=2e-5,atol=1e-18)


@pytest.mark.parametrize('kind',['samples','batch_graph'])
def test_resource_preflight_rejects_before_native_solver(monkeypatch,kind):
    # Distinct caps must reject excessive receiver storage or retained batch graph.
    import rnn_fd
    def forbidden_solver(*args,**kwargs):
        pytest.fail('Native solver invoked before resource preflight rejection')
    monkeypatch.setattr(rnn_fd,'rnn2D',forbidden_solver)
    if kind=='samples':
        geometry=dict(nz=6,nx=8,xs=torch.zeros(1,16,dtype=torch.long),zs=torch.ones(1,16,dtype=torch.long),
            xr=torch.zeros(1,16,4096,dtype=torch.long),zr=torch.ones(1,16,4096,dtype=torch.long))
        kw=dict(geometry=geometry,wavelet=torch.ones(16))
    else:
        geometry=dict(nz=64,nx=64,xs=torch.zeros(1,8,dtype=torch.long),zs=torch.ones(1,8,dtype=torch.long),
            xr=torch.zeros(1,8,1,dtype=torch.long),zr=torch.ones(1,8,1,dtype=torch.long))
        kw=dict(geometry=geometry,wavelet=torch.ones(64),shot_batch_size=8)
    with pytest.raises(ValueError,match='recorded samples|batch graph'):
        native(**kw)


def test_resource_diagnostics_report_exact_sizes():
    d=native().estimate(torch.full((1,6,8),2000.))['diagnostics']
    assert d['recorded_samples']==144
    assert d['max_batch_time_grid_entries']==2304  # 2 sources * 12 steps * 8 * 12
    assert d['max_recorded_samples']==1_000_000
    assert d['max_batch_time_grid_entries_limit']==2_000_000


def test_operator_identity_binds_geometry_wavelet_not_grouping():
    from ifwi_research.illumination import native_operator_identity
    geometry=dict(nz=6,nx=8,xs=torch.tensor([[2,5,6]]),zs=torch.ones(1,3,dtype=torch.long),
        xr=torch.tensor([[[1,3,5,7]]]*3).reshape(1,3,4),zr=torch.ones(1,3,4,dtype=torch.long))
    wavelet=torch.sin(torch.arange(12)*.4)
    saved={k:v.clone() if isinstance(v,torch.Tensor) else v for k,v in geometry.items()}
    saved_wavelet=wavelet.clone()
    kw=dict(geometry=geometry,wavelet=wavelet,dt_s=.001,spacing_m=20.,npad=2,
            free_surface=True,pml_vmax_mps=2500.)
    identity=native_operator_identity(**kw)
    assert len(identity)==64
    assert identity==native(shot_batch_size=1).operator_identity
    assert identity==native(shot_batch_size=3).operator_identity
    assert identity==native().estimate(torch.full((1,6,8),2000.))['operator_identity']
    assert identity==native_operator_identity(**{**kw,'wavelet':wavelet[None].double()})
    assert identity!=native_operator_identity(**{**kw,'wavelet':wavelet+.01})
    changed={**geometry,'xr':geometry['xr'].clone()}; changed['xr'][0,0,0]=2
    assert identity!=native_operator_identity(**{**kw,'geometry':changed})
    assert identity!=native_operator_identity(**{**kw,'pml_vmax_mps':2400.})
    assert torch.equal(wavelet,saved_wavelet)
    for k,v in geometry.items():
        assert torch.equal(v,saved[k]) if isinstance(v,torch.Tensor) else v==saved[k]


def test_masked_result_identity_cannot_match_unmasked_operator():
    estimator=native(); v=torch.full((1,6,8),2000.)
    base=estimator.operator_identity
    ones=torch.ones(1,3,12,4)
    assert estimator.estimate(v,data_mask=ones)['operator_identity']==base
    mask=ones.clone(); mask[0,0,0,0]=0.
    masked=estimator.estimate(v,data_mask=mask)['operator_identity']
    assert masked!=base
    assert estimator.operator_identity==base
    assert estimator.estimate(v,data_mask=mask.double())['operator_identity']==masked
    other=ones.clone(); other[0,0,0,1]=0.
    assert estimator.estimate(v,data_mask=other)['operator_identity']!=masked
