"""Contract-only fixtures: synthetic declarations do NOT certify trained quality."""
import hashlib
import importlib
import torch
import pytest


def module():
    return importlib.import_module('ifwi_research.conditional_dfm')


def checkpoint(tmp_path, **changes):
    m = module()
    model = m.MaskedTwoTimeTransport(width=4)
    report = tmp_path / 'contract_only_report.txt'
    report.write_text('SYNTHETIC CONTRACT TEST ONLY. No training or scientific qualification.')
    meta = dict(schema='ifwi_masked_conditional_dfm_v1', architecture='masked_two_time_cnn_v1',
                width=4, mean_mps=2500., scale_mps=100., spacing_m=10., units='m/s',
                training=dict(steps=1, toy=False, contract='shared_background_estimate_anchors_mask_v1',
                              time_semantics='gaussian_0_to_geology_1_two_time_transport',
                              anchor_semantics='estimated_not_truth',
                              train_model_hashes=['a'*64], val_model_hashes=['b'*64]),
                qualification=dict(status='qualified', report_ref=str(report),
                                   report_sha256=hashlib.sha256(report.read_bytes()).hexdigest()))
    meta.update(changes)
    path=tmp_path/'model.pt'
    torch.save(dict(meta=meta, state_dict=model.state_dict()), path)
    return path


def test_transport_has_independent_linear_time_expectation():
    model=module().MaskedTwoTimeTransport(width=4)
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        model.net[-1].bias.fill_(2.)
    x=torch.ones(3,1,2,2)
    c=torch.zeros(1,3,2,2)
    assert torch.equal(model(x, .25, .75, c), torch.full_like(x,2.))
    state=x
    for t,r in [(0., .2),(.2,.7),(.7,1.)]: state=model(state,t,r,c)
    torch.testing.assert_close(state, torch.full_like(x,3.))
    with pytest.raises(ValueError): model(x,.8,.2,c)
    with pytest.raises(ValueError): model(x,0.,1.,c.expand(3,-1,-1,-1))


def test_frozen_sampler_is_seeded_gaussian_not_corrupted_estimate(tmp_path):
    m=module(); adapter=m.FrozenConditionalDFM(checkpoint(tmp_path),10.,nfe=3,seed=3)
    bg=torch.full((1,3,4),2500.,requires_grad=True)
    anchors=torch.full_like(bg,3000.); mask=torch.ones_like(bg)
    rng_before=torch.random.get_rng_state().clone()
    result=adapter.generate(bg,anchors,mask,sample_id=7)
    assert torch.equal(torch.random.get_rng_state(),rng_before)
    expected=2500+100*torch.randn((1,1,3,4), generator=torch.Generator().manual_seed((3*1000003+7)%2**63))[:,0]
    torch.testing.assert_close(result,expected)
    assert not result.requires_grad and adapter.calls==3
    assert not adapter.model.training and all(not p.requires_grad for p in adapter.model.parameters())
    assert torch.equal(result,adapter.generate(bg+100, anchors,mask,sample_id=7))
    assert adapter.calls==6


@pytest.mark.parametrize('change', [dict(schema='legacy'),dict(units='km/s'),dict(training={}),dict(qualification={}),dict(width=100000)])
def test_rejects_incompatible_or_unqualified(tmp_path,change):
    with pytest.raises((ValueError,KeyError)):
        module().FrozenConditionalDFM(checkpoint(tmp_path,**change),10.)


def test_rejects_overlap_and_changed_report(tmp_path):
    path=checkpoint(tmp_path); data=torch.load(path,weights_only=True)
    data['meta']['training']['val_model_hashes']=['a'*64]; torch.save(data,path)
    with pytest.raises(ValueError,match='disjoint'): module().FrozenConditionalDFM(path,10.)
    path=checkpoint(tmp_path); data=torch.load(path,weights_only=True); data['meta']['qualification']['report_sha256']=''; torch.save(data,path)
    with pytest.raises(ValueError,match='report'): module().FrozenConditionalDFM(path,10.)


def test_resource_check_precedes_sampling(tmp_path,monkeypatch):
    adapter=module().FrozenConditionalDFM(checkpoint(tmp_path),10.,max_cells=4)
    def forbidden(*a,**kw): raise AssertionError('allocated noise before guard')
    monkeypatch.setattr(torch,'randn',forbidden)
    with pytest.raises(ValueError,match='cells'): adapter.generate(torch.ones(1,3,3)*2500,torch.ones(1,3,3)*2500,torch.zeros(1,3,3),sample_id=0)
    assert adapter.calls==0


def test_units_mask_and_spacing_fail_closed(tmp_path):
    m=module(); path=checkpoint(tmp_path)
    with pytest.raises(ValueError,match='spacing'): m.FrozenConditionalDFM(path,20.)
    a=m.FrozenConditionalDFM(path,10.); v=torch.full((1,2,2),2500.)
    for bg,mask in [(v/1000,torch.ones_like(v)),(v,torch.full_like(v,1.1)),(v*float('nan'),torch.ones_like(v))]:
        with pytest.raises(ValueError): a.generate(bg,v,mask,sample_id=0)


def test_condition_uses_detached_estimates_and_binary_mask(tmp_path):
    adapter=module().FrozenConditionalDFM(checkpoint(tmp_path),10.)
    bg=torch.full((1,2,2),2600.,requires_grad=True)
    estimate=torch.full((1,2,2),2800.,requires_grad=True)
    mask=torch.tensor([[[1.,0.],[0.,1.]]])
    seen=[]
    hook=adapter.model.register_forward_pre_hook(lambda model,args: seen.append(args[3].clone()))
    adapter.generate(bg,estimate,mask,sample_id=1)
    hook.remove()
    expected=torch.stack((torch.ones(2,2),3*mask[0],mask[0]))[None]
    torch.testing.assert_close(seen[0],expected)
    assert not seen[0].requires_grad
    assert bg.grad is None and estimate.grad is None
    assert torch.equal(bg,torch.full_like(bg,2600.))


@pytest.mark.parametrize('field,value',[('steps',0),('toy',True),('time_semantics','denoise_current_estimate'),('anchor_semantics','true_known_pixels')])
def test_training_declaration_rejects_unsafe_contract(tmp_path,field,value):
    path=checkpoint(tmp_path); data=torch.load(path,weights_only=True)
    data['meta']['training'][field]=value; torch.save(data,path)
    with pytest.raises(ValueError): module().FrozenConditionalDFM(path,10.)


def test_loader_preserves_global_rng(tmp_path):
    path=checkpoint(tmp_path)
    before=torch.random.get_rng_state().clone()
    module().FrozenConditionalDFM(path,10.)
    assert torch.equal(before,torch.random.get_rng_state())


@pytest.mark.parametrize('shape',[(1,1),(3,5),(12,13)])
def test_multiscale_identity_transport_and_gradients(shape):
    m=module(); model=m.make_transport('masked_two_time_unet_v1',width=4)
    x=torch.randn(2,1,*shape,requires_grad=True); c=torch.zeros(1,3,*shape)
    torch.testing.assert_close(model(x,0.,1.,c),x)
    with torch.no_grad(): model.output.bias.fill_(2.)
    result=model(x,.25,.75,c)
    torch.testing.assert_close(result,x+1.)
    result.sum().backward()
    torch.testing.assert_close(x.grad,torch.ones_like(x))
    assert any(p.grad is not None for p in model.parameters())


def test_multiscale_factory_and_checkpoint_roundtrip(tmp_path):
    m=module(); path=checkpoint(tmp_path)
    model=m.make_transport('masked_two_time_unet_v1',width=4)
    data=torch.load(path,weights_only=True)
    data['meta']['architecture']='masked_two_time_unet_v1'
    data['state_dict']=model.state_dict(); torch.save(data,path)
    adapter=m.FrozenConditionalDFM(path,10.)
    v=torch.full((1,5,7),2500.)
    result=adapter.generate(v,v,torch.ones_like(v),sample_id=1)
    assert result.shape==v.shape and torch.isfinite(result).all()
    with pytest.raises(ValueError): m.make_transport('unknown',width=4)


def test_multiscale_zero_interval_and_strict_architecture_state(tmp_path):
    m=module(); model=m.make_transport('masked_two_time_unet_v1',width=4)
    x=torch.randn(1,1,3,5); c=torch.zeros(1,3,3,5)
    assert model(x,.4,.4,c) is x
    with pytest.raises(ValueError): model(x,.5,.2,c)
    with pytest.raises(ValueError): model(x,0.,1.,torch.zeros(1,1,3,5))
    path=checkpoint(tmp_path); data=torch.load(path,weights_only=True)
    data['meta']['architecture']='masked_two_time_unet_v1'; torch.save(data,path)
    with pytest.raises(ValueError,match='model state'): m.FrozenConditionalDFM(path,10.)
    with pytest.raises(ValueError,match='width'): m.make_transport('masked_two_time_unet_v1',width=100000)


def test_inference_identity_binds_seed_and_time_grid(tmp_path):
    m=module(); path=checkpoint(tmp_path)
    a=m.FrozenConditionalDFM(path,10.,seed=3,nfe=1)
    same=m.FrozenConditionalDFM(path,10.,seed=3,nfe=1)
    seed=m.FrozenConditionalDFM(path,10.,seed=4,nfe=1)
    steps=m.FrozenConditionalDFM(path,10.,seed=3,nfe=2)
    assert a.inference_identity==same.inference_identity
    assert a.inference_identity!=seed.inference_identity
    assert a.inference_identity!=steps.inference_identity
    old=a.inference_identity; a.nfe=2
    assert a.inference_identity!=old
