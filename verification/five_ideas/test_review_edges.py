from pathlib import Path
import importlib.util
import hashlib,json
import pytest
import torch
from torch import nn


def test_gradient_identity_bypasses_reanchoring_roundoff():
    from ifwi_research.gradient_attention import GradientAttentionVelocity
    class Field(nn.Module):
        def __init__(self): super().__init__(); self.v=nn.Parameter(torch.full((1,4,5),1e8))
        def forward(self): return self.v
    base=Field(); wrapped=GradientAttentionVelocity(base,base().detach(),strength=0.)
    with torch.no_grad(): base.v.fill_(3.)
    # A + (3 - A) in float32 loses the small update when A is very large.
    assert torch.equal(wrapped(),base())


def test_prior_metadata_rejects_nonfinite_normalization(tmp_path):
    from ifwi_research.dfm import TwoTimeTransport,FrozenDFMPrior
    model=TwoTimeTransport(8)
    meta=dict(schema='ifwi_small_dfm_v1',trained_steps=1,width=8,spacing_m=20.,mean_mps=float('nan'),
              scale_mps=1000.,condition_kernel=3,patch_size=8,toy_only=False)
    path=tmp_path/'bad.pt'; torch.save({'meta':meta,'model':model.state_dict()},path)
    with pytest.raises(ValueError,match='normalization'): FrozenDFMPrior(path,20.)


def test_prior_penalty_disallows_broadcasting(tmp_path):
    from ifwi_research.dfm import TwoTimeTransport,FrozenDFMPrior
    meta=dict(schema='ifwi_small_dfm_v1',trained_steps=1,width=8,spacing_m=20.,mean_mps=2000.,
              scale_mps=1000.,condition_kernel=3,patch_size=8,toy_only=False)
    path=tmp_path/'test.pt'; torch.save({'meta':meta,'model':TwoTimeTransport(8).state_dict()},path)
    prior=FrozenDFMPrior(path,20.)
    with pytest.raises(ValueError,match='grid'): prior.penalty(torch.ones(1,8,8),torch.ones(1,1,8))


def test_publish_rejects_destination_symlink(tmp_path):
    spec=importlib.util.spec_from_file_location('pub',Path(__file__).resolve().parents[2]/'tools'/'publish_five_ideas.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    package=tmp_path/'pkg'; package.mkdir(); (package/'folder').mkdir(); (package/'folder'/'a.py').write_text('x=1\n')
    work=tmp_path/'work'; work.mkdir(); outside=tmp_path/'outside'; outside.mkdir()
    try: (work/'folder').symlink_to(outside,target_is_directory=True)
    except OSError: pytest.skip('OS does not permit creating a directory symlink')
    sha=hashlib.sha256((package/'folder'/'a.py').read_bytes()).hexdigest()
    (package/'DELIVERY_MANIFEST.json').write_text(json.dumps({'files':[{'path':'folder/a.py','sha256':sha}]}))
    with pytest.raises(ValueError,match='path'): module.validate_delivery(package,work)
