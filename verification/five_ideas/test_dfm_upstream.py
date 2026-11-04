"""DFM algorithm contracts against frozen official numerical reference outputs."""
from pathlib import Path
import json
import math
import numpy as np
import pytest
import torch
from torch import nn

SHA = '619663f0e7cbd17e58d98501daaf389d2810077b'
REFERENCE = Path(__file__).parent / 'data/dfm_upstream_reference.json'


def test_exact_resume_identity_includes_the_vendored_transport_core():
    import hashlib
    from ifwi_research.run import source_identity
    root=Path(__file__).resolve().parents[2]
    name='ifwi_research/_vendor/dfm_sinkhorn.py'
    assert source_identity(root).get(name)==hashlib.sha256((root/name).read_bytes()).hexdigest()


def test_drift_matches_pinned_official_sinkhorn_reference():
    from ifwi_research.dfm import drift_field
    ref = json.loads(REFERENCE.read_text())
    gen = torch.tensor(ref['gen'], dtype=torch.float64)
    pos = torch.tensor(ref['pos'], dtype=torch.float64)
    actual = drift_field(gen, pos, temperature=1.)
    torch.testing.assert_close(actual, torch.tensor(ref['drift'], dtype=torch.float64), rtol=1e-12, atol=1e-12)


def test_loss_and_stopped_gradient_match_official_reference():
    from ifwi_research.dfm import grouped_drift_loss
    ref = json.loads(REFERENCE.read_text())
    gen = torch.tensor(ref['gen'], dtype=torch.float64, requires_grad=True)
    pos = torch.tensor(ref['pos'], dtype=torch.float64, requires_grad=True)
    loss = grouped_drift_loss(gen, pos, temperature=1.)
    loss.backward()
    assert float(loss.detach()) == pytest.approx(ref['loss'], rel=1e-12)
    torch.testing.assert_close(gen.grad, torch.tensor(ref['gradient'], dtype=torch.float64), rtol=1e-12, atol=1e-12)
    assert pos.grad is None


def test_sinkhorn_balances_both_marginals_and_isolates_groups():
    from ifwi_research._vendor.dfm_sinkhorn import _sinkhorn_from_logits, compute_drift_grouped_sinkhorn
    logits = torch.tensor([[[0., -1., -3.], [-2., 0., -1.]]], dtype=torch.float64)
    rows = torch.tensor([.5, .5], dtype=torch.float64)
    cols = torch.tensor([.2, .3, .5], dtype=torch.float64)
    plan = _sinkhorn_from_logits(logits, row_marginals=rows, col_marginals=cols, num_iters=100)
    torch.testing.assert_close(plan.sum(-1)[0], rows, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(plan.sum(-2)[0], cols, rtol=1e-10, atol=1e-10)
    gen = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) / 10
    pos = gen.flip(1) + .4
    both = compute_drift_grouped_sinkhorn(gen, pos)
    for i in range(2):
        single = compute_drift_grouped_sinkhorn(gen[i:i+1], pos[i:i+1])
        torch.testing.assert_close(both[i], single[0])


def test_conditional_loss_uses_the_official_paired_endpoint_coupling():
    from ifwi_research.dfm import conditional_group_loss
    ref = json.loads(REFERENCE.read_text())
    class IdentityTransport(nn.Module):
        def __init__(self):
            super().__init__(); self.offset = nn.Parameter(torch.zeros((), dtype=torch.float64))
        def forward(self, x, t, r, condition):
            self.seen = (x.detach().clone(), condition.detach().clone())
            return x + self.offset
    model = IdentityTransport()
    clean = torch.tensor(ref['clean'], dtype=torch.float64).reshape(1, 1, 4, 4)
    loss = conditional_group_loss(model, clean, t=.2, r=.8, group_size=4,
                                  generator=torch.Generator().manual_seed(71), kernel=3)
    assert float(loss.detach()) == pytest.approx(ref['paired_loss'], rel=1e-12)
    expected_xt = torch.tensor(ref['paired_xt'], dtype=torch.float64).reshape(4, 1, 4, 4)
    torch.testing.assert_close(model.seen[0], expected_xt, rtol=1e-12, atol=1e-12)


def test_ordered_logit_normal_time_pairs_are_seeded_and_match_upstream_law():
    from ifwi_research.dfm import sample_time_pair
    rng = torch.Generator().manual_seed(22)
    expected_rng = torch.Generator().manual_seed(22)
    for _ in range(12):
        expected = [float(torch.sigmoid(torch.randn((), generator=expected_rng) * 2.5 - 1)),
                    float(torch.sigmoid(torch.randn((), generator=expected_rng) * 2.5 + 1))]
        assert sample_time_pair(rng) == tuple(sorted(expected))


def test_transport_sampler_preserves_one_condition_over_every_interval():
    from ifwi_research.dfm import sample_transport
    class ConstantVelocity(nn.Module):
        def __init__(self): super().__init__(); self.calls = []
        def forward(self, x, t, r, condition):
            self.calls.append((t, r, condition))
            return x + (r-t) * 2
    model = ConstantVelocity()
    x = torch.zeros(2, 1, 4, 4)
    condition = torch.ones(1, 1, 4, 4)
    actual = sample_transport(model, x, condition, nfe=4)
    torch.testing.assert_close(actual, x + 2)
    assert model.calls[0][0] == 0 and model.calls[-1][1] == 1
    assert all(c is condition for _, _, c in model.calls)


def test_legacy_prior_is_rejected_with_a_retraining_message(tmp_path):
    from ifwi_research.dfm import TwoTimeTransport, FrozenDFMPrior
    meta = dict(schema='ifwi_small_dfm_v1', trained_steps=20, width=8,
                spacing_m=15., mean_mps=3000., scale_mps=1500., condition_kernel=3, patch_size=8)
    path = tmp_path / 'legacy.pt'
    torch.save(dict(meta=meta, model=TwoTimeTransport(8).state_dict()), path)
    with pytest.raises(ValueError, match='[Rr]etrain|legacy|Legacy'):
        FrozenDFMPrior(path, 15.)


def test_corrected_training_marks_upstream_and_validates_generation(tmp_path):
    from ifwi_research.train_dfm import train
    from ifwi_research.dfm import FrozenDFMPrior
    td, vd = tmp_path/'train', tmp_path/'val'; td.mkdir(); vd.mkdir()
    z, x = np.mgrid[:8, :10]
    np.save(td/'train.npy', (1800 + 20*z + 3*x).astype('float32'))
    np.save(vd/'val.npy', (1900 + 22*z + 4*x).astype('float32'))
    out = tmp_path/'trained'
    train(td, vd, out, spacing_m=15., patch_size=4, width=8, group_size=4,
          groups_per_step=1, steps=2, validate_every=1)
    payload = torch.load(out/'best.pt', weights_only=True)
    assert payload['meta']['schema'] == 'ifwi_small_dfm_v2'
    assert payload['meta']['upstream_commit'] == SHA
    assert payload['meta']['inference_weights'] == 'ema'
    assert payload['meta']['drift_operator'] == 'grouped_sinkhorn_euclidean'
    last = json.loads((out/'history.jsonl').read_text().splitlines()[-1])
    assert set(last['generation_rmse_mps']) == {'1', '2', '4'}
    assert last['validation_start_t'] == 0
    prior = FrozenDFMPrior(out/'best.pt', 15.)
    assert prior.start_t == 0 and prior.nfe == 4


def test_default_prior_starts_at_noise_and_keeps_estimated_condition(tmp_path):
    from ifwi_research.dfm import TwoTimeTransport, FrozenDFMPrior, smooth
    meta = dict(schema='ifwi_small_dfm_v2', upstream_commit=SHA,
        drift_operator='grouped_sinkhorn_euclidean', trained_steps=1, width=8,
        spacing_m=15., mean_mps=3000., scale_mps=1500., condition_kernel=3, patch_size=8)
    path=tmp_path/'prior.pt'
    torch.save(dict(meta=meta,model=TwoTimeTransport(8).state_dict()),path)
    prior=FrozenDFMPrior(path,15.)
    class CaptureIdentity(nn.Module):
        def __init__(self): super().__init__(); self.seen=[]
        def forward(self,x,t,r,condition):
            self.seen.append((x.clone(),condition)); return x
    prior.model=CaptureIdentity()
    velocity=torch.full((1,8,8),2200.)
    actual=prior.proposal(velocity,1)
    rng=torch.Generator().manual_seed(3*1000003+9176)
    noise=torch.randn((1,1,8,8),generator=rng)
    torch.testing.assert_close(prior.model.seen[0][0],noise)
    torch.testing.assert_close(actual,(noise*1500+3000)[:,0])
    condition=prior.model.seen[0][1]
    torch.testing.assert_close(condition,smooth(((velocity-3000)/1500)[:,None],3))
    assert len(prior.model.seen)==4 and prior.calls==4
    assert all(c is condition for _,c in prior.model.seen)


@pytest.mark.parametrize('tau', [0, -1, math.nan, math.inf])
def test_bad_kernel_temperatures_fail_closed(tau):
    from ifwi_research.dfm import drift_field
    with pytest.raises(ValueError): drift_field(torch.zeros(3, 2), torch.ones(3, 2), tau)
