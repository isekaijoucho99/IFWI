"""Original FD integration: no alternative solver and no mocked physical gradient."""
import copy
import json
import numpy as np
import pytest
import torch

from verification.residual_ifwi.test_experiment import api, fields, small_config
from verification.residual_ifwi.test_paper_supplement import tiny_spec
from experiments.diffusion_prior.guidance import add_guidance
from experiments.diffusion_prior.io import save_base
from experiments.diffusion_prior.model import UNet


def make_checkpoint(tmp_path):
    cfg = dict(channels=32, mult=[1, 2], n_res=1, heads=4, attn_from=1)
    meta = dict(units='m/s', mean_mps=3000., scale_mps=1500., grid_spacing_m=15., patch_size=8)
    path = tmp_path/'prior.pt'
    # This is deliberately labeled untrained; opt-in below is required.
    save_base(path, UNet(**cfg), cfg, dict(steps=100, beta_start=1e-4, beta_end=.02), meta, updates=0)
    return path


def prior_cfg(path):
    return dict(checkpoint=str(path), allow_untrained=True, warmup=0, every=1,
                eta_start=.2, eta_end=.02, anneal_updates=5, patch_stride=4, patch_batch_size=2)


def test_disabled_update_identical_to_original_api():
    e = api()
    truth, initial = fields()
    cfg = small_config()
    problem = e.build_forward_problem(truth, cfg, 'cpu')
    torch.manual_seed(3)
    a = e.model_from_config(initial, cfg)
    b = copy.deepcopy(a)
    oa = torch.optim.Adam(a.parameters(), lr=1e-4)
    ob = torch.optim.Adam(b.parameters(), lr=1e-4)
    for update in range(1, 4):
        ra = e.train_update(a, problem, oa)
        rb = e.train_update(b, problem, ob, prior=None, update=update)
        assert ra['data_mse'] == rb['data_mse']
        assert all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))


def test_one_injection_after_all_shot_batches():
    e = api()
    truth, initial = fields()
    cfg = small_config()
    cfg['training']['shot_batch_size'] = 1
    problem = e.build_forward_problem(truth, cfg, 'cpu')
    m = e.model_from_config(initial, cfg)
    class CountingPrior:
        calls = 0
        def apply(self, model, update):
            self.calls += 1
            assert update == 1
            assert all(p.grad is not None for p in model.parameters())
            return dict(active=True, **add_guidance(model, torch.ones_like(model())*1e-7, ratio=.025))
    prior = CountingPrior()
    report = e.train_update(m, problem, torch.optim.Adam(m.parameters(), lr=1e-4), prior=prior, update=1)
    assert prior.calls == 1 and report['diffusion_prior']['active']
    assert report['diffusion_prior']['effective_ratio'] == pytest.approx(.025)


def test_prior_checkpoint_restore_replays_exactly(tmp_path):
    import core_runner
    path = make_checkpoint(tmp_path)
    spec = tiny_spec(tmp_path, 'diffusion_tiny', method='E3')
    spec['diffusion_prior'] = prior_cfg(path)
    a = core_runner.run(spec, tmp_path/'continuous', 'cpu', save_figures=False, shared_dir=tmp_path/'shared')
    b = core_runner.run(spec, tmp_path/'resumed', 'cpu', resume=a/'checkpoints/update_0001.pt',
                        save_figures=False, shared_dir=tmp_path/'shared')
    x = torch.load(a/'checkpoint.pt', weights_only=False, map_location='cpu')
    y = torch.load(b/'checkpoint.pt', weights_only=False, map_location='cpu')
    assert all(torch.equal(x['model'][k], y['model'][k]) for k in x['model'])
    assert len(x['history']) == len(y['history']) == 3
    assert x['provenance']['diffusion_prior']['base_sha256']
    for p, q in zip(x['history'], y['history']):
        assert p['data_mse'] == q['data_mse']
    # Replacing the prior at the same path must invalidate resume.
    ck = torch.load(path, weights_only=True)
    key = next(iter(ck['model']))
    ck['model'][key] += .01
    torch.save(ck, path)
    with pytest.raises(ValueError, match='identity mismatch'):
        core_runner.run(spec, tmp_path/'bad', 'cpu', resume=a/'checkpoints/update_0001.pt',
                        save_figures=False, shared_dir=tmp_path/'shared')


def test_unsupported_acceleration_is_not_silently_accepted(tmp_path):
    import core_runner
    spec = tiny_spec(tmp_path, 'prior_incompatible', method='E3')
    spec['diffusion_prior'] = dict(mode='resize_only', warmup=0, every=1)
    spec['speedup'] = {'source_encoding': {'supershots': 1, 'seed': 3}}
    with pytest.raises(ValueError, match='diffusion_prior'):
        core_runner.check_spec(spec)


def test_prior_sources_are_hashed():
    import core_runner
    hashes = core_runner.source_hashes()
    assert 'experiments/diffusion_prior/guidance.py' in hashes
    assert 'experiments/diffusion_prior/model.py' in hashes


def test_example_controls_keep_the_same_physical_experiment():
    from experiments.diffusion_prior.examples import build_examples
    import core_runner
    specs = build_examples('base.pt','adapter.pt','random.pt',updates=1000)
    fields = []
    for item in specs.values():
        core_runner.check_spec(item)
        cfg = copy.deepcopy(item['experiment'])
        cfg.pop('experiment_name')
        fields.append(cfg)
        assert 'target' not in item
    assert all(f == fields[0] for f in fields)
    assert 'fourier' not in fields[0]['model']
    assert fields[0]['training']['frequency_schedule']
