"""Offline data split, checkpoint, and LoRA adaptation smoke tests."""
import numpy as np
import pytest
import torch


def arrays(root):
    for split, offset in [('train',0),('val',1)]:
        d = root/split
        d.mkdir(parents=True)
        for i in range(2):
            arr = np.linspace(1800+offset*50+i, 4000+i*100, 12*13, dtype=np.float32).reshape(12,13)
            np.save(d/f'{i}.npy', arr)
    return root/'train', root/'val'


def options(root):
    train_dir, val_dir = arrays(root/'data')
    return dict(train_dir=train_dir, val_dir=val_dir, steps=3, batch_size=2,
                eval_every=1, val_batches=1, seed=9, device='cpu',
                meta=dict(units='m/s',mean_mps=3000.,scale_mps=1500.,grid_spacing_m=15.,patch_size=8),
                model_config=dict(channels=32,mult=[1,2],n_res=1,heads=4,attn_from=1),
                schedule_config=dict(steps=20,beta_start=1e-4,beta_end=.02))


def test_duplicate_models_across_splits_rejected(tmp_path):
    from experiments.diffusion_prior.training import train
    cfg = options(tmp_path)
    np.save(cfg['val_dir']/'different_name.npy', np.load(cfg['train_dir']/'0.npy'))
    with pytest.raises(ValueError, match='overlap'):
        train(out=tmp_path/'out', **cfg)


def test_offline_resume_matches_uninterrupted(tmp_path):
    from experiments.diffusion_prior.training import train
    cfg = options(tmp_path)
    a = train(out=tmp_path/'full', **cfg)
    part = {**cfg,'steps':1}
    train(out=tmp_path/'part', **part)
    b = train(out=tmp_path/'resume', resume=tmp_path/'part'/'last_training.pt', **cfg)
    x = torch.load(tmp_path/'full'/'last_training.pt', weights_only=False)
    y = torch.load(tmp_path/'resume'/'last_training.pt', weights_only=False)
    assert all(torch.equal(x['model'][k], y['model'][k]) for k in x['model'])
    assert x['history'] == y['history']
    assert a.exists() and b.exists()


def test_lora_training_exports_compact_adapter_without_changing_base(tmp_path):
    from experiments.diffusion_prior.training import train
    from experiments.diffusion_prior.io import file_hash, load_prior
    cfg = options(tmp_path)
    base = train(out=tmp_path/'base', **{**cfg, 'steps':1})
    h = file_hash(base)
    adapted = train(out=tmp_path/'adapt', base=base, rank=2, alpha=2., **{**cfg,'steps':2})
    assert file_hash(base) == h
    ck = torch.load(adapted, weights_only=True)
    assert ck['updates'] > 0 and ck['adapter']
    assert all('.lora_' in k for k in ck['adapter'])
    net, _, _, _ = load_prior(base, 'cpu', 15., adapter=adapted)
    assert not any(p.requires_grad for p in net.parameters())


def test_generator_split_and_physical_metadata(tmp_path):
    from experiments.diffusion_prior.training import generate_dataset, VelocityDataset
    meta = dict(units='m/s',mean_mps=3000.,scale_mps=1500.,grid_spacing_m=15.,patch_size=8)
    generate_dataset(tmp_path/'data', train_models=2, val_models=1, shape=(16,24), grid_spacing_m=15., seed=4, profile='folded')
    d = VelocityDataset(tmp_path/'data'/'train', meta)
    assert len(d.records) == 2
    batch = d.sample(2, np.random.default_rng(1))
    assert batch.shape == (2,1,8,8) and torch.isfinite(batch).all()
    with pytest.raises(ValueError, match='spacing'):
        VelocityDataset(tmp_path/'data'/'train', {**meta, 'grid_spacing_m':20.})
