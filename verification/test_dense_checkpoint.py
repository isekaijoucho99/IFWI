"""Dense checkpoint compatibility, rejection atomicity, and exact CPU resume."""
import copy
import importlib.util
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(1)
from experiments import encoding_checkpoint as legacy

TARGET = ROOT / 'experiments/dense_checkpoint.py'
checkpoint = None
if TARGET.exists():
    spec = importlib.util.spec_from_file_location('dense_checkpoint_target', TARGET)
    checkpoint = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checkpoint)


@pytest.fixture(autouse=True)
def require_module():
    assert checkpoint is not None, 'dense_checkpoint.py not implemented'


class SmallProgressive(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, dtype=torch.float64)
        self.register_buffer('completed_updates', torch.tensor(0, dtype=torch.int64))

    def forward(self, x):
        return self.linear(x)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def metadata():
    return {'protocol': 'dense-test-v1', 'mode': 'gated', 'sources': 48, 'seed': 12}


def step(model, optimizer):
    x = torch.randn(4, 3, dtype=torch.float64) + random.random()
    y = torch.from_numpy(np.random.randn(4, 2)).double()
    optimizer.zero_grad(set_to_none=True)
    (model(x) - y).square().mean().backward()
    optimizer.step()
    model.completed_updates.add_(1)


def initialized():
    seed(12)
    model = SmallProgressive()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=.02)
    step(model, optimizer)
    return model, optimizer


def raw_payload(model, optimizer, kind='Adam'):
    return dict(schema=checkpoint.SCHEMA, optimizer_type=kind, completed=1,
                metadata=metadata(), model=copy.deepcopy(model.state_dict()),
                optimizer=copy.deepcopy(optimizer.state_dict()),
                optimizer_parameter_names=[['linear.weight', 'linear.bias']],
                rng=legacy._capture_rng())


def snapshot(model, optimizer):
    return (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()), legacy._capture_rng())


def assert_unchanged(model, optimizer, before):
    assert legacy._exact(model.state_dict(), before[0])
    assert legacy._exact(optimizer.state_dict(), before[1])
    assert legacy._exact(legacy._capture_rng(), before[2])


def test_adam_resume_reproduces_model_optimizer_and_all_cpu_rng(tmp_path):
    model, optimizer = initialized()
    path = tmp_path / 'adam.pt'
    before_rng = legacy._capture_rng()
    assert checkpoint.save_dense_checkpoint(model, optimizer, path, 1, metadata()) == path
    assert legacy._exact(legacy._capture_rng(), before_rng)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['schema'] != legacy.SCHEMA
    step(model, optimizer)
    expected = snapshot(model, optimizer)
    expected_draws = (random.random(), np.random.randn(3), torch.randn(3))
    seed(99)
    resumed = SmallProgressive()
    resumed_optimizer = torch.optim.Adam(resumed.parameters(), lr=1e-3, weight_decay=.02)
    assert checkpoint.load_dense_checkpoint(resumed, resumed_optimizer, path, metadata()) == 1
    step(resumed, resumed_optimizer)
    assert_unchanged(resumed, resumed_optimizer, expected)
    assert random.random() == expected_draws[0]
    assert np.array_equal(np.random.randn(3), expected_draws[1])
    assert torch.equal(torch.randn(3), expected_draws[2])


def test_zero_checkpoint_restores_empty_adam_state(tmp_path):
    model = SmallProgressive()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = checkpoint.save_dense_checkpoint(model, optimizer, tmp_path / 'zero.pt', 0, metadata())
    step(model, optimizer)
    assert checkpoint.load_dense_checkpoint(model, optimizer, path, metadata()) == 0
    assert optimizer.state_dict()['state'] == {}
    assert int(model.completed_updates) == 0


@pytest.mark.parametrize('damage', ['schema', 'metadata', 'top_extra', 'model_extra', 'optimizer_extra',
    'state_extra', 'group_extra', 'names', 'group_ids', 'moment_shape', 'moment_dtype',
    'negative_variance', 'moment_nan', 'step', 'buffer', 'rng', 'optimizer_type'])
def test_malformed_adam_rejected_before_any_mutation_without_publication(tmp_path, damage):
    source, source_optimizer = initialized()
    payload = raw_payload(source, source_optimizer)
    state = next(iter(payload['optimizer']['state'].values()))
    if damage == 'schema': payload['schema'] = legacy.SCHEMA
    elif damage == 'metadata': payload['metadata']['sources'] = 52
    elif damage == 'top_extra': payload['surprise'] = True
    elif damage == 'model_extra': payload['model']['surprise'] = torch.tensor(1)
    elif damage == 'optimizer_extra': payload['optimizer']['surprise'] = True
    elif damage == 'state_extra': state['surprise'] = torch.tensor(1)
    elif damage == 'group_extra': payload['optimizer']['param_groups'][0]['surprise'] = True
    elif damage == 'names': payload['optimizer_parameter_names'][0].reverse()
    elif damage == 'group_ids': payload['optimizer']['param_groups'][0]['params'][1] = 0
    elif damage == 'moment_shape': state['exp_avg'] = torch.zeros(7, dtype=torch.float64)
    elif damage == 'moment_dtype': state['exp_avg_sq'] = state['exp_avg_sq'].float()
    elif damage == 'negative_variance': state['exp_avg_sq'].view(-1)[0] = -1
    elif damage == 'moment_nan': state['exp_avg'].view(-1)[0] = float('nan')
    elif damage == 'step': state['step'].fill_(2)
    elif damage == 'buffer': payload['model']['completed_updates'].fill_(2)
    elif damage == 'rng': payload['rng']['torch_cpu'] = torch.zeros(3, dtype=torch.uint8)
    else: payload['optimizer_type'] = 'MaskedAdam'
    path = tmp_path / 'bad.pt'
    torch.save(payload, path)
    seed(37)
    target = SmallProgressive()
    target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-3, weight_decay=.02)
    before = snapshot(target, target_optimizer)
    with pytest.raises(ValueError):
        checkpoint.load_dense_checkpoint(target, target_optimizer, path, metadata())
    assert_unchanged(target, target_optimizer, before)


@pytest.mark.parametrize('option,value', [('lr', .3), ('weight_decay', .1), ('betas', (.8, .9)),
                                          ('eps', .01), ('amsgrad', True)])
def test_group_hyperparameter_mismatch_rejected_before_mutation_without_publication(tmp_path, option, value):
    source, source_optimizer = initialized()
    path = tmp_path / 'source.pt'
    torch.save(raw_payload(source, source_optimizer), path)
    options = dict(lr=1e-3, weight_decay=.02)
    options[option] = value
    target = SmallProgressive()
    target_optimizer = torch.optim.Adam(target.parameters(), **options)
    before = snapshot(target, target_optimizer)
    with pytest.raises(ValueError):
        checkpoint.load_dense_checkpoint(target, target_optimizer, path, metadata())
    assert_unchanged(target, target_optimizer, before)


def test_cpu_checkpoint_does_not_initialize_cuda(tmp_path, monkeypatch):
    model, optimizer = initialized()
    assert not torch.cuda.is_initialized()
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU checkpoint attempted CUDA access')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', forbidden)
    monkeypatch.setattr(torch.cuda, 'device_count', forbidden)
    path = checkpoint.save_dense_checkpoint(model, optimizer, tmp_path / 'cpu.pt', 1, metadata())
    assert checkpoint.load_dense_checkpoint(model, optimizer, path, metadata()) == 1
    assert not torch.cuda.is_initialized()


def test_publication_retries_same_prepared_payload_without_reserialization(tmp_path, monkeypatch):
    model, optimizer = initialized()
    real_link, real_save = legacy.os.link, torch.save
    links, saves = [], []
    def transient_link(source, target):
        links.append((source, target))
        if len(links) < 3:
            error = PermissionError('synthetic transient publication')
            error.winerror = 5
            raise error
        return real_link(source, target)
    def counted_save(*args, **kwargs):
        saves.append(True)
        return real_save(*args, **kwargs)
    monkeypatch.setattr(legacy.os, 'link', transient_link)
    monkeypatch.setattr(legacy.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(torch, 'save', counted_save)
    path = checkpoint.save_dense_checkpoint(model, optimizer, tmp_path / 'retry.pt', 1, metadata())
    assert path.is_file()
    assert len(links) == 3 and len(set(links)) == 1 and len(saves) == 1
    assert Path(links[0][0]).is_file()


def test_existing_target_never_overwritten_without_publication(tmp_path):
    model, optimizer = initialized()
    target = tmp_path / 'existing.pt'
    target.write_bytes(b'user evidence')
    with pytest.raises(FileExistsError):
        checkpoint.save_dense_checkpoint(model, optimizer, target, 1, metadata())
    assert target.read_bytes() == b'user evidence'


@pytest.mark.parametrize('damage', ['completed', 'step', 'metadata'])
def test_invalid_save_does_not_create_prepared_file_without_publication(tmp_path, damage):
    model, optimizer = initialized()
    completed, meta = 1, metadata()
    if damage == 'completed': completed = 2
    elif damage == 'step': next(iter(optimizer.state.values()))['step'].fill_(2)
    else: meta['bad'] = (1, 2)
    before = snapshot(model, optimizer)
    with pytest.raises(ValueError):
        checkpoint.save_dense_checkpoint(model, optimizer, tmp_path / 'invalid.pt', completed, meta)
    assert not list(tmp_path.iterdir())
    assert_unchanged(model, optimizer, before)


def gated_pair():
    from experiments.gated_siren import GatedSirenVelocity
    from ifwi_research.masked_adam import MaskedAdam
    model = GatedSirenVelocity(torch.full((3, 4), 3000.), dx_m=15., warmup_updates=4)
    return model, MaskedAdam(model.parameters(), lr=.001, weight_decay=.02)


def gated_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    prediction = model()
    target = torch.randn_like(prediction) + float(np.random.randn()) + random.random()
    (prediction - target).square().mean().backward()
    optimizer.step(masks=model.optimizer_masks())
    model.set_progress(int(model.completed_updates) + 1)


def test_gated_resume_reproduces_local_ages_unlock_and_all_rng(tmp_path):
    seed(12)
    model, optimizer = gated_pair()
    gated_step(model, optimizer)
    assert any(bool((state['age'] == 0).any()) for state in optimizer.state.values())
    path = checkpoint.save_dense_checkpoint(model, optimizer, tmp_path / 'gated.pt', 1, metadata())
    for _ in range(4):
        gated_step(model, optimizer)
    expected = snapshot(model, optimizer)
    seed(67)
    resumed, resumed_optimizer = gated_pair()
    assert checkpoint.load_dense_checkpoint(resumed, resumed_optimizer, path, metadata()) == 1
    for _ in range(4):
        gated_step(resumed, resumed_optimizer)
    assert_unchanged(resumed, resumed_optimizer, expected)
    assert resumed.active_channels == 128
    assert all(state['age'].dtype == torch.int64 for state in resumed_optimizer.state.values())


@pytest.mark.parametrize('damage', ['age_dtype', 'age_shape', 'age_negative', 'age_future',
    'missing_age', 'moment_shape', 'moment_dtype', 'negative_variance', 'state_extra', 'optimizer_type'])
def test_malformed_masked_state_rejected_before_any_mutation_without_publication(tmp_path, damage):
    from ifwi_research.masked_adam import MaskedAdam
    seed(12)
    source = SmallProgressive()
    optimizer = MaskedAdam(source.parameters(), lr=.001, weight_decay=.02)
    step(source, optimizer)
    payload = raw_payload(source, optimizer, kind='MaskedAdam')
    state = next(iter(payload['optimizer']['state'].values()))
    if damage == 'age_dtype': state['age'] = state['age'].double()
    elif damage == 'age_shape': state['age'] = torch.ones(7, dtype=torch.int64)
    elif damage == 'age_negative': state['age'].view(-1)[0] = -1
    elif damage == 'age_future': state['age'].view(-1)[0] = 2
    elif damage == 'missing_age': del state['age']
    elif damage == 'moment_shape': state['exp_avg'] = torch.zeros(7, dtype=torch.float64)
    elif damage == 'moment_dtype': state['exp_avg_sq'] = state['exp_avg_sq'].float()
    elif damage == 'negative_variance': state['exp_avg_sq'].view(-1)[0] = -1
    elif damage == 'state_extra': state['step'] = torch.tensor(1.)
    else: payload['optimizer_type'] = 'Adam'
    path = tmp_path / 'bad-masked.pt'
    torch.save(payload, path)
    target = SmallProgressive()
    target_optimizer = MaskedAdam(target.parameters(), lr=.001, weight_decay=.02)
    before = snapshot(target, target_optimizer)
    with pytest.raises(ValueError):
        checkpoint.load_dense_checkpoint(target, target_optimizer, path, metadata())
    assert_unchanged(target, target_optimizer, before)


def test_invalid_cuda_rng_bytes_rejected_before_any_mutation_without_publication(tmp_path, monkeypatch):
    # CUDA is simulated at its external boundary: the real CPU generator still
    # validates CPU RNG, while isolated CUDA generators reject malformed bytes.
    source, source_optimizer = initialized()
    payload = raw_payload(source, source_optimizer)
    valid = torch.zeros(16, dtype=torch.uint8)
    invalid = torch.zeros(1, dtype=torch.uint8)
    payload['rng']['cuda_initialized'] = True
    payload['rng']['torch_cuda'] = [valid.clone(), invalid]
    path = tmp_path / 'invalid-cuda-rng.pt'
    torch.save(payload, path)

    seed(99)
    target = SmallProgressive()
    target_optimizer = torch.optim.Adam(target.parameters(), lr=1e-3, weight_decay=.02)
    actual_generator = torch.Generator
    validated_devices = []
    class IsolatedCudaGenerator:
        def __init__(self, device):
            self.device = str(device)
        def set_state(self, state):
            validated_devices.append(self.device)
            if not torch.equal(state, valid):
                raise RuntimeError('malformed CUDA RNG bytes')
    def generator(device='cpu'):
        if str(device).startswith('cuda:'):
            return IsolatedCudaGenerator(device)
        return actual_generator(device=device)
    def global_cuda_restore(states):
        # Model the late failure of the global CUDA setter without touching GPU.
        if any(not torch.equal(state, valid) for state in states):
            raise ValueError('malformed CUDA RNG bytes')
    def forbidden(*args, **kwargs):
        raise AssertionError('CPU test attempted real CUDA initialization')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: True)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: 2)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [valid.clone(), valid.clone()])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', global_cuda_restore)
    monkeypatch.setattr(torch, 'Generator', generator)
    before = snapshot(target, target_optimizer)
    with pytest.raises(ValueError, match='malformed CUDA RNG bytes'):
        checkpoint.load_dense_checkpoint(target, target_optimizer, path, metadata())
    assert_unchanged(target, target_optimizer, before)
    assert validated_devices == ['cuda:0', 'cuda:1']
