"""Focused new tests; the historical unpublished 148-test suite is not included."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))
CONFIGS = ROOT / 'experiments' / 'configs'
SCRIPT = ROOT / 'scripts' / 'compare_lr_control.py'


def configs():
    paths = [CONFIGS / ('modern13_' + name + '.yaml') for name in ('constant_lr', 'cosine_lr')]
    for path in paths:
        assert path.is_file(), f'Missing approved LR configuration: {path.name}'
    return [yaml.safe_load(path.read_text()) for path in paths]


def audit_module():
    assert SCRIPT.is_file(), 'Missing read-only paired LR audit script'
    spec = importlib.util.spec_from_file_location('compare_lr_control', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configs_are_exact_single_factor_pair():
    constant, cosine = configs()
    assert constant['seed'] == cosine['seed'] == 3
    assert constant['model']['neuron'] == [2, 128, 128, 128, 128, 1]
    assert constant['training'] == {'max_iterations': 4001, 'log_interval': 100, 'alpha': 0,
                                    'clip_grad': None, 'shot_batch_size': None}
    assert constant['optimizer']['use_scheduler'] is False
    assert cosine['optimizer']['use_scheduler'] is True
    for item in (constant, cosine):
        assert item.get('execution', {}).get('protocol') != 'original_baseline'
        assert item['optimizer']['scheduler_params'] == {'warmup_epochs': 0, 'max_epochs': 4001, 'eta_min': 1e-5}
        item.pop('experiment_name'); item.pop('description')
        item['optimizer'].pop('use_scheduler')
    assert constant == cosine


def test_audit_accepts_only_the_approved_pair():
    module = audit_module()
    constant, cosine = configs()
    report = module.validate_pair(constant, cosine)
    assert report['only_treatment'] == 'optimizer.use_scheduler'
    assert report['upstream_commit'] == '22fa517bc9c9217e9c08d47c585a56a916702a91'
    assert len(report['constant_config_sha256']) == 64
    for path, value in [(('seed',), 42), (('training', 'clip_grad'), .25),
                        (('model', 'bias'), False), (('loss', 'use_prior'), True),
                        (('optimizer', 'optimizer_type'), 'adamw')]:
        changed = copy.deepcopy(cosine)
        at = changed
        for key in path[:-1]: at = at[key]
        at[path[-1]] = value
        with pytest.raises(ValueError, match='configuration'):
            module.validate_pair(constant, changed)
    same_but_wrong = copy.deepcopy(constant)
    same_but_wrong['seed'] = 42
    also_wrong = copy.deepcopy(cosine)
    also_wrong['seed'] = 42
    with pytest.raises(ValueError, match='configuration'):
        module.validate_pair(same_but_wrong, also_wrong)


def test_existing_scheduler_endpoints_monotonic_and_resume():
    from improved_modules.optimizers import create_improved_optimizer
    _, cosine = configs()
    network = torch.nn.Linear(2, 1)
    optimizer, scheduler, _ = create_improved_optimizer(network, cosine['optimizer'])
    rates = []
    for step in range(4001):
        scheduler.step(step); rates.append(optimizer.param_groups[0]['lr'])
    assert rates[0] == pytest.approx(1e-4)
    assert rates[2000] == pytest.approx(5.5e-5)
    assert rates[-1] == pytest.approx(1e-5)
    assert all(a >= b for a, b in zip(rates, rates[1:]))
    with pytest.raises(ValueError, match='horizon'): scheduler.step(4001)
    for split in (1, 100, 3500):
        scheduler.step(split - 1)
        saved = scheduler.state_dict()
        other_opt, other, _ = create_improved_optimizer(torch.nn.Linear(2, 1), cosine['optimizer'])
        other.load_state_dict(saved)
        assert other_opt.param_groups[0]['lr'] == rates[split - 1]
        other.step(split)
        assert other_opt.param_groups[0]['lr'] == rates[split]


def test_constant_optimizer_matches_original_adam():
    from improved_modules.optimizers import create_improved_optimizer
    constant, _ = configs()
    torch.manual_seed(3)
    a = torch.nn.Linear(2, 1)
    b = copy.deepcopy(a)
    opt_a, scheduler, _ = create_improved_optimizer(a, constant['optimizer'])
    opt_b = torch.optim.Adam(b.parameters(), lr=1e-4)
    assert scheduler is None
    x = torch.tensor([[1., 2.], [-2., 1.]])
    for _ in range(4):
        for net, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad(); net(x).square().mean().backward(); opt.step()
    assert all(torch.equal(a.state_dict()[key], value) for key, value in b.state_dict().items())


def test_compare_function_exists_before_report_implementation():
    assert callable(getattr(audit_module(), 'compare_runs', None)), 'Missing validated read-only run comparison'
