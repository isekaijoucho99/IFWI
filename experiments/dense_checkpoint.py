"""Isolated complete dense-run snapshots for Adam and local-age MaskedAdam.

The encoding checkpoint schema stays unchanged. Publication reuses its bounded,
atomic, non-overwriting hard-link protocol and keeps the prepared payload.
"""
import copy
import os
from pathlib import Path
import random
import uuid

import numpy as np
import torch

from experiments.encoding_checkpoint import (
    CheckpointPublishError, _capture_rng, _cpu_tree, _exact, _finite,
    _metadata_text, _parameter_names, _publish_once_prepared, _require,
    _validate_rng,
)
from ifwi_research.masked_adam import MaskedAdam

SCHEMA = 'ifwi_dense_checkpoint_v1'
_KEYS = {'schema', 'optimizer_type', 'completed', 'metadata', 'model',
         'optimizer', 'optimizer_parameter_names', 'rng'}


def _optimizer_type(optimizer):
    if type(optimizer) is torch.optim.Adam:
        return 'Adam'
    if type(optimizer) is MaskedAdam:
        return 'MaskedAdam'
    raise ValueError('Dense checkpoint requires Adam or MaskedAdam')


def _validate(payload, model, optimizer, metadata):
    """Validate the entire saved contract before any target or RNG mutation."""
    try:
        _require(type(payload) is dict and set(payload) == _KEYS, 'Checkpoint keys differ')
        _require(payload['schema'] == SCHEMA, 'Wrong dense checkpoint schema')
        kind = _optimizer_type(optimizer)
        _require(payload['optimizer_type'] == kind, 'Optimizer type differs')
        completed = payload['completed']
        _require(type(completed) is int and 0 <= completed <= torch.iinfo(torch.int64).max,
                 'completed must be a nonnegative int64 integer')
        _require(_metadata_text(payload['metadata']) == _metadata_text(metadata),
                 'Checkpoint metadata differs')

        saved, current = payload['model'], model.state_dict()
        _require(isinstance(saved, dict) and set(saved) == set(current), 'Model state keys differ')
        _finite(saved, 'model')
        for name, value in current.items():
            other = saved[name]
            _require(isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor)
                     and value.shape == other.shape and value.dtype == other.dtype
                     and value.layout == other.layout == torch.strided,
                     'Model state shape/dtype/layout differs: ' + name)
            if name == 'completed_updates' or name.endswith('.completed_updates'):
                _require(other.ndim == 0 and other.dtype == torch.int64
                         and int(other.item()) == completed,
                         'Gated progress buffer differs from completed')

        saved_optimizer = payload['optimizer']
        _require(type(saved_optimizer) is dict and set(saved_optimizer) == {'state', 'param_groups'},
                 'Optimizer state keys differ')
        _finite(saved_optimizer, 'optimizer')
        groups, states = saved_optimizer['param_groups'], saved_optimizer['state']
        target_groups = optimizer.state_dict()['param_groups']
        names = payload['optimizer_parameter_names']
        _require(_exact(names, _parameter_names(model, optimizer)),
                 'Optimizer parameter order/names differ')
        _require(type(groups) is list and len(groups) == len(target_groups)
                 and type(states) is dict, 'Optimizer groups/state differ')
        identifiers = []
        for group, target_group, group_names in zip(groups, target_groups, names):
            _require(type(group) is dict and set(group) == set(target_group),
                     'Optimizer group keys differ')
            # This also checks IDs: otherwise load_state_dict can renumber IDs
            # and a post-load mismatch would be discovered after mutation.
            _require(_exact(group, _cpu_tree(target_group)),
                     'Optimizer group hyperparameters/parameter IDs differ')
            _require(type(group['params']) is list
                     and all(type(identifier) is int for identifier in group['params']),
                     'Invalid optimizer parameter IDs')
            identifiers.extend(group['params'])
        _require(len(identifiers) == len(set(identifiers)), 'Duplicate optimizer parameter IDs')
        _require(all(type(identifier) is int for identifier in states), 'Invalid optimizer state IDs')
        _require((completed == 0 and not states) or set(states) == set(identifiers),
                 'Incomplete optimizer state')
        for group, group_names in zip(groups, names):
            for identifier, name in zip(group['params'], group_names):
                if identifier not in states:
                    continue
                state, parameter = states[identifier], saved[name]
                expected = {'exp_avg', 'exp_avg_sq', 'age' if kind == 'MaskedAdam' else 'step'}
                if kind == 'Adam' and group['amsgrad']:
                    expected.add('max_exp_avg_sq')
                _require(type(state) is dict and set(state) == expected,
                         'Optimizer per-parameter state keys differ: ' + name)
                for key in expected - {'step', 'age'}:
                    moment = state[key]
                    _require(isinstance(moment, torch.Tensor) and moment.shape == parameter.shape
                             and moment.dtype == parameter.dtype and moment.layout == torch.strided,
                             'Optimizer moment shape/dtype/layout differs: ' + name + '.' + key)
                    if key != 'exp_avg':
                        _require(bool((moment >= 0).all()), 'Negative optimizer variance: ' + name)
                if kind == 'MaskedAdam':
                    age = state['age']
                    _require(isinstance(age, torch.Tensor) and age.shape == parameter.shape
                             and age.dtype == torch.int64 and age.layout == torch.strided,
                             'MaskedAdam age shape/dtype/layout differs: ' + name)
                    _require(bool(((age >= 0) & (age <= completed)).all()),
                             'MaskedAdam age outside completed range: ' + name)
                    zero = age == 0
                    _require(bool((state['exp_avg'][zero] == 0).all())
                             and bool((state['exp_avg_sq'][zero] == 0).all()),
                             'Uninitialized local age has nonzero moments: ' + name)
                else:
                    step = state['step']
                    if isinstance(step, torch.Tensor):
                        _require(step.ndim == 0 and not step.is_complex()
                                 and step.dtype != torch.bool, 'Adam step must be a real scalar')
                        step = step.item()
                    _require(type(step) in (int, float) and step == completed,
                             'Adam step differs from completed')

        rng = payload['rng']
        _validate_rng(rng)
        _require(set(rng['numpy']) == {'bit_generator', 'keys', 'pos', 'has_gauss', 'cached_gaussian'},
                 'NumPy RNG keys differ')
        _require(torch.cuda.is_initialized() == rng['cuda_initialized'],
                 'CUDA initialization differs from saved run')
        if rng['cuda_initialized']:
            _require(torch.cuda.device_count() == len(rng['torch_cuda']), 'CUDA RNG device count differs')
            # Byte dtype/shape alone does not establish a valid CUDA RNG state.
            # Use isolated generators so rejection precedes every global restore.
            for device_index, state in enumerate(rng['torch_cuda']):
                torch.Generator(device=f'cuda:{device_index}').set_state(state.cpu())
        return completed
    except (KeyError, TypeError, RuntimeError, ValueError, AttributeError) as error:
        raise ValueError('Invalid complete dense checkpoint: ' + str(error)) from error


def save_dense_checkpoint(model, optimizer, path, completed, metadata) -> Path:
    """Serialize once, fsync, then atomically publish without overwriting."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    kind = _optimizer_type(optimizer)
    _metadata_text(metadata)
    payload = dict(schema=SCHEMA, optimizer_type=kind, completed=completed,
                   metadata=copy.deepcopy(metadata), model=_cpu_tree(model.state_dict()),
                   optimizer=_cpu_tree(optimizer.state_dict()),
                   optimizer_parameter_names=_parameter_names(model, optimizer), rng=_capture_rng())
    _validate(payload, model, optimizer, metadata)
    source = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.prepared')
    with source.open('xb') as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    _publish_once_prepared(source, path)
    return path


def load_dense_checkpoint(model, optimizer, path, metadata) -> int:
    """Prevalidate and restore full model, optimizer, and initialized RNG state."""
    _metadata_text(metadata)
    _optimizer_type(optimizer)
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    completed = _validate(payload, model, optimizer, metadata)
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    _require(_exact(model.state_dict(), payload['model'])
             and _exact(optimizer.state_dict(), payload['optimizer']), 'Loaded model/optimizer differs')
    rng = payload['rng']
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n['bit_generator'], np.asarray(n['keys'], dtype=np.uint32),
                         n['pos'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(rng['torch_cpu'].cpu())
    if rng['cuda_initialized']:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng['torch_cuda']])
    _require(_exact(_capture_rng(), rng), 'Restored RNG differs')
    return completed
