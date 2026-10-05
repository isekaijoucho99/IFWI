"""One-factor-at-a-time protocol, independent of training implementation."""
import copy
import numpy as np


def backward_shot_batches(model, wavelet, shots):
    """Exact full-data mean gradient (up to roundoff), one optimizer update outside."""
    geometry = {key: getattr(model.rnn, key) for key in ('xs', 'zs', 'xr', 'zr')}
    count = shots.shape[1]
    total_loss = None
    total_parts = {}
    try:
        for start in range(0, count, model.shot_batch_size):
            stop = min(start + model.shot_batch_size, count)
            for key, value in geometry.items():
                setattr(model.rnn, key, value[:, start:stop] if not isinstance(value, int) else value)
            v, loss, parts, handle = model.objective(wavelet, shots[:, start:stop])
            if handle is not None:
                handle.remove()
                raise ValueError('Shot accumulation does not support gradient hooks')
            weight = (stop - start) / count
            (loss * weight).backward()
            detached = loss.detach() * weight
            total_loss = detached if total_loss is None else total_loss + detached
            for key, value in parts.items():
                total_parts[key] = total_parts.get(key, 0) + value.detach() * weight
            before = v.detach()
            del v, loss, parts
        return before, total_loss, total_parts, None
    finally:
        for key, value in geometry.items():
            setattr(model.rnn, key, value)


def source_positions(nx, data):
    original = np.arange(20, nx - 10, 20, dtype=np.int64)
    if not len(original):
        raise ValueError('Grid too narrow for the original acquisition geometry')
    count = data.get('num_shots', len(original))
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= original[-1] - original[0] + 1:
        raise ValueError('num_shots must be an integer fitting the fixed acquisition aperture')
    positions = np.rint(np.linspace(original[0], original[-1], count)).astype(np.int64)
    if len(np.unique(positions)) != count:
        raise ValueError('Duplicate source positions')
    return positions


def validate_case(baseline, case, preliminary=False):
    """Reject any extra change, including width changes disguised as depth."""
    candidate = copy.deepcopy(case['config'])
    base = copy.deepcopy(baseline)
    for config in (base, candidate):
        config.pop('experiment_name', None)
        config.pop('description', None)
    factor, value = case['factor'], case['value']
    expected = copy.deepcopy(base)
    hidden = base['model']['neuron'][1:-1]
    if factor == 'shots':
        expected['data']['num_shots'] = value
    elif factor == 'depth':
        expected['model']['neuron'] = [2] + [hidden[0]] * value + [1]
    elif factor == 'width':
        expected['model']['neuron'] = [2] + [value] * len(hidden) + [1]
    elif factor == 'omega':
        expected['model']['omega_0'] = value
    elif factor != 'baseline':
        raise ValueError('Unknown treatment factor')
    if candidate != expected:
        raise ValueError('Confounded parameter experiment: ' + case['name'])
    if baseline.get('execution', {}).get('protocol') == 'original_baseline':
        from baseline_protocol import validate_baseline_protocol
        registered = {'baseline': (0,), 'shots': (25, 49), 'depth': (6, 8),
                      'width': (256, 512), 'omega': (10, 20, 50)}
        name = 'baseline' if factor == 'baseline' else f'{factor}_{value}'
        if (isinstance(value, bool) or not isinstance(value, int)
                or value not in registered.get(factor, ()) or case['name'] != name):
            raise ValueError('Unregistered original baseline treatment: ' + case['name'])
        validate_baseline_protocol(baseline, preliminary=preliminary)
        validate_baseline_protocol(case['config'], allow_treatment=True, preliminary=preliminary)


def make_cases(base, shot_batch_size=None, preliminary=False):
    if shot_batch_size is not None and (isinstance(shot_batch_size, bool) or not isinstance(shot_batch_size, int) or shot_batch_size < 1):
        raise ValueError('shot_batch_size must be a positive integer')
    base = copy.deepcopy(base)
    original = base.get('execution', {}).get('protocol') == 'original_baseline'
    if original:
        from baseline_protocol import validate_baseline_protocol
        if shot_batch_size is not None:
            raise ValueError('Original baseline requires full shot input; shot_batch_size must be null')
        validate_baseline_protocol(base, preliminary=preliminary)
    neurons = base['model']['neuron']
    width = neurons[1]
    if neurons != [2] + [width]*4 + [1] or width not in (128,256) or base['model']['omega_0'] != 30:
        raise ValueError('This registered sweep requires a 4x128 or 4x256 omega=30 baseline')
    if base['data'].get('downsample', 4) != 4 or base['data'].get('noise_level', 0) != 0:
        raise ValueError('This sweep requires the original downsample=4 noise-free data')
    base['data']['num_shots'] = 13
    base['training']['shot_batch_size'] = shot_batch_size
    base['experiment_name'] = 'sweep_baseline'
    base['description'] = f'Random-initialized 4x{width} SIREN baseline, omega=30; shared seed and initialization rule'
    cases = [dict(name='baseline', factor='baseline', value=0, config=base)]
    for factor, values in [('shots', [25, 49]), ('depth', [6, 8]),
                           ('width', [256,512] if width==128 else [384,512]), ('omega', [10, 20, 50])]:
        for value in values:
            config = copy.deepcopy(base)
            name = f'{factor}_{value}'
            config['experiment_name'] = 'sweep_' + name
            config['description'] = f'OFAT {factor}={value}; other settings match baseline'
            if factor == 'shots': config['data']['num_shots'] = value
            elif factor == 'depth': config['model']['neuron'] = [2] + [width] * value + [1]
            elif factor == 'width': config['model']['neuron'] = [2] + [value] * 4 + [1]
            else: config['model']['omega_0'] = value
            case = dict(name=name, factor=factor, value=value, config=config)
            validate_case(base, case, preliminary=preliminary)
            cases.append(case)
    return cases
