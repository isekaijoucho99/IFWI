"""Fixed protocol and file identities of the verified original random IFWI run.

This module deliberately imports no training code. Launchers can reject drift
before constructing a model, writing a suite, or dispatching a GPU process.
"""
import copy
import hashlib
import math
from pathlib import Path
from types import MappingProxyType


ORIGINAL_PROTOCOL = 'original_baseline'
ORIGINAL_WRAPPER_REFERENCE = 'experiments/baseline_reference/ifwi_experiment.py'

# Audited against IFWI 1/11479806 and its overnight_20260920_215223/random run.
# These identities must not be recomputed from a caller's source snapshot.
ORIGINAL_SOURCE_HASHES = MappingProxyType({
    'ifwi_modules.py': '900adab18822769946a64109fa77314d71d396403e70179b815094cd952bb043',
    'rnn_fd.py': 'eeeebd821462fadb52bb9b12a06932f2c9f764e3e7482b316e12a94d26ec9fd0',
    'generator.py': 'c37e49c631a2f654808455b92a8125604d7dbf46b6dac7c63a17d736ef79f69c',
    'plot_functions.py': '53d70c622cc2e4fef19aded39835c73e4feca06ddaa6e6d40ff043bdc337737b',
    'data/vel_marmousi_376x1151.csv': '9b79c401fb161d0edb58a668b572cac7de29f18e1d141f08caee9b55377cfb4d',
    ORIGINAL_WRAPPER_REFERENCE: '748137c90c47bbd9e032cc70f07d25b4bc027a30580511ff74e2079134728459',
})


def baseline_config() -> dict:
    """Return a fresh canonical profile; the author train/save flow stays intact."""
    return {
        'experiment_name': 'legacy_random_baseline',
        'description': 'Verified original random IFWI',
        'seed': 3,
        'execution': {'protocol': ORIGINAL_PROTOCOL},
        'model': {'network_type': 'vanilla', 'neuron': [2, 128, 128, 128, 128, 1],
                  'omega_0': 30, 'activation': 'sine', 'dropout': False,
                  'prob': .2, 'outermost_linear': True, 'bias': True},
        'loss': {'type': 'standard_mse', 'use_prior': False},
        'optimizer': {'optimizer_type': 'adam', 'learning_rate': .0001,
                      'use_scheduler': False},
        'training': {'max_iterations': 4001, 'log_interval': 100, 'alpha': 0,
                     'clip_grad': None, 'shot_batch_size': None},
        'data': {'model_file': 'vel_marmousi_376x1151.csv', 'downsample': 4,
                 'noise_level': 0., 'dz': 15, 'dt': .0019, 'nt': 1000,
                 'num_shots': 13},
        'evaluation': {'depth_threshold': .5, 'corner_size': .25, 'save_plots': True},
        'gradient_preconditioner': {'enabled': False},
    }


def _comparison_profile(config):
    result = copy.deepcopy(config)
    result.pop('experiment_name', None)
    result.pop('description', None)
    evaluation = result.get('evaluation')
    if isinstance(evaluation, dict):
        evaluation.pop('save_plots', None)
    return result


def _first_difference(expected, actual, path='config'):
    """Find a concrete error path, including unknown settings and bool/int traps."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return path
        for key in expected:
            location = path + '.' + key
            if key not in actual:
                return location
            difference = _first_difference(expected[key], actual[key], location)
            if difference:
                return difference
        for key in actual:
            if key not in expected:
                return path + '.' + str(key)
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return path
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _first_difference(left, right, f'{path}[{index}]')
            if difference:
                return difference
        return None
    if isinstance(expected, bool) or isinstance(actual, bool):
        if type(actual) is not type(expected) or actual != expected:
            return path
    elif isinstance(expected, int) and not isinstance(actual, int):
        return path
    elif expected != actual:
        return path
    return None


def validate_baseline_protocol(config: dict, allow_treatment: bool = False,
                               preliminary: bool = False, allow_custom: bool = False) -> None:
    """Require the original profile, optionally with one registered OFAT change.

    Preliminary validation permits only positive integer update/log budgets.
    It does not relax seed, data, solver, objective, optimizer, or batching.
    Custom values require both allow flags and may combine the supported parameters.
    """
    if not isinstance(config, dict):
        raise ValueError('Original baseline protocol requires a config mapping')
    training = config.get('training')
    if not isinstance(training, dict):
        raise ValueError('Original baseline protocol mismatch: config.training')
    for key in ('max_iterations', 'log_interval'):
        value = training.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Original baseline protocol requires a positive integer: training.' + key)
    expected = _comparison_profile(baseline_config())
    if preliminary:
        for key in ('max_iterations', 'log_interval'):
            expected['training'][key] = training[key]
    actual = _comparison_profile(config)
    candidates = [expected]
    if allow_treatment and allow_custom:
        data, model = actual.get('data'), actual.get('model')
        if not isinstance(data, dict) or not isinstance(model, dict):
            raise ValueError('Original baseline protocol requires data and model mappings')
        shots = data.get('num_shots')
        if not isinstance(shots, int) or isinstance(shots, bool) or not 2 <= shots <= 241:
            raise ValueError('Custom num_shots must be an integer between 2 and 241')
        neurons = model.get('neuron')
        if (not isinstance(neurons, list) or len(neurons) < 3
                or not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in neurons)
                or neurons[0] != 2 or neurons[-1] != 1 or len(set(neurons[1:-1])) != 1):
            raise ValueError('Custom network must have input 2, positive uniform hidden widths, and output 1')
        omega = model.get('omega_0')
        if (not isinstance(omega, (int, float)) or isinstance(omega, bool)
                or omega <= 0 or not math.isfinite(omega)):
            raise ValueError('Custom omega_0 must be finite and positive')
        expected['data']['num_shots'] = shots
        expected['model']['neuron'] = neurons
        expected['model']['omega_0'] = omega
    elif allow_treatment:
        treatments = [
            ('data', 'num_shots', 25), ('data', 'num_shots', 49),
            ('model', 'neuron', [2] + [128] * 6 + [1]),
            ('model', 'neuron', [2] + [128] * 8 + [1]),
            ('model', 'neuron', [2] + [256] * 4 + [1]),
            ('model', 'neuron', [2] + [512] * 4 + [1]),
            ('model', 'omega_0', 10), ('model', 'omega_0', 20),
            ('model', 'omega_0', 50),
        ]
        for section, key, value in treatments:
            candidate = copy.deepcopy(expected)
            candidate[section][key] = value
            candidates.append(candidate)
    if any(_first_difference(candidate, actual) is None for candidate in candidates):
        return
    difference = _first_difference(expected, actual)
    qualifier = ''
    if allow_treatment and not allow_custom:
        qualifier = ' (only one registered OFAT parameter may change)'
    raise ValueError('Original baseline protocol mismatch: ' + difference + qualifier)


def validate_original_sources(root: Path) -> dict:
    """Reject changed or missing author code, runner reference, and truth CSV."""
    root = Path(root).resolve()
    hashes = {}
    for name, expected in ORIGINAL_SOURCE_HASHES.items():
        try:
            actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError('Original baseline source missing or unreadable: ' + name) from error
        if actual != expected:
            raise ValueError('Original baseline source differs: ' + name)
        hashes[name] = actual
    return hashes
