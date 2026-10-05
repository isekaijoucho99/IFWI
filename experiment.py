"""Run one original IFWI experiment: baseline by default, configurable parameters."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'experiments'))
from baseline_protocol import baseline_config, validate_baseline_protocol, validate_original_sources

PRESETS = {'baseline': {}}
for factor, values in [('shots', (25, 49)), ('depth', (6, 8)),
                       ('width', (256, 512)), ('omega', (10, 20, 50))]:
    PRESETS.update({f'{factor}_{value}': {factor: value} for value in values})


def build_run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', choices=PRESETS, default='baseline')
    parser.add_argument('-shots', '--shots', type=int, help='Shot count, 2..241; baseline 13')
    parser.add_argument('-depth', '--depth', type=int, help='Hidden layer count; baseline 4')
    parser.add_argument('-width', '--width', type=int, help='Neurons in each hidden layer; baseline 128')
    parser.add_argument('-omega', '--omega', type=float, help='Positive SIREN omega; baseline 30')
    parser.add_argument('--epochs', type=int, help='Total updates: baseline 4001 (labels 0..4000)')
    parser.add_argument('--log-interval', type=int, help='Best/checkpoint interval; baseline 100')
    parser.add_argument('--device', help='Default: cuda:0 if available, otherwise cpu')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/single_experiments')
    parser.add_argument('--resume', type=Path, help='Original numbered checkpoint; repeat its parameter settings')
    parser.add_argument('--dry-run', action='store_true', help='Print settings without training or writing files')
    parser.add_argument('--list-presets', action='store_true', help='List presets without training')
    args = parser.parse_args(argv)
    config = baseline_config()
    baseline_values = dict(shots=13, depth=4, width=128, omega=30)
    values = dict(baseline_values)
    values.update(PRESETS[args.preset])
    for factor in values:
        override = getattr(args, factor)
        if override is not None:
            values[factor] = override
    omega = values['omega']
    if isinstance(omega, float) and omega.is_integer():
        values['omega'] = int(omega)
    config['data']['num_shots'] = values['shots']
    config['model']['neuron'] = [2] + [values['width']] * max(values['depth'], 0) + [1]
    config['model']['omega_0'] = values['omega']
    if args.epochs is not None:
        config['training']['max_iterations'] = args.epochs
    if args.log_interval is not None:
        config['training']['log_interval'] = args.log_interval
    args.preliminary = (config['training']['max_iterations'] != 4001
                        or config['training']['log_interval'] != 100)
    changed = [(factor, value) for factor, value in values.items()
               if value != baseline_values[factor]]
    name = '_'.join(f'{factor}_{value}' for factor, value in changed) if changed else 'baseline'
    config['experiment_name'] = name
    config['description'] = 'Original random IFWI baseline' if not changed else f'Parameters {name}; original common settings'
    try:
        # A nonpositive depth must not be hidden by constructing a baseline shape.
        if values['depth'] < 1:
            raise ValueError('Hidden layer count must be positive')
        validate_baseline_protocol(config, allow_treatment=True, preliminary=args.preliminary, allow_custom=True)
    except ValueError as error:
        parser.error(str(error))
    return args, config


def main(argv=None):
    args, config = build_run(argv)
    if args.list_presets:
        for name, values in PRESETS.items():
            print(f"{name:12s} " + (', '.join(f'{key}={value}' for key, value in values.items()) or '13 shots, 4x128, omega=30'))
        return 0
    validate_original_sources(ROOT)
    if args.dry_run:
        print(json.dumps(dict(config=config, preliminary=args.preliminary,
                              registered_preset=config['experiment_name'] in PRESETS,
                              device=args.device or 'auto', output_dir=str(args.output_dir)), indent=2))
        return 0
    import torch
    from baseline_experiment import run_config
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
    run_config(config, args.output_dir, device=device, seed=config['seed'], resume=args.resume,
               preliminary=args.preliminary, allow_custom=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
