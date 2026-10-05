"""Import the completed legacy IFWI trajectory; do not perform optimizer updates."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import contextlib
import sys

import numpy as np
import torch
from run_experiment import load_config, prepare_data, build_model, evaluate_model_metrics, save_plot
from experiment_runtime import write_json, velocity, comparison_contract, source_hashes

ROOT = Path(__file__).resolve().parents[1]


def validate_legacy_config(legacy, config, completed_updates):
    strict = config.get('execution', {}).get('protocol') == 'original_baseline'
    if strict:
        from baseline_protocol import validate_baseline_protocol
        from baseline_experiment import _legacy_config
        validate_baseline_protocol(config, allow_treatment=False)
        canonical = _legacy_config(config, config['seed'])
        for key, value in canonical.items():
            if key not in legacy or type(legacy[key]) is not type(value) or legacy[key] != value:
                # Author JSON may use 0 instead of 0.0 for a numeric setting.
                if (key not in legacy or isinstance(legacy[key], bool) or
                    not isinstance(value, (int, float)) or not isinstance(legacy[key], (int, float)) or
                    legacy[key] != value):
                    raise ValueError('Legacy baseline mismatch: ' + key)
        if legacy.get('backend', 'reference') != 'reference':
            raise ValueError('Legacy baseline mismatch: backend')
        if legacy.get('gradient_clip') is not None:
            raise ValueError('Legacy baseline mismatch: gradient_clip')
        unknown = set(legacy) - set(canonical) - {'backend', 'gradient_clip'}
        if unknown:
            raise ValueError('Legacy baseline unknown configuration field: ' + sorted(unknown)[0])
    expected = dict(mode='random', seed=config['seed'], mean=3., std=1., dz=config['data']['dz'],
        dt=config['data']['dt'], nt=config['data']['nt'], frequency=8,
        source_depth_index=1, receiver_depth_index=2,
        learning_rate=config['optimizer']['learning_rate'], alpha=0, noise=0, dropout=0, pretrained=None)
    for key, value in expected.items():
        if legacy.get(key) != value: raise ValueError('Legacy baseline mismatch: '+key)
    if legacy['epochs'] != completed_updates or completed_updates != config['training']['max_iterations']:
        raise ValueError('Legacy and treatment update budgets must match, including zero-based epoch offset')
    mc=config['model']
    if mc['neuron'] != [2,128,128,128,128,1] or mc['omega_0'] != 30 or mc.get('activation') != 'sine' or not mc.get('outermost_linear') or mc.get('dropout'):
        raise ValueError('Legacy checkpoint requires the original 4x128 omega=30 SIREN')
    if config['training'].get('clip_grad') is not None or config['training'].get('alpha',0) or config['loss'].get('use_prior') or config['optimizer'].get('use_scheduler') or config['data'].get('noise_level',0):
        raise ValueError('Legacy import requires the unmodified deterministic MSE protocol')


def import_original_baseline(legacy, config, output_dir, device='cpu'):
    """Verify and copy the author run using its original model and saved losses."""
    from baseline_protocol import ORIGINAL_SOURCE_HASHES, validate_original_sources
    from baseline_experiment import (prepare_original_data, build_original_model,
                                     _velocity, _spatial_metrics, LiveLog, plot_result)
    legacy = Path(legacy).resolve()
    metadata = json.loads((legacy / 'config.json').read_text(encoding='utf-8-sig'))
    validate_legacy_config(metadata, config, metadata['epochs'])
    pinned_sources = validate_original_sources(ROOT)
    author_sources = legacy / 'author_sources'
    if not author_sources.exists():
        author_sources = legacy.parents[2]
    for name in ('ifwi_modules.py', 'rnn_fd.py', 'generator.py'):
        if hashlib.sha256((author_sources / name).read_bytes()).hexdigest() != ORIGINAL_SOURCE_HASHES[name]:
            raise ValueError('Legacy solver/network source differs: ' + name)
    epochs = metadata['epochs']
    checkpoint_path = legacy / 'checkpoints' / f'MarmousiI_random-checkpoint-{epochs}.pth'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = {'epoch', 'state_dict', 'optimizer', 'train_loss', 'best_loss',
                'best_loss_epoch', 'best_loss_model'}
    if 'schema' in checkpoint or not required.issubset(checkpoint):
        raise ValueError('Strict import requires an original author checkpoint')
    validate_legacy_config(metadata, config, checkpoint['epoch'])
    if isinstance(checkpoint['epoch'], bool) or len(checkpoint['train_loss']) != epochs:
        raise ValueError('Original checkpoint history must contain all completed updates')
    best_epoch = checkpoint['best_loss_epoch']
    if (isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or
        not 0 <= best_epoch < epochs or best_epoch % metadata['log_interval'] != 0):
        raise ValueError('Original best epoch must be an author logging checkpoint')
    expected_updates = list(range(1, epochs + 1, metadata['log_interval']))
    if expected_updates[-1] != epochs:
        expected_updates.append(epochs)
    snapshots = [legacy / 'checkpoints' / f'MarmousiI_random-checkpoint-{update}.pth'
                 for update in expected_updates]
    missing = [path.name for path in snapshots if not path.is_file()]
    if missing:
        raise ValueError('Original baseline snapshot missing: ' + missing[0])
    history = np.loadtxt(legacy / 'loss.csv', delimiter=',', skiprows=1, ndmin=2)
    if history.shape != (epochs, 4):
        raise ValueError('Original loss CSV must contain one three-loss row per completed update')
    np.testing.assert_array_equal(history[:, 0], np.arange(1, epochs + 1))
    np.testing.assert_array_equal(history[:, 1:], np.asarray(checkpoint['train_loss']))
    if not np.isfinite(history).all():
        raise ValueError('Original loss history contains nonfinite values')
    if checkpoint['best_loss'] != history[best_epoch, 1]:
        raise ValueError('Original best loss differs from its preupdate history row')
    out = Path(output_dir) / 'legacy_baseline_import'
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'config.json', config)
    shutil.copy2(legacy / 'config.json', out / 'baseline_config.json')
    write_json(out / 'status.json', dict(state='preparing', completed_updates=epochs))
    with (out / 'progress.log').open('w', encoding='utf8', buffering=1) as log:
        with contextlib.redirect_stdout(LiveLog(sys.stdout, log)):
            try:
                # Match the original seed -> data/forward -> model order.
                torch.manual_seed(config['seed'])
                torch.cuda.manual_seed_all(config['seed'])
                np.random.seed(config['seed'])
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                data = prepare_original_data(config, device)
                truth = data['vp_true'].squeeze().cpu().numpy()
                np.testing.assert_array_equal(truth, np.load(legacy / 'true_velocity.npy').squeeze())
                model = build_original_model(config, data, device)
                initial = _velocity(model)
                original_initial = np.load(legacy / 'initial_velocity.npy').squeeze()
                np.testing.assert_allclose(initial, original_initial, rtol=1e-6, atol=.01)
                model.vel_net.load_state_dict(checkpoint['state_dict'])
                final = _velocity(model)
                model.vel_net.load_state_dict(checkpoint['best_loss_model'])
                best = _velocity(model)
                original_best = np.load(legacy / 'best_velocity.npy').squeeze()
                np.testing.assert_allclose(best, original_best, rtol=1e-5, atol=.2)
                (out / 'checkpoints').mkdir()
                for path in snapshots:
                    shutil.copy2(path, out / 'checkpoints' / path.name)
                for name in ('initial_velocity.npy', 'best_velocity.npy', 'true_velocity.npy', 'loss.csv'):
                    shutil.copy2(legacy / name, out / name)
                if (legacy / 'progress.log').is_file():
                    shutil.copy2(legacy / 'progress.log', out / 'legacy_progress.log')
                np.save(out / 'legacy_initial_velocity.npy', original_initial)
                np.save(out / 'v_true.npy', truth)
                np.save(out / 'last_velocity.npy', final)
                np.save(out / 'v_pred.npy', original_best)
                np.save(out / 'observed.npy', data['shots'].cpu().numpy())
                geometry = data['geometry']
                write_json(out / 'acquisition.json', dict(num_shots=int(geometry['xs'].shape[1]),
                    source_x_indices=geometry['xs'].tolist(), source_z_indices=geometry['zs'].tolist(),
                    receiver_x_indices=geometry['xr'].tolist(), receiver_z_indices=geometry['zr'].tolist()))
                write_json(out / 'comparison_contract.json', comparison_contract(config, data))
                metrics = dict(relative_model_error=float(np.linalg.norm(original_best - truth) / np.linalg.norm(truth)),
                    velocity_rmse_mps=float(np.sqrt(np.mean((original_best - truth) ** 2))),
                    final_data_loss=float(history[-1, 2]))
                if (legacy / 'metrics.json').is_file():
                    original_metrics = json.loads((legacy / 'metrics.json').read_text(encoding='utf-8-sig'))
                    if set(original_metrics) != set(metrics):
                        raise ValueError('Original metrics must retain the three author output fields')
                    for key in metrics:
                        if not np.isclose(original_metrics[key], metrics[key], rtol=1e-6, atol=1e-9):
                            raise ValueError('Original metric differs from the archived data: ' + key)
                    metrics = original_metrics
                write_json(out / 'metrics.json', metrics)
                final_metrics = _spatial_metrics(truth, final, config)
                final_metrics.update(completed_updates=epochs, selection_metric='legacy_preupdate_data_loss',
                                     last_training_data_loss_before_update=float(history[-1, 2]))
                write_json(out / 'final_metrics.json', final_metrics)
                write_json(out / 'training_summary.json', dict(completed_updates=epochs,
                    imported_completed_run=True, executed_updates=0,
                    parameter_count=sum(p.numel() for p in model.vel_net.parameters()),
                    total_seconds=None, peak_cuda_bytes=None, best_update=best_epoch + 1,
                    best_loss=checkpoint['best_loss'], selection_metric='legacy_preupdate_data_loss',
                    snapshot_updates=expected_updates, endpoint_waveform_evaluated=False,
                    model_state_timing='postupdate', selection_loss_timing='preupdate'))
                write_json(out / 'legacy_provenance.json', dict(legacy_run=str(legacy),
                    checkpoint=str(checkpoint_path),
                    checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                    snapshot_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in snapshots},
                    original_config=metadata, original_source_hashes=pinned_sources,
                    original_initial_max_difference_mps=float(np.abs(initial - original_initial).max()),
                    observations='Reconstructed by the original forward solver; original observations were not archived',
                    epoch_label=epochs - 1, completed_updates=epochs))
                if config.get('evaluation', {}).get('save_plots', True):
                    plot_result(out, truth, original_initial, original_best, history[:, 1:])
                write_json(out / 'status.json', dict(state='completed', completed_updates=epochs))
                print(f"IMPORTED completed legacy baseline: {epochs} updates; full RMSE={final_metrics['full_rmse']:.4f} m/s; zero new optimizer updates", flush=True)
            except BaseException as error:
                write_json(out / 'status.json', dict(state='failed', completed_updates=epochs, error=str(error)))
                raise
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-run', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    args=parser.parse_args()
    legacy=args.legacy_run.resolve()
    metadata=json.loads((legacy/'config.json').read_text(encoding='utf-8-sig'))
    config=load_config(args.config)
    strict = config.get('execution', {}).get('protocol') == 'original_baseline'
    if strict and config.get('seed', args.seed) != args.seed:
        raise ValueError('Original baseline config seed differs from the requested seed')
    config['seed']=args.seed
    if strict:
        return import_original_baseline(legacy, config, args.output_dir, device=args.device)
    checkpoint=legacy/'checkpoints'/f"MarmousiI_random-checkpoint-{metadata['epochs']}.pth"
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    validate_legacy_config(metadata,config,ck['epoch'])
    author_sources=legacy/'author_sources'
    if not author_sources.exists(): author_sources=legacy.parents[2]
    for name in ('ifwi_modules.py','rnn_fd.py','generator.py'):
        if hashlib.sha256((author_sources/name).read_bytes()).digest() != hashlib.sha256((ROOT/name).read_bytes()).digest():
            raise ValueError('Legacy solver/network source differs: '+name)
    out=args.output_dir/'legacy_baseline_import'
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',config)
    write_json(out/'status.json',dict(state='preparing',completed_updates=ck['epoch']))
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    try:
        data=prepare_data(config,args.device)
        truth=data['vp_true'].squeeze().cpu().numpy()
        np.testing.assert_array_equal(truth,np.load(legacy/'true_velocity.npy').squeeze())
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
        model=build_model(config,data,args.device)
        initial=velocity(model)
        original_initial=np.load(legacy/'initial_velocity.npy').squeeze()
        np.testing.assert_allclose(initial,original_initial,rtol=1e-6,atol=.01)
        print('Verified legacy truth, random initialization, core sources and completed update budget',flush=True)
        np.save(out/'initial_velocity.npy',initial)
        np.save(out/'legacy_initial_velocity.npy',original_initial)
        np.save(out/'v_true.npy',truth)
        np.save(out/'observed.npy',data['shots'].cpu().numpy())
        write_json(out/'acquisition.json',dict(num_shots=int(data['geometry']['xs'].shape[1]),
            **{key:data['geometry'][key].tolist() for key in ('xs','zs','xr','zr')}))
        # Keep the existing acquisition schema for the common result auditor.
        acquisition=json.loads((out/'acquisition.json').read_text())
        for key,label in [('xs','source_x_indices'),('zs','source_z_indices'),('xr','receiver_x_indices'),('zr','receiver_z_indices')]:
            acquisition[label]=acquisition.pop(key)
        write_json(out/'acquisition.json',acquisition)
        write_json(out/'comparison_contract.json',comparison_contract(config,data))
        model.vel_net.load_state_dict(ck['state_dict'])
        final=velocity(model)
        np.save(out/'last_velocity.npy',final)
        final_metrics=evaluate_model_metrics(model,data,config)
        final_metrics['completed_updates']=ck['epoch']
        write_json(out/'final_metrics.json',final_metrics)
        model.vel_net.load_state_dict(ck['best_loss_model'])
        best=velocity(model)
        np.testing.assert_allclose(best,np.load(legacy/'best_velocity.npy').squeeze(),rtol=1e-5,atol=.2)
        np.save(out/'best_velocity.npy',best); np.save(out/'v_pred.npy',best)
        best_metrics=evaluate_model_metrics(model,data,config)
        count=sum(p.numel() for p in model.vel_net.parameters())
        best_metrics.update(parameter_count=count,best_update=ck['best_loss_epoch']+1,
            selection_metric='legacy_preupdate_data_loss')
        write_json(out/'metrics.json',best_metrics)
        write_json(out/'training_summary.json',dict(completed_updates=ck['epoch'],
            imported_completed_run=True,executed_updates=0,parameter_count=count,
            total_seconds=None,peak_cuda_bytes=None,best_update=ck['best_loss_epoch']+1,
            best_loss=ck['best_loss'],selection_metric='legacy_preupdate_data_loss'))
        shutil.copy2(legacy/'loss.csv',out/'legacy_loss_history.csv')
        write_json(out/'legacy_provenance.json',dict(legacy_run=str(legacy),checkpoint=str(checkpoint),
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            original_config=metadata,source_hashes=source_hashes(),
            original_initial_max_difference_mps=float(np.abs(initial-original_initial).max()),
            observations='Reconstructed from original acquisition; original GPU observations were not archived',
            epoch_label=ck['epoch']-1,completed_updates=ck['epoch']))
        save_plot(truth,final,data['params']['dz'],out/'final_result.png',model_title=f'Legacy final model ({ck["epoch"]} updates)')
        write_json(out/'status.json',dict(state='completed',completed_updates=ck['epoch']))
        print(f"IMPORTED completed legacy baseline: {ck['epoch']} updates; full RMSE={final_metrics['full_rmse']:.4f} m/s; zero new optimizer updates",flush=True)
    except BaseException as exc:
        write_json(out/'status.json',dict(state='failed',completed_updates=ck['epoch'],error=str(exc)))
        raise
    return 0


if __name__ == '__main__': raise SystemExit(main())
