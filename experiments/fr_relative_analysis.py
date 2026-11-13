"""CPU-only acceptance and tables for a completed twelve-case 500-update run.

This module imports no trainer, torch, propagator or model constructor. It never
restarts work, fills missing evaluations, selects checkpoints or overwrites an
analysis directory. Missing/mismatched evidence is a validation failure.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from experiments.residual_ifwi_report import velocity_metrics

ROOT = Path(__file__).resolve().parents[1]
COUNTERS = ('forward_fields_attempt', 'forward_fields_complete',
            'backward_fields_attempt', 'backward_fields_complete')
FIELDS = dict(full_raw=52, full_preconditioned=52, global1=1,
              interleaved2=2, interleaved4=4, interleaved4_relative_fr=4)
CONDITIONS = ('clean', 'noisy')
THRESHOLDS = (500., 475., 450., 425.)
METRICS = ('rmse_mps', 'mae_mps', 'r2', 'ssim')


class AnalysisError(ValueError):
    """Saved evidence is missing, incomplete, inconsistent or out of scope."""


def require(condition, message):
    if not condition:
        raise AnalysisError(message)


def read_json(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return json.load(stream, parse_constant=lambda s: (_ for _ in ()).throw(AnalysisError('Nonfinite JSON: '+s)))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def physical_budget(stage):
    if stage == 'preflight':
        return dict(forward_fields=34, backward_fields=34, optimizer_updates=0)
    if stage == 'timing':
        return dict(forward_fields=1380, backward_fields=1380, optimizer_updates=72)
    require(stage == 'formal', 'Unknown stage')
    return dict(forward_fields=118632, backward_fields=115460, optimizer_updates=6024)


def first_crossings(rows, thresholds=THRESHOLDS):
    result = []
    for threshold in thresholds:
        hit = next((row for row in rows if row['rmse_mps'] <= threshold), None)
        result.append(dict(threshold_rmse_mps=threshold,
            status='REACHED' if hit is not None else 'NOT_REACHED',
            first_observed_update=None if hit is None else hit['update'],
            training_seconds=None if hit is None else hit['training_seconds'],
            elapsed_seconds=None if hit is None else hit['elapsed_seconds']))
    return result


def validate_curve(rows, target=500):
    require([row['update'] for row in rows] == list(range(0, target+1, 10)),
            'Model curve has missing, duplicate, reordered or unexpected updates')
    for key in ('training_seconds', 'elapsed_seconds', 'evaluation_seconds'):
        values = [row[key] for row in rows]
        require(all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values),
                'Invalid curve time: '+key)
        require(all(b >= a for a,b in zip(values, values[1:])), 'Time reversal: '+key)
    require(rows[0]['training_seconds'] == 0, 'Initial curve is not update zero')


def waveform_mse(prediction, target):
    require(prediction.shape == target.shape and prediction.ndim == 4,
            'Waveform dimensions differ; broadcasting is forbidden')
    require(np.isfinite(prediction).all() and np.isfinite(target).all(), 'Nonfinite waveform array')
    return float(np.mean((prediction.astype(np.float64)-target.astype(np.float64))**2))


def check_metric(name, saved, recomputed):
    if saved is None or recomputed is None:
        require(saved is None and recomputed is None, 'Undefined metric differs: '+name)
    else:
        require(isinstance(saved, (int, float)) and math.isfinite(saved)
                and math.isfinite(recomputed) and math.isclose(saved, recomputed, rel_tol=2e-8, abs_tol=1e-10),
                'Saved metric differs from CPU recomputation: '+name)


def verify_identity(launch, end):
    require(isinstance(launch.get('run_id'), str) and bool(launch['run_id']), 'Missing unique launch identity')
    require(type(launch.get('pid')) is int and launch['pid'] > 0, 'Invalid launch PID')
    for key in ('run_id', 'pid'):
        require(key in launch and end.get(key) == launch[key], 'Launch/exit identity differs: '+key)
    for key in ('owner_pid', 'parent_pid'):
        if key in launch:
            require(end.get(key) == launch[key], 'Launch/exit identity differs: '+key)
    require(end.get('status') == 'COMPLETED' and type(end.get('exit_code')) is int and end['exit_code'] == 0,
            'No successful terminal receipt; partial results are read-only')


def verify_plan(plan, source_root=ROOT):
    common = {k:v for k,v in plan.items() if k not in ('signature', 'status', 'budgets')}
    require(canonical_hash(common) == plan['signature'], 'Plan signature mismatch')
    source_root = Path(source_root).resolve()
    require(bool(plan['source_sha256']), 'Missing frozen source hashes')
    for name, expected in plan['source_sha256'].items():
        path = (source_root/name).resolve()
        require(path.is_relative_to(source_root), 'Frozen source path escapes source root')
        require(file_hash(path) == expected, 'Frozen source changed: '+name)


def _load_array(path, expected_hash=None):
    path = Path(path)
    if expected_hash is not None:
        require(file_hash(path) == expected_hash, 'Array hash mismatch: '+str(path))
    array = np.load(path, allow_pickle=False)
    require(np.isfinite(array).all(), 'Nonfinite array: '+str(path))
    return array


def verify_assets(plan, dense_directory):
    arrays = {}
    for name in ('truth', 'background', 'wavelet'):
        item = plan['input_manifest'][name]
        arrays[name] = _load_array(item['path'], item['sha256'])
        require(hashlib.sha256(np.asarray(arrays[name]).tobytes()).hexdigest()
                == plan['physical']['input_array_sha256'][name], 'Physical input bytes differ: '+name)
    require(arrays['truth'].shape == arrays['background'].shape, 'Truth/background shapes differ')
    require(canonical_hash(plan['physical']) == plan['data_key'], 'Physical data key mismatch')
    directory = Path(dense_directory)
    manifest = read_json(directory/'data_manifest.json')
    require(manifest == plan['dense_data_manifest'] and manifest['data_key'] == plan['data_key'],
            'Frozen dense-data manifest differs')
    expected = (1, 52, plan['config']['forward']['nt'],
                len(range(0, arrays['truth'].shape[1], plan['config']['acquisition']['receiver_stride'])))
    for name in CONDITIONS:
        arrays[name] = _load_array(directory/(name+'.npy'), manifest[name+'_sha256'])
        require(arrays[name].dtype == np.float32 and arrays[name].shape == expected,
                'Dense observation shape/dtype differs: '+name)
    require(arrays['wavelet'].shape == (expected[2],), 'Wavelet dimensions differ')
    snr = float(10*np.log10(np.sum(arrays['clean'].astype(np.float64)**2)
                          /np.sum((arrays['noisy'].astype(np.float64)-arrays['clean'].astype(np.float64))**2)))
    check_metric('measured_snr_db', manifest['measured_snr_db'], snr)
    return arrays


def _read_ledger(directory, launch):
    rows = []
    with (directory/'ledger.jsonl').open(encoding='utf-8-sig') as stream:
        for line in stream:
            require(line.endswith('\n') and bool(line.strip()), 'Incomplete or blank ledger row')
            row = json.loads(line)
            for key in ('run_id', 'pid', 'owner_pid', 'parent_pid'):
                require(row.get(key) == launch[key], 'Ledger identity differs: '+key)
            rows.append(row)
    require(bool(rows), 'Empty stage ledger')
    return rows


def verify_stage(directory, stage, plan):
    directory = Path(directory)
    require(not (directory/'failure.json').exists() and not (directory/'control/STOP_ACK.json').exists(),
            'Stage contains a failure/stop receipt; no silent new attempt')
    launch = read_json(directory/'control/LAUNCH.json')
    end = read_json(directory/'control/EXIT.json')
    verify_identity(launch, end)
    require(read_json(directory/'plan.json') == plan, 'Stage plan differs: '+stage)
    report = read_json(directory/'result.json')
    require(report.get('status') == 'COMPLETED' and report.get('stage') == stage
            and report.get('signature') == plan['signature'], 'Stage result differs: '+stage)
    require(report.get('target_updates') == 500 and not report.get('resume_root'),
            'Collector accepts the unique fresh500 stage only')
    expected = physical_budget(stage)
    saved_budget = read_json(directory/'budget.json')
    require(all(saved_budget.get(key) == value for key,value in expected.items()), 'Budget differs: '+stage)
    rows = _read_ledger(directory, launch)
    counters = {key:0 for key in COUNTERS}
    phases = defaultdict(lambda: {key:0 for key in COUNTERS})
    for row in rows:
        present = [key for key in COUNTERS if key in row]
        if not present:
            continue
        phase = row.get('phase', 'other')
        for key in COUNTERS:
            value = row.get(key, 0)
            require(type(value) is int and value >= 0, 'Invalid physical counter')
            counters[key] += value; phases[phase][key] += value
    require(counters == report['cost_totals'] == end['cost_totals'], 'Ledger/result/exit physical costs differ')
    require(dict(phases) == report['cost_by_phase'], 'Ledger/result phase costs differ')
    for kind in ('forward', 'backward'):
        for state in ('attempt', 'complete'):
            require(counters[kind+'_fields_'+state] == expected[kind+'_fields'], 'Extra/missing physics work: '+stage)
    if stage in ('timing', 'formal'):
        multiplier = 1 if stage == 'timing' else 2
        for phase, fields in (('warmup', 115*2*multiplier),
                              ('training', 115*(10 if stage == 'timing' else 500)*multiplier)):
            require(phase in phases and all(phases[phase][key] == fields for key in COUNTERS),
                    'Training/warmup cost attribution differs: '+phase)
        if stage == 'formal':
            require(phases.get('evaluation') == dict(forward_fields_attempt=3172,forward_fields_complete=3172,
                    backward_fields_attempt=0,backward_fields_complete=0), 'Shared-initial/evaluation cost differs')
    if stage == 'preflight':
        require(report.get('passed') is True and len(report['cases']) == 2
                and all(case.get('passed') is True for case in report['cases']), 'Physical correctness gate failed')
        require(report['optimizer_updates'] == dict(warmup=0, training=0), 'Preflight contains Adam work')
    else:
        _verify_updates(rows, report, stage)
    return dict(directory=directory, launch=launch, end=end, report=report, ledger=rows)


def _verify_updates(rows, report, stage):
    conditions = ('noisy',) if stage == 'timing' else CONDITIONS
    expected_keys = {(c,a) for c in conditions for a in FIELDS}
    by_case = {key:{'warmup_update':[], 'update':[]} for key in expected_keys}
    for row in rows:
        if row['event'] in ('update', 'warmup_update'):
            key = (row['condition'], row['arm'])
            require(key in by_case, 'Unexpected condition/arm in ledger')
            by_case[key][row['event']].append(row['result'])
    expected_measured = list(range(3,13)) if stage == 'timing' else list(range(1,501))
    for key, events in by_case.items():
        require([r['completed_updates'] for r in events['warmup_update']] == [1,2], 'Duplicate/missing warmup commits')
        require([r['completed_updates'] for r in events['update']] == expected_measured, 'Duplicate/missing measured commits')
        for row in events['warmup_update']+events['update']:
            require(row['fields'] == FIELDS[key[1]] and row['reference_shots'] == 52, 'Update field/normalization mismatch')
            require(row['batch_fields'] == ([13]*4 if FIELDS[key[1]] == 52 else [FIELDS[key[1]]]), 'Unexpected batching')
            require(math.isfinite(row['wall_seconds']) and row['wall_seconds'] > 0
                    and math.isfinite(row['cuda_ms']) and row['cuda_ms'] > 0, 'Invalid synchronized update time')
    expected = dict(warmup=2*len(expected_keys), training=len(expected_measured)*len(expected_keys))
    require(report['optimizer_updates'] == expected, 'Optimizer receipt differs from returned commits')
    cases = report['cases']
    require(len(cases) == len(expected_keys) and {(c['condition'],c['arm']) for c in cases} == expected_keys,
            'Missing/duplicate/unexpected case summaries')
    if stage == 'timing':
        recovered = [(r['condition'],r['arm']) for r in rows if r['event'] == 'exact_checkpoint_recovery']
        require(len(recovered) == len(expected_keys) and set(recovered) == expected_keys, 'Missing/duplicate recovery checks')


def verify_pipeline(root, plan, stages):
    root = Path(root)
    launch = read_json(root/'PIPELINE_LAUNCH.json')
    end = read_json(root/'PIPELINE_EXIT.json')
    verify_identity(launch, end)
    phases = end.get('completed_phases', [])
    require([p['stage'] for p in phases] == ['preflight', 'timing', 'formal'], 'Pipeline stage completion chain differs')
    ids = [launch['run_id']]+[stages[s]['launch']['run_id'] for s in ('preflight','timing','formal')]
    require(len(set(ids)) == 4, 'Launch identities were reused')
    for phase in phases:
        stage = phase['stage']; saved = stages[stage]
        require((root/phase['directory']).resolve() == saved['directory'].resolve(), 'Pipeline output link differs')
        require(phase['receipt'] == saved['end'], 'Pipeline/child terminal receipt differs')
        require(phase.get('result_sha256') == file_hash(saved['directory']/'result.json'), 'Pipeline result hash differs')
        require(math.isfinite(phase['phase_wall_seconds']) and phase['phase_wall_seconds'] >= saved['report']['stage_wall_seconds'],
                'Invalid phase wall time')
    return launch, end


def _saved_case(directory, report_case, stage):
    case = read_json(directory/'case.json')
    require(case == report_case, 'Case file/result differs')
    require(case['completed_updates'] == (12 if stage == 'timing' else 500), 'Incomplete case')
    require(case['measured_updates'] == (10 if stage == 'timing' else 500) and case['warmup_updates'] == 2,
            'Unexpected case update counts')
    for key in ('training_seconds', 'case_elapsed_seconds', 'evaluation_seconds', 'wall_seconds_median',
                'cuda_ms_median', 'peak_allocated_bytes', 'peak_reserved_bytes', 'optimizer_tensor_bytes', 'trainable_parameters'):
        require(isinstance(case[key], (int,float)) and math.isfinite(case[key]) and case[key] >= 0, 'Invalid case scalar: '+key)
    require(case['case_elapsed_seconds'] >= case['training_seconds'], 'Case wall shorter than training wall')
    return case


def _verify_checkpoints(directory, case, ledger):
    expected = [100,200,300,400,500]
    paths = sorted(directory.glob('update_*.pt'))
    require([p.name for p in paths] == [f'update_{k:04d}.pt' for k in expected], 'Missing/extra formal snapshots')
    require(Path(case['checkpoint']).resolve() == paths[-1].resolve(), 'Final checkpoint path differs')
    events = [r for r in ledger if r['event'] == 'checkpoint' and r['condition'] == case['condition'] and r['arm'] == case['arm']]
    require([r['completed_updates'] for r in events] == expected, 'Checkpoint event counts differ')
    for path, event in zip(paths, events):
        require(Path(event['path']).resolve() == path.resolve() and file_hash(path) == event['sha256'], 'Checkpoint bytes/path differ')


def verify_case_arrays(directory, case, arrays, shared, ledger):
    files = sorted(directory.glob('metrics_*.json'))
    require([p.name for p in files] == [f'metrics_{k:04d}.json' for k in range(0,501,10)], 'Missing/extra model metric files')
    rows = [read_json(path) for path in files]
    validate_curve(rows)
    require([p.name for p in sorted(directory.glob('velocity_*.npy'))]
            == [f'velocity_{k:04d}.npy' for k in range(0,501,10)], 'Missing/extra velocity arrays')
    require([p.name for p in sorted(directory.glob('prediction_*.npy'))]
            == [f'prediction_{k:04d}.npy' for k in range(0,501,100)], 'Missing/extra waveform arrays')
    metric_events = [r['result'] for r in ledger if r['event'] == 'metrics'
                     and r['condition'] == case['condition'] and r['arm'] == case['arm']]
    require(metric_events == rows, 'Saved curve differs from producer ledger')
    hashes = {}
    for row in rows:
        k = row['update']; path = directory/f'velocity_{k:04d}.npy'
        vel = _load_array(path)
        for name, value in velocity_metrics(vel, arrays['truth']).items():
            check_metric(name, row.get(name), value)
        require(all(name in row for name in ('clean_mse','noisy_mse')) == (k%100 == 0), 'Waveform metric cadence differs')
        if k == 0:
            require(np.array_equal(vel, arrays['background']), 'Case initial velocity differs')
            for name in METRICS+('clean_mse','noisy_mse'):
                check_metric(name, row.get(name), shared['metrics'].get(name))
            require(file_hash(directory/'prediction_0000.npy') == shared['prediction_sha256'], 'Case initial prediction differs')
        if k == 500:
            pred = _load_array(directory/'prediction_0500.npy')
            for name in CONDITIONS:
                check_metric(name+'_mse', row[name+'_mse'], waveform_mse(pred, arrays[name]))
            hashes = dict(velocity_sha256=file_hash(path), prediction_sha256=file_hash(directory/'prediction_0500.npy'))
    check_metric('final_training_seconds', rows[-1]['training_seconds'], case['training_seconds'])
    require(case['thresholds'] == first_crossings(rows), 'Saved threshold crossings differ')
    _verify_checkpoints(directory, case, ledger)
    return rows, hashes


def _csv(path, rows):
    rows = list(rows)
    names = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader()
        writer.writerows({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows)


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def _markdown(summary, cases, costs):
    lines = ['# 52-shot IFWI: twelve complete 500-update cases', '',
        f"Pipeline `{summary['run_id']}`, frozen plan `{summary['signature']}`. All three stage receipts, field ledgers and returned Adam updates were verified. This is one seed and the training acquisition; clean-reference waveform error is not held-out-shot performance.", '',
        'The matched encoding control is `full_preconditioned`. `full_raw` separates the existing illumination rule. The relative-FR arm changes representation and parameterization together; its parameter count and optimizer memory are reported without a single-component attribution.', '',
        '| Condition | Arm | RMSE m/s | SSIM | Clean MSE | Noisy MSE | Training s | Case wall s | Parameters | Peak GiB |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in cases:
        lines.append(f"| {row['condition']} | {row['arm']} | {row['final_rmse_mps']:.6g} | {row['final_ssim']:.6g} | {row['final_clean_mse']:.7g} | {row['final_noisy_mse']:.7g} | {row['training_seconds']:.3f} | {row['case_elapsed_seconds']:.3f} | {row['trainable_parameters']} | {row['peak_allocated_bytes']/2**30:.4f} |")
    lines += ['', 'Every row is retained, including deteriorations. No truth-based method, checkpoint or parameter selection was performed. RMSE/SSIM curves use every10 updates; waveform metrics use every100. All saved model-curve arrays and all twelve final waveform arrays were recomputed on CPU. Intermediate waveform values remain producer-recorded; their raw arrays are retained. The one shared initial waveform was independently checked and its identical copies verified.', '',
        'Thresholds 500/475/450/425m/s were fixed before formal training. `threshold_times.csv` reports first observed crossings with no interpolation; NOT_REACHED has empty time/update fields. Unreached thresholds provide no equal-accuracy speedup evidence. A fixed-update time ratio is not an equal-accuracy speedup.', '',
        'Training seconds sum synchronized returned update intervals and exclude warmup, construction, evaluation and checkpoint I/O. Case wall includes those costs within each case. Condition construction and complete pipeline/preflight overhead are separately retained in ledgers and phase costs. CUDA timing and peak allocation/reservation are reported separately. Full microbatch13 was feasible; no claim that13 is the largest feasible batch is made.', '',
        'For N52/B2 or4, cyclic interleaved shifts only relabel groups and do not change which source pairs share a group. Fresh signs still redraw every update. Background-relative scaling is not exact log-velocity parameterization or a full illumination correction. This combination is a local adaptation, without an assumed novelty or performance guarantee.', '',
        '| Stage | F attempt/complete | B attempt/complete | Adam commits | Phase wall s |', '|---|---:|---:|---:|---:|']
    for row in costs:
        lines.append(f"| {row['stage']} | {row['forward_fields_attempt']}/{row['forward_fields_complete']} | {row['backward_fields_attempt']}/{row['backward_fields_complete']} | {row['optimizer_updates']} | {row['phase_wall_seconds']:.3f} |")
    lines += ['', f"Total physical work: {summary['total_F']}F/{summary['total_B']}B and {summary['total_Adam']} Adam commits. Reused data synthesis cost in this pipeline is zero; prior data-generation cost is not reclassified as new work. Collection used CPU NumPy/SSIM only:0 PDE F/B,0 neural evaluations,0 GPU work,0 Adam updates.", '']
    return '\n'.join(lines)


def collect(pipeline_root, *, dense_data=None, source_root=ROOT):
    root = Path(pipeline_root).resolve(strict=True)
    output = root/'analysis500'
    require(not output.exists(), 'analysis500 already exists; collection never overwrites or repeats an attempt')
    started = time.perf_counter()
    plan = read_json(root/'formal/plan.json'); verify_plan(plan, source_root)
    require(plan.get('schema') == 'fr_relative_52_v1' and plan.get('conditions') == list(CONDITIONS), 'Unexpected protocol')
    require([a['name'] for a in plan['arms']] == list(FIELDS)
            and [a['fields'] for a in plan['arms']] == list(FIELDS.values()), 'Unexpected configurations')
    for arm in plan['arms']:
        require(arm['preconditioned'] == (arm['name'] != 'full_raw')
                and arm['model'] == ('relative_fr' if arm['name'] == 'interleaved4_relative_fr' else 'plain'),
                'Model/preconditioning control differs')
    require(plan['model_metric_interval'] == 10 and plan['waveform_metric_interval'] == 100
            and plan['thresholds_rmse_mps'] == list(THRESHOLDS), 'Evaluation/threshold protocol differs')
    stages = {s:verify_stage(root/s, s, plan) for s in ('preflight','timing','formal')}
    launch, end = verify_pipeline(root, plan, stages)
    if dense_data is None:
        request = read_json(root/'EXPERIMENT_REQUEST.json')
        dense_data = request['dense_data_directory']
    arrays = verify_assets(plan, dense_data)
    shared_dir = root/'formal/shared_initial'
    shared = read_json(shared_dir/'manifest.json')
    vel = _load_array(shared_dir/'velocity.npy', shared['velocity_sha256'])
    pred = _load_array(shared_dir/'prediction.npy', shared['prediction_sha256'])
    require(np.array_equal(vel, arrays['background']), 'Shared initial velocity differs from frozen background')
    for name, value in velocity_metrics(vel, arrays['truth']).items():
        check_metric(name, shared['metrics'].get(name), value)
    for name in CONDITIONS:
        check_metric(name+'_mse', shared['metrics'][name+'_mse'], waveform_mse(pred, arrays[name]))
    case_rows=[]; curve_rows=[]; threshold_rows=[]; timing_rows=[]; update_rows=[]; artifact_hashes=[]
    for stage in ('timing','formal'):
        saved = stages[stage]
        expected_cases = {(c,a) for c in (('noisy',) if stage == 'timing' else CONDITIONS) for a in FIELDS}
        actual_dirs = {(p.parent.name,p.name) for c in (('noisy',) if stage == 'timing' else CONDITIONS)
                       for p in (root/stage/c).iterdir() if p.is_dir()}
        require(actual_dirs == expected_cases, 'Extra/missing case directories')
        for report_case in saved['report']['cases']:
            condition, arm = report_case['condition'],report_case['arm']
            directory = root/stage/condition/arm
            case = _saved_case(directory, report_case, stage)
            events = [r for r in saved['ledger'] if r['event'] in ('update','warmup_update')
                      and r['condition'] == condition and r['arm'] == arm]
            for event in events:
                result = event['result']
                update_rows.append(dict(stage=stage,condition=condition,arm=arm,
                    phase='warmup' if event['event']=='warmup_update' else 'training',
                    update=result['completed_updates'],loss=result['loss'],wall_seconds=result['wall_seconds'],
                    cuda_ms=result['cuda_ms'],fields=result['fields'],NT=result['NT']))
            warm = [e['result'] for e in events if e['event']=='warmup_update']
            measured = [e['result'] for e in events if e['event']=='update']
            check_metric('case training_seconds', case['training_seconds'], sum(r['wall_seconds'] for r in measured))
            base = {k:v for k,v in case.items() if k not in ('thresholds','checkpoint')}
            base.update(warmup_seconds=sum(r['wall_seconds'] for r in warm),
                training_cuda_ms=sum(r['cuda_ms'] for r in measured),warmup_cuda_ms=sum(r['cuda_ms'] for r in warm),
                training_F=sum(r['fields'] for r in measured),training_B=sum(r['fields'] for r in measured),
                warmup_F=sum(r['fields'] for r in warm),warmup_B=sum(r['fields'] for r in warm))
            if stage == 'timing':
                timing_rows.append(base); continue
            rows, hashes = verify_case_arrays(directory, case, arrays, shared, saved['ledger'])
            artifact_hashes.append(dict(condition=condition,arm=arm,**hashes))
            for name in METRICS+('clean_mse','noisy_mse'):
                base['initial_'+name] = rows[0][name]
                base['final_'+name] = rows[-1][name]
                base['change_'+name] = None if rows[0][name] is None else rows[-1][name]-rows[0][name]
            case_rows.append(base)
            curve_rows.extend(dict(condition=condition,arm=arm,**row) for row in rows)
            threshold_rows.extend(dict(condition=condition,arm=arm,**row) for row in first_crossings(rows))
    phase_costs=[]; cost_rows=[]
    phase_receipts = {p['stage']:p for p in end['completed_phases']}
    for stage,saved in stages.items():
        report=saved['report']
        cost_rows.append(dict(stage=stage,**report['cost_totals'],optimizer_updates=sum(report['optimizer_updates'].values()),
            stage_wall_seconds=report['stage_wall_seconds'],phase_wall_seconds=phase_receipts[stage]['phase_wall_seconds']))
        phase_costs.extend(dict(stage=stage,phase=phase,**totals) for phase,totals in report['cost_by_phase'].items())
    summary=dict(schema='fr_relative_analysis_v1',status='VERIFIED_COMPLETE_500',run_id=launch['run_id'],
        signature=plan['signature'],pipeline_root=str(root),pipeline_launch=launch,pipeline_exit=end,
        total_F=sum(c['forward_fields_complete'] for c in cost_rows),
        total_B=sum(c['backward_fields_complete'] for c in cost_rows),
        total_Adam=sum(c['optimizer_updates'] for c in cost_rows),
        cpu_verification_wall_seconds=time.perf_counter()-started,
        collection_physical_F=0,collection_physical_B=0,collection_neural_evaluations=0,
        collection_GPU_work=0,collection_Adam_updates=0,final_array_hashes=artifact_hashes,
        shared_initial=shared,source_sha256=plan['source_sha256'],collector_sha256=file_hash(__file__),
        verification_scope='All model-curve arrays, shared initial waveforms and twelve final waveforms recomputed; intermediate waveform scores are producer-recorded.')
    require((summary['total_F'],summary['total_B'],summary['total_Adam']) == (120046,116874,6096), 'Complete pipeline budget differs')
    # All validation precedes the first output mutation. A failed write leaves a
    # visible one-off failed attempt, which is never removed or overwritten.
    output.mkdir(exist_ok=False)
    receipt=dict(pid=os.getpid(),parent_pid=os.getppid(),pipeline_run_id=launch['run_id'],
                 utc=datetime.now(timezone.utc).isoformat(),collector_sha256=summary['collector_sha256'])
    _write_json(output/'COLLECTION_START.json', receipt)
    try:
        for name,rows in (('cases',case_rows),('curves',curve_rows),('timing',timing_rows),
                          ('updates',update_rows),('stage_costs',cost_rows),('cost_by_phase',phase_costs),('threshold_times',threshold_rows)):
            _csv(output/(name+'.csv'), rows)
        _write_json(output/'summary.json', summary)
        with (output/'results.md').open('x', encoding='utf-8') as stream:
            stream.write(_markdown(summary, case_rows, cost_rows))
        _write_json(output/'COLLECTION_EXIT.json',dict(receipt,status='COMPLETED',exit_code=0,
            wall_seconds=time.perf_counter()-started,physical_F=0,physical_B=0,neural_evaluations=0,GPU_work=0,Adam_updates=0))
    except BaseException as error:
        if not (output/'COLLECTION_EXIT.json').exists():
            _write_json(output/'COLLECTION_EXIT.json',dict(receipt,status='FAILED',exit_code=1,error=str(error)))
        raise
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pipeline-root',type=Path,required=True)
    parser.add_argument('--dense-data',type=Path)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    args=parser.parse_args(argv)
    result=collect(args.pipeline_root,dense_data=args.dense_data,source_root=args.source_root)
    print(json.dumps({k:result[k] for k in ('status','run_id','total_F','total_B','total_Adam')},sort_keys=True))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
