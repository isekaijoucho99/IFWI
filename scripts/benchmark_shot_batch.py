"""Compare shot microbatches using saved observations, with one Adam update per epoch."""
import argparse
import copy
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments'))
from run_experiment import build_model
from experiment_runtime import write_json, velocity
from parameter_sweep import make_cases
from generator import wGenerator


def saved_data(run, config, device):
    geom = json.loads((run/'acquisition.json').read_text(encoding='utf-8-sig'))
    truth = np.load(run/'v_true.npy')
    nz, nx = truth.shape
    dt, nt = config['data']['dt'], config['data']['nt']
    return dict(vp_true=torch.from_numpy(truth[None]).to(device),
        shots=torch.from_numpy(np.load(run/'observed.npy')).to(device),
        wavelet=wGenerator(dt*torch.arange(nt, dtype=torch.float32), 8).ricker().to(device),
        geometry=dict(nz=nz, nx=nx, xs=torch.tensor(geom['source_x_indices']),
            zs=torch.tensor(geom['source_z_indices']), xr=torch.tensor(geom['receiver_x_indices']),
            zr=torch.tensor(geom['receiver_z_indices'])),
        params=dict(dz=config['data']['dz'], dt=dt, nt=nt))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-run', required=True, type=Path)
    parser.add_argument('--shots49-run', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--updates', type=int, default=3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.updates < 2: parser.error('At least two updates are needed to exclude the warmup')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    base = json.loads((args.baseline_run/'config.json').read_text(encoding='utf-8'))
    cases = {case['name']: case for case in make_cases(base, shot_batch_size=13)}
    jobs = [(name, batch) for name in ('baseline', 'depth_8', 'width_512') for batch in (7, 13)]
    jobs.append(('shots_49', 13))
    rows = []
    references = {}
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    for name, batch in jobs:
        config = copy.deepcopy(cases[name]['config'])
        config['training']['shot_batch_size'] = batch
        write_json(out/'active.json', dict(state='running', name=name, batch_size=batch,
            completed_tests=len(rows), total_tests=len(jobs)))
        source = args.shots49_run if name == 'shots_49' else args.baseline_run
        data = saved_data(source, config, args.device)
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
        model = build_model(config, data, args.device)
        opt = torch.optim.Adam(model.vel_net.parameters(), lr=config['optimizer']['learning_rate'])
        torch.cuda.synchronize(args.device)
        torch.cuda.reset_peak_memory_stats(args.device)
        initial = velocity(model)
        seconds, losses = [], []
        print(f'START {name} batch={batch}', flush=True)
        for update in range(args.updates):
            start = time.perf_counter()
            _, parts = model.train_one_epoch(opt, wavelet=data['wavelet'], shots=data['shots'])
            torch.cuda.synchronize(args.device)
            elapsed = time.perf_counter()-start
            seconds.append(elapsed); losses.append(parts[0])
            print(f'BENCH {name} batch={batch} update={update+1}/{args.updates} '
                  f'seconds={elapsed:.3f} loss={parts[0]:.8e}', flush=True)
        final = velocity(model)
        peak = torch.cuda.max_memory_allocated(args.device)/1024**3
        np.save(out/f'{name}_batch{batch}_initial.npy', initial)
        np.save(out/f'{name}_batch{batch}_final.npy', final)
        row = dict(name=name, batch_size=batch, updates=args.updates,
            seconds=seconds, steady_mean_seconds=float(np.mean(seconds[1:])),
            peak_allocated_gib=peak, loss_before_updates=losses,
            all_parameters_finite=all(bool(torch.isfinite(p).all()) for p in model.vel_net.parameters()))
        if batch == 7:
            references[name] = dict(initial=initial, final=final, losses=losses, row=row)
        elif name in references:
            ref = references[name]
            np.testing.assert_array_equal(initial, ref['initial'])
            np.testing.assert_allclose(losses, ref['losses'], rtol=2e-4, atol=1e-7)
            # GPU reduction order can cause small Adam differences; disclose the measured error.
            row['max_final_difference_mps'] = float(np.abs(final-ref['final']).max())
            row['speedup_vs_batch7'] = ref['row']['steady_mean_seconds']/row['steady_mean_seconds']
        rows.append(row)
        write_json(out/'benchmark.json', rows)
        print(f'END {name} batch={batch} mean={row["steady_mean_seconds"]:.3f}s peak={peak:.3f}GiB', flush=True)
        del model, opt, data
        gc.collect(); torch.cuda.empty_cache()
    write_json(out/'active.json', dict(state='completed', completed_tests=len(rows), total_tests=len(jobs)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
