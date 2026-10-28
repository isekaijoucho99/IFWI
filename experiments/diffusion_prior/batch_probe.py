"""Measure safe G2AuthorUNet microbatch throughput on the TARGET CUDA GPU.

Usage (from repo root)::

    python -m experiments.diffusion_prior.batch_probe --model-channels 32 \
       --patch-size 128 --effective-batch 16 --candidates 1 2 4 \
       --mixed-precision bf16 --out outputs/g2_batch_probe.json

Each candidate is tested in a FRESH subprocess, so a failed CUDA allocation
cannot poison the following trials or a real training process. Probe does a
real optimizer step with Adam, EMA, mixed precision, checkpointing and gradient
accumulation. It uses random synthetic tensors, not the training dataset.
No checkpoint weights are changed; re-run a new training job with suggested
batch settings. Changing batch settings is NOT an exact resume of an old run.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from .author_unet import check_memory_budget, paper_configuration


def next_batch_is_safe(measurements, next_microbatch, free_bytes, safety_fraction=.75):
    """Estimate the *next* peak from previous CUDA measurements; fail closed.

    With one observation scale the WHOLE peak linearly (conservative).
    With two, extrapolate marginal activation cost with a 50% buffer.
    The isolated child process still performs an actual CUDA check.
    """
    if not measurements or not 0 < safety_fraction < 1 or free_bytes <= 0:
        raise ValueError('Invalid memory extrapolation inputs')
    (current, peak) = measurements[-1]
    if next_microbatch <= current or current < 1 or peak <= 0:
        raise ValueError('Candidate batch size must increase after a positive measurement')
    if len(measurements) == 1:
        projected = peak*next_microbatch/current
    else:
        previous, previous_peak = measurements[-2]
        if previous >= current or previous_peak <= 0:
            raise ValueError('Non-increasing calibration batch sizes')
        # A flat allocator reading can hide new activation allocations: do
        # not extrapolate zero cost per additional sample from a short probe.
        slope = max((peak-previous_peak)/(current-previous),
                    .25*(2**30), .05*peak/current)
        projected = peak+1.5*slope*(next_microbatch-current)
    return projected <= safety_fraction*free_bytes


def select_fastest(records, free_bytes, effective_batch, safety_fraction=.75):
    """Choose the fastest *observed* configuration meeting headroom policy."""
    if type(effective_batch) is not int or effective_batch < 1 or free_bytes <= 0 or not 0 < safety_fraction < 1:
        raise ValueError('Invalid effective batch/memory fraction')
    good = [r for r in records if (r['peak_reserved_bytes'] <= safety_fraction*free_bytes
                                   and effective_batch % r['microbatch'] == 0
                                   and r['samples_per_second'] > 0)]
    if not good:
        raise ValueError('No measured microbatch satisfies the memory and effective batch bounds')
    chosen = max(good, key=lambda r: (r['samples_per_second'], -r['microbatch']))
    return dict(microbatch=int(chosen['microbatch']),
                accumulate_steps=effective_batch//int(chosen['microbatch']),
                effective_batch=effective_batch,
                measured_samples_per_second=float(chosen['samples_per_second']),
                measured_peak_reserved_gib=round(chosen['peak_reserved_bytes']/2**30, 3))


def _gpu_worker(args):
    """Real per-configuration optimizer/EMA probe. Runs in isolated subprocess."""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA not available: use a CUDA-enabled PyTorch build on the target GPU')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(args.device)
    free_start, total = torch.cuda.mem_get_info(args.device)
    model_config = paper_configuration(args.model_channels)
    check_memory_budget(model_config, free_start, mode='base_train',
                        microbatch=args.worker_batch, patch_size=args.patch_size)

    from .model import build_model, DDPM
    net = build_model(model_config).to(args.device).train()
    net.gradient_checkpointing = not args.no_activation_checkpointing
    ddpm = DDPM(steps=1000).to(args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-6)
    ema = [p.detach().clone() for p in net.parameters()]
    amp = args.mixed_precision != 'none'
    amp_dtype = torch.bfloat16 if args.mixed_precision == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=(args.mixed_precision == 'fp16'))
    accum = args.effective_batch//args.worker_batch

    def step():
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            # The real trainer samples patches on CPU, then copies them to GPU.
            x = torch.randn(args.worker_batch, 1, args.patch_size, args.patch_size).to(args.device)
            with (torch.autocast('cuda', dtype=amp_dtype) if amp else nullcontext()):
                loss = ddpm.loss(net, x)/accum
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite probe loss')
            scaler.scale(loss).backward()
            del x, loss
        scaler.unscale_(optimizer)
        for p in net.parameters():
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise FloatingPointError('Invalid probe gradient')
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            for p, avg in zip(net.parameters(), ema):
                avg.lerp_(p.detach(), .001)
        torch.cuda.synchronize(args.device)

    for _ in range(args.warmup_steps):
        step()
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    for _ in range(args.timed_steps):
        step()
    elapsed = time.perf_counter()-started
    with torch.inference_mode():
        net.eval()
        x = torch.randn(args.worker_batch, 1, args.patch_size, args.patch_size).to(args.device)
        t = torch.zeros(args.worker_batch, dtype=torch.long, device=args.device)
        # Actual training validation uses FP32 even when optimization uses BF16.
        # Cover that memory peak too (not only the mixed-precision training step).
        noisy = ddpm.q_sample(x,t,torch.randn_like(x))
        predicted = net(noisy,t)
        _ = ddpm.predict_clean(noisy, t, predicted)
        torch.cuda.synchronize(args.device)
    report = dict(microbatch=args.worker_batch, accumulation=accum,
                  effective_batch=args.effective_batch,
                  seconds_per_optimizer_step=elapsed/args.timed_steps,
                  samples_per_second=args.effective_batch*args.timed_steps/elapsed,
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                  free_before_bytes=free_start, total_bytes=total,
                  architecture='g2_author', channels=args.model_channels,
                  patch_size=args.patch_size, mixed_precision=args.mixed_precision,
                  activation_checkpointing=net.gradient_checkpointing,
                  device_name=torch.cuda.get_device_name(args.device))
    print(json.dumps(report, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--model-channels', type=int, default=32)
    ap.add_argument('--patch-size', type=int, default=128)
    ap.add_argument('--effective-batch', type=int, default=16)
    ap.add_argument('--candidates', nargs='+', type=int, default=[1,2,4,8])
    ap.add_argument('--mixed-precision', choices=['bf16','fp16','none'], default='bf16')
    ap.add_argument('--no-activation-checkpointing', action='store_true')
    ap.add_argument('--safety-fraction', type=float, default=.75,
                    help='Maximum allowed measured reserved memory as fraction of initially free VRAM')
    ap.add_argument('--warmup-steps', type=int, default=1)
    ap.add_argument('--timed-steps', type=int, default=2)
    ap.add_argument('--out', default='outputs/g2_batch_probe.json')
    ap.add_argument('--worker-batch', type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if (not args.device.startswith('cuda') or type(args.patch_size) is not int or args.patch_size < 16
            or args.patch_size % 16 or args.effective_batch < 1 or args.warmup_steps < 0
            or args.timed_steps < 1 or not 0 < args.safety_fraction <= .85):
        ap.error('Invalid CUDA device, patch size, effective batch, probe iterations, or safety fraction')
    paper_configuration(args.model_channels)
    if args.worker_batch is not None:
        if args.worker_batch <= 0 or args.effective_batch % args.worker_batch:
            ap.error('Worker microbatch must divide the effective batch')
        _gpu_worker(args)
        return
    if (not args.candidates or args.candidates != sorted(set(args.candidates))
            or args.candidates[0] != 1 or any(m < 1 or args.effective_batch % m for m in args.candidates)):
        ap.error('Candidates must be unique increasing divisors of effective batch, starting at 1')
    if not torch.cuda.is_available():
        ap.error('A CUDA-capable GPU is required to collect a memory profile')
    torch.cuda.set_device(args.device)
    free, total = torch.cuda.mem_get_info(args.device)
    if free < 4*2**30:
        ap.error('Less than 4 GiB free VRAM; stop other GPU jobs before profiling')
    report = dict(device=torch.cuda.get_device_name(args.device),
                  device_capability=list(torch.cuda.get_device_capability(args.device)),
                  cuda_runtime=torch.version.cuda, torch_version=torch.__version__,
                  initial_free_bytes=free, total_bytes=total,
                  model_config=paper_configuration(args.model_channels),
                  patch_size=args.patch_size, effective_batch=args.effective_batch,
                  mixed_precision=args.mixed_precision,
                  activation_checkpointing=not args.no_activation_checkpointing,
                  safety_fraction=args.safety_fraction, records=[], rejected=[])
    print(f'GPU: {report["device"]}; free {free/2**30:.2f} GiB; target '
          f'<= {args.safety_fraction:.0%} of free VRAM. Each probe is isolated.', flush=True)
    history = []
    for candidate in args.candidates:
        if history and not next_batch_is_safe(history, candidate, free, args.safety_fraction):
            report['rejected'].append({'microbatch':candidate,'reason':'Projected peak exceeds headroom'})
            break
        command = [sys.executable, '-m', 'experiments.diffusion_prior.batch_probe',
                   '--device', args.device, '--model-channels', str(args.model_channels),
                   '--patch-size', str(args.patch_size), '--effective-batch', str(args.effective_batch),
                   '--mixed-precision', args.mixed_precision, '--warmup-steps', str(args.warmup_steps),
                   '--timed-steps', str(args.timed_steps), '--worker-batch', str(candidate)]
        if args.no_activation_checkpointing:
            command.append('--no-activation-checkpointing')
        print(f'Probing microbatch={candidate}...', flush=True)
        try:
            child = subprocess.run(command, text=True, capture_output=True, timeout=1800)
        except subprocess.TimeoutExpired:
            report['rejected'].append({'microbatch':candidate,'reason':'Probe timed out'})
            break
        if child.returncode != 0:
            report['rejected'].append({'microbatch':candidate,
                                      'reason':child.stderr[-1000:] or f'exit {child.returncode}'})
            break
        try:
            measured = json.loads(child.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError(f'Probe did not return a memory profile: {child.stdout[-500:]}') from exc
        report['records'].append(measured)
        history.append((candidate, measured['peak_reserved_bytes']))
        print(f'  peak reserved={measured["peak_reserved_bytes"]/2**30:.2f} GiB; '
              f'{measured["samples_per_second"]:.2f} patches/s', flush=True)
        if measured['peak_reserved_bytes'] > free*args.safety_fraction:
            report['rejected'].append({'microbatch':candidate,'reason':'Measured peak exceeds budget'})
            break
        # Do not increase batch if external processes consumed free VRAM mid-probe.
        now_free, _ = torch.cuda.mem_get_info(args.device)
        free = min(free, now_free)
    report['free_after_probes_bytes'] = free
    if report['records']:
        try:
            report['recommendation'] = select_fastest(report['records'], free,
                                                   args.effective_batch, args.safety_fraction)
        except ValueError:
            report['recommendation'] = None
        if report['recommendation'] is not None:
            selected = report['recommendation']
            print(f'Recommended: --batch-size {selected["microbatch"]} '
                  f'--accumulate-steps {selected["accumulate_steps"]} '
                  f'(new run only; not bitwise-resumable from another batch setting)', flush=True)
    else:
        report['recommendation'] = None
    if report['recommendation'] is None:
        print('No viable configuration measured. Keep current settings or investigate CUDA errors.', flush=True)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing memory profile {dest}')
    dest.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f'Wrote {dest}', flush=True)


if __name__ == '__main__':
    main()
