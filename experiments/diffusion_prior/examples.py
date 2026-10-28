"""Generate controlled, fixed-budget core_runner specs; never launches training."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path


def build_examples(checkpoint, adapter=None, untrained_checkpoint=None, updates=1000, seed=3,
                   backbone='ms', background='smooth400'):
    import paper_supplement as ps
    if updates < 1 or backbone not in ('ms', 'fr') or background not in ('smooth400', 'gaussian', 'linear_depth'):
        raise ValueError('Invalid example experiment configuration')
    cfg = ps.experiment('E2MS' if backbone == 'ms' else 'E3', seed=seed)
    cfg['training'].update(epochs=updates, checkpoint_interval=min(50, updates), shot_batch_size=1)
    override = {'kind':'smooth400'} if background == 'smooth400' else (
        copy.deepcopy(ps.LINEAR_BACKGROUND) if background == 'linear_depth' else None)
    baseline = ps.spec(f'{backbone}_baseline_s{seed}', cfg, background=override)
    settings = dict(checkpoint=str(checkpoint), warmup=50, every=5, seed=seed, ratio=.025,
                    eta_start=.2, eta_end=.02, anneal_updates=400, patch_batch_size=4,
                    weighting='kl', allow_untrained=False)
    methods = [('01_baseline',None),
               ('02_resize_only',dict(mode='resize_only',warmup=50,every=5,ratio=.025,resize_shape=[128,128])),
               ('03_diffusion',settings)]
    if adapter:
        methods.append(('04_lora',{**settings,'adapter':str(adapter)}))
    if untrained_checkpoint:
        methods.append(('05_untrained',{**settings,'checkpoint':str(untrained_checkpoint),'allow_untrained':True}))
    specs = {}
    for label, prior in methods:
        spec = copy.deepcopy(baseline)
        spec['name'] = f'{backbone}_{label}_s{seed}'
        spec['experiment']['experiment_name'] = spec['name']
        if prior is not None:
            spec['diffusion_prior'] = prior
        specs[label] = spec
    return specs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--untrained-checkpoint')
    ap.add_argument('--out', default='configs/diffusion_prior')
    ap.add_argument('--updates', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--backbone', choices=['ms','fr'], default='ms')
    ap.add_argument('--background', choices=['smooth400','gaussian','linear_depth'], default='smooth400')
    args = vars(ap.parse_args())
    out = Path(args.pop('out'))
    specs = build_examples(**args)
    out.mkdir(parents=True, exist_ok=True)
    for name in specs:
        if (out/f'{name}.json').exists():
            raise FileExistsError(f'Refusing to overwrite {(out/name)}.json; use a new output folder')
    from core_runner import check_spec
    for name, spec in specs.items():
        check_spec(spec)
        (out/f'{name}.json').write_text(json.dumps(spec,indent=2),encoding='utf-8')
        print(out/f'{name}.json')


if __name__ == '__main__':
    main()
