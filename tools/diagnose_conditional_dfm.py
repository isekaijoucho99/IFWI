"""CPU-only synthetic convergence study; never grants prior qualification.

Runs the SAME fixture/data/lr/width as the 20-step smoke, with a longer budget
and multiple initialization seeds. Outputs initial/raw/EMA and per-draw/mean
validation metrics. Do not infer real geological efficacy from this experiment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifwi_research.conditional_dfm import ARCHITECTURES
from ifwi_research.train_conditional_dfm import TrainingConfig, group_drift_loss, train
from tools.smoke_conditional_dfm import create_manifest


def affine_control(steps=300):
    """Endpoint-only toy learnability check; not a geological forward model."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(410)
        weights = torch.nn.Parameter(torch.tensor([1., 0.]))
        optimizer = torch.optim.Adam([weights], lr=.03)
        for _ in range(steps):
            noise = torch.randn(4, 8, 1, 8, 8)
            condition = torch.randn(4, 1, 1, 8, 8).expand_as(noise)
            prediction = weights[0]*noise + weights[1]*condition
            loss = group_drift_loss(prediction, condition).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        noise = torch.randn(4, 8, 1, 8, 8)
        condition = torch.randn(4, 1, 1, 8, 8).expand_as(noise)
        error = weights[0]*noise + weights[1]*condition - condition
        return dict(steps=steps, weights=weights.detach().tolist(),
                    heldout_normalized_rmse=float(error.detach().square().mean().sqrt()))


def run(out, *, steps=1000, seeds=(3, 7, 19)):
    torch.set_num_threads(1)
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    manifest = create_manifest(out/'fixture')
    report = dict(scope='Synthetic CPU convergence only; all checkpoints unqualified',
                  affine_control=affine_control(), runs=[])
    for architecture in ARCHITECTURES:
        for seed in seeds:
            config = TrainingConfig(architecture=architecture, steps=steps, seed=seed,
                patch_size=16, width=16, group_size=8, groups_per_step=2,
                validate_every=max(1, steps//10), learning_rate=5e-4, ema_decay=.9)
            summary = train(manifest, out/f'{architecture}_seed{seed}', spacing_m=20., config=config)
            report['runs'].append(dict(architecture=architecture, seed=seed, summary=summary))
            (out/'diagnosis.json').write_text(json.dumps(report, indent=2, allow_nan=False))
            print(json.dumps(dict(architecture=architecture, seed=seed,
                diagnostics=summary['training_diagnostics'])), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True); parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[3, 7, 19])
    run(**vars(parser.parse_args()))
