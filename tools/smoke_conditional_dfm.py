"""Tiny synthetic CPU exercise. Outputs are ALWAYS unqualified research weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifwi_research.train_conditional_dfm import TrainingConfig, train
from ifwi_research.conditional_dfm import ARCHITECTURE, ARCHITECTURES


def create_manifest(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    dataset = out/'synthetic'; dataset.mkdir()
    records = []
    z, x = np.mgrid[:24, :24]
    for index, split in enumerate(('train', 'train', 'train', 'validation', 'test')):
        # Distinct analytic fixtures, NOT geological independence evidence.
        clean = 1800+120*index+25*z+90*np.sin(x/(2+index))*(z>8+index)
        background = 1800+120*index+22*z+np.zeros_like(x)
        mask = ((z<8) | ((x<4)&(z<15))).astype('float32')
        anchor = clean+45*np.cos(x/3)+20*np.sin(z/5)
        record = dict(model_id=f'synthetic_{index}', family_id=f'analytic_fixture_{index}',
                      split=split, spacing_m=20., condition_source='synthetic_smoke')
        for name, value in dict(clean=clean, background=background, anchor=anchor, mask=mask).items():
            path = dataset/f'{index}_{name}.npy'
            np.save(path, value.astype('float32'))
            record[name+'_path'] = path.name
        records.append(record)
    manifest = dataset/'manifest.json'
    manifest.write_text(json.dumps(dict(schema='ifwi_conditional_dataset_v1', records=records), indent=2))
    return manifest


def run(out, *, steps=20, architecture=ARCHITECTURE):
    manifest = create_manifest(out)
    torch.set_num_threads(1)
    summary = train(manifest, Path(out)/'training', spacing_m=20., config=TrainingConfig(
        architecture=architecture, steps=steps, patch_size=16, width=16, group_size=8, groups_per_step=2,
        validate_every=max(1, steps//4), learning_rate=5e-4, ema_decay=.9))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--architecture', choices=ARCHITECTURES, default=ARCHITECTURE)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))
