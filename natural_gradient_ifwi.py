"""Explicit experimental damped natural-gradient / matched Adam IFWI runner."""
import argparse
from pathlib import Path
import json
import torch
import yaml
from experiments.natural_gradient_experiment import run_experiment,validate_config

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('outputs/natural_gradient'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=1)
    group=p.add_mutually_exclusive_group();group.add_argument('--resume',type=Path);group.add_argument('--initialize-from',type=Path)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.threads<1:p.error('--threads must be positive')
    cfg=validate_config(yaml.safe_load(a.config.read_text(encoding='utf-8-sig')))
    if a.dry_run: print(json.dumps(cfg,indent=2));return
    torch.set_num_threads(a.threads)
    print(run_experiment(cfg,a.output_dir,a.device,resume=a.resume,initialize_from=a.initialize_from))

if __name__=='__main__':main()
