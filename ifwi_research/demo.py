"""Standalone small acoustic test; NOT a Marmousi/Overthrust benchmark."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import torch
from .engine import ResearchEngine,ResearchConfig
from .toy_wave import TinyINR,ToyWaveObjective
from .dfm import FrozenDFMPrior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',default='baseline',choices=['baseline','routing','wavelet','interface','gradient_attention','dfm'])
    p.add_argument('--steps',type=int,default=10); p.add_argument('--out',required=True)
    p.add_argument('--device',default='cpu'); p.add_argument('--prior')
    args=p.parse_args()
    if args.steps<1: p.error('--steps must be positive')
    torch.set_num_threads(1); torch.manual_seed(3)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    h,w=16,24
    depth=torch.arange(h,device=args.device)[:,None].expand(h,w)
    background=(2000+12*depth.float())[None]; truth=background.clone(); truth[:,7:12,8:17]+=250
    objective=ToyWaveObjective(truth,shots=4,nt=80,spacing_m=20.,dt_s=.002)
    prior=FrozenDFMPrior(args.prior,20.,args.device) if args.prior else None
    cfg=ResearchConfig(method=args.method,attention_block=2,audit_every=5,prior_start=1)
    engine=ResearchEngine(TinyINR(background,width=16),background,objective,20.,cfg,prior=prior)
    started=time.perf_counter()
    with (out/'history.jsonl').open('w',encoding='utf-8') as log:
        for _ in range(args.steps):
            report=engine.step()
            # Evaluation only; never chooses an update or an optimizer state.
            with torch.no_grad(): report['oracle_rmse_mps']=float((engine.model()-truth).square().mean().sqrt())
            log.write(json.dumps(report,allow_nan=False)+'\n')
    torch.save(engine.state_dict(),out/'checkpoint.pt')
    if args.device.startswith('cuda'): torch.cuda.synchronize(args.device)
    summary={'method':args.method,'updates':args.steps,'elapsed_seconds':time.perf_counter()-started,
             'is_geological_benchmark':False,'solver':'standalone tiny FD fixture, not repository RNN-FD',
             'final_oracle_rmse_mps':report['oracle_rmse_mps'],'counters':objective.counters,
             'torch':str(torch.__version__),'device':args.device}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
