"""Generate matched experiment manifests and summarize measured time-error fronts.

Generation never starts GPU jobs. Use run commands from the printed manifest.
No velocity truth is used to select configurations. Pareto curves are post hoc.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path

RECIPES={
 'baseline':dict(method='baseline',safeguard=False),
 'baseline_guard':dict(method='baseline',safeguard=True),
 'routing':dict(method='routing',options=dict(top_k=3,samples=2,refresh_every=20)),
 'routing_uniform':dict(method='routing',options=dict(top_k=0,samples=5,exploration=1.,diversity=0.)),
 'wavelet':dict(method='wavelet',options=dict(high_start=0.,ramp_updates=200,strength=.5)),
 'interfaces':dict(method='interfaces',options=dict(interfaces=3,width_m=60.,warmup=100,ramp_updates=100)),
 'gradient_attention':dict(method='gradient_attention',safeguard=True,options=dict(warmup=100,fit_every=5,ratio=.1)),
}


def generate(out,*,seeds=(0,1,3),datasets=('marmousi','overthrust'),include_dfm=False,
             steps=1000,noise_sigma=0.,background='reference'):
    from paper_supplement import MARMOUSI,OVERTHRUST
    if type(steps) is not int or steps<1 or not math.isfinite(noise_sigma) or noise_sigma<0:
        raise ValueError('Invalid experiment budget or noise')
    if not seeds or any(type(s) is not int or s<0 for s in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('Distinct nonnegative integer seeds required')
    if not datasets or set(datasets)-{'marmousi','overthrust'} or len(set(datasets))!=len(datasets):
        raise ValueError('Unknown/duplicated dataset')
    if background not in ('reference','gaussian','linear'): raise ValueError('Unknown background condition')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    (out/'specs').mkdir(); (out/'recipes').mkdir(); jobs=[]
    recipes=copy.deepcopy(RECIPES)
    if include_dfm:
        for name,blend in [('dfm_dual',.5),('dfm_direct',1.),('dfm_multistep',0.)]:
            recipes[name]=dict(method='dfm',safeguard=True,options=dict(blend=blend,ratio=.1))
    for dataset in datasets:
        cfg=copy.deepcopy(MARMOUSI if dataset=='marmousi' else OVERTHRUST)
        cfg['training']['frequency_schedule']=[dict(max_hz=3.,updates=100),dict(max_hz=5.,updates=100),dict(max_hz=None)]
        cfg['training']['epochs']=steps
        for seed in seeds:
            cfg['seed']=seed; cfg['experiment_name']=f'{dataset}_research_s{seed}'
            spec=dict(name=cfg['experiment_name'],experiment=copy.deepcopy(cfg),control=dict(kind='zero_output'),
                      robustness=dict(condition=f'{dataset}_{background}',noise_sigma=noise_sigma,noise_seed=991))
            if background=='reference':
                spec['background']=dict(kind='smooth400') if dataset=='marmousi' else dict(
                    kind='slowness_gaussian',slowness_sigma_m=135.,velocity_sigma_m=187.5)
            elif background=='linear': spec['background']=dict(kind='linear_depth',top_mps=1500.,gradient_per_s=2.)
            spec_path=f'specs/{dataset}_s{seed}.json'
            (out/spec_path).write_text(json.dumps(spec,indent=2)+'\n',encoding='utf-8')
            for name,original in recipes.items():
                recipe=copy.deepcopy(original); recipe.update(steps=steps,eval_every=50)
                if recipe['method']=='dfm':
                    spacing=int(cfg['data']['grid_spacing_m'])
                    recipe['options']['checkpoint']=f'weights/dfm_{spacing}m/best.pt'
                recipe_path=f'recipes/{dataset}_{name}.json'
                (out/recipe_path).write_text(json.dumps(recipe,indent=2)+'\n',encoding='utf-8')
                jobs.append(dict(name=f'{dataset}_{name}_s{seed}',spec=spec_path,recipe=recipe_path))
    (out/'manifest.json').write_text(json.dumps(jobs,indent=2)+'\n',encoding='utf-8')
    return out


def pareto_front(points):
    """Indices nondominated in BOTH measured seconds and velocity error; keep ties."""
    if any(len(p)!=2 or not all(math.isfinite(x) and x>=0 for x in p) for p in points):
        raise ValueError('Expected finite nonnegative cost/error pairs')
    return [i for i,p in enumerate(points) if not any(
        q[0]<=p[0] and q[1]<=p[1] and (q[0]<p[0] or q[1]<p[1]) for q in points)]


def summarize(directories,out):
    rows=[]
    for directory in directories:
        directory=Path(directory)
        run=json.loads((directory/'run.json').read_text(encoding='utf-8'))
        evaluations=json.loads((directory/'evaluations.json').read_text(encoding='utf-8'))
        for item in evaluations:
            rows.append(dict(run=str(directory),method=run['recipe']['method'],smoke=run['smoke'],
                update=item['update'],seconds=item['seconds'],rmse_mps=item['velocity']['rmse_mps'],
                full_band_loss=item['full_band_loss'],costs=item['costs']))
    if not rows: raise ValueError('No evaluated points')
    # Do not aggregate different problems/seeds as though they form one fair front.
    signatures=[]
    for directory in directories:
        run=json.loads((Path(directory)/'run.json').read_text(encoding='utf-8'))
        identity=run['identity']; signatures.append((identity['truth'],identity['background'],identity['noise'],
            identity['spec']['experiment']['seed'],json.dumps(identity['spec']['experiment']['acquisition'],sort_keys=True),
            json.dumps(identity['spec']['experiment']['forward'],sort_keys=True),identity['device']))
    if any(s!=signatures[0] for s in signatures[1:]):
        raise ValueError('Build each Pareto front on the same dataset/background/noise/seed/acquisition/solver/device')
    front=pareto_front([(r['seconds'],r['rmse_mps']) for r in rows])
    out=Path(out)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(dict(points=rows,pareto_indices=front,
        note='Measured post-hoc front, not an online truth-based stopping rule; no pretrained-cost amortization included'),indent=2),encoding='utf-8')
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate'); g.add_argument('--out',required=True); g.add_argument('--steps',type=int,default=1000)
    g.add_argument('--seeds',type=int,nargs='+',default=[0,1,3]); g.add_argument('--datasets',nargs='+',default=['marmousi','overthrust'])
    g.add_argument('--include-dfm',action='store_true'); g.add_argument('--noise-sigma',type=float,default=0.)
    g.add_argument('--background',choices=['reference','gaussian','linear'],default='reference')
    s=sub.add_parser('summarize'); s.add_argument('--runs',nargs='+',required=True); s.add_argument('--out',required=True)
    a=p.parse_args()
    if a.command=='generate':
        print(generate(a.out,seeds=a.seeds,datasets=a.datasets,include_dfm=a.include_dfm,steps=a.steps,
                       noise_sigma=a.noise_sigma,background=a.background))
    else: print(summarize(a.runs,a.out))

if __name__=='__main__': main()
