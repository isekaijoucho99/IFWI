"""Train the small grouped DFM prior on independent .npy velocity models.

python -m research_ideas.train_dfm --train-data TRAIN --validation-data VALID \
    --out weights/dfm --steps 20000 --spacing-m 15 --device cuda:0
Synthetic smoke data are plumbing tests, never evidence of a useful prior.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .dfm import TwoTimeTransport,grouped_dfm_loss


def model_digest(array):
    a=np.ascontiguousarray(array,dtype=np.float32)
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


def load_models(paths,patch_size):
    arrays=[]; manifest=[]
    for path in paths:
        path=Path(path); a=np.load(path,allow_pickle=False).astype(np.float32)
        if a.ndim!=2 or min(a.shape)<patch_size or not np.isfinite(a).all() or a.min()<100 or a.max()>15000:
            raise ValueError(f'Expected a sufficiently large 2D physical m/s model: {path}')
        arrays.append(a); manifest.append(dict(path=str(path.resolve()),sha256=model_digest(a)))
    if not arrays: raise ValueError('Empty velocity dataset')
    return arrays,manifest


def train(train_paths,validation_paths,out,*,steps=2000,patch_size=32,width=32,group_size=8,
          groups=2,learning_rate=1e-4,spacing_m=15.,seed=3,device='cpu',smoke_only=False):
    for name,n in [('steps',steps),('patch_size',patch_size),('groups',groups),('group_size',group_size)]:
        if type(n) is not int or n<1: raise ValueError(f'{name} must be a positive integer')
    if group_size<2 or patch_size<4 or not math.isfinite(spacing_m) or spacing_m<=0:
        raise ValueError('Require group_size >=2, patch >=4 and positive spacing')
    if not math.isfinite(learning_rate) or learning_rate<=0: raise ValueError('Invalid training rate')
    train_arrays,train_manifest=load_models(train_paths,patch_size)
    val_arrays,val_manifest=load_models(validation_paths,patch_size)
    if {m['sha256'] for m in train_manifest}&{m['sha256'] for m in val_manifest}:
        raise ValueError('The same independent model occurs in training and validation')
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(seed)
    if str(device).startswith('cuda') and not torch.cuda.is_available(): raise ValueError('CUDA is unavailable')
    net=TwoTimeTransport(width).to(device); optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate)
    random=np.random.default_rng(seed); generator=torch.Generator(device=device).manual_seed(seed+1)
    center,scale=3000.,1500.; history=[]; best=float('inf'); best_state=None; best_step=0
    batch=groups*group_size; started=time.perf_counter()
    def patches(arrays,n):
        selected=[]
        for _ in range(n):
            a=arrays[int(random.integers(len(arrays)))]; h,w=a.shape
            z=int(random.integers(h-patch_size+1)); x=int(random.integers(w-patch_size+1))
            selected.append(a[z:z+patch_size,x:x+patch_size])
        return (torch.from_numpy(np.stack(selected)[:,None]).to(device)-center)/scale
    for update in range(1,steps+1):
        data=patches(train_arrays,batch)
        noise=torch.randn(data.shape,generator=generator,device=device)
        times=torch.rand(groups,2,generator=generator,device=device).sort(1).values
        # Explicit endpoint examples train the direct-generation mode as well.
        times[0]=torch.tensor([0.,1.],device=device)
        optimizer.zero_grad(set_to_none=True)
        loss=grouped_dfm_loss(net,noise,data,times[:,0],times[:,1],group_size)
        if not torch.isfinite(loss): raise FloatingPointError('DFM loss is not finite')
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1.); optimizer.step()
        row=dict(update=update,training_drift_loss=float(loss.detach()))
        if update==1 or update%max(1,steps//10)==0 or update==steps:
            with torch.no_grad():
                target=patches(val_arrays,max(8,batch))
                sample=net(torch.randn(target.shape,generator=generator,device=device),0.,1.)
                q,p=sample.flatten(1),target.flatten(1)
                energy=(2*torch.cdist(q,p).mean()-torch.cdist(q,q).mean()-torch.cdist(p,p).mean())/math.sqrt(q.shape[1])
                value=float(energy); row['validation_energy_distance']=value
                row['sample_std']=float(sample.std())
                if value<best:
                    best=value; best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}; best_step=update
        history.append(row)
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)
    metadata=dict(width=width,patch_size=patch_size,spacing_m=spacing_m,center_mps=center,scale_mps=scale,
        trained_steps=best_step,total_training_steps=steps,smoke_only=bool(smoke_only),
        training_models=train_manifest,validation_models=val_manifest,group_size=group_size,groups=groups,
        seed=seed,selection='validation energy distance, no inversion benchmark labels',
        seconds=time.perf_counter()-started)
    checkpoint=out/'best.pt'
    torch.save(dict(schema='ifwi_dfm_v1',model=best_state,metadata=metadata),checkpoint)
    (out/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False),encoding='utf-8')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return checkpoint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data',nargs='+'); p.add_argument('--validation-data',nargs='+')
    p.add_argument('--out',required=True); p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--patch-size',type=int,default=32); p.add_argument('--width',type=int,default=32)
    p.add_argument('--group-size',type=int,default=8); p.add_argument('--groups',type=int,default=2)
    p.add_argument('--learning-rate',type=float,default=1e-4); p.add_argument('--spacing-m',type=float,default=15.)
    p.add_argument('--seed',type=int,default=3); p.add_argument('--device',default='cpu')
    p.add_argument('--synthetic-smoke',action='store_true'); args=p.parse_args()
    def expand(paths):
        return [f for path in paths for f in (sorted(Path(path).glob('*.npy')) if Path(path).is_dir() else [Path(path)])]
    if args.synthetic_smoke:
        folder=Path(args.out).parent/(Path(args.out).name+'_smoke_data'); folder.mkdir(parents=True,exist_ok=False)
        n=max(16,args.patch_size*2); z=np.linspace(0,1,n,dtype=np.float32)[:,None]; x=np.linspace(0,1,n,dtype=np.float32)[None]
        for name,phase in [('train',0.),('validation',1.3)]:
            a=1800+1400*z+150*np.sin(5*x+phase)*(z>.5)
            np.save(folder/(name+'.npy'),a.astype(np.float32))
        train_paths=[folder/'train.npy']; val_paths=[folder/'validation.npy']
    else:
        if not args.train_data or not args.validation_data: p.error('Independent train and validation model paths are required')
        train_paths,val_paths=expand(args.train_data),expand(args.validation_data)
    result=train(train_paths,val_paths,args.out,steps=args.steps,patch_size=args.patch_size,width=args.width,
        group_size=args.group_size,groups=args.groups,learning_rate=args.learning_rate,
        spacing_m=args.spacing_m,seed=args.seed,device=args.device,smoke_only=args.synthetic_smoke)
    print(result)


if __name__=='__main__': main()
