"""Train the small conditional DFM on independent geological-model NPY files.

Split FULL geological models into train/val before patch extraction. Identical
arrays are rejected across splits; parent-model/crop leakage must also be audited
by the researcher. No test-model labels are loaded by the inversion engine.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from .dfm import TwoTimeTransport, conditional_group_loss, smooth


def array_hash(array: np.ndarray) -> str:
    a=np.ascontiguousarray(array,dtype='<f4')
    return hashlib.sha256(str(tuple(a.shape)).encode()+a.tobytes()).hexdigest()


def model_inventory(directory: str | Path,patch_size: int) -> list[dict]:
    paths=sorted(Path(directory).glob('*.npy'))
    if not paths: raise ValueError(f'No independent-model .npy files in {directory}')
    result=[]
    for path in paths:
        a=np.load(path,allow_pickle=False)
        if a.ndim!=2 or min(a.shape)<patch_size or not np.isfinite(a).all() or a.min()<=0:
            raise ValueError(f'{path}: expected finite positive m/s HxW model, at least patch_size on each axis')
        result.append({'path':str(path.resolve()),'shape':list(a.shape),'hash':array_hash(a)})
    return result


def validate_split(train: list[dict],val: list[dict]) -> None:
    ta={x['hash'] for x in train}; va={x['hash'] for x in val}
    if ta&va: raise ValueError('Train/validation model overlap (canonical float32 content hash)')
    if len(ta)!=len(train) or len(va)!=len(val): raise ValueError('Duplicate models within a split')


def _patch(entry,side,rng,device,mean,scale):
    # Model-at-a-time load; does not accumulate the whole dataset on the GPU.
    a=np.load(entry['path'],mmap_mode='r',allow_pickle=False)
    iz=int(torch.randint(a.shape[0]-side+1,(1,),generator=rng)); ix=int(torch.randint(a.shape[1]-side+1,(1,),generator=rng))
    patch=np.array(a[iz:iz+side,ix:ix+side],dtype=np.float32,copy=True)
    return (torch.from_numpy(patch)[None,None].to(device)-mean)/scale


@torch.no_grad()
def _validation(model,inventory,side,device,mean,scale,kernel):
    rng=torch.Generator().manual_seed(941); error=0.; noisy_error=0.
    for entry in inventory:
        clean=_patch(entry,side,rng,device,mean,scale)
        noise=torch.randn(clean.shape,generator=rng).to(device)
        state=.8*clean+.2*noise
        pred=model(state,.8,1.,smooth(clean,kernel))
        error+=float((pred-clean).square().mean())*scale**2
        noisy_error+=float((state-clean).square().mean())*scale**2
    return {'validation_rmse_mps':math.sqrt(error/len(inventory)),
            'noisy_input_rmse_mps':math.sqrt(noisy_error/len(inventory))}


def train(train_dir,val_dir,out,*,spacing_m: float,patch_size: int=64,width: int=32,
          group_size: int=16,groups_per_step: int=2,steps: int=20000,
          validate_every: int=100,seed: int=3,device: str='cpu',learning_rate: float=1e-4,
          mean_mps: float=3000.,scale_mps: float=1500.,condition_kernel: int=9,
          temperature: float=1.) -> dict:
    for name,value in [('spacing_m',spacing_m),('learning_rate',learning_rate),('scale_mps',scale_mps),('temperature',temperature)]:
        if not math.isfinite(value) or value<=0: raise ValueError(f'{name} must be finite positive')
    if not math.isfinite(mean_mps): raise ValueError('Invalid mean_mps')
    for name,value in [('patch_size',patch_size),('group_size',group_size),('groups_per_step',groups_per_step),('steps',steps),('validate_every',validate_every)]:
        if type(value) is not int or value<1: raise ValueError(f'{name} must be a positive integer')
    if group_size<2 or patch_size<4 or condition_kernel%2!=1 or condition_kernel<1:
        raise ValueError('group_size>=2, patch_size>=4 and odd positive condition_kernel required')
    train_files=model_inventory(train_dir,patch_size); val_files=model_inventory(val_dir,patch_size)
    validate_split(train_files,val_files)
    out=Path(out); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(seed)
    model=TwoTimeTransport(width).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=learning_rate)
    rng=torch.Generator().manual_seed(seed); best=float('inf'); started=time.perf_counter()
    meta={'schema':'ifwi_small_dfm_v1','width':width,'spacing_m':spacing_m,'mean_mps':mean_mps,
          'scale_mps':scale_mps,'condition_kernel':condition_kernel,'patch_size':patch_size,
          'toy_only':False,'seed':seed,'group_size':group_size,'groups_per_step':groups_per_step,
          'temperature':temperature,'train_model_hashes':[x['hash'] for x in train_files],
          'val_model_hashes':[x['hash'] for x in val_files],
          'time_sampling':'20% endpoint 0->1; else t=0.9U, r=t+(1-t)*(0.1+0.9U)',
          'notes':'Small-CNN research prototype; trained_steps is not a quality certificate'}
    (out/'dataset_manifest.json').write_text(json.dumps({'train':train_files,'validation':val_files},indent=2),encoding='utf-8')
    def save(path,step):
        payload={'meta':{**meta,'trained_steps':step},'model':model.state_dict(),
                 'optimizer':opt.state_dict(),'rng_state':rng.get_state()}
        temp=path.with_suffix('.tmp'); torch.save(payload,temp); temp.replace(path)
    with (out/'history.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,steps+1):
            opt.zero_grad(set_to_none=True); value=0.
            for _ in range(groups_per_step):
                index=int(torch.randint(len(train_files),(1,),generator=rng))
                clean=_patch(train_files[index],patch_size,rng,device,mean_mps,scale_mps)
                if float(torch.rand((),generator=rng))<.2: t,r=0.,1.
                else:
                    t=.9*float(torch.rand((),generator=rng))
                    r=t+(1-t)*(.1+.9*float(torch.rand((),generator=rng)))
                loss=conditional_group_loss(model,clean,t=t,r=r,group_size=group_size,generator=rng,
                    kernel=condition_kernel,temperature=temperature)/groups_per_step
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite drift regression loss')
                loss.backward(); value+=float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            record={'step':step,'drift_surrogate_loss':value}
            if step%validate_every==0 or step==steps:
                model.eval(); record.update(_validation(model,val_files,patch_size,device,mean_mps,scale_mps,condition_kernel)); model.train()
                if record['validation_rmse_mps']<best:
                    best=record['validation_rmse_mps']; save(out/'best.pt',step)
                save(out/'last.pt',step)
            if str(device).startswith('cuda'): torch.cuda.synchronize(device)
            record['elapsed_seconds']=time.perf_counter()-started
            log.write(json.dumps(record,allow_nan=False)+'\n'); log.flush()
    summary={'steps':steps,'best_validation_rmse_mps':best,'device':device,'torch':str(torch.__version__),
             'elapsed_seconds':time.perf_counter()-started,
             'warning':'Validation denoising is not evidence of FWI improvement. No pretrained weights shipped.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-dir',required=True); p.add_argument('--val-dir',required=True); p.add_argument('--out',required=True)
    p.add_argument('--grid-spacing-m',type=float,required=True,dest='spacing_m')
    for name,default in [('patch-size',64),('width',32),('group-size',16),('groups-per-step',2),('steps',20000),('validate-every',100),('seed',3),('condition-kernel',9)]:
        p.add_argument('--'+name,type=int,default=default)
    for name,default in [('learning-rate',1e-4),('mean-mps',3000.),('scale-mps',1500.),('temperature',1.)]:
        p.add_argument('--'+name,type=float,default=default)
    p.add_argument('--device',default='cpu')
    print(json.dumps(train(**vars(p.parse_args())),indent=2))

if __name__=='__main__': main()
