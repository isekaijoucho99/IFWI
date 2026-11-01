"""Paired original-FDTD/ROM measurements and a bounded residual-IFWI bridge."""
from __future__ import annotations
import copy
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from experiments.residual_ifwi_experiment import (_simulate,waveform_mse,train_update,
    VelocityParameterization,evaluate_data_mse)
from experiments.residual_ifwi_report import velocity_metrics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def evaluate(v,problem,rom=None,backward=False):
    value=v.detach().clone().requires_grad_(backward);gradient=torch.zeros_like(value) if backward else None
    pieces=[];loss_value=0.
    items=problem['propagators'] if rom is None else [dict(solver=rom,first=0,last=problem['observed'].shape[1])]
    with torch.set_grad_enabled(backward):
        for item in items:
            prediction=_simulate(item['solver'],value,problem['wavelet'])
            target=problem['observed'][:,item['first']:item['last']]
            loss=waveform_mse(prediction,target,problem['observed'].shape[1])
            if backward:gradient+=torch.autograd.grad(loss,value)[0].detach()
            loss_value+=float(loss.detach());pieces.append(prediction.detach())
    return torch.cat(pieces,dim=1),gradient,loss_value


def synchronized(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


def paired_timings(functions,device,repeats):
    # Warm both; alternate order. Timed functions include assembly and backward.
    for fn in functions.values():
        result=fn();del result
    synchronized(device);rows={name:[] for name in functions};outputs={}
    for repeat in range(repeats):
        names=list(functions)
        if repeat%2:names.reverse()
        for name in names:
            synchronized(device)
            resident=torch.cuda.memory_allocated(device) if str(device).startswith('cuda') else 0
            if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter();result=functions[name]();synchronized(device)
            elapsed=time.perf_counter()-start
            peak=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else 0
            reserved=torch.cuda.max_memory_reserved(device) if str(device).startswith('cuda') else 0
            rows[name].append(dict(seconds=elapsed,resident_allocated_mib=resident/2**20,
                peak_allocated_mib=peak/2**20,incremental_peak_mib=(peak-resident)/2**20,peak_reserved_mib=reserved/2**20))
            outputs[name]=result
    summary={name:dict(samples=values,median_seconds=float(np.median([v['seconds'] for v in values])),
        max_peak_allocated_mib=max(v['peak_allocated_mib'] for v in values),
        max_incremental_peak_mib=max(v['incremental_peak_mib'] for v in values)) for name,values in rows.items()}
    return summary,outputs


def error_metrics(actual,reference):
    actual=np.asarray(actual,dtype='float64');reference=np.asarray(reference,dtype='float64')
    norm=float(np.linalg.norm(reference));error=float(np.linalg.norm(actual-reference))
    return dict(relative=error/norm if norm>1e-20 else None,absolute=error,reference_norm=norm,
        cosine=float(np.vdot(actual.ravel(),reference.ravel()).real/(np.linalg.norm(actual)*norm)) if norm>1e-20 and np.linalg.norm(actual)>0 else None)


class SnapshotObserver(torch.nn.Module):
    def __init__(self,base,indices,nz,nx):
        super().__init__();self.base=base;self.indices=set(indices);self.nz=nz;self.nx=nx;self.count=0;self.fields=[]
    def forward(self,*args,**kwargs):
        output=self.base(*args,**kwargs)
        if self.count in self.indices:
            top=0 if self.base.freeSurface else self.base.npad
            self.fields.append(output[1][:,:,top:top+self.nz,self.base.npad:self.base.npad+self.nx].detach().cpu().clone())
        self.count+=1;return output


@torch.no_grad()
def original_snapshots(v,problem,indices):
    fields=[];nz,nx=v.shape[-2:]
    for item in problem['propagators']:
        solver=item['solver'];base=solver.fd;observer=SnapshotObserver(base,indices,nz,nx);solver.fd=observer
        try:_simulate(solver,v,problem['wavelet'])
        finally:solver.fd=base
        fields.append(torch.stack(observer.fields).squeeze(1))
    return torch.cat(fields,dim=1).numpy()


def make_model(background,cfg,device):
    m=cfg['model']
    return VelocityParameterization(background,cfg['data']['grid_spacing_m'],m['neurons'],m['omega_0'],
        m['mean_kmps'],m['std_kmps'],cfg['parameterization']).to(device)


def siren_gradient_check(background,cfg,problem,rom,device):
    torch.manual_seed(3);model=make_model(background,cfg,device)
    # Nonzero head activates hidden-layer sensitivities; this is a declared
    # diagnostic state, not claimed as a trained inversion checkpoint.
    with torch.no_grad():model.net.linear[-1].weight.normal_(0,.002)
    v=model();_,a,_=evaluate(v,problem,backward=True);_,b,_=evaluate(v,problem,rom,backward=True)
    parameters=tuple(model.parameters())
    ga=torch.autograd.grad(v,parameters,grad_outputs=a,retain_graph=True)
    gb=torch.autograd.grad(v,parameters,grad_outputs=b)
    result=error_metrics(torch.cat([p.ravel() for p in gb]).cpu().numpy(),torch.cat([p.ravel() for p in ga]).cpu().numpy())
    result['reference_parameter_gradient_norms']=[float(g.norm()) for g in ga]
    result['state']='same initialized hidden weights with a fixed-seed nonzero output head'
    return result


def inversion_pilot(background,truth,cfg,problem,rom,out,device,updates):
    torch.manual_seed(3);base=make_model(background,cfg,device);results=[]
    encoded={**problem,'propagators':[dict(solver=rom,first=0,last=problem['observed'].shape[1])]}
    for name,p in [('fdtd',problem),('rom',encoded)]:
        model=copy.deepcopy(base);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'])
        rows=[];arm=out/name;arm.mkdir()
        for step in range(1,updates+1):
            before=copy.deepcopy(model.state_dict());old_opt=copy.deepcopy(optimizer.state_dict())
            try:
                report=train_update(model,p,optimizer);synchronized(device)
            except BaseException:
                model.load_state_dict(before);optimizer.load_state_dict(old_opt);raise
            with torch.no_grad():
                mse=evaluate_data_mse(model,problem);velocity=model().detach().cpu().numpy()[0]
            row=dict(update=step,full_original_fdtd_mse=mse,**velocity_metrics(velocity,truth));rows.append(row)
            np.save(arm/f'velocity_{step:04d}.npy',velocity)
            temporary=arm/'checkpoint.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=step),temporary)
            temporary.replace(arm/'checkpoint.pt');save_json(arm/'history.json',rows)
        results.append(dict(method=name,history=rows))
    return results
