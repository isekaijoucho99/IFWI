"""Bounded dynamic-ROM IFWI experiment; no native audit steers online decisions."""
from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
import psutil
import platform
import scipy
import torch
import yaml
from threadpoolctl import threadpool_limits,threadpool_info
from experiments.residual_ifwi_experiment import (build_forward_problem,validate_config,train_update,require_finite)
from experiments.residual_ifwi_data import load_velocity_data
from experiments.residual_ifwi_report import velocity_metrics
from experiments.rom.discrete import DiscretePMLSystem
from experiments.rom.projector import SparseProjector
from experiments.rom.adaptive import RefreshController
from experiments.rom.solver import PressureROM
from experiments.rom.benchmark import make_model,evaluate,error_metrics,save_json,sha,synchronized

ROOT=Path(__file__).resolve().parents[1]


def cpu_state(module):return {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}


def memory(device):
    info=psutil.Process().memory_info()
    result=dict(host_resident_mib=info.rss/2**20,host_process_peak_mib=getattr(info,'peak_wset',info.rss)/2**20,
        host_private_commit_mib=getattr(info,'private',info.rss)/2**20,
        host_peak_commit_mib=getattr(info,'peak_pagefile',info.rss)/2**20)
    if str(device).startswith('cuda'):
        result.update(cuda_allocated_mib=torch.cuda.memory_allocated(device)/2**20,
            cuda_reserved_mib=torch.cuda.memory_reserved(device)/2**20,
            cuda_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20)
    return result


def audit_state(base,record,problem,system,sources,receivers,options,truth,device):
    model=copy.deepcopy(base);model.load_state_dict(record['model']);v=model()
    pred,g_ref,fd_mse=evaluate(v,problem,backward=True)
    if record['rom'] is None:
        approximate,g_rom,rom_mse=pred,g_ref,fd_mse
    else:
        state=record['rom']
        rom=PressureROM(system,state['basis'].numpy(),sources,receivers,len(problem['wavelet']),
            nfft=options['nfft'],max_hz=options['max_hz'],alias_tolerance=options['alias_tolerance'],
            frequency_batch_size=options['frequency_batch_size'],
            preprojected=(state['stiffness_r'],state['pml_r'])).to(device)
        rom.load_state_dict(state)
        approximate,g_rom,rom_mse=evaluate(v,problem,rom,backward=True)
    parameters=tuple(model.parameters())
    a=torch.autograd.grad(v,parameters,grad_outputs=g_ref,retain_graph=True)
    b=torch.autograd.grad(v,parameters,grad_outputs=g_rom)
    we=error_metrics(approximate.cpu().numpy(),pred.cpu().numpy())
    ge=error_metrics(g_rom.cpu().numpy(),g_ref.cpu().numpy())
    pe=error_metrics(torch.cat([p.ravel() for p in b]).cpu().numpy(),torch.cat([p.ravel() for p in a]).cpu().numpy())
    def passes(error,tolerance):
        return error['relative']<tolerance if error['relative'] is not None else error['absolute']<1e-12
    return dict(completed_updates=record['completed_updates'],used_for_update=record['used_for_update'],
        backend=record['backend'],final_basis_is_preceding_update_basis=not record['used_for_update'],
        waveform=we,velocity_gradient=ge,parameter_gradient=pe,fdtd_mse=fd_mse,rom_mse=rom_mse,
        passes_accuracy_gate=passes(we,.01) and passes(ge,.05) and passes(pe,.05),
        **velocity_metrics(v.detach().cpu().numpy()[0],truth))


def run_arm(name,base,initial_cache,initial_rom,initial_probe,prototype,problem,options,out,device,updates):
    model=copy.deepcopy(base);optimizer=torch.optim.Adam(model.parameters(),lr=options['learning_rate'])
    controller=RefreshController(prototype.system,prototype.sources,prototype.receivers,prototype.wavelet,
        prototype.observed,prototype.projector,options,device)
    controller.cache=initial_cache;controller.initial_probe=initial_probe
    arm=out/name;arm.mkdir();records=[];rows=[];last_rom=None
    result=dict(method=name,completed_updates=0,status='running',steps=rows)
    synchronized(device)
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    for step in range(updates):
        stamp=time.perf_counter();velocity=model().detach()
        saved=cpu_state(model);saved_optimizer=copy.deepcopy(optimizer.state_dict())
        events=[]
        if name=='fdtd':rom=None
        elif name=='frozen' or step==0:rom=initial_rom
        else:_,rom,events=controller.prepare(velocity,name)
        backend='fdtd' if rom is None else 'rom'
        saved_rom=None if rom is None else cpu_state(rom)
        training_problem=problem if rom is None else {**problem,'propagators':[dict(solver=rom,first=0,last=len(prototype.sources))]}
        record=dict(completed_updates=step,used_for_update=True,backend=backend,model=saved,rom=saved_rom)
        records.append(record)
        try:
            train_update(model,training_problem,optimizer)
            require_finite(model(),'post-update velocity');synchronized(device)
        except (ValueError,RuntimeError,FloatingPointError) as error:
            model.load_state_dict(saved);optimizer.load_state_dict(saved_optimizer);record['used_for_update']=False
            result.update(status='numerical_failure',failure=repr(error));break
        result['completed_updates']=step+1;last_rom=saved_rom
        temporary=arm/'checkpoint.tmp'
        torch.save(dict(model=cpu_state(model),optimizer=optimizer.state_dict(),completed_updates=step+1,
            cache_anchor_velocity=controller.cache.velocity,cache_requires_refactorization_on_resume=True),temporary)
        temporary.replace(arm/'checkpoint.pt')
        row=dict(update=step+1,backend=backend,online_seconds=time.perf_counter()-stamp,
            events=events,memory=memory(device));rows.append(row)
        print(json.dumps(dict(event='update',method=name,**row)),flush=True)
        save_json(arm/'progress.json',result)
    if controller.cache is not initial_cache:controller.cache.close()
    synchronized(device);result['online_total_seconds']=time.perf_counter()-start
    result['memory_after_online']=memory(device)
    if result['status']=='running':result['status']='completed'
    records.append(dict(completed_updates=result['completed_updates'],used_for_update=False,
        backend='rom' if last_rom is not None else 'fdtd',model=cpu_state(model),rom=last_rom))
    export_start=time.perf_counter()
    for index,record in enumerate(records):torch.save(record,arm/f'state_{index:03d}.pt')
    if not (arm/'checkpoint.pt').exists():
        temporary=arm/'checkpoint.tmp'
        torch.save(dict(model=cpu_state(model),optimizer=optimizer.state_dict(),completed_updates=result['completed_updates'],
            cache_anchor_velocity=controller.cache.velocity,cache_requires_refactorization_on_resume=True),temporary)
        temporary.replace(arm/'checkpoint.pt')
    result['audit_artifact_export_seconds']=time.perf_counter()-export_start
    save_json(arm/'online.json',result)
    return result,records


def main():
    wall=time.perf_counter();ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='experiments/configs/rom_marmousi_refresh.yaml')
    ap.add_argument('--small',action='store_true');ap.add_argument('--device',default='cpu')
    ap.add_argument('--updates',type=int,choices=range(1,6));ap.add_argument('--output',required=True)
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    provided=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'));cfg=provided['experiment'];options=provided['refresh']
    if args.small:
        cfg['data']['crop_shape']=[20,31];cfg['forward']['nt']=96;cfg['acquisition']['source_x_indices']=[15]
        cfg['model']['neurons']=[2,16,16,16,16,1]
        options.update(rank=16,primal_rank=8,training_frequencies=16,updates=2)
    if args.updates:options['updates']=args.updates
    if not 1<=options['updates']<=5:raise ValueError('This pilot allows at most five updates')
    if not set(options['methods'])<=set(['fdtd','frozen','fresh','correction1','correction2','correction3']):raise ValueError('Unknown method')
    blas_guard=threadpool_limits(limits=options.get('blas_threads',1),user_api='blas')
    options['nfft']=2**int(np.ceil(np.log2(options['pad_factor']*cfg['forward']['nt'])))
    options['learning_rate']=cfg['training']['learning_rate'];validate_config(cfg)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    sources_files=['rnn_fd.py','ifwi_modules.py','experiments/residual_ifwi_experiment.py',
        'experiments/residual_ifwi_data.py','experiments/residual_ifwi_report.py',
        'experiments/run_rom_refresh_pilot.py',*[f'experiments/rom/{name}.py' for name in ['refresh','owned_lu','adaptive','projector','solver','discrete','paper','benchmark']]]
    hashes={name:sha(ROOT/name) for name in sources_files}
    save_json(out/'config.json',dict(experiment=cfg,refresh=options,arguments=vars(args),source_sha256=hashes))
    for name in sources_files:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    common=time.perf_counter();truth,background,provenance=load_velocity_data(cfg,ROOT)
    problem=build_forward_problem(truth,cfg,args.device);synchronized(args.device)
    common_seconds=time.perf_counter()-common
    np.save(out/'true_velocity.npy',truth);np.save(out/'initial_velocity.npy',background);save_json(out/'provenance.json',provenance)
    prep_start=time.perf_counter()
    system=DiscretePMLSystem.from_solver(problem['propagators'][0]['solver']);geometry=problem['geometry_tensors']
    source=system.physical_indices(np.stack([geometry['zs'][0].numpy(),geometry['xs'][0].numpy()],axis=-1))
    receiver=system.physical_indices(np.stack([geometry['zr'][0].numpy(),geometry['xr'][0].numpy()],axis=-1).reshape(-1,2)).reshape(geometry['xr'][0].shape)
    projector=SparseProjector(system,args.device)
    controller=RefreshController(system,source,receiver,problem['wavelet'],problem['observed'],projector,options,args.device)
    velocity=torch.tensor(background,device=args.device)[None]
    _,initial_rom,initial_info=controller.fresh(velocity);initial_cache=controller.cache;initial_cache.pinned=True
    if initial_info['adjoint_source_discrepancy_after']>options['source_consistency_tolerance']:
        save_json(out/'initialization_failure.json',initial_info)
        raise ValueError('Initial basis fails the same adjoint-source consistency check as online refreshes')
    controller.initial_probe=initial_info['probe'];synchronized(args.device)
    initial_seconds=time.perf_counter()-prep_start
    # Warm both physics paths without an optimizer step or network random draw.
    warm=time.perf_counter();evaluate(velocity,problem,backward=True);evaluate(velocity,problem,initial_rom,backward=True)
    synchronized(args.device);warm_seconds=time.perf_counter()-warm
    torch.manual_seed(cfg['seed']);base=make_model(background,cfg,args.device)
    save_json(out/'preparation.json',dict(common_data_seconds=common_seconds,initial_rom_seconds=initial_seconds,
        warmup_seconds=warm_seconds,initial_construction=initial_info,memory=memory(args.device),
        runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
            torch=str(torch.__version__),cuda=torch.version.cuda,
            device_name=torch.cuda.get_device_name(args.device) if args.device.startswith('cuda') else 'CPU',
            outer_threadpools=[{k:v for k,v in item.items() if k!='filepath'} for item in threadpool_info()])))
    results=[];trajectories=[]
    for method in options['methods']:
        result,records=run_arm(method,base,initial_cache,initial_rom,controller.initial_probe,controller,
            problem,options,out,args.device,options['updates'])
        results.append(result);trajectories.append(records)
    # Only now compute reference gradients. No audit influences any arm's updates.
    for result,records in zip(results,trajectories):
        stamp=time.perf_counter();audits=[]
        for index,record in enumerate(records):
            try:row=audit_state(base,record,problem,system,source,receiver,options,truth,args.device)
            except (ValueError,RuntimeError,FloatingPointError) as error:
                row=dict(completed_updates=record['completed_updates'],used_for_update=record['used_for_update'],
                    failure=repr(error),passes_accuracy_gate=False)
            audits.append(row);save_json(out/result['method']/'audit.json',audits)
            print(json.dumps(dict(event='posthoc_audit',method=result['method'],**row)),flush=True)
        result['audit_seconds']=time.perf_counter()-stamp;result['audits']=audits
        result['all_used_gradients_pass']=all(row['passes_accuracy_gate'] for row in audits if row['used_for_update'])
    reference=next((r for r in results if r['method']=='fdtd'),None)
    for result in results:
        result['online_speedup']=reference['online_total_seconds']/result['online_total_seconds'] if reference else None
        result['including_initial_preparation_seconds']=result['online_total_seconds']+(0 if result['method']=='fdtd' else initial_seconds)
        result['including_preparation_audit_export_seconds']=result['including_initial_preparation_seconds']+result['audit_seconds']+result['audit_artifact_export_seconds']
        result['same_accuracy_online_acceleration']=(result['completed_updates']==options['updates']
            and result['all_used_gradients_pass'] and result['online_speedup'] is not None and result['online_speedup']>1)
    initial_cache.close()
    summary=dict(initial_preparation_seconds=initial_seconds,common_data_seconds=common_seconds,warmup_seconds=warm_seconds,
        methods=results,experiment_wall_seconds=time.perf_counter()-wall,
        source_hashes_unchanged=hashes=={name:sha(ROOT/name) for name in sources_files},
        accounting='Online totals include refresh, indicators, transfers, basis snapshots, optimizer and fallbacks; ex-post audit and artifact export separate',
        limits=['Five updates maximum; no long inversion convergence claim',
            'PDE/probe residuals are indicators without a certified PML stability constant',
            'Probe growth always uses the initial-basis reference, even after LU reanchoring',
            'Final post-update state is also audited with the preceding update basis; it would be refreshed before another update',
            'Shared initial LU cache remains resident across paired arms; process memory is not standalone deployment memory'])
    save_json(out/'summary.json',summary);print('REFRESH_PILOT_COMPLETE '+str(out),flush=True)
    blas_guard.restore_original_limits()


if __name__=='__main__':main()
