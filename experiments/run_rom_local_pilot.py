"""At most three guarded IFWI updates, using a previously frozen ROM basis.

This diagnostic does not override the benchmark's failed held-out admission.
Each candidate trajectory state is checked against original FDTD before any
further optimizer update. No refresh, source encoding or new network is added.
"""
from pathlib import Path
import argparse
import copy
import json
import time
import numpy as np
import torch
from experiments.residual_ifwi_experiment import build_forward_problem,train_update
from experiments.residual_ifwi_report import velocity_metrics
from experiments.rom.discrete import DiscretePMLSystem
from experiments.rom.solver import PressureROM
from experiments.rom.benchmark import make_model,evaluate,error_metrics,save_json,sha,synchronized


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--rank',type=int,required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--updates',type=int,choices=[1,2,3],default=3);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    root=Path(args.run);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    config=json.loads((root/'config.json').read_text());cfg=config['experiment'];opts=config['rom']
    bg=np.load(root/'initial_velocity.npy');truth=np.load(root/'true_velocity.npy')
    problem=build_forward_problem(truth,cfg,args.device)
    system=DiscretePMLSystem.from_solver(problem['propagators'][0]['solver'])
    geom=problem['geometry_tensors']
    sources=system.physical_indices(np.stack([geom['zs'][0].numpy(),geom['xs'][0].numpy()],axis=-1))
    receivers=system.physical_indices(np.stack([geom['zr'][0].numpy(),geom['xr'][0].numpy()],axis=-1).reshape(-1,2)).reshape(geom['xr'][0].shape)
    rom=PressureROM(system,np.load(root/f'basis_{args.rank}.npy'),sources,receivers,len(problem['wavelet']),
        nfft=2**int(np.ceil(np.log2(opts['pad_factor']*len(problem['wavelet'])))),
        max_hz=opts['max_hz'],alias_tolerance=opts['alias_tolerance'],
        frequency_batch_size=opts['frequency_batch_size']).to(args.device)
    encoded={**problem,'propagators':[dict(solver=rom,first=0,last=len(sources))]}
    torch.manual_seed(cfg['seed']);base=make_model(bg,cfg,args.device)
    result=dict(purpose='Guarded local trajectory diagnostic; held-out benchmark admission remains failed',
        rank=args.rank,config=config,arguments=vars(args),basis_sha256=sha(root/f'basis_{args.rank}.npy'),
        source_sha256=sha(Path(__file__)),arms=[])
    (out/'run_rom_local_pilot.py').write_bytes(Path(__file__).read_bytes())
    save_json(out/'summary.json',result)
    for label,training_problem in [('fdtd',problem),('rom',encoded)]:
        model=copy.deepcopy(base);optimizer=torch.optim.Adam(model.parameters(),lr=cfg['training']['learning_rate'])
        arm=out/label;arm.mkdir();record=dict(method=label,states=[],update_seconds=[],last_validated_update=None,status='running')
        result['arms'].append(record)
        for step in range(args.updates+1):
            try:
                velocity=model();a,ga,msea=evaluate(velocity,problem,backward=True);b,gb,mseb=evaluate(velocity,problem,rom,backward=True)
                parameters=tuple(model.parameters())
                pa=torch.autograd.grad(velocity,parameters,grad_outputs=ga,retain_graph=True)
                pb=torch.autograd.grad(velocity,parameters,grad_outputs=gb)
                we=error_metrics(b.cpu().numpy(),a.cpu().numpy());ge=error_metrics(gb.cpu().numpy(),ga.cpu().numpy())
                pe=error_metrics(torch.cat([x.ravel() for x in pb]).cpu().numpy(),torch.cat([x.ravel() for x in pa]).cpu().numpy())
                physical=velocity.detach().cpu().numpy()[0]
                passed=(we['relative']<opts['waveform_tolerance'] and ge['relative'] is not None
                    and ge['relative']<opts['gradient_tolerance'] and pe['relative'] is not None and pe['relative']<opts['gradient_tolerance'])
                row=dict(completed_updates=step,waveform_relative_error=we['relative'],velocity_gradient_relative_error=ge['relative'],
                    parameter_gradient_relative_error=pe['relative'],fdtd_mse=msea,rom_mse=mseb,
                    passes_local_accuracy_gate=passed,**velocity_metrics(physical,truth))
            except (RuntimeError,ValueError,FloatingPointError) as error:
                record.update(status='candidate_validation_failure',completed_updates_at_failure=step,failure=repr(error))
                save_json(out/'summary.json',result);break
            record['states'].append(row);np.save(arm/f'velocity_{step:04d}.npy',physical)
            print(json.dumps(dict(method=label,**row)),flush=True)
            if passed:
                record['last_validated_update']=step
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),completed_updates=step),arm/'validated_checkpoint.pt')
            if label=='rom' and not passed:
                record['status']='stopped_by_local_accuracy_gate';save_json(out/'summary.json',result);break
            if step==args.updates:
                record['status']='completed_bounded_pilot';save_json(out/'summary.json',result);break
            del velocity,a,b,ga,gb,pa,pb
            old_model=copy.deepcopy(model.state_dict());old_optimizer=copy.deepcopy(optimizer.state_dict())
            synchronized(args.device);start=time.perf_counter()
            try:
                train_update(model,training_problem,optimizer);synchronized(args.device)
            except (RuntimeError,ValueError,FloatingPointError) as error:
                model.load_state_dict(old_model);optimizer.load_state_dict(old_optimizer)
                record.update(status='numerical_failure',failure=repr(error));save_json(out/'summary.json',result);break
            record['update_seconds'].append(time.perf_counter()-start)
            save_json(out/'summary.json',result)
    save_json(out/'summary.json',result);print('LOCAL_PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()
