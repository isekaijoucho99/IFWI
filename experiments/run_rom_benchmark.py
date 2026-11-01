"""Standalone ROM benchmark; never edits the original IFWI configuration/entry."""
from pathlib import Path
import argparse
import csv
import json
import time
import numpy as np
import torch
import yaml
from experiments.residual_ifwi_experiment import default_config,validate_config,build_forward_problem
from experiments.residual_ifwi_data import load_velocity_data
from experiments.rom.discrete import DiscretePMLSystem
from experiments.rom.basis import build_pressure_basis
from experiments.rom.solver import PressureROM
from experiments.rom.benchmark import (sha,save_json,evaluate,paired_timings,error_metrics,
    original_snapshots,siren_gradient_check,inversion_pilot)

ROOT=Path(__file__).resolve().parents[1]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config');ap.add_argument('--small',action='store_true')
    ap.add_argument('--device',default='cpu');ap.add_argument('--output',required=True);args=ap.parse_args()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
    cfg=default_config();options=dict(ranks=[16,32,64,96],training_frequencies=64,max_hz=32.,
        alias_tolerance=1e-12,pad_factor=4,repeats=3,states=['background','perturbed','truth'],pilot_updates=3,
        waveform_tolerance=.01,gradient_tolerance=.05,source_constraints=True,frequency_batch_size=128,
        rom_dtype='float64',reference_dtype='float32',basis_mode='primal')
    cfg['acquisition'].update(num_shots=1,source_x_indices=[144]);cfg['training']['shot_batch_size']=1
    if args.config:
        provided=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'));cfg=provided['experiment'];options.update(provided['rom'])
    if args.small:
        cfg['data']['crop_shape']=[20,31];cfg['forward']['nt']=96
        cfg['acquisition'].update(num_shots=1,source_x_indices=[15]);cfg['model']['neurons']=[2,16,16,16,16,1]
        options.update(ranks=[8,16],training_frequencies=16,repeats=1,states=['background','perturbed'],pilot_updates=0)
    if options['rom_dtype']!='float64' or options['reference_dtype']!='float32':
        raise ValueError('This validation CLI fixes original float32 FDTD and float64 ROM precision')
    if options['basis_mode'] not in ('primal','primal_adjoint'):raise ValueError('Unknown basis mode')
    validate_config(cfg);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    files=['rnn_fd.py','ifwi_modules.py','experiments/residual_ifwi_experiment.py','experiments/residual_ifwi_data.py',
        'experiments/run_rom_benchmark.py',*[f'experiments/rom/{name}.py' for name in ['paper','discrete','basis','solver','benchmark','stability']]]
    hashes={name:sha(ROOT/name) for name in files}
    save_json(out/'config.json',dict(experiment=cfg,rom=options,arguments=vars(args),source_sha256=hashes))
    for name in files:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
    truth,background,provenance=load_velocity_data(cfg,ROOT);save_json(out/'provenance.json',provenance)
    np.save(out/'true_velocity.npy',truth);np.save(out/'initial_velocity.npy',background)
    problem=build_forward_problem(truth,cfg,args.device);system=DiscretePMLSystem.from_solver(problem['propagators'][0]['solver'])
    geom=problem['geometry_tensors'];source=system.physical_indices(np.stack([geom['zs'][0].numpy(),geom['xs'][0].numpy()],axis=-1))
    rec_shape=geom['xr'][0].shape
    receivers=system.physical_indices(np.stack([geom['zr'][0].numpy(),geom['xr'][0].numpy()],axis=-1).reshape(-1,2)).reshape(rec_shape)
    wave=problem['wavelet'].cpu().numpy();np.save(out/'wavelet.npy',wave)
    nfft=2**int(np.ceil(np.log2(options['pad_factor']*len(wave))))
    snapshots=np.unique(np.linspace(len(wave)//4,len(wave)-1,4).round().astype(int)).tolist()
    z,x=np.meshgrid(np.linspace(0,np.pi,len(background)),np.linspace(0,2*np.pi,background.shape[1]),indexing='ij')
    states=dict(background=background,perturbed=(background*(1+.02*np.sin(z)*np.cos(x))).astype('float32'),truth=truth)
    adjoint_residual=None;calibration_seconds=0.
    if options['basis_mode']=='primal_adjoint':
        start=time.perf_counter()
        prediction,_,_=evaluate(torch.tensor(background,device=args.device)[None],problem)
        adjoint_residual=(prediction-problem['observed']).cpu().numpy()[0]
        calibration_seconds=time.perf_counter()-start
    references={}
    for name in options['states']:
        v=torch.tensor(states[name],device=args.device)[None]
        fields=original_snapshots(v,problem,snapshots);references[name]=fields
        np.savez_compressed(out/f'fdtd_{name}_snapshots.npz',fields=fields,indices=snapshots)
    rows=[];failures=[];accepted=[];offline=[]
    for rank in options['ranks']:
        print(json.dumps(dict(event='build_basis',rank=rank)),flush=True)
        try:
            basis,info=build_pressure_basis(system,background,source,wave,rank,
                training_frequencies=options['training_frequencies'],max_hz=options['max_hz'],nfft=nfft,
                alias_tolerance=options['alias_tolerance'],preserve_sources=options['source_constraints'],
                progress=lambda v:print(json.dumps(v),flush=True),receiver_indices=receivers,
                adjoint_residual=adjoint_residual)
            info['adjoint_calibration_fdtd_seconds']=calibration_seconds
            np.save(out/f'basis_{rank}.npy',basis);save_json(out/f'offline_{rank}.json',info);offline.append(info)
            start=time.perf_counter()
            rom=PressureROM(system,basis,source,receivers,len(wave),nfft=nfft,max_hz=options['max_hz'],
                alias_tolerance=options['alias_tolerance'],frequency_batch_size=options['frequency_batch_size']).to(args.device)
            info['static_projection_seconds']=time.perf_counter()-start
            save_json(out/f'offline_{rank}.json',info)
            torch.save(rom.state_dict(),out/f'rom_{rank}.pt')
            rank_rows=[]
            for name in options['states']:
                v=torch.tensor(states[name],device=args.device)[None]
                forward,_=paired_timings(dict(fdtd=lambda:evaluate(v,problem),rom=lambda:evaluate(v,problem,rom)),args.device,options['repeats'])
                backward,values=paired_timings(dict(fdtd=lambda:evaluate(v,problem,backward=True),rom=lambda:evaluate(v,problem,rom,backward=True)),args.device,options['repeats'])
                reference,g_ref,fd_loss=values['fdtd'];prediction,g_rom,rom_loss=values['rom']
                we=error_metrics(prediction.cpu().numpy(),reference.cpu().numpy());ge=error_metrics(g_rom.cpu().numpy(),g_ref.cpu().numpy())
                with torch.no_grad():
                    fields=rom.wavefields(v,problem['wavelet'],snapshots).cpu().numpy()
                top=0 if system.free_surface else system.npad
                fields=fields[:,:,top:top+truth.shape[0],system.npad:system.npad+truth.shape[1]]
                fe=error_metrics(fields,references[name]);np.savez_compressed(out/f'rom_{rank}_{name}_snapshots.npz',fields=fields,indices=snapshots)
                row=dict(rank=rank,state=name,source_count=len(source),waveform_relative_error=we['relative'],
                    selected_wavefield_relative_error=fe['relative'],velocity_gradient_relative_error=ge['relative'],
                    velocity_gradient_cosine=ge['cosine'],velocity_gradient_absolute_error=ge['absolute'],
                    fdtd_mse=fd_loss,rom_mse=rom_loss,
                    fdtd_forward_seconds=forward['fdtd']['median_seconds'],rom_forward_seconds=forward['rom']['median_seconds'],
                    fdtd_forward_backward_seconds=backward['fdtd']['median_seconds'],rom_forward_backward_seconds=backward['rom']['median_seconds'],
                    fdtd_incremental_peak_mib=backward['fdtd']['max_incremental_peak_mib'],rom_incremental_peak_mib=backward['rom']['max_incremental_peak_mib'],
                    fdtd_peak_allocated_mib=backward['fdtd']['max_peak_allocated_mib'],rom_peak_allocated_mib=backward['rom']['max_peak_allocated_mib'])
                rows.append(row);rank_rows.append(row)
                save_json(out/f'timing_{rank}_{name}.json',dict(forward=forward,forward_backward=backward,rom=rom.specification()))
                print(json.dumps(dict(event='comparison',**row)),flush=True);save_json(out/'progress.json',dict(comparisons=rows,failures=failures))
            hidden=siren_gradient_check(background,cfg,problem,rom,args.device);save_json(out/f'siren_gradient_{rank}.json',hidden)
            checks=[r for r in rank_rows if r['state']!='truth']
            if all(r['waveform_relative_error']<options['waveform_tolerance'] and r['velocity_gradient_relative_error'] is not None and r['velocity_gradient_relative_error']<options['gradient_tolerance'] for r in checks) and hidden['relative']<options['gradient_tolerance']:
                accepted.append(rank)
            del rom
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            failure=dict(rank=rank,error=repr(error));failures.append(failure);print(json.dumps(dict(event='rank_failure',**failure)),flush=True)
            save_json(out/'progress.json',dict(comparisons=rows,failures=failures))
    pilot=None
    if accepted and options['pilot_updates']:
        rank=min(accepted);basis=np.load(out/f'basis_{rank}.npy')
        rom=PressureROM(system,basis,source,receivers,len(wave),nfft=nfft,max_hz=options['max_hz'],
            alias_tolerance=options['alias_tolerance'],frequency_batch_size=options['frequency_batch_size']).to(args.device)
        pilot_out=out/'inversion_pilot';pilot_out.mkdir()
        pilot=inversion_pilot(background,truth,cfg,problem,rom,pilot_out,args.device,options['pilot_updates'])
    result=dict(device=args.device,source_hashes_unchanged=hashes=={name:sha(ROOT/name) for name in files},
        offline_truth_snapshots_used=False,comparisons=rows,failures=failures,offline=offline,
        data_conditioned_adjoint_calibration=adjoint_residual is not None,
        shared_pml_reference_vmax_mps=system.reference_vmax,
        shared_pml_profile_origin='Unchanged original baseline uses truth maximum to fix PML coefficients',
        accepted_ranks=accepted,inversion_pilot=pilot,snapshot_indices=snapshots,
        limits=['Paper elastic FEM theorem does not certify the discrete PML ROM',
            'Original FDTD float32; ROM float64 with complex128 solves',
            'Memory uses common resident data; incremental graph peaks are reported separately',
            'Frequency cutoff, contour error, fixed-basis material drift and inversion quality require separate validation'])
    save_json(out/'summary.json',result)
    if rows:
        with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('COMPLETE '+str(out),flush=True)


if __name__=='__main__':main()
