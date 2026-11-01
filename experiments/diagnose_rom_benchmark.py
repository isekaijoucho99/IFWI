"""Post-run PML pole and inverse-transform sensitivity diagnostics.

Eigenvalues are numerical diagnostics, not stability certificates. ARPACK only
computes selected Ritz values; failure to find an exterior pole proves nothing.
This deliberately runs separately from the timed benchmark.
"""
from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator,eigs,ArpackNoConvergence
from experiments.residual_ifwi_experiment import build_forward_problem
from experiments.rom.discrete import DiscretePMLSystem
from experiments.rom.solver import PressureROM
from experiments.rom.benchmark import evaluate,error_metrics,save_json


def pole_probe(rom,velocity,system):
    m,md,br=[a.detach().cpu().numpy() for a in rom.projected_model(velocity)]
    k=rom.stiffness_r.detach().cpu().numpy();p=rom.pml_r.detach().cpu().numpy()
    alpha=system.alphas;r=len(m);blocks=len(alpha)+2;dimension=r*blocks
    d=m+system.dt*md
    a=np.linalg.solve(d,2*m+system.dt*md-k-p.sum(0))
    previous=np.linalg.solve(d,-m)
    aux=np.stack([np.linalg.solve(d,-value*term) for value,term in zip(alpha,p)])
    def mv(x):
        x=np.asarray(x).reshape(blocks,r)
        top=a@x[0]+previous@x[1]+np.einsum('kij,kj->i',aux,x[2:])
        return np.concatenate([top,x[0],(alpha[:,None]*x[2:]+x[0]).ravel()])
    operator=LinearOperator((dimension,dimension),matvec=mv,dtype=np.dtype('float64'))
    start=time.perf_counter();status='selected_ritz_values';message=None
    try:
        values,vectors=eigs(operator,k=6,which='LM',tol=1e-7,maxiter=2000,ncv=30,
            v0=np.random.default_rng(3).normal(size=dimension))
    except ArpackNoConvergence as error:
        values,vectors=error.eigenvalues,error.eigenvectors
        status='arpack_incomplete';message=str(error)
    residuals=[]
    if values is not None and len(values):
        for value,vector in zip(values,vectors.T):
            residuals.append(float(np.linalg.norm(operator@vector-value*vector)/np.linalg.norm(vector)))
    else:values=np.array([])
    rho=float(np.max(abs(values))) if len(values) else None
    contour=float(abs(rom.z[0].item()))
    return dict(rank=r,realization_dimension=dimension,method='matrix_free_ARPACK_LM',
        status=status,message=message,seconds=time.perf_counter()-start,
        largest_observed_pole_modulus=rho,contour_radius=contour,
        observed_poles_outside_unit_disk=int(np.sum(abs(values)>1+1e-7)),
        observed_poles_outside_contour=int(np.sum(abs(values)>contour)),
        values=[[float(v.real),float(v.imag)] for v in values],ritz_absolute_residuals=residuals,
        limit='Selected numerical eigenvalues are not a certificate; unobserved poles and nonnormal growth remain possible')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True)
    ap.add_argument('--ranks',nargs='+',type=int,required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--skip-poles',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.fp32_precision='ieee';torch.backends.cudnn.conv.fp32_precision='ieee'
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    root=Path(args.run);config=json.loads((root/'config.json').read_text());cfg=config['experiment'];opts=config['rom']
    truth=np.load(root/'true_velocity.npy');bg=np.load(root/'initial_velocity.npy')
    problem=build_forward_problem(truth,cfg,args.device)
    system=DiscretePMLSystem.from_solver(problem['propagators'][0]['solver'])
    geom=problem['geometry_tensors']
    sources=system.physical_indices(np.stack([geom['zs'][0].numpy(),geom['xs'][0].numpy()],axis=-1))
    receivers=system.physical_indices(np.stack([geom['zr'][0].numpy(),geom['xr'][0].numpy()],axis=-1).reshape(-1,2)).reshape(geom['xr'][0].shape)
    wave=problem['wavelet'];v=torch.tensor(bg,device=args.device)[None]
    ref,gref,_=evaluate(v,problem,backward=True);result=[]
    for rank in args.ranks:
        basis=np.load(root/f'basis_{rank}.npy');rows=[]
        item=dict(rank=rank,poles=None,contour_and_bandwidth=rows)
        result.append(item)
        nfft=2**int(np.ceil(np.log2(opts['pad_factor']*len(wave))))
        for alias,cutoff in [(1e-8,32.),(1e-12,32.),(1e-16,32.),(1e-12,None)]:
            row=dict(alias_tolerance=alias,max_hz=cutoff);rom=None
            try:
                rom=PressureROM(system,basis,sources,receivers,len(wave),nfft=nfft,
                    max_hz=cutoff,alias_tolerance=alias,frequency_batch_size=64).to(args.device)
                pred,grad,_=evaluate(v,problem,rom,backward=True)
                row.update(waveform=error_metrics(pred.cpu().numpy(),ref.cpu().numpy()),
                    velocity_gradient=error_metrics(grad.cpu().numpy(),gref.cpu().numpy()))
            except (FloatingPointError,ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                row['failure']=repr(error)
            rows.append(row)
            print(json.dumps(dict(rank=rank,**row)),flush=True)
            save_json(root/'diagnostics.json',result)
            if rom is not None and alias==1e-12 and cutoff==32. and not args.skip_poles:
                try:item['poles']=pole_probe(rom,v,system)
                except (FloatingPointError,ValueError,RuntimeError,np.linalg.LinAlgError) as error:
                    item['poles']=dict(status='numerical_failure',failure=repr(error))
                save_json(root/'diagnostics.json',result)
            del rom
    print('DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':main()
