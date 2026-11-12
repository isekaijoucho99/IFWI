"""Frozen cached Marmousi assets and homogeneous residual model controls."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch
from rnn_fd import rnn2D
from experiments.residual_ifwi_experiment import model_from_config
from experiments.progressive_siren import ProgressiveSirenVelocity

ARMS=(dict(name='full13',grouping='identity',fields=13,progressive=False),
      dict(name='single',grouping='global',fields=1,progressive=False),
      dict(name='global2',grouping='global',fields=2,progressive=False),
      dict(name='interleaved2',grouping='interleaved',fields=2,progressive=False),
      dict(name='interleaved3',grouping='interleaved',fields=3,progressive=False),
      dict(name='interleaved2_progressive',grouping='interleaved',fields=2,progressive=True))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _asset(base,meta):
    p=(base/meta['path']).resolve()
    if sha(p)!=meta['sha256']:raise ValueError('Cached input hash mismatch: '+str(p))
    array=np.load(p,allow_pickle=False)
    if list(array.shape)!=meta['shape'] or str(array.dtype)!=meta['dtype']:
        raise ValueError('Cached input shape/dtype differs: '+str(p))
    if not np.isfinite(array).all():raise ValueError('Nonfinite cached data')
    return array,dict(path=str(p),sha256=meta['sha256'],raw_shape=list(array.shape),raw_dtype=str(array.dtype))


def load_assets(index_path,config_path,initial_checkpoint=None):
    index_path=Path(index_path).resolve();config_path=Path(config_path).resolve()
    index=json.loads(index_path.read_text(encoding='utf-8-sig'))
    cfg=json.loads(config_path.read_text(encoding='utf-8-sig'))
    if 'config'in cfg:cfg=cfg['config']
    clean_meta=index['marmousi_clean'];noise_meta=index['marmousi_noise_sigma4']
    arrays={};manifest={}
    for name in ('truth','background','wavelet'):
        if clean_meta[name]['sha256']!=noise_meta[name]['sha256']:
            raise ValueError('Clean/noisy physical inputs differ: '+name)
        arrays[name],manifest[name]=_asset(index_path.parent,clean_meta[name])
    for label,meta in [('clean',clean_meta['observed']),('noisy',noise_meta['observed'])]:
        raw,manifest[label]=_asset(index_path.parent,meta)
        arrays[label]=np.asarray(raw,dtype=np.float32)
        manifest[label]['used_dtype']='float32'
        manifest[label]['used_array_sha256']=hashlib.sha256(arrays[label].tobytes(order='C')).hexdigest()
    shape=arrays['clean'].shape
    if shape!=arrays['noisy'].shape or len(shape)!=4 or shape[:2]!=(1,13):
        raise ValueError('This protocol requires cached [1,13,T,R] records')
    if arrays['background'].shape!=arrays['truth'].shape or arrays['wavelet'].shape!=(shape[2],):
        raise ValueError('Cached model/wavelet dimensions differ')
    if cfg.get('parameterization','residual')!='residual' or int(cfg['acquisition']['num_shots'])!=13:
        raise ValueError('Homogeneous residual13-shot configuration required')
    if cfg['forward']['nt']!=shape[2] or cfg['forward']['order']!=2:
        raise ValueError('Cached records and numerical configuration disagree')
    cfg=copy.deepcopy(cfg)
    cfg['parameterization']='residual';cfg['seed']=3
    cfg.setdefault('initialization',{})
    if cfg['initialization'].get('pretrain') is not None or any(cfg['model'].get(k)for k in ('fourier','trainable_omega','low_rank')):
        raise ValueError('This control suite requires plain fixed SIREN and zero-head residual initialization')
    cfg['training'].update(epochs=1000,shot_batch_size=13,learning_rate=1e-4)
    cfg['training'].pop('shots_per_update',None);cfg['training'].pop('frequency_schedule',None)
    clean=arrays['clean'].astype(np.float64);noise=arrays['noisy'].astype(np.float64)-clean
    signal=float(np.mean(clean**2))
    noise_power=float(np.mean(noise**2))
    measured_snr=10*np.log10(signal/noise_power)if signal>0 and noise_power>0 else None
    manifest.update(dataset_index=dict(path=str(index_path),sha256=sha(index_path)),
        reference_config=dict(path=str(config_path),sha256=sha(config_path)))
    canonical=None
    if initial_checkpoint is not None:
        path=Path(initial_checkpoint).resolve();manifest['initial_checkpoint']=dict(path=str(path),sha256=sha(path))
        # Admit only NumPy's known legacy RNG representation, never arbitrary globals.
        reconstruct=np._core.multiarray._reconstruct
        allowed=[reconstruct,(reconstruct,'numpy.core.multiarray._reconstruct'),np.ndarray,np.dtype,type(np.dtype(np.uint32))]
        with torch.serialization.safe_globals(allowed):canonical=torch.load(path,map_location='cpu',weights_only=True)
        if canonical.get('completed_updates')!=0 or canonical['optimizer']['state']:
            raise ValueError('Only a true zero-update empty-Adam canonical checkpoint is accepted')
    return dict(**arrays,config=cfg,manifest=manifest,measured_snr_db=measured_snr,
        nominal_noise_label='sigma4 (~-12dB), fixed cached realization',canonical_checkpoint=canonical)


def make_model(assets,arm,device):
    cfg=assets['config'];background=assets['background'];torch.manual_seed(3);np.random.seed(3);random.seed(3)
    base=model_from_config(background,cfg)
    canonical=assets.get('canonical_checkpoint')
    if canonical is not None:base.load_state_dict(canonical['model'],strict=True)
    expected=torch.from_numpy(np.asarray(background,dtype=np.float32))[None]
    with torch.no_grad():
        if not torch.equal(base(),expected):raise ValueError('Canonical residual output differs from fixed background')
    if arm['progressive']:
        spec=cfg['model'];torch.manual_seed(3)
        model=ProgressiveSirenVelocity(background,cfg['data']['grid_spacing_m'],spec['neurons'],
            mean_kmps=spec['mean_kmps'],std_kmps=spec['std_kmps'],warmup_updates=200,
            low_omega=6.,high_omega=30.)
        # The high-frequency branch shares the canonical30-frequency initialization.
        model.high.load_state_dict(base.state_dict(),strict=True)
    else:model=base
    model=model.to(device)
    with torch.no_grad():
        if not torch.equal(model().cpu(),expected):raise ValueError('An arm changed the common initial physical velocity')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.999),eps=1e-8)
    return model,optimizer


def build_problem(assets,condition,device):
    if condition not in ('clean','noisy'):raise ValueError('Explicit clean or noisy condition required')
    truth,cfg=assets['truth'],assets['config'];nz,nx=truth.shape;acq,fd=cfg['acquisition'],cfg['forward']
    positions=acq['source_x_indices']
    if positions is None:
        positions=np.arange(acq['source_start_index'],nx-acq['source_end_margin'],acq['source_spacing_index'])
        positions=positions[np.linspace(0,len(positions)-1,13).round().astype(int)].tolist()
    if positions!=sorted(positions)or len(set(positions))!=13:
        raise ValueError('Interleaving requires13 distinct sources in fixed spatial order')
    xs=torch.as_tensor(positions,dtype=torch.long)[None];zs=torch.full_like(xs,acq['source_depth_index'])
    xr=torch.arange(0,nx,acq['receiver_stride'],dtype=torch.long)[None,None].repeat(1,13,1)
    zr=torch.full_like(xr,acq['receiver_depth_index']);geometry=dict(nz=nz,nx=nx,xs=xs,zs=zs,xr=xr,zr=zr)
    solver=rnn2D(nz=nz,nx=nx,xs=xs,zs=zs,xr=xr,zr=zr,dz=cfg['data']['grid_spacing_m'],
        dt=fd['dt_s'],npad=fd['npad'],order=2,vmax=float(truth.max()),log_para=fd['pml_reflection'],
        freeSurface=fd['free_surface'],dtype=torch.float32,device=device)
    observed=torch.as_tensor(assets[condition],dtype=torch.float32,device=device)
    wave=torch.as_tensor(assets['wavelet'],dtype=torch.float32,device=device)
    if observed.shape!=(1,13,fd['nt'],xr.shape[-1]):raise ValueError('Cached acquisition dimensions disagree')
    return dict(config=cfg,observed=observed,wavelet=wave,propagators=[dict(solver=solver,first=0,last=13)],
        geometry_tensors=geometry,geometry=dict(nz=nz,nx=nx,source_x_indices=positions,
        receiver_x_indices=xr[0,0].tolist()),frequency_scales={},reference_shots=13)
