"""Dense acquisition through the common vectorized source-code adapter.

Physical chunks use a sliced geometry and a K-by-K identity code, while encoded
updates use all N source locations. Raw velocity gradients and pressure energies
are summed before one illumination direction, network VJP, and optimizer commit.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch

from core_speed import SourceEncoder, normalized_encoding_loss
from generator import wGenerator
from rnn_fd import rnn2D
from experiments.encoding_batch_correctness import _clear_graph_cache
from experiments.encoding_training_step import (
    CommittedUpdateError, UncertainOptimizerError, _record, _sync,
)
from experiments.residual_ifwi_experiment import check_velocity
from experiments.residual_ifwi_report import velocity_metrics
from ifwi_research.encoding_illumination import PressureEnergy, illumination_direction


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def generate_sources(nx, dx_m, nshots, spacing_m=75., margin_cells=10):
    """Center exactly spaced source cells within [margin, nx-1-margin].

    In the actual nx=288, dx=15m grid, 48 and 52 shots fit at 75m with
    margin=10. A requested margin=20 cannot accommodate 52 shots at 75m.
    Neither rounding the spacing nor reducing the declared margin is allowed.
    """
    nx = _integer(nx, 'nx'); nshots = _integer(nshots, 'nshots')
    margin_cells = _integer(margin_cells, 'margin_cells', 0)
    if (not math.isfinite(dx_m) or not math.isfinite(spacing_m)
            or dx_m <= 0 or spacing_m <= 0):
        raise ValueError('Positive finite grid and source spacing required')
    step = spacing_m / dx_m
    if not math.isclose(step, round(step), rel_tol=0., abs_tol=1e-9) or round(step) < 1:
        raise ValueError('Source spacing must be an exact positive number of grid cells')
    span = (nshots-1)*round(step)
    if span > nx-1-2*margin_cells or nx <= 2*margin_cells:
        raise ValueError(f'{nshots} shots at {spacing_m:g}m do not fit nx={nx} with margin={margin_cells}')
    first = (nx-1-span)//2
    return [first+i*round(step) for i in range(nshots)]


def build_dense_problem(truth, cfg, device, observed=None, wavelet=None):
    """Build detached dense assets without using legacy 13-shot observations.

    Missing observations may be generated only for tiny CPU verification
    problems (at most two million grid/shot/time samples). Real dense assets
    must be supplied explicitly; this factory never launches a GPU asset solve.
    Acquisition options are source_spacing_m and source_margin_cells (10 by
    default), or explicit legal sorted source_x_indices.
    """
    device = torch.device(device)
    shape = tuple(truth.shape)
    if len(shape) != 2 or min(shape) < 3:
        raise ValueError('truth must be a [nz,nx] physical grid')
    cfg = copy.deepcopy(cfg); acq, fd = cfg['acquisition'], cfg['forward']
    nz, nx = shape; nshots = _integer(acq['num_shots'], 'num_shots')
    nt = _integer(fd['nt'], 'nt')
    if observed is None and (device.type != 'cpu' or nz*nx*nshots*nt > 2_000_000):
        raise ValueError('Provide dense observed assets; automatic generation is restricted to tiny CPU checks')
    vp = torch.as_tensor(truth, dtype=torch.float32, device=device).detach()[None]
    check_velocity(vp, cfg)
    if fd['order'] != 2:
        raise ValueError('Dense acquisition preserves the original order-2 FD')
    margin = _integer(acq.get('source_margin_cells', 10), 'source_margin_cells', 0)
    positions = acq.get('source_x_indices')
    if positions is None:
        positions = generate_sources(nx, cfg['data']['grid_spacing_m'], nshots,
                                     acq.get('source_spacing_m', 75.), margin)
    else:
        positions = [_integer(p, 'source_x_indices', 0) for p in positions]
        if len(positions) != nshots or positions != sorted(set(positions)):
            raise ValueError('Dense sources must be distinct and sorted, with exactly num_shots entries')
        if not positions or positions[0] < margin or positions[-1] >= nx-margin:
            raise ValueError('Explicit dense sources violate the declared acquisition margin')
        if 'source_spacing_m' in acq and len(positions) > 1:
            spacings = np.diff(positions)*cfg['data']['grid_spacing_m']
            if not np.allclose(spacings, acq['source_spacing_m'], rtol=0., atol=1e-9):
                raise ValueError('Explicit source coordinates disagree with source_spacing_m')
    source_z = _integer(acq['source_depth_index'], 'source_depth_index', 0)
    receiver_z = _integer(acq['receiver_depth_index'], 'receiver_depth_index', 0)
    if max(source_z, receiver_z) >= nz:
        raise ValueError('Source and receiver depths must lie inside the grid')
    stride = _integer(acq['receiver_stride'], 'receiver_stride')
    xs = torch.as_tensor(positions, dtype=torch.long)[None]
    zs = torch.full_like(xs, source_z)
    xr = torch.arange(0,nx,stride,dtype=torch.long)[None,None].repeat(1,nshots,1)
    zr = torch.full_like(xr,receiver_z)
    geometry = dict(nz=nz,nx=nx,xs=xs,zs=zs,xr=xr,zr=zr)
    acq.update(source_x_indices=positions,source_margin_cells=margin)
    wave = (wGenerator(fd['dt_s']*torch.arange(nt,dtype=torch.float32),fd['frequency_hz']).ricker()
            if wavelet is None else wavelet)
    wave = torch.as_tensor(wave,dtype=torch.float32,device=device).detach()
    if wave.shape != (nt,) or not torch.isfinite(wave).all():
        raise ValueError('Wavelet must contain nt finite samples')
    expected = (1,nshots,nt,xr.shape[-1])
    data = (torch.zeros(expected,dtype=torch.float32,device=device) if observed is None else
            torch.as_tensor(observed,dtype=torch.float32,device=device).detach())
    if data.shape != expected or not torch.isfinite(data).all():
        raise ValueError('Dense observations must have matching finite [1,N,T,R] shape')
    # The base solver only supplies immutable FD configuration to SourceEncoder.
    # It never executes; every actual solve below uses the vectorized adapter.
    seed_geometry = {key:(value[:,:1] if isinstance(value,torch.Tensor) else value)
                     for key,value in geometry.items()}
    seed_solver = rnn2D(**seed_geometry,dz=cfg['data']['grid_spacing_m'],dt=fd['dt_s'],
        npad=fd['npad'],order=2,vmax=float(vp.max()),log_para=fd['pml_reflection'],
        freeSurface=fd['free_surface'],dtype=torch.float32,device=device)
    problem = dict(config=cfg,observed=data,wavelet=wave,geometry_tensors=geometry,
        geometry=dict(nz=nz,nx=nx,source_x_indices=positions,source_z_index=source_z,
                      receiver_x_indices=xr[0,0].tolist(),receiver_z_index=receiver_z,
                      num_shots=nshots,num_receivers=xr.shape[-1],nt=nt,
                      source_margin_cells=margin,
                      source_spacings_m=(np.diff(positions)*cfg['data']['grid_spacing_m']).tolist()),
        propagators=[dict(solver=seed_solver,first=0,last=1)],
        reference_shots=nshots,frequency_scales={})
    if observed is None:
        acquisition = DenseAcquisition(problem,batch_size=cfg.get('training',{}).get('shot_batch_size') or 13)
        with torch.no_grad():
            for batch in acquisition.batches(0):
                solver = batch['propagators'][0]['solver']
                try:
                    prediction = solver(vmodel=vp,segment_wavelet=wave,option=0)[2]
                    if not torch.isfinite(prediction).all():
                        raise ValueError('Nonfinite generated dense observations')
                    first,last = batch['shot_range']; data[:,first:last].copy_(prediction)
                finally:
                    _clear_graph_cache(solver)
    return problem


class DenseAcquisition:
    """Cache chunk solvers, with no retained wavefield graphs between chunks."""
    encoder_type = SourceEncoder

    def __init__(self,problem,batch_size=13,grouping='identity',fields=None,seed=3):
        self.base = problem
        self.batch_size = _integer(batch_size,'batch_size')
        self.reference_shots = int(problem['observed'].shape[1])
        self.grouping,self.seed = grouping,int(seed)
        self.fields = self.reference_shots if grouping == 'identity' else _integer(1 if fields is None else fields,'fields')
        if grouping == 'identity' and fields not in (None,self.reference_shots):
            raise ValueError('Physical identity acquisition uses all original shots')
        self.encoders = []
        self.ranges = []
        if grouping == 'identity':
            for first in range(0,self.reference_shots,self.batch_size):
                last = min(first+self.batch_size,self.reference_shots)
                geometry = {key:(value[:,first:last] if isinstance(value,torch.Tensor) else value)
                            for key,value in problem['geometry_tensors'].items()}
                sub = {**problem,'observed':problem['observed'][:,first:last],
                       'geometry_tensors':geometry}
                self.encoders.append(self.encoder_type(sub,last-first,seed,grouping='identity'))
                self.ranges.append((first,last))
        else:
            self.encoders.append(self.encoder_type(problem,self.fields,seed,grouping=grouping))
            self.ranges.append((0,self.reference_shots))
        self._identity = self if grouping == 'identity' else None

    def batches(self,update):
        for encoder,shot_range in zip(self.encoders,self.ranges):
            yield {**encoder.problem(update),'reference_shots':self.reference_shots,
                   'shot_range':shot_range}

    def physical(self):
        """An evaluation view sharing cached full-shot chunks across evaluations."""
        if self._identity is None:
            self._identity = DenseAcquisition(self.base,self.batch_size,seed=self.seed)
        return self._identity


def train_dense_step(model,optimizer,acquisition,completed_updates,*,record_cost=None,
                     check_stop=None,phase='training',arm_name='arm',
                     illumination_preconditioning=True):
    """Commit exactly one update, summing raw chunk gradients and energies first.

    With illumination_preconditioning=False the summed physical velocity
    gradient goes directly to the network VJP, without illumination observers
    or energy maps; loss and PDE cost stay the same.
    STOP checks precede the optimizer. An optimizer exception is uncertain;
    failures after it returns carry the committed count, matching legacy APIs.
    """
    if type(illumination_preconditioning) is not bool:
        raise ValueError('illumination_preconditioning must be a bool')
    check_stop = check_stop or (lambda:None); check_stop()
    device = next(model.parameters()).device
    _sync(device); tick = time.perf_counter()
    start = end = None; committed = optimizer_started = False
    if device.type == 'cuda':
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
    if hasattr(model,'set_progress'): model.set_progress(completed_updates)
    optimizer.zero_grad(set_to_none=True)
    velocity = model(); check_velocity(velocity,acquisition.base['config'])
    raw = torch.zeros_like(velocity)
    source = receiver = None
    if illumination_preconditioning:
        source = torch.zeros_like(velocity[0],dtype=torch.float64)
        receiver = torch.zeros_like(source)
    total_loss = 0.; batch_fields = []; matrices = []
    nt = int(acquisition.base['wavelet'].shape[-1])
    try:
        for problem in acquisition.batches(completed_updates+1):
            solver = problem['propagators'][0]['solver']; fd = solver.fd
            target = problem['observed']; fields = int(target.shape[1])
            observer = (PressureEnergy(velocity.shape[-2],velocity.shape[-1],fd.npad,fd.freeSurface)
                        if illumination_preconditioning else None)
            leaf = velocity.detach().requires_grad_(True)
            handle = fd.register_forward_hook(observer.observe) if observer is not None else None
            outputs = prediction = loss = None
            try:
                check_stop()
                _record(record_cost,'forward_fields_attempt',fields,phase,arm_name,completed_updates+1,nt)
                part = time.perf_counter()
                outputs = solver(vmodel=leaf,segment_wavelet=problem['wavelet'],option=0)
                _sync(device)
                _record(record_cost,'forward_fields_complete',fields,phase,arm_name,completed_updates+1,nt,time.perf_counter()-part)
                prediction = outputs[2]
                if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                    raise ValueError('Invalid predicted dense records')
                loss = normalized_encoding_loss(prediction,target,acquisition.reference_shots)
                if not torch.isfinite(loss): raise ValueError('Nonfinite dense objective')
                check_stop()
                _record(record_cost,'backward_fields_attempt',fields,phase,arm_name,completed_updates+1,nt)
                part = time.perf_counter(); loss.backward(); _sync(device)
                _record(record_cost,'backward_fields_complete',fields,phase,arm_name,completed_updates+1,nt,time.perf_counter()-part)
                if observer is not None and (observer.forward_samples,observer.adjoint_samples) != (nt,nt):
                    raise ValueError('Incomplete dense illumination observation')
                if leaf.grad is None: raise ValueError('Missing physical velocity gradient')
                raw.add_(leaf.grad)
                if observer is not None:
                    source.add_(observer.source); receiver.add_(observer.receiver)
                total_loss += float(loss.detach()); batch_fields.append(fields)
                matrices.append(problem['encoding_matrix'].detach().cpu().tolist())
            finally:
                if handle is not None: handle.remove()
                _clear_graph_cache(solver); leaf.grad = None
                outputs = prediction = loss = None
        if illumination_preconditioning:
            direction,illumination = illumination_direction(raw,source,receiver)
        else:
            if not torch.isfinite(raw).all():
                raise ValueError('Nonfinite dense physical velocity gradient')
            direction = raw
            illumination = dict(enabled=False,raw_gradient_norm=float(raw.double().norm()))
        parameters = tuple(model.parameters())
        gradients = torch.autograd.grad(velocity,parameters,grad_outputs=direction)
        for parameter,gradient in zip(parameters,gradients):
            if not torch.isfinite(gradient).all(): raise ValueError('Nonfinite dense parameter gradient')
            parameter.grad = gradient
        check_stop()
        masks = model.optimizer_masks() if hasattr(model,'optimizer_masks') else None
        optimizer_started = True
        if masks is None: optimizer.step()
        else: optimizer.step(masks=masks)
        committed = True
        if hasattr(model,'set_progress'): model.set_progress(completed_updates+1)
        with torch.no_grad():
            updated = model(); check_velocity(updated,acquisition.base['config'])
            vmin,vmax = float(updated.min()),float(updated.max())
        if end is not None: end.record()
        _sync(device)
        return dict(completed_updates=completed_updates+1,loss=total_loss,
            wall_seconds=time.perf_counter()-tick,
            cuda_ms=start.elapsed_time(end) if start is not None else None,
            velocity_min=vmin,velocity_max=vmax,illumination=illumination,
            fields=sum(batch_fields),batch_fields=batch_fields,NT=nt,
            reference_shots=acquisition.reference_shots,encoding_matrices=matrices,
            grouping=acquisition.grouping,
            illumination_preconditioning=illumination_preconditioning)
    except Exception as error:
        if committed: raise CommittedUpdateError(completed_updates+1,error) from error
        if optimizer_started: raise UncertainOptimizerError(completed_updates+1,error) from error
        raise


def evaluate_dense(model,acquisition,assets,*,record_cost=None,check_stop=None,label='evaluation'):
    """Predict all N identity shots once at the cap; score clean and noisy data."""
    check_stop = check_stop or (lambda:None); check_stop()
    device = next(model.parameters()).device
    _sync(device); tick = time.perf_counter(); predictions = []
    physical = acquisition.physical()
    with torch.no_grad():
        velocity = model(); check_velocity(velocity,physical.base['config'])
        for problem in physical.batches(0):
            check_stop()
            solver = problem['propagators'][0]['solver']
            fields,nt = int(problem['observed'].shape[1]),int(problem['observed'].shape[2])
            try:
                _record(record_cost,'forward_fields_attempt',fields,'evaluation',label,0,nt)
                part = time.perf_counter()
                outputs = solver(vmodel=velocity,segment_wavelet=problem['wavelet'],option=0)
                _sync(device)
                _record(record_cost,'forward_fields_complete',fields,'evaluation',label,0,nt,time.perf_counter()-part)
                prediction = outputs[2]
                if prediction.shape != problem['observed'].shape or not torch.isfinite(prediction).all():
                    raise ValueError('Invalid dense evaluation waveform')
                predictions.append(prediction.detach().cpu().numpy().copy())
            finally:
                _clear_graph_cache(solver)
        v = velocity[0].detach().cpu().numpy().copy()
    pred = np.concatenate(predictions,axis=1)
    metrics = velocity_metrics(v,assets['truth'])
    for name in ('clean','noisy'):
        target = np.asarray(assets[name])
        if target.shape != pred.shape or not np.isfinite(target).all():
            raise ValueError('Dense evaluation asset dimensions or values differ')
        metrics[name+'_mse'] = float(np.mean((pred.astype(np.float64)-target.astype(np.float64))**2))
    metrics['evaluation_wall_seconds'] = time.perf_counter()-tick
    return metrics,pred,v
