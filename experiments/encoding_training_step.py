"""One unchanged-Adam update with a single batched physical adjoint.

The shared velocity VJP receives the existing instantaneous illumination
direction. Neither truth nor validation records enter the update.
"""
from __future__ import annotations
import platform
import time
import numpy as np
import torch
from core_speed import normalized_encoding_loss
from experiments.residual_ifwi_experiment import check_velocity
from experiments.residual_ifwi_report import velocity_metrics
from experiments.encoding_batch_correctness import _clear_graph_cache
from ifwi_research.encoding_illumination import PressureEnergy, illumination_direction


class CommittedUpdateError(RuntimeError):
    """Failed post-update validation; this state is not a safe checkpoint."""
    def __init__(self,completed_updates,cause):
        self.completed_updates=completed_updates
        super().__init__(f'Adam committed update {completed_updates}, post-check failed: {cause}')


class UncertainOptimizerError(RuntimeError):
    """Adam raised inside step; only the prior on-disk snapshot is safe."""
    def __init__(self,attempted_update,cause):
        self.attempted_update=attempted_update
        super().__init__(f'Adam update {attempted_update} started but did not return; commit uncertain: {cause}')


def configure_runtime(device):
    device=torch.device(device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; no CPU fallback')
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.fp32_precision='ieee'
    torch.backends.cudnn.conv.fp32_precision='ieee'
    runtime=dict(python=platform.python_version(),torch=str(torch.__version__),numpy=np.__version__,
        device=str(device),threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
        cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
        matmul_fp32=torch.backends.cuda.matmul.fp32_precision,conv_fp32=torch.backends.cudnn.conv.fp32_precision,
        cuda=str(torch.version.cuda),cudnn=torch.backends.cudnn.version())
    if device.type=='cuda':
        torch.cuda.set_device(device)
        runtime.update(gpu=torch.cuda.get_device_name(device),total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
    return runtime


def _sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def _record(callback,counter,fields,phase,arm,update,nt,elapsed=0.):
    if callback is not None:
        callback(phase+'.'+counter,phase=phase,arm=arm,update=update,fields=fields,NT=nt,
            wall_seconds=elapsed,**{counter:fields})


def train_step(model,optimizer,encoder,completed_updates,*,record_cost=None,check_stop=None,
               phase='training',arm_name='arm'):
    """Return only after a committed update; never check STOP after Adam commits.

    Caller publishes completed_updates before checking STOP again. Stops before
    Adam leave parameters and Adam untouched, while attempted PDE costs remain.
    """
    check_stop=check_stop or (lambda:None)
    check_stop()
    device=next(model.parameters()).device
    _sync(device);tick=time.perf_counter()
    start=end=None;committed=False;optimizer_started=False
    if device.type=='cuda':
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record()
    if hasattr(model,'set_progress'):model.set_progress(completed_updates)
    optimizer.zero_grad(set_to_none=True)
    velocity=model()
    problem=encoder.problem(completed_updates+1)
    check_velocity(velocity,problem['config'])
    if len(problem['propagators'])!=1:raise ValueError('Update requires one batched solver')
    solver=problem['propagators'][0]['solver'];fd=solver.fd
    target=problem['observed'];fields=int(target.shape[1]);nt=int(target.shape[2])
    observer=PressureEnergy(velocity.shape[-2],velocity.shape[-1],fd.npad,fd.freeSurface)
    handle=fd.register_forward_hook(observer.observe)
    leaf=velocity.detach().requires_grad_(True)
    prediction=outputs=loss=None
    try:
        check_stop()
        _record(record_cost,'forward_fields_attempt',fields,phase,arm_name,completed_updates+1,nt)
        part=time.perf_counter()
        outputs=solver(vmodel=leaf,segment_wavelet=problem['wavelet'],option=0)
        _sync(device)
        _record(record_cost,'forward_fields_complete',fields,phase,arm_name,completed_updates+1,nt,time.perf_counter()-part)
        prediction=outputs[2]
        if prediction.shape!=target.shape or not torch.isfinite(prediction).all():raise ValueError('Invalid predicted records')
        loss=normalized_encoding_loss(prediction,target,problem['reference_shots'])
        if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
        check_stop()
        _record(record_cost,'backward_fields_attempt',fields,phase,arm_name,completed_updates+1,nt)
        part=time.perf_counter();loss.backward();_sync(device)
        _record(record_cost,'backward_fields_complete',fields,phase,arm_name,completed_updates+1,nt,time.perf_counter()-part)
        if (observer.forward_samples,observer.adjoint_samples)!=(nt,nt):raise ValueError('Incomplete illumination observation')
        if leaf.grad is None:raise ValueError('Missing physical velocity gradient')
        direction,illumination=illumination_direction(leaf.grad,observer.source,observer.receiver)
        params=tuple(model.parameters())
        gradients=torch.autograd.grad(velocity,params,grad_outputs=direction)
        for parameter,gradient in zip(params,gradients):
            if not torch.isfinite(gradient).all():raise ValueError('Nonfinite parameter gradient')
            parameter.grad=gradient
        check_stop()
        optimizer_started=True
        optimizer.step()
        committed=True
        if hasattr(model,'set_progress'):model.set_progress(completed_updates+1)
        with torch.no_grad():
            updated=model();check_velocity(updated,problem['config'])
            vmin,vmax=float(updated.min()),float(updated.max())
        if end is not None:end.record()
        _sync(device)
        return dict(completed_updates=completed_updates+1,loss=float(loss.detach()),
            wall_seconds=time.perf_counter()-tick,cuda_ms=start.elapsed_time(end)if start is not None else None,
            velocity_min=vmin,velocity_max=vmax,illumination=illumination,fields=fields,NT=nt,
            alpha=float(model.alpha)if hasattr(model,'alpha')else None,
            encoding_matrix=problem['encoding_matrix'].detach().cpu().tolist())
    except Exception as error:
        if committed:raise CommittedUpdateError(completed_updates+1,error)from error
        if optimizer_started:raise UncertainOptimizerError(completed_updates+1,error)from error
        raise
    finally:
        handle.remove();_clear_graph_cache(solver);leaf.grad=None
        prediction=outputs=loss=None


def evaluate(model,problem,assets,*,record_cost=None,check_stop=None,label='evaluation'):
    """One physical13-shot prediction supplies both clean and noisy errors."""
    check_stop=check_stop or (lambda:None);check_stop()
    if len(problem['propagators'])!=1 or problem['observed'].shape[1]!=13:
        raise ValueError('Evaluation requires unchanged physical13-shot acquisition')
    solver=problem['propagators'][0]['solver'];nt=int(problem['wavelet'].shape[-1])
    device=next(model.parameters()).device
    _sync(device);tick=time.perf_counter()
    try:
        with torch.no_grad():
            velocity=model();check_velocity(velocity,problem['config'])
            _record(record_cost,'forward_fields_attempt',13,'evaluation',label,0,nt)
            outputs=solver(vmodel=velocity,segment_wavelet=problem['wavelet'],option=0)
            _sync(device)
            _record(record_cost,'forward_fields_complete',13,'evaluation',label,0,nt,time.perf_counter()-tick)
            prediction=outputs[2].detach().cpu().numpy().copy()
            v=velocity[0].detach().cpu().numpy().copy()
        if not np.isfinite(prediction).all():raise ValueError('Nonfinite evaluation waveform')
        metrics=velocity_metrics(v,assets['truth'])
        for name in ('clean','noisy'):
            target=assets[name]
            if prediction.shape!=target.shape:raise ValueError('Evaluation data dimensions differ')
            metrics[name+'_mse']=float(np.mean((prediction.astype(np.float64)-target.astype(np.float64))**2))
        metrics['evaluation_wall_seconds']=time.perf_counter()-tick
        return metrics,prediction,v
    finally:_clear_graph_cache(solver)
