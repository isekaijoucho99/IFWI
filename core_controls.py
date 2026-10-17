"""Initialization controls; inversion still uses the unchanged historical update."""
import copy
import math
import time
import torch
from experiments.residual_ifwi_experiment import _rng_state, _restore_rng


def sync(model):
    device = next(model.parameters()).device
    if device.type == 'cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def small_output(model, rms_mps):
    if model.parameterization != 'residual' or not math.isfinite(rms_mps) or rms_mps <= 0:
        raise ValueError('Small-output control requires residual representation and positive RMS')
    layer = model.net.linear[-1]
    layer = getattr(layer, 'layer', layer)
    layer.bias.zero_()
    layer.weight.normal_(0., 1.)
    delta = model() - model.fixed_init_mps
    actual = delta.square().mean().sqrt()
    if not torch.isfinite(actual) or actual <= 0: raise ValueError('Degenerate random output')
    layer.weight.mul_(rms_mps / actual)
    delta = model() - model.fixed_init_mps
    return {'distribution_before_rescaling': 'normal(0,1)', 'target_rms_mps': rms_mps,
            'actual_rms_mps': float(delta.square().mean().sqrt()),
            'max_abs_mps': float(delta.abs().max()), 'bias': 0.}


def prefit_background(model, updates, lr, emit=lambda message: None, checkpoint=None, saved=None, keep_state=False):
    """Fit the direct network to the background (best state kept). keep_state additionally returns the
    final Adam state under 'optimizer_state' (popped by the runner), so the inversion can continue with
    the pretraining moments instead of a reset; the moments belong to the last prefit step."""
    if model.parameterization != 'absolute': raise ValueError('Prefit control must be absolute IFWI')
    sync(model); started = time.perf_counter()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history=[]; best_loss=float('inf'); best_state=None; best_update=0; first=0; prior_seconds=0.
    if saved:
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        first=saved['completed_updates']; history=saved['history']
        best_loss=saved['best_loss']; best_state=saved['best_state']; best_update=saved['best_update']
        prior_seconds=saved['seconds']; _restore_rng(saved['rng'])
    scale=1000.*model.std_kmps
    target=model.fixed_init_mps.detach()
    for update in range(first, updates+1):
        opt.zero_grad(set_to_none=True)
        loss=((model()-target)/scale).square().mean()
        if not torch.isfinite(loss): raise FloatingPointError('Non-finite background prefit')
        value=float(loss.detach())
        if not history or history[-1]['completed_updates'] != update:
            history.append({'completed_updates':update, 'normalized_mse':value,
                            'rmse_vs_background_mps':math.sqrt(value)*scale})
        if value < best_loss:
            best_loss=value; best_update=update; best_state=copy.deepcopy(model.state_dict())
        if update % 500 == 0 or update == updates:
            sync(model)
            emit(f'Prefit {update}/{updates}: background RMSE={math.sqrt(value)*scale:.6f} m/s')
            if checkpoint:
                checkpoint({'phase':'prefit','model':copy.deepcopy(model.state_dict()),
                    'optimizer':copy.deepcopy(opt.state_dict()), 'completed_updates':update,
                    'history':history, 'best_state':best_state, 'best_loss':best_loss,
                    'best_update':best_update, 'rng':_rng_state(),
                    'seconds':prior_seconds+time.perf_counter()-started})
        if update == updates: break
        loss.backward(); opt.step()
    model.load_state_dict(best_state)
    sync(model)
    report={'target':'fixed_background_only', 'updates':updates,'learning_rate':lr,
            'best_update':best_update, 'rmse_vs_background_mps':math.sqrt(best_loss)*scale,
            'seconds':prior_seconds+time.perf_counter()-started, 'history':history,
            'optimizer_reset_for_inversion':not keep_state}
    if keep_state:
        report['optimizer_state']=copy.deepcopy(opt.state_dict())
    return report


def training_state(model, optimizer, completed, timing):
    return {'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'completed_updates':completed,'timing':copy.deepcopy(timing),'rng':_rng_state()}


def restore_training_state(saved, model, optimizer):
    model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
    _restore_rng(saved['rng'])
    return saved['completed_updates'], copy.deepcopy(saved['timing'])


@torch.no_grad()
def layer_snapshot(model):
    result=[]
    for wrapped in model.net.linear:
        layer=getattr(wrapped,'layer',wrapped)
        key='lamb' if hasattr(layer,'lamb') else 'weight'
        result.append((key,getattr(layer,key).detach().clone(),layer.weight.detach().clone()))
    return result


@torch.no_grad()
def layer_changes(model, before):
    rows=[]
    for index,(wrapped,(key,parameter,weight)) in enumerate(zip(model.net.linear,before)):
        layer=getattr(wrapped,'layer',wrapped)
        delta=getattr(layer,key)-parameter
        dw=delta @ layer.bases if key == 'lamb' else delta
        norm=float(torch.linalg.vector_norm(dw)); prior=float(torch.linalg.vector_norm(weight))
        rows.append({'layer':index,'effective_weight_absolute_delta':norm,
            'effective_weight_relative_delta':norm/prior if prior>0 else None,
            'weight_norm_before':prior,'calculation':'delta_lambda @ B' if key=='lamb' else 'delta_weight'})
    return rows
