"""Physical batch/serial oracles, not training or throughput measurements.

The caller owns dtype, geometry, IEEE/cuDNN flags and runtime settings. Every
solve uses the same cached observations and the original-shot-mean objective.
Outputs are detached CPU tensors so the next oracle never retains a second
wavefield graph. Fixed tolerances are reported, never adjusted after a failure.
"""
from __future__ import annotations

import math
import time
import traceback

import torch

from core_speed import SourceEncoder, normalized_encoding_loss
from ifwi_research.batched_sources import build_code_solver
from ifwi_research.encoding_illumination import PressureEnergy, illumination_direction
from ifwi_research.suite_control import StopRequested


TOLERANCES = {
    'prediction': {'relative_l2': 1e-5, 'absolute_rms': 1e-12},
    'loss': {'relative_l2': 1e-5, 'absolute_rms': 1e-20},
    'gradient': {'relative_l2': 3e-4, 'absolute_rms': 1e-25},
    'source': {'relative_l2': 3e-4, 'absolute_rms': 1e-25},
    'adjoint': {'relative_l2': 3e-4, 'absolute_rms': 1e-25},
    'H': {'relative_l2': 3e-4, 'absolute_rms': 1e-25},
    'direction': {'relative_l2': 3e-4, 'absolute_rms': 1e-25},
}


class CorrectnessError(RuntimeError):
    def __init__(self, report):
        self.report = report
        failed = [case['name'] for case in report.get('cases', []) if not case.get('passed')]
        super().__init__('Batch correctness failed without retuning: '+', '.join(failed))


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


class _Costs:
    def __init__(self, callback):
        self.callback = callback
        self.totals = dict(forward_fields_attempt=0, forward_fields_complete=0,
                           backward_fields_attempt=0, backward_fields_complete=0)

    def record(self, counter, fields, *, case, implementation, nt, wall_seconds):
        fields = int(fields)
        if self.callback is not None:
            self.callback('correctness.'+case+'.'+implementation+'.'+counter,
                phase='correctness', case=case, implementation=implementation,
                fields=fields, NT=int(nt), wall_seconds=float(wall_seconds),
                **{counter: fields})
        self.totals[counter] += fields


def _clear_graph_cache(solver):
    # Both names occur in project FD adapters. Detached configuration buffers
    # stay intact; only the graph-bearing result of the last velocity padding
    # is dropped. Each native forward reconstructs this transient padding.
    for owner in (solver, solver.fd):
        for name in ('velocity', 'vpadding'):
            value = getattr(owner, name, None)
            if isinstance(value, torch.Tensor) and (value.requires_grad or value.grad_fn is not None):
                setattr(owner, name, None)


def _cpu(value):
    return value.detach().to(device='cpu', copy=True)


def _solve(solver, velocity, wavelet, target, reference_shots, costs, *, case, implementation):
    """Exactly one physical forward and one backward on a fresh velocity leaf."""
    fields, nt = int(target.shape[1]), int(target.shape[2])
    leaf = velocity.detach().clone().requires_grad_(True)
    fd = solver.fd
    observer = PressureEnergy(velocity.shape[-2], velocity.shape[-1], fd.npad, fd.freeSurface)
    handle = fd.register_forward_hook(observer.observe)
    outputs = prediction = loss = None
    try:
        with torch.enable_grad():
            _sync(leaf.device)
            costs.record('forward_fields_attempt', fields, case=case,
                implementation=implementation, nt=nt, wall_seconds=0.)
            tick = time.perf_counter()
            outputs = solver(vmodel=leaf, segment_wavelet=wavelet, option=0)
            _sync(leaf.device)
            costs.record('forward_fields_complete', fields, case=case,
                implementation=implementation, nt=nt, wall_seconds=time.perf_counter()-tick)
            prediction = outputs[2]
            if prediction.shape != target.shape or not torch.isfinite(prediction).all():
                raise ValueError('Oracle waveform shape/nonfinite result')
            loss = normalized_encoding_loss(prediction, target, reference_shots)
            if not torch.isfinite(loss):
                raise ValueError('Oracle loss is nonfinite')
            # An identical bug in the helper would otherwise let batch and
            # serial agree on the wrong common scale. Independently recompute
            # the intended scalar from detached CPU records before backward.
            prediction_cpu, target_cpu = _cpu(prediction), _cpu(target)
            models, _, time_samples, receivers = target_cpu.shape
            explicit_loss = (prediction_cpu.double()-target_cpu.double()).square().sum()
            explicit_loss = explicit_loss/models/reference_shots/time_samples/receivers
            if not _metric(_cpu(loss), explicit_loss, TOLERANCES['loss'])['passed']:
                raise ValueError('Loss normalization differs from explicit original-shot mean')
            costs.record('backward_fields_attempt', fields, case=case,
                implementation=implementation, nt=nt, wall_seconds=0.)
            tick = time.perf_counter()
            loss.backward()
            _sync(leaf.device)
            costs.record('backward_fields_complete', fields, case=case,
                implementation=implementation, nt=nt, wall_seconds=time.perf_counter()-tick)
            if leaf.grad is None or not torch.isfinite(leaf.grad).all():
                raise ValueError('Missing/nonfinite shared-velocity gradient')
            if (observer.forward_samples, observer.adjoint_samples) != (nt, nt):
                raise ValueError('Incomplete pressure/adjoint observation')
            if observer.source is None or observer.receiver is None:
                raise ValueError('Missing pressure/adjoint energy')
            return dict(prediction=prediction_cpu, loss=_cpu(loss),
                gradient=_cpu(leaf.grad), source=_cpu(observer.source),
                adjoint=_cpu(observer.receiver))
    finally:
        handle.remove()
        _clear_graph_cache(solver)
        leaf.grad = None
        # FD result tuples include final pressure/PML fields; none escape.
        outputs = prediction = loss = None


def _decorate(result):
    result['H'] = (result['source'].double()*result['adjoint'].double()).sqrt()
    result['direction'], result['illumination_summary'] = illumination_direction(
        result['gradient'], result['source'], result['adjoint'])
    return result


def _combine(rows):
    if not rows:
        raise ValueError('Oracle requires at least one field')
    result = dict(prediction=torch.cat([row['prediction'] for row in rows], dim=1))
    # Accumulate CPU oracle sums in double, then use the velocity's native
    # dtype for the shared gradient/direction. No averaging by code count.
    for key in ('loss', 'gradient', 'source', 'adjoint'):
        total = torch.stack([row[key].double() for row in rows]).sum(0)
        result[key] = total.to(rows[0][key].dtype)
    return _decorate(result)


def _validate(problem, velocity):
    observed, wavelet = problem['observed'], problem['wavelet']
    if (not isinstance(velocity, torch.Tensor) or velocity.ndim != 3 or velocity.shape[0] != 1
            or observed.ndim != 4 or observed.shape[0] != 1 or observed.shape[1] < 3):
        raise ValueError('One shared [1,nz,nx] velocity and at least three physical shots required')
    if problem.get('encoding_matrix') is not None:
        raise ValueError('Pass the cached physical-shot problem, not an already encoded problem')
    geometry = problem['geometry_tensors']
    if tuple(velocity.shape[-2:]) != (geometry['nz'], geometry['nx']):
        raise ValueError('Velocity geometry differs from the cached problem')
    if velocity.device != observed.device or wavelet.device != observed.device:
        raise ValueError('Do not move or cast a problem silently in the correctness oracle')
    if velocity.dtype != problem['propagators'][0]['solver'].fd.dtype:
        raise ValueError('Velocity dtype differs from the unchanged FD runtime')
    if (int(observed.shape[2]) != int(wavelet.shape[-1]) or
            int(observed.shape[2]) != int(problem['config']['forward']['nt'])):
        raise ValueError('Cached waveform and propagation horizons differ')
    if not all(bool(torch.isfinite(value).all()) for value in (velocity, observed, wavelet)):
        raise ValueError('Nonfinite cached correctness input')
    return int(observed.shape[1])


def serial_reference(problem, velocity, encoding_matrix, *, record_cost=None,
                     update=1, case='serial_reference', _costs=None):
    """Serial one-code solves with the SAME 1/N coefficient, then sums.

    This helper is an oracle only. It returns detached CPU tensors, per-field
    physical cost totals, and never updates a model or optimizer. CPU export
    occurs before the next row so only one field graph is resident at a time.
    """
    reference_shots = _validate(problem, velocity)
    if (encoding_matrix.ndim != 2 or encoding_matrix.shape[1] != reference_shots
            or encoding_matrix.shape[0] < 1 or not torch.isfinite(encoding_matrix).all()
            or encoding_matrix.device != velocity.device):
        raise ValueError('Finite [code_channels,original_shots] matrix on the same device required')
    costs = _Costs(record_cost) if _costs is None else _costs
    before = dict(costs.totals)
    solver = build_code_solver(problem, 1)
    rows = []
    for index in range(encoding_matrix.shape[0]):
        matrix = encoding_matrix[index:index+1].detach()
        solver.fd.weights = matrix
        target = torch.einsum('ks,mstr->mktr', matrix.to(problem['observed'].dtype), problem['observed'])
        rows.append(_solve(solver, velocity, problem['wavelet'], target, reference_shots,
            costs, case=case, implementation='serial_row'+str(index)))
    result = _combine(rows)
    result['cost_totals'] = {key: costs.totals[key]-before[key] for key in before}
    result['update'] = int(update)
    return result


def native_reference(problem, velocity, *, record_cost=None, case='identity', _costs=None):
    """The native all-physical-shot solve, including any configured shot splits."""
    reference_shots = _validate(problem, velocity)
    costs = _Costs(record_cost) if _costs is None else _costs
    before = dict(costs.totals)
    rows, cursor = [], 0
    for index, item in enumerate(problem['propagators']):
        first, last = int(item['first']), int(item['last'])
        if first != cursor or not first < last <= reference_shots:
            raise ValueError('Native shot partitions must cover all shots once in order')
        rows.append(_solve(item['solver'], velocity, problem['wavelet'],
            problem['observed'][:, first:last], reference_shots, costs,
            case=case, implementation='native_batch'+str(index)))
        cursor = last
    if cursor != reference_shots:
        raise ValueError('Incomplete native shot coverage')
    result = _combine(rows)
    result['cost_totals'] = {key: costs.totals[key]-before[key] for key in before}
    return result


def _metric(actual, expected, tolerance):
    if actual.shape != expected.shape:
        return dict(passed=False, actual_shape=list(actual.shape), reference_shape=list(expected.shape))
    a, b = actual.double().reshape(-1), expected.double().reshape(-1)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(passed=False, finite=False)
    difference = a-b
    error = float(torch.linalg.vector_norm(difference))
    reference = float(torch.linalg.vector_norm(b))
    actual_norm = float(torch.linalg.vector_norm(a))
    allowed = tolerance['relative_l2']*reference+tolerance['absolute_rms']*math.sqrt(a.numel())
    cosine = float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b))) if actual_norm and reference else None
    return dict(passed=error <= allowed, finite=True, error_l2=error,
        reference_l2=reference, actual_l2=actual_norm,
        relative_l2=error/max(reference, 1e-300),
        maximum_absolute_error=float(difference.abs().max()),
        norm_ratio=actual_norm/reference if reference else None,
        cosine=cosine, allowed_error_l2=allowed,
        relative_l2_tolerance=tolerance['relative_l2'],
        absolute_rms_tolerance=tolerance['absolute_rms'])


def run_correctness(problem, velocity, *, record_cost=None, update=1) -> dict:
    """Five batch/serial comparisons plus identity/native, with no Adam step.

    Fixed tolerances: predictions/loss1e-5 relative; raw gradient, pressure
    energies, H and preconditioned direction3e-4. Small RMS floors are stated
    explicitly in the report so tiny gradients cannot pass on a large atol.
    Any failed comparison raises CorrectnessError(report), without retuning.
    """
    reference_shots = _validate(problem, velocity)
    costs = _Costs(record_cost)
    report = dict(schema='encoding_batch_correctness_v1', phase='correctness',
        passed=False, status='RUNNING', reference_shots=reference_shots,
        velocity_shape=list(velocity.shape), dtype=str(velocity.dtype),
        device=str(velocity.device), NT=int(problem['observed'].shape[2]),
        update=int(update), tolerances={key: dict(value) for key,value in TOLERANCES.items()},
        cases=[], Adam_updates=0)
    started = time.perf_counter()
    seed = int(problem['config'].get('seed', 3))
    cases = [('global', 1), ('global', 2), ('global', 3),
             ('interleaved', 2), ('interleaved', 3), ('identity', reference_shots)]

    def finalize():
        report.update(cost_totals=dict(costs.totals),
            actual_forward_fields=costs.totals['forward_fields_complete'],
            actual_backward_fields=costs.totals['backward_fields_complete'],
            wall_seconds=time.perf_counter()-started)

    try:
        for grouping, count in cases:
            case = dict(name=grouping+'_B'+str(count), grouping=grouping,
                code_channels=count, passed=False)
            report['cases'].append(case)
            encoder = SourceEncoder(problem, count, seed, grouping=grouping)
            encoded = encoder.problem(update)
            matrix = encoded['encoding_matrix'].detach().clone()
            case['encoding_matrix'] = matrix.cpu().tolist()
            case['group_sizes'] = encoded['group_sizes']
            column_energy = matrix.detach().cpu().double().square().sum(0)
            case['maximum_code_column_energy_error'] = float((column_energy-1).abs().max())
            if not torch.allclose(column_energy, torch.ones_like(column_energy), rtol=1e-6, atol=1e-12):
                raise ValueError('Code column energy differs from the common shot-mean normalization')
            case['oracle'] = 'native_physical_shots' if grouping == 'identity' else 'serial_code_rows'
            if len(encoded['propagators']) != 1:
                raise ValueError('Production encoded path contains multiple wave solvers')
            solver = encoded['propagators'][0]['solver']
            # _solve exports and releases the entire graph before the next call.
            batch = _decorate(_solve(solver, velocity, encoded['wavelet'], encoded['observed'],
                reference_shots, costs, case=case['name'], implementation='batch'))
            del solver, encoded, encoder
            if grouping == 'identity':
                serial = native_reference(problem, velocity, case=case['name'], _costs=costs)
            else:
                serial = serial_reference(problem, velocity, matrix, update=update,
                    case=case['name'], _costs=costs)
            case['metrics'] = {name: _metric(batch[name], serial[name], tolerance)
                               for name,tolerance in TOLERANCES.items()}
            case['passed'] = all(value['passed'] for value in case['metrics'].values())
            finalize()
            if not case['passed']:
                report.update(status='FAIL', passed=False)
                raise CorrectnessError(report)
            del batch, serial, matrix
        report.update(status='PASS', passed=True)
        finalize()
        return report
    except CorrectnessError:
        raise
    except StopRequested as error:
        finalize()
        report.update(status='STOPPED_BY_USER',passed=False)
        error.report=report
        raise
    except Exception as error:
        finalize()
        report.update(status='ERROR', passed=False,
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise CorrectnessError(report) from error
