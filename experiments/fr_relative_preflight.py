"""Small fixed-budget physical oracles for the 52-shot experiment.

The physical oracle covers the first actual 13-shot microbatch, not a complete
52-shot serial solve. The encoded oracle covers all 52 sources with B=4. Both
use the common 1/52 objective denominator. No optimizer is constructed here.
Runtime flags, data hashes, exclusive GPU ownership and receipts belong to the
calling runner. This module never launches itself or retries a failed solve.
"""
from __future__ import annotations

import time
import traceback

import torch

from core_speed import SourceEncoder
from experiments.dense_encoding_physics import DenseAcquisition, build_dense_problem
from experiments.encoding_batch_correctness import (
    CorrectnessError, TOLERANCES, _Costs, _combine, _decorate, _metric, _solve,
)
from ifwi_research.batched_sources import build_code_solver
from ifwi_research.suite_control import StopRequested


EXPECTED_COST = dict(forward_fields_attempt=34, forward_fields_complete=34,
                     backward_fields_attempt=34, backward_fields_complete=34)


def _physical_subset(problem, first, last):
    """Slice physical injection geometry and data, retaining the global config."""
    if not 0 <= first < last <= int(problem['observed'].shape[1]):
        raise ValueError('Invalid physical shot subset')
    geometry = {key: (value[:, first:last] if isinstance(value, torch.Tensor) else value)
                for key, value in problem['geometry_tensors'].items()}
    return {**problem, 'geometry_tensors': geometry,
            'observed': problem['observed'][:, first:last]}


def _serial_rows(problem, velocity, matrix, reference_shots, counters, control, case):
    """Reuse one one-code solver; preserve the global objective denominator."""
    solver = build_code_solver(problem, 1)
    rows = []
    for index in range(matrix.shape[0]):
        control.check_stop()
        weights = matrix[index:index+1].detach()
        solver.fd.weights = weights
        target = torch.einsum('ks,mstr->mktr', weights.to(problem['observed'].dtype),
                              problem['observed'])
        rows.append(_solve(solver, velocity, problem['wavelet'], target,
                           reference_shots, counters, case=case,
                           implementation='serial_row'+str(index)))
    return _combine(rows)


def _compare(batch, serial, case):
    case['metrics'] = {name: _metric(batch[name], serial[name], tolerance)
                       for name, tolerance in TOLERANCES.items()}
    case['passed'] = all(item['passed'] for item in case['metrics'].values())
    return case['passed']


def parameter_vjp_checks(model, raw_velocity_gradient):
    """Independent autograd checks without an FD solve or optimizer update.

    The raw VJP is checked against a scalar weighted-velocity objective. The
    relative-scale chain rule is checked against the raw network output with
    the explicit fixed-background factor. This verifies the parameterization,
    not an assertion that background scaling is a full Hessian compensation.
    """
    if not hasattr(model, 'relative_scale') or not hasattr(model, 'raw_output'):
        raise ValueError('A background-relative velocity model is required')
    direction = raw_velocity_gradient.detach()
    velocity = model()
    if direction.shape != velocity.shape or not torch.isfinite(direction).all():
        raise ValueError('A finite raw velocity gradient matching the model is required')
    parameters = tuple(model.parameters())
    names = tuple(name for name, _ in model.named_parameters())
    actual = torch.autograd.grad(velocity, parameters, grad_outputs=direction,
                                 retain_graph=True)
    scalar = torch.autograd.grad((velocity*direction).sum(), parameters)
    background = model.fixed_init_mps.detach()
    rms = background.double().square().mean().sqrt().to(background.dtype)
    factor = background/rms * (1000*model.std_kmps)
    explicit = torch.autograd.grad(model.raw_output(), parameters,
                                   grad_outputs=direction*factor)
    raw = {name: _metric(a.detach().cpu(), b.detach().cpu(), TOLERANCES['gradient'])
           for name, a, b in zip(names, actual, scalar)}
    relative = {name: _metric(a.detach().cpu(), b.detach().cpu(), TOLERANCES['gradient'])
                for name, a, b in zip(names, actual, explicit)}
    return dict(passed=all(x['passed'] for x in list(raw.values())+list(relative.values())),
                raw_vjp=raw, relative_chain_rule=relative,
                parameter_count=sum(p.numel() for p in parameters),
                physical_forward_fields=0, physical_backward_fields=0, Adam_updates=0)


def run_preflight(control, costs, assets, plan, device):
    """Check actual K13 and B4 arithmetic at a fixed 34F/34B, zero-Adam budget.

    The caller supplies the previously hash-verified ``noisy`` 52-shot array.
    F/B counters retain attempts and completions if an exception interrupts the
    operation. Failed comparisons raise CorrectnessError carrying that report;
    stop requests propagate with their partial report, without retry.
    """
    cfg = plan['config']
    if cfg['acquisition']['num_shots'] != 52:
        raise ValueError('This preflight is fixed to the actual 52-shot plan')
    if 'noisy' not in assets or tuple(assets['noisy'].shape)[1] != 52:
        raise ValueError('The existing 52-shot noisy observations are required')
    device = torch.device(device)
    counters = _Costs(costs.record)
    started = time.perf_counter()
    report = dict(schema='fr_relative_preflight_v1', status='RUNNING', passed=False,
                  reference_shots=52, Adam_updates=0, device=str(device),
                  tolerances={key: dict(value) for key,value in TOLERANCES.items()},
                  expected_cost_totals=dict(EXPECTED_COST), cases=[],
                  scope='one physical microbatch13 plus all-source interleaved B4',
                  cpu_parameterization_checks='verification/test_fr_relative_preflight.py',
                  cpu_raw_update_check='verification/test_dense_preconditioning_control.py')

    def finalize():
        report.update(cost_totals=dict(counters.totals),
                      wall_seconds=time.perf_counter()-started)

    try:
        control.check_stop()
        problem = build_dense_problem(assets['truth'], cfg, device,
                                       observed=assets['noisy'], wavelet=assets['wavelet'])
        velocity = torch.as_tensor(assets['background'], device=device,
                                    dtype=torch.float32).detach()[None]
        subset = _physical_subset(problem, 0, 13)
        case = dict(name='physical_micro13_vs_serial1', code_channels=13,
                    scope='physical_shots_0_to_12_of_52', passed=False,
                    objective_reference_shots=52)
        report['cases'].append(case)
        control.record('preflight_case_start', name=case['name'], scope=case['scope'])
        encoder = SourceEncoder(subset, 13, 3, grouping='identity')
        encoded = encoder.problem(1)
        matrix = encoded['encoding_matrix'].detach().clone()
        solver = encoded['propagators'][0]['solver']
        control.check_stop()
        batch = _decorate(_solve(solver, velocity, encoded['wavelet'], encoded['observed'],
                                 52, counters, case=case['name'], implementation='batch13'))
        del solver, encoded, encoder
        serial = _serial_rows(subset, velocity, matrix, 52, counters, control, case['name'])
        _compare(batch, serial, case)
        finalize()
        control.record('preflight_case_complete', result=case, cost_totals=dict(counters.totals))
        if not case['passed']:
            report['status'] = 'FAIL'
            raise CorrectnessError(report)
        del batch, serial, matrix, subset

        case = dict(name='interleaved4_vs_serial_rows', code_channels=4,
                    scope='all_52_sources_four_code_rows', passed=False,
                    objective_reference_shots=52)
        report['cases'].append(case)
        control.record('preflight_case_start', name=case['name'], scope=case['scope'])
        acquisition = DenseAcquisition(problem, batch_size=13, grouping='interleaved', fields=4, seed=3)
        encoded, = acquisition.batches(1)
        matrix = encoded['encoding_matrix'].detach().clone()
        case['encoding_matrix'] = matrix.cpu().tolist()
        case['group_sizes'] = (matrix != 0).sum(dim=1).cpu().tolist()
        solver = encoded['propagators'][0]['solver']
        if not torch.equal(solver.fd.weights, matrix):
            raise ValueError('Reported interleaved code differs from the FD injection matrix')
        explicit_target = torch.einsum('ks,mstr->mktr', matrix.to(problem['observed'].dtype),
                                       problem['observed'])
        if not torch.equal(encoded['observed'], explicit_target):
            raise ValueError('Interleaved encoded target and injection matrix disagree')
        if not torch.equal(matrix.square().sum(0), torch.ones(52,device=device)):
            raise ValueError('Each original shot must appear once with unit squared code weight')
        control.check_stop()
        batch = _decorate(_solve(solver, velocity, encoded['wavelet'], encoded['observed'],
                                 52, counters, case=case['name'], implementation='batch4'))
        del solver, encoded, acquisition
        serial = _serial_rows(problem, velocity, matrix, 52, counters, control, case['name'])
        _compare(batch, serial, case)
        finalize()
        control.record('preflight_case_complete', result=case, cost_totals=dict(counters.totals))
        if not case['passed']:
            report['status'] = 'FAIL'
            raise CorrectnessError(report)
        if counters.totals != EXPECTED_COST:
            raise ValueError('Preflight actual cost differs from its fixed 34F/34B budget')
        report.update(status='PASS', passed=True)
        finalize()
        return report
    except CorrectnessError:
        finalize()
        raise
    except StopRequested as error:
        finalize()
        report.update(status='STOPPED_BY_USER', passed=False)
        error.report = report
        raise
    except Exception as error:
        finalize()
        report.update(status='ERROR', passed=False, error_type=type(error).__name__,
                      error=str(error), traceback=traceback.format_exc())
        raise CorrectnessError(report) from error
