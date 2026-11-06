"""Candidate usefulness and matched-update diagnostics, not qualification.

Ordinary conditions contain only the data-only snapshot. Truth is used after
updates for metrics, or in the explicitly requested and labeled oracle branch.
"""
from __future__ import annotations
import copy
from dataclasses import replace
import math
import torch
from torch import nn
from .conditional_dfm import FrozenConditionalDFM
from .engine import ResearchConfig, ResearchEngine
from .illumination_prior import IlluminationSoftPrior
from .safety import rng_state, restore_rng


class DiagnosticConditionalDFM(FrozenConditionalDFM):
    """Explicit evaluation-only loader; never modifies qualification metadata.

    Allows UNQUALIFIED and toy declarations only in this diagnostic module.
    All schema, physical units, split, training-contract and state checks remain.
    A successful short comparison is not geological scientific qualification.
    """
    diagnostic_only = True


def _rmse(a,b,mask=None):
    error=(a.detach().double()-b.detach().double()).square()
    if mask is not None: error=error[mask]
    return float(error.mean().sqrt()) if error.numel() else None


def _truth_metrics(value,truth,prior):
    if truth is None: return None
    prior._grid(truth,'Evaluation truth')
    return dict(all_rmse_mps=_rmse(value,truth),
        weak_rmse_mps=_rmse(value,truth,prior.illumination<=prior.config.weak_threshold),
        strong_rmse_mps=_rmse(value,truth,prior.illumination>=prior.config.strong_threshold))


def evaluate_candidates(prior,estimate,objective,*,sample_ids,truth=None):
    """Score fixed single draws without selecting a target or mutating its cache.

    Every sample sees the same snapshot estimate, background and estimated mask.
    Physical and truth metrics are observations only, never a best-of-N selector.
    Source call counts include these extra diagnostic generations separately.
    """
    from .safety import InvalidProposalError
    ids=list(sample_ids)
    if not ids or any(type(i) is not int or i<1 for i in ids) or len(set(ids))!=len(ids):
        raise ValueError('Unique positive predeclared sample IDs required')
    prior.prepare_context(prior.cutoff,operator_identity=objective.operator_identity)
    prior._grid(estimate,'Snapshot estimate')
    start_costs=copy.deepcopy(objective.counters)
    physical_before=objective.evaluate(estimate,cutoff=prior.cutoff,gradient=False).value
    start_calls=prior.calls; rows=[]
    for sample_id in ids:
        # Independent wrappers prevent refresh cadence/cache from repeating draws.
        candidate_prior=IlluminationSoftPrior(prior.source,prior.background,prior.illumination,
            spacing_m=prior.spacing_m,config=prior.config,cutoff=prior.cutoff,
            operator_identity=prior.operator_identity)
        try:
            candidate=candidate_prior.proposal(estimate,sample_id)
            if not objective.valid(candidate): raise InvalidProposalError('Candidate violates physical solver bounds/CFL',reason='solver_validity')
            loss=objective.evaluate(candidate,cutoff=prior.cutoff,gradient=False).value
            if not math.isfinite(loss): raise InvalidProposalError('Nonfinite candidate physical objective',reason='nonfinite')
            rows.append(dict(sample_id=sample_id,valid=True,physical_objective=loss,
                physical_objective_delta=loss-physical_before,
                estimate_rms_change_mps=_rmse(candidate,estimate),
                strong_anchor_rms_change_mps=_rmse(candidate,estimate,prior.anchor_mask.bool()),
                truth_evaluation_only=_truth_metrics(candidate,truth,prior)))
        except (InvalidProposalError,FloatingPointError) as exc:
            rows.append(dict(sample_id=sample_id,valid=False,reason=str(exc)))
    valid=[row for row in rows if row['valid']]
    return dict(selection='fixed_single_draws_no_truth_selection',draw_count=len(ids),
        valid_draw_count=len(valid),invalid_fraction=1-len(valid)/len(ids),draws=rows,
        network_evaluations=prior.calls-start_calls,
        costs={k:objective.counters[k]-start_costs[k] for k in start_costs},
        estimate=dict(physical_objective=physical_before,truth_evaluation_only=_truth_metrics(estimate,truth,prior)),
        mean_valid_candidate_physical_objective=(sum(r['physical_objective'] for r in valid)/len(valid) if valid else None),
        diagnostic_only=True,qualification_changed=False)


class _OracleSource:
    """Truth-fed control ONLY: not a DFM, checkpoint, or deployable prior."""
    def __init__(self,truth,prior):
        from .run import tensor_hash
        self.truth=truth.detach().clone(); self.model=nn.Identity().eval(); self.calls=0
        self.scale_mps=prior.scale_mps; self.spacing_m=prior.spacing_m
        self.checkpoint_hash='oracle_diagnostic_only:'+tensor_hash(truth)
        self.meta={'schema':'oracle_truth_diagnostic_not_dfm'}
    def generate(self,background_mps,anchor_mps,anchor_mask,*,sample_id):
        self.calls+=1
        return self.truth.clone()


def compare_coupling(snapshot,prior,*,steps=3,ratio=.025,oracle_truth=None,evaluation_truth=None,include_dfm=True,artifacts=None):
    """Run matched guarded branches from identical parameters, optimizer and RNG.

    The baseline is guarded too: otherwise guard differences confound prior
    influence. The scheduled LR multiplier is fixed at 1 in this stage-local
    diagnostic; the reported base LR is identical for every branch.
    """
    if type(steps) is not int or steps<1: raise ValueError('Positive fixed update budget required')
    if not math.isfinite(ratio) or not 0<=ratio<1: raise ValueError('ratio must lie in [0,1)')
    prior.prepare_context(prior.cutoff,operator_identity=snapshot.objective.operator_identity)
    initial_rng=rng_state(); initial=snapshot.model().detach().clone()
    initial_counters=copy.deepcopy(snapshot.objective.counters)
    truth=evaluation_truth
    if truth is not None: prior._grid(truth,'Evaluation truth')
    priors={'physics_only':None}
    if include_dfm: priors['dfm']=prior
    if oracle_truth is not None:
        prior._grid(oracle_truth,'Oracle diagnostic truth')
        priors['oracle']=IlluminationSoftPrior(_OracleSource(oracle_truth,prior),prior.background,prior.illumination,
            spacing_m=prior.spacing_m,config=prior.config,cutoff=prior.cutoff,operator_identity=prior.operator_identity)
    reports={}; models={}
    try:
        for name,source in priors.items():
            restore_rng(initial_rng)
            cfg=replace(snapshot.config,method='baseline' if source is None else 'dfm',
                prior_start=snapshot.completed+1,prior_every=1,guard_every=1,prior_ratio=ratio,
                prior_invalid_policy='physical_only')
            cloned_prior=copy.deepcopy(source)
            before_calls=getattr(cloned_prior,'calls',0)
            # Never reuse targets from a different estimation trajectory.
            if cloned_prior is not None:
                cloned_prior._target=None; cloned_prior._refreshed_at=None
                cloned_prior._refreshes=0; cloned_prior._cache_hits=0
            # Sequential branches share immutable solver/data construction; counters
            # are private. RepositoryObjective contains imported modules and the
            # solver's last autograd field, so it is not deep-copyable.
            objective=copy.copy(snapshot.objective)
            objective.counters=copy.deepcopy(initial_counters)
            engine=ResearchEngine(copy.deepcopy(snapshot.model),initial,objective,
                prior.spacing_m,cfg,prior=cloned_prior)
            engine.optimizer.load_state_dict(copy.deepcopy(snapshot.optimizer.state_dict()))
            engine.completed=snapshot.completed
            history=[engine.step(prior.cutoff) for _ in range(steps)]
            final=engine.model().detach().clone()
            final_loss=engine.objective.evaluate(final,cutoff=prior.cutoff,gradient=False).value
            models[name]=(final,[p.detach().clone() for p in engine.model.parameters()])
            if artifacts is not None:
                artifacts[name]={'velocity':final.clone(),'engine':copy.deepcopy(engine.state_dict())}
            reports[name]=dict(history=history,final_physical_objective=final_loss,
                optimizer={'name':type(engine.optimizer).__name__,
                    'parameter_groups':[{k:v for k,v in group.items() if k!='params'} for group in engine.optimizer.param_groups]},
                prior_calls=getattr(cloned_prior,'calls',0)-before_calls,
                velocity_rms_change_mps=_rmse(final,initial),
                truth_evaluation_only=_truth_metrics(final,truth,prior),
                oracle_truth_used_for_updates=name=='oracle',
                source_schema=getattr(getattr(cloned_prior,'source',None),'meta',{}).get('schema'),
                costs={k:engine.objective.counters[k]-initial_counters[k] for k in initial_counters})
        reference,reference_params=models['physics_only']
        for name,(velocity,params) in models.items():
            reports[name]['velocity_rmse_vs_physics_mps']=_rmse(velocity,reference)
            reports[name]['parameter_l2_vs_physics']=float(sum((p.double()-q.double()).square().sum()
                for p,q in zip(params,reference_params)).sqrt())
            reports[name]['actual_model_difference']=not torch.equal(velocity,reference)
        return dict(**reports,fixed_updates=steps,start_update=snapshot.completed,
            cutoff_hz=prior.cutoff,learning_rate=snapshot.config.learning_rate,
            diagnostic_only=True,qualification_changed=False,
            interpretation='A nonzero gradient or accepted step alone does not prove useful prior influence; compare actual model differences and metrics.')
    finally: restore_rng(initial_rng)


def restore_snapshot(engine,snapshot,spec,background):
    """Restore only a data-only baseline snapshot under matching problem inputs.

    Diagnostic branching is intentionally not an exact resume of an old source
    version. Its source identity must be saved beside the new result separately.
    """
    from .run import tensor_hash
    state=snapshot['engine']; identity=snapshot['identity']
    if (state.get('schema')!='ifwi_five_ideas_engine_v1' or state['config']['method']!='baseline'
            or state.get('prior_hash') is not None or state.get('prior_state') is not None
            or state.get('router') is not None):
        raise ValueError('A data-only baseline engine snapshot is required')
    expected=dict(baseline=spec,initial_hash=tensor_hash(background),
        observations_hash=tensor_hash(engine.objective.problem['observed']))
    for name,value in expected.items():
        if identity.get(name)!=value: raise ValueError(f'Snapshot {name} mismatch')
    if type(state['completed']) is not int or state['completed']<0:
        raise ValueError('Invalid completed update count')
    engine.model.load_state_dict(state['model'],strict=True)
    engine.optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    engine.completed=state['completed']
    engine.objective.counters=copy.deepcopy(state['counters'])
    restore_rng(state['rng'])
