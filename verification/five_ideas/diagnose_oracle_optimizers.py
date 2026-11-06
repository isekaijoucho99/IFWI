"""Fixed-budget synthetic warmup/optimizer controls; no production changes."""
import copy,json,sys,hashlib
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from ifwi_research.validate_conditional_dfm import synthetic_snapshot,sensitivity_for_objective
from ifwi_research.conditional_validation import _OracleSource,compare_coupling
from ifwi_research.illumination_prior import IlluminationSoftPrior
from ifwi_research.engine import ResearchEngine
from ifwi_research.safety import protect_physical_gradient,rng_state,restore_rng

torch.set_num_threads(1);out=ROOT/'outputs'/'oracle-influence-diagnostic'; out.mkdir(parents=True,exist_ok=True)
from ifwi_research.run import source_identity

def source_hashes():
    identities=source_identity(ROOT)
    for driver in ('diagnose_oracle_influence.py','diagnose_oracle_optimizers.py'):
        path=ROOT/'verification'/'five_ideas'/driver
        identities[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return identities

INITIAL_SOURCE_HASHES=source_hashes()
snap,bg,truth=synthetic_snapshot(); initial_rng=rng_state()
class Context:scale_mps=1000.;spacing_m=20.
def make_prior(snapshot):
    estimate=snapshot.model().detach();sense=sensitivity_for_objective(snapshot.objective,estimate)
    return IlluminationSoftPrior(_OracleSource(truth,Context()),estimate,sense['illumination'].to(estimate),spacing_m=20.,operator_identity=snapshot.objective.operator_identity)
def flat(xs):return torch.cat([x.detach().double().flatten() for x in xs])
def gradients(snapshot,prior):
    v=snapshot.model(); ps=list(snapshot.model.parameters()); measured=snapshot.objective.evaluate(v.detach(),gradient=True)
    g=torch.autograd.grad(v,ps,measured.gradient,retain_graph=True)
    raw=torch.autograd.grad(prior.penalty(v,prior.proposal(v.detach(),snapshot.completed+1)),ps)
    comp,report=protect_physical_gradient(g,raw,.025)
    return dict(physical_loss=measured.value,protection=report,nonzero_physical_gradients=int((flat(g)!=0).sum()),
     nonzero_physical_hidden_gradients=int((flat(g[:-2])!=0).sum()),parameter_count=flat(g).numel(),sign_flips=int(((flat(g).sign()!=flat(comp).sign())&(flat(g)!=0)).sum()),
     optimizer_state_entries=len(snapshot.optimizer.state),optimizer_steps=[float(v['step']) for v in snapshot.optimizer.state.values() if 'step' in v],
     per_parameter=[dict(name=n,physical_norm=float(a.double().norm()),raw_prior_norm=float(b.double().norm())) for (n,p),a,b in zip(snapshot.model.named_parameters(),g,raw)])
def compact(comp):
    out={k:v for k,v in comp.items() if k not in ('physics_only','oracle')}
    for name in ('physics_only','oracle'):
        branch=comp[name];out[name]={k:v for k,v in branch.items() if k!='history'}
        out[name]['steps']=[r['step'] for r in branch['history']]
    return out
results=dict(schema='synthetic_warmed_optimizer_controls_v1',
 torch_version=str(torch.__version__),mkldnn_enabled=torch.backends.mkldnn.enabled,threads=torch.get_num_threads(),scientific_qualification=False,qualification_changed=False,
 notices=['Warmup budgets 5 and 10 fixed before observing results, using ordinary unguarded data-only Adam (existing baseline defaults); no truth-based selection.',
          'Matched baseline and oracle branches use identical optimizer type/settings, model, optimizer state and RNG for each control. SGD uses no Adam state.',
          'Optimizer/LR controls are sensitivity experiments, not tuned comparisons, convergence evidence, or production recommendations.'])

# Direct native comparison from zero-head start for predeclared optimizer settings.
initial_prior=make_prior(snap)
results['initial_optimizer_controls']=[]
for kind,lr,eps in [('adam',1e-4,1e-8),('adam',1e-4,1e-6),('sgd',1e-4,None),('sgd',1e-2,None)]:
    branches={};velocities={};params={}
    for name in ('physics_only','oracle'):
        restore_rng(initial_rng);prior=None if name=='physics_only' else copy.deepcopy(initial_prior)
        cfg=replace(snap.config,method='baseline' if prior is None else 'dfm',prior_start=1,prior_every=1,guard_every=1,prior_ratio=.025,learning_rate=lr)
        obj=copy.copy(snap.objective);obj.counters=copy.deepcopy(snap.objective.counters)
        e=ResearchEngine(copy.deepcopy(snap.model),bg,obj,20.,cfg,prior=prior)
        e.optimizer=(torch.optim.Adam(e.model.parameters(),lr=lr,eps=eps) if kind=='adam' else torch.optim.SGD(e.model.parameters(),lr=lr))
        records=[e.step() for _ in range(3)];v=e.model().detach();velocities[name]=v;params[name]=flat(e.model.parameters())
        branches[name]=dict(final_physical_objective=obj.evaluate(v,gradient=False).value,
          velocity_rms_change_mps=float((v.double()-bg.double()).square().mean().sqrt()),
          steps=[r['step'] for r in records],optimizer_state_entries=len(e.optimizer.state))
    results['initial_optimizer_controls'].append(dict(optimizer=kind,lr=lr,eps=eps,ratio=.025,**branches,
      velocity_rmse_vs_physics_mps=float((velocities['oracle'].double()-velocities['physics_only'].double()).square().mean().sqrt()),
      parameter_l2_vs_physics=float((params['oracle']-params['physics_only']).norm())))

# Fixed data-only unguarded baseline trajectory: snapshots at 5 and 10 updates.
restore_rng(initial_rng);warm_records=[];results['warmed_snapshots']=[]
for completed in range(1,11):
    warm_records.append(snap.step())
    if completed in (5,10):
        prior=make_prior(snap)
        gr=gradients(snap,prior)
        comp=compare_coupling(snap,prior,steps=3,ratio=.025,oracle_truth=truth,evaluation_truth=truth,include_dfm=False)
        results['warmed_snapshots'].append(dict(warmup_updates=completed,initial_diagnostics=gr,comparison=compact(comp)))
results['warmup_history']=warm_records
results['source_hashes']=source_hashes()
assert INITIAL_SOURCE_HASHES==results['source_hashes'], 'Relevant source changed during controls; rerun'
(out/'warmed_optimizer_controls.json').write_text(json.dumps(results,indent=2,allow_nan=False))
print(json.dumps(results,indent=2,allow_nan=False))
