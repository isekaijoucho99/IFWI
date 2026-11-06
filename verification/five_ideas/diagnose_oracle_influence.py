"""Synthetic diagnostic only; not a geological qualification or production change.
Run from repository root with python verification/five_ideas/diagnose_oracle_influence.py.
"""
from pathlib import Path
import sys,copy,json,hashlib
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
import torch
import ifwi_research.engine as engine_mod
from ifwi_research.engine import ResearchEngine
from ifwi_research.validate_conditional_dfm import synthetic_snapshot,sensitivity_for_objective
from ifwi_research.conditional_validation import _OracleSource,compare_coupling
from ifwi_research.illumination_prior import IlluminationSoftPrior,SoftPriorConfig
from ifwi_research.safety import protect_physical_gradient,rng_state,restore_rng

torch.set_num_threads(1)
OUT=ROOT/'outputs'/'oracle-influence-diagnostic'; OUT.mkdir(parents=True,exist_ok=True)
engine,background,truth=synthetic_snapshot()
sense=sensitivity_for_objective(engine.objective,background)
class Context: scale_mps=1000.; spacing_m=20.
prior=IlluminationSoftPrior(_OracleSource(truth,Context()),background,sense['illumination'].to(background),spacing_m=20.,operator_identity=engine.objective.operator_identity)
initial_rng=rng_state()
from ifwi_research.run import source_identity

def source_hashes():
    identities=source_identity(ROOT)
    for driver in ('diagnose_oracle_influence.py','diagnose_oracle_optimizers.py'):
        path=ROOT/'verification'/'five_ideas'/driver
        identities[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return identities

INITIAL_SOURCE_HASHES=source_hashes()
def flat(xs): return torch.cat([x.detach().double().flatten() for x in xs])
def norm(xs): return float(flat(xs).norm())
def delta(a,b):
    d=a.detach().double()-b.detach().double()
    return dict(l2=float(d.norm()),rmse=float(d.square().mean().sqrt()),max_abs=float(d.abs().max()),changed=int((a!=b).sum()),total=a.numel())
def metric(a,b): return delta(flat(a),flat(b))
def clone_model(dtype): return copy.deepcopy(engine.model).to(dtype=dtype)
def model_values(m):
    with torch.no_grad(): return m().clone(), (m.raw_output()*1000*m.std_kmps).clone()
def gradients(model,oracle,objective):
    v=model(); params=list(model.parameters()); measured=objective.evaluate(v.detach(),gradient=True)
    gp=torch.autograd.grad(v,params,measured.gradient,retain_graph=True)
    tar=oracle.proposal(v.detach(),1); penalty=oracle.penalty(v,tar)
    raw=torch.autograd.grad(penalty,params)
    return gp,raw,measured,penalty

gp,raw,measured,penalty=gradients(engine.model,prior,engine.objective)
mixed,protection=protect_physical_gradient(gp,raw,.025)
fg,fr,fm=map(flat,(gp,raw,mixed))
result=dict(schema='oracle_zero_effect_diagnostic_v1',synthetic_only=True,scientific_qualification=False,qualification_changed=False,
    native_operator_identity=engine.objective.operator_identity,threads=torch.get_num_threads(),torch_version=str(torch.__version__),mkldnn_enabled=torch.backends.mkldnn.enabled,
    initial=dict(physical_loss=measured.value,penalty=float(penalty.detach()),physical_velocity_gradient_norm=float(measured.gradient.double().norm()),
      physical_parameter_gradient_norm=norm(gp),raw_prior_parameter_gradient_norm=norm(raw),protection=protection,
      parameter_count=fg.numel(),nonzero_physical_parameter_gradients=int((fg!=0).sum()),nonzero_raw_prior_parameter_gradients=int((fr!=0).sum()),
      physical_min_nonzero=float(fg[fg!=0].abs().min()),physical_max_abs=float(fg.abs().max()),raw_physical_cosine=float(torch.dot(fr,fg)/(fr.norm()*fg.norm())),
      protected_sign_flips=int(((fm.sign()!=fg.sign())&(fg!=0)).sum()),new_nonzero_coordinates=int(((fg==0)&(fm!=0)).sum()),
      per_parameter=[dict(name=n,numel=p.numel(),physical_norm=float(g.double().norm()),prior_norm=float(r.double().norm())) for (n,p),g,r in zip(engine.model.named_parameters(),gp,raw)],
      velocity_ulp_mps=float(torch.nextafter(background.flatten()[0],torch.tensor(float('inf')))-background.flatten()[0])))

# Frozen native-FP32 gradients: dtype replay isolates optimizer and decoder arithmetic.
# This is NOT FP64 wave propagation or a valid alternative physical trajectory.
def replay(grads,dtype,optimizer='adam',eps=1e-8,lr=1.25e-5):
    m=clone_model(dtype); ps=list(m.parameters())
    opt=(torch.optim.Adam(ps,lr=lr,eps=eps) if optimizer=='adam' else torch.optim.SGD(ps,lr=lr))
    for p,g in zip(ps,grads):
        # eps=0 is a sign-limit control: skip all-zero blocks to avoid 0/0.
        p.grad=None if eps==0 and not torch.count_nonzero(g) else g.detach().to(dtype).clone()
    before=[p.detach().clone() for p in ps]; opt.step(); v,res=model_values(m)
    return dict(model=m,params=[p.detach().clone() for p in ps],velocity=v,residual=res,update=metric(ps,before),state=copy.deepcopy(opt.state_dict()))
result['replay_notice']='FP64 optimizer/INR replay of captured native FP32 gradients, not FP64 physical inversion. Identical native gradient inputs and initial parameters; no new solver or qualification.'
replay_rows=[]
for kind,eps in [('adam',1e-8),('adam',0.),('adam',1e-6),('sgd',1e-8)]:
    for dtype in (torch.float32,torch.float64):
        base=replay(gp,dtype,kind,eps)
        for ratio in (.025,.1,.25,.5,.9):
            comp,rep=protect_physical_gradient(gp,raw,ratio); other=replay(comp,dtype,kind,eps)
            row=dict(optimizer=kind,eps=eps,zero_gradient_blocks_skipped=(eps==0),dtype=str(dtype),lr=1.25e-5,ratio=ratio,applied_norm=rep['applied_proposal_norm'],
                sign_flips=int(((flat(comp).sign()!=fg.sign())&(fg!=0)).sum()),
                physical_update=base['update'],prior_update=other['update'],parameter_difference=metric(base['params'],other['params']),
                velocity_difference=delta(base['velocity'],other['velocity']),residual_difference=delta(base['residual'],other['residual']))
            # Decode identical final FP32 parameters in FP64 to isolate output quantization.
            if dtype==torch.float32 and kind=='adam' and eps==1e-8:
                bv,br=model_values(copy.deepcopy(base['model']).double()); ov,orr=model_values(copy.deepcopy(other['model']).double())
                row['same_fp32_parameters_fp64_decode']=delta(bv,ov)
            replay_rows.append(row)
result['optimizer_dtype_replays']=replay_rows

# Cap is an upper bound, not a requested norm or hidden amplification.
result['proposal_magnitude_sweep']=[]
for magnitude in (1e-8,1e-6,1e-4,1e-3,1.,1e3):
    comp,rep=protect_physical_gradient(gp,[r*magnitude for r in raw],.025)
    result['proposal_magnitude_sweep'].append(dict(raw_multiplier=magnitude,**rep))
result['native_fp64_probe']={}
try:
    engine.objective.evaluate(background.double(),gradient=True)
    result['native_fp64_probe']['status']='unexpected_success'
except Exception as exc:
    result['native_fp64_probe']=dict(status='unsupported_same_operator',error_type=type(exc).__name__,error=str(exc),
       contract='RepositoryObjective.operator_identity and NativeSensitivity bind float32/order-2. No kernel changes attempted.')

# Instrument existing guard, without changing algorithm or production source.
original_guard=engine_mod.guarded_optimizer_step
branches={}
def run_branch(name,ratio=.025):
    restore_rng(initial_rng)
    source=None if name=='physics_only' else copy.deepcopy(prior)
    if source:
        source._target=None; source._refreshed_at=None; source._refreshes=0; source._cache_hits=0
    cfg=replace(engine.config,method='baseline' if source is None else 'dfm',prior_start=1,prior_every=1,guard_every=1,prior_ratio=ratio)
    obj=copy.copy(engine.objective); obj.counters=copy.deepcopy(engine.objective.counters)
    test=ResearchEngine(copy.deepcopy(engine.model),background,obj,20.,cfg,prior=source)
    test.optimizer.load_state_dict(copy.deepcopy(engine.optimizer.state_dict()))
    rounds=[]
    for update in range(1,4):
        trace=[]; captured={}
        def wrapped(model,opt,physical_value,reference_loss,**kwargs):
            captured['physical']=[g.clone() if g is not None else torch.zeros_like(p) for p,g in zip(model.parameters(),kwargs['fallback_gradients'])]
            captured['composed']=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in model.parameters()]
            def measure():
                value=physical_value(); v,res=model_values(model)
                trace.append(dict(value=value,velocity=v,params=[p.detach().clone() for p in model.parameters()],residual=res))
                return value
            return original_guard(model,opt,measure,reference_loss,**kwargs)
        engine_mod.guarded_optimizer_step=wrapped
        try: record=test.step()
        finally: engine_mod.guarded_optimizer_step=original_guard
        vel,res=model_values(test.model)
        rounds.append(dict(record=record,trace=trace,physical=captured['physical'],composed=captured['composed'],
                           velocity=vel,residual=res,params=[p.detach().clone() for p in test.model.parameters()],state=copy.deepcopy(test.optimizer.state_dict())))
    return rounds
branches['physics_only']=run_branch('physics_only'); branches['oracle']=run_branch('oracle')
# Controlled step-2 factorial: identical parameters and physical gradient,
# change only previous Adam moments and/or the current protected proposal.
common=copy.deepcopy(engine.model)
with torch.no_grad():
    for p,q in zip(common.parameters(),branches['physics_only'][0]['params']): p.copy_(q)
common_prior=copy.deepcopy(prior)
g2,r2,eval2,_=gradients(common,common_prior,engine.objective)
c2,rep2=protect_physical_gradient(g2,r2,.025)
conditioned=[]; cases={}
for moments in ('physics_only','oracle'):
    for direction,gs in [('physical',g2),('composed',c2)]:
        m=copy.deepcopy(common); ps=list(m.parameters()); opt=torch.optim.Adam(ps,lr=1.25e-5)
        opt.load_state_dict(copy.deepcopy(branches[moments][0]['state']))
        for group in opt.param_groups: group['lr']=1.25e-5
        for p,g in zip(ps,gs): p.grad=g.clone()
        opt.step(); v,res=model_values(m); loss=engine.objective.evaluate(v,gradient=False).value
        displacement=[p.detach()-q.detach() for p,q in zip(ps,common.parameters())]
        cases[(moments,direction)]={'params':[p.detach().clone() for p in ps],'velocity':v,'residual':res}
        conditioned.append(dict(previous_moments=moments,current_gradient=direction,physical_before=eval2.value,physical_after=loss,
            strict_guard_would_accept=loss<eval2.value,physical_first_order_change=float(torch.dot(flat(g2),flat(displacement))),
            parameter_update_norm=norm(displacement),velocity_change=delta(v,common().detach())))
result['step2_moment_gradient_factorial']=dict(common_start='Accepted first-step physics-only parameters, identical across four cases',lr=1.25e-5,cases=conditioned,
    current_prior_effect_with_physical_moments=delta(cases[('physics_only','composed')]['velocity'],cases[('physics_only','physical')]['velocity']),
    previous_moment_effect_with_physical_gradient=delta(cases[('oracle','physical')]['velocity'],cases[('physics_only','physical')]['velocity']),
    current_prior_effect_with_oracle_moments=delta(cases[('oracle','composed')]['velocity'],cases[('oracle','physical')]['velocity']),
    previous_moment_effect_with_composed_gradient=delta(cases[('oracle','composed')]['velocity'],cases[('physics_only','composed')]['velocity']))
result['guard_trace']=[]
for b,o in zip(branches['physics_only'],branches['oracle']):
    def state_difference():
        diffs={}
        for key in ('exp_avg','exp_avg_sq'):
            aa=[s[key] for s in b['state']['state'].values() if key in s]
            bb=[s[key] for s in o['state']['state'].values() if key in s]
            diffs[key]=metric(aa,bb) if aa and bb else None
        return diffs
    result['guard_trace'].append(dict(update=b['record']['update'],physical_gradient_norm=norm(b['physical']),oracle_physical_gradient_norm=norm(o['physical']),
      oracle_composed_gradient_norm=norm(o['composed']),oracle_contribution_norm=norm([c-g for c,g in zip(o['composed'],o['physical'])]),
      baseline_step=b['record']['step'],oracle_step=o['record']['step'],prior_report=o['record']['prior'],
      accepted_velocity_difference=delta(b['velocity'],o['velocity']),accepted_residual_difference=delta(b['residual'],o['residual']),
      accepted_parameter_difference=metric(b['params'],o['params']),optimizer_state_entries=len(o['state']['state']),optimizer_state_difference=state_difference(),
      trials=[dict(trial=i+1,kind='adam' if i<b['record']['step']['optimizer_trials'] else 'physical_sgd_recovery',
        baseline_loss=tb['value'],oracle_loss=to['value'],velocity_difference=delta(tb['velocity'],to['velocity']),
        residual_difference=delta(tb['residual'],to['residual']),parameter_difference=metric(tb['params'],to['params']))
        for i,(tb,to) in enumerate(zip(b['trace'],o['trace']))]))

# Actual guarded native-physical trajectories. Ratio is the only changed setting.
result['native_ratio_sweep']=[]
for ratio in (0.,.025,.1,.25,.5,.9):
    restore_rng(initial_rng)
    comp=compare_coupling(engine,prior,steps=3,ratio=ratio,oracle_truth=truth,evaluation_truth=truth,include_dfm=False)
    row=dict(ratio=ratio)
    for name in ('physics_only','oracle'):
        branch=comp[name]
        row[name]={k:v for k,v in branch.items() if k!='history'}
        row[name]['steps']=[r['step'] for r in branch['history']]
        row[name]['prior_ratios']=[r['prior']['ratio'] if r['prior'] else None for r in branch['history']]
    result['native_ratio_sweep'].append(row)

# Small raw-prior control uses uncapped magnitude; this is just a norm-cap demonstration.
result['source_hashes']=source_hashes()
assert INITIAL_SOURCE_HASHES==result['source_hashes'], 'Relevant source changed during diagnostic; rerun against a stable source version'
(OUT/'diagnostics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps(dict(out=str(OUT/'diagnostics.json'),initial=result['initial'],native_fp64_probe=result['native_fp64_probe'],
  ratios=[dict(ratio=r['ratio'],model_rmse=r['oracle']['velocity_rmse_vs_physics_mps'],parameter_l2=r['oracle']['parameter_l2_vs_physics'],
    kinds=[s['update_kind'] for s in r['oracle']['steps']]) for r in result['native_ratio_sweep']]),indent=2))
