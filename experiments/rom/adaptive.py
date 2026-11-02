"""Online fixed-rank basis refresh, with all decisions independent of FDTD audits."""
import time
import numpy as np
import torch
from experiments.residual_ifwi_experiment import _simulate,require_finite
from experiments.rom.refresh import (ReferenceCache,primal_space,enriched_space,scalar_stats,galerkin_probe)
from experiments.rom.solver import PressureROM


class RefreshController:
    def __init__(self,system,sources,receivers,wavelet,observed,projector,options,device):
        self.system=system;self.sources=sources;self.receivers=receivers
        self.wavelet=wavelet;self.observed=observed;self.projector=projector
        self.options=options;self.device=device;self.cache=None;self.initial_probe=None

    def make_rom(self,basis):
        o=self.options
        return PressureROM(self.system,basis,self.sources,self.receivers,len(self.wavelet),
            nfft=o['nfft'],max_hz=o['max_hz'],alias_tolerance=o['alias_tolerance'],
            frequency_batch_size=o['frequency_batch_size'],preprojected=self.projector.project(basis)).to(self.device)

    @torch.no_grad()
    def prediction(self,rom,v):
        prediction=_simulate(rom,v,self.wavelet)
        require_finite(prediction,'online provisional ROM prediction')
        return prediction.cpu().numpy()[0]

    def _finish(self,cache,primal,dual,v,pilot_prediction,order):
        target=self.observed.detach().cpu().numpy()[0];events=[]
        basis=enriched_space(cache,primal,dual,self.options['rank']);rom=self.make_rom(basis)
        prediction=self.prediction(rom,v)
        used_residual=pilot_prediction-target
        def discrepancy(used,current):
            a=cache.adjoint_rhs(used);b=cache.adjoint_rhs(current-target)
            result=float(np.linalg.norm(a-b)/max(np.linalg.norm(b),1e-300))
            if not np.isfinite(result):raise FloatingPointError('Nonfinite adjoint-source discrepancy')
            return result
        before=discrepancy(used_residual,prediction)
        if before>self.options['source_consistency_tolerance']:
            used_residual=prediction-target
            dual,stats=cache.transport_adjoint(v.cpu().numpy()[0],used_residual,order)
            events.append(dict(event='adjoint_source_consistency_pass',**scalar_stats(stats)))
            basis=enriched_space(cache,primal,dual,self.options['rank']);rom=self.make_rom(basis)
            prediction=self.prediction(rom,v)
        probe=galerkin_probe(cache,rom,basis,v,prediction-target,count=self.options['probe_frequencies'])
        return basis,rom,dict(events=events,adjoint_source_discrepancy_before=before,
            adjoint_source_discrepancy_after=discrepancy(used_residual,prediction),probe=probe)

    def fresh(self,v):
        start=time.perf_counter();o=self.options;velocity=v.detach().cpu().numpy()[0]
        cache=ReferenceCache(self.system,velocity,self.sources,self.receivers,self.wavelet.cpu().numpy(),None,
            training_frequencies=o['training_frequencies'],max_hz=o['max_hz'],nfft=o['nfft'],alias_tolerance=o['alias_tolerance'],
            workers=o.get('cache_workers',1))
        primal=primal_space(cache,cache.primal,o['primal_rank']);pilot=self.make_rom(primal)
        prediction=self.prediction(pilot,v);cache.complete_adjoint(prediction-self.observed.cpu().numpy()[0])
        basis,rom,meta=self._finish(cache,primal,cache.adjoint,v,prediction,1)
        # Publish only after every reference field/RHS and final basis exist.
        self.cache=cache
        return basis,rom,dict(event='full_refactorization',cache=cache.info,
            seconds=time.perf_counter()-start,**meta)

    def corrected(self,v,order):
        stamp=time.perf_counter();cache=self.cache;o=self.options;events=[]
        fields,stats=cache.transport_primal(v.cpu().numpy()[0],order)
        events.append(dict(event='primal_transport',**scalar_stats(stats)))
        if stats['aggregate_relative_residual']>o['transport_tolerance']:
            return None,None,dict(events=events,reason='primal_transport_indicator',seconds=time.perf_counter()-stamp)
        primal=primal_space(cache,fields,o['primal_rank']);pilot=self.make_rom(primal)
        prediction=self.prediction(pilot,v)
        dual,stats=cache.transport_adjoint(v.cpu().numpy()[0],prediction-self.observed.cpu().numpy()[0],order)
        events.append(dict(event='adjoint_transport',**scalar_stats(stats)))
        if stats['aggregate_relative_residual']>o['transport_tolerance']:
            return None,None,dict(events=events,reason='adjoint_transport_indicator',seconds=time.perf_counter()-stamp)
        basis,rom,meta=self._finish(cache,primal,dual,v,prediction,order)
        return basis,rom,dict(events=events,construction=meta,seconds=time.perf_counter()-stamp)

    def probe_growth(self,probe):
        return max(probe[key]/max(self.initial_probe[key],1e-12) for key in ['primal','adjoint'])

    def prepare(self,v,method):
        """Return (basis, solver or None for native fallback, decision log)."""
        if method not in ('fresh','correction1','correction2','correction3'):raise ValueError('Unknown online refresh method')
        events=[];basis=None;rom=None
        if method!='fresh':
            try:
                basis,rom,meta=self.corrected(v,int(method[-1]));events.append(meta)
                if rom is not None:
                    growth=self.probe_growth(meta['construction']['probe']);meta['heldout_probe_growth']=growth
                    if growth>self.options['probe_growth_limit']:
                        events.append(dict(reason='heldout_final_space_probe_growth',growth=growth));rom=None
                    if meta['construction']['adjoint_source_discrepancy_after']>self.options['source_consistency_tolerance']:
                        events.append(dict(reason='adjoint_source_consistency'));rom=None
                    for event in meta['construction']['events']:
                        if event['aggregate_relative_residual']>self.options['transport_tolerance']:
                            events.append(dict(reason='consistency_transport_indicator'));rom=None
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                events.append(dict(event='correction_failure',failure=repr(error)))
        if rom is None:
            try:
                basis,rom,meta=self.fresh(v);events.append(meta)
                growth=self.probe_growth(meta['probe']);meta['heldout_probe_growth']=growth
                if growth>self.options['probe_growth_limit']:
                    events.append(dict(event='native_fallback',reason='fresh_space_probe_growth',growth=growth));rom=None;basis=None
                if meta['adjoint_source_discrepancy_after']>self.options['source_consistency_tolerance']:
                    events.append(dict(event='native_fallback',reason='fresh_adjoint_source_discrepancy'));rom=None;basis=None
            except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                events.append(dict(event='native_fallback',reason='full_refresh_failure',failure=repr(error)));rom=None;basis=None
        return basis,rom,events
