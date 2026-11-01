import json
import sys
import numpy as np
import torch
from experiments.residual_ifwi_experiment import default_config,build_forward_problem
from experiments.rom.discrete import DiscretePMLSystem
import experiments.run_rom_local_pilot as pilot


def test_local_pilot_persists_nonfinite_candidate_failure_without_an_optimizer_step(tmp_path,monkeypatch):
    config=default_config();config['model']['neurons']=[2,8,8,8,8,1]
    config['acquisition'].update(num_shots=1,source_x_indices=[2],source_depth_index=1,receiver_depth_index=2)
    config['forward'].update(nt=24,dt_s=.001,npad=2,frequency_hz=25.)
    config['training']['shot_batch_size']=1
    truth=np.full((4,5),2800,dtype='float32');background=np.full((4,5),2700,dtype='float32')
    problem=build_forward_problem(truth,config,'cpu');system=DiscretePMLSystem.from_solver(problem['propagators'][0]['solver'])
    run=tmp_path/'input';run.mkdir();out=tmp_path/'output'
    np.save(run/'true_velocity.npy',truth);np.save(run/'initial_velocity.npy',background)
    np.save(run/f'basis_{system.dimension}.npy',np.eye(system.dimension))
    options=dict(pad_factor=4,max_hz=None,alias_tolerance=1e-12,frequency_batch_size=64,
        waveform_tolerance=.01,gradient_tolerance=.05)
    (run/'config.json').write_text(json.dumps(dict(experiment=config,rom=options)))
    original_evaluate=pilot.evaluate
    def fail_rom(velocity,problem,rom=None,backward=False):
        if rom is not None:raise FloatingPointError('injected invalid candidate')
        return original_evaluate(velocity,problem,backward=backward)
    def forbidden_update(*args,**kwargs):raise AssertionError('An invalid candidate must not be updated')
    monkeypatch.setattr(pilot,'evaluate',fail_rom);monkeypatch.setattr(pilot,'train_update',forbidden_update)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda count:None)
    monkeypatch.setattr(sys,'argv',['pilot','--run',str(run),'--rank',str(system.dimension),
        '--output',str(out),'--device','cpu','--updates','1'])
    pilot.main();result=json.loads((out/'summary.json').read_text())
    assert len(result['arms'])==2
    assert all(arm['status']=='candidate_validation_failure' for arm in result['arms'])
    assert all(arm['completed_updates_at_failure']==0 for arm in result['arms'])
    assert all(arm['last_validated_update'] is None for arm in result['arms'])
    assert (out/'run_rom_local_pilot.py').is_file()
