"""Synthetic artifacts exercise audit failures; these are not inversion results."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_lr_control import ROOT, SCRIPT, audit_module, configs


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape, str(value.dtype))).encode() + value.tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def paired_runs(tmp_path):
    module = audit_module()
    assert callable(getattr(module, 'compare_runs', None)), 'Missing validated read-only run comparison'
    constant, cosine = configs()
    truth = np.linspace(2000, 4500, 94 * 288, dtype=np.float32).reshape(94, 288)
    observed = np.zeros((1, 13, 1000, 288), dtype=np.float32)
    initial = np.full_like(truth, 3000)
    sources = json.loads((ROOT / 'experiments/configs/lr_control_sources.json').read_text())['sources']
    contract = dict(data=constant['data'], truth=array_hash(truth[None]), observed=array_hash(observed),
                    wavelet='a' * 64, params={'dz': 15, 'dt': .0019, 'nt': 1000}, seed=3,
                    budget=4001, clip_grad=None,
                    backbone={k:constant['model'][k] for k in ('neuron','omega_0','activation','outermost_linear','dropout')},
                    evaluation={'depth_threshold': .5, 'corner_size': .25}, sources=sources)
    for label, config in [('constant', constant), ('cosine', cosine)]:
        path = tmp_path / label; path.mkdir()
        write_json(path / 'config.json', config)
        write_json(path / 'comparison_contract.json', contract)
        write_json(path / 'environment.json', {'python':'test-python','device':'cpu','packages':{'torch':'test'},'source_hashes':sources})
        write_json(path / 'status.json', {'state':'completed','completed_updates':4001})
        write_json(path / 'training_summary.json', {'completed_updates':4001,'best_update':4000,'best_loss':.2,
                    'selection_metric':'total_loss','selection_score':.2,'resumed_from_updates':0,'executed_updates':4001})
        for filename, array in [('v_true',truth),('observed',observed),('initial_velocity',initial),
                                 ('last_velocity',truth + 100),('best_velocity',truth + 120)]:
            np.save(path / (filename + '.npy'), array)
        for filename, error, loss in [('final_metrics.json',100,.3),('metrics.json',120,.2)]:
            metrics = dict(full_rmse=float(error),deep_rmse=float(error),data_mse=loss,
                           selection_metric='total_loss',deep_ssim=.5,deep_gradient_fidelity=.9)
            metrics['completed_updates' if filename.startswith('final') else 'best_update'] = 4001 if filename.startswith('final') else 4000
            write_json(path / filename, metrics)
        fields = ['completed_updates','loss_before_update','data_loss_before_update','prior_loss_before_update',
                  'data_mse_before_update','loss_after_update','data_loss_after_update','data_mse_after_update',
                  'gradient_norm','learning_rates','cutoff_hz']
        with (path / 'loss_history.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
            for update in range(1,4002):
                lr = 1e-4 if label == 'constant' else 1e-5 + 9e-5 * (1 + math.cos(math.pi * (update-1) / 4000)) / 2
                after = (.2 if update == 4000 else .3 if update == 4001 else .8) if update % 100 == 0 or update == 4001 else ''
                writer.writerow(dict(completed_updates=update,loss_before_update=1.,data_loss_before_update=1.,
                    prior_loss_before_update=0.,data_mse_before_update=1.,loss_after_update=after,data_loss_after_update=after,
                    data_mse_after_update=after,gradient_norm=2.,learning_rates=json.dumps([lr]),cutoff_hz=''))
    return module, tmp_path / 'constant', tmp_path / 'cosine'


def hashes(path):
    return {p.relative_to(path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in path.rglob('*') if p.is_file()}


def test_compare_outputs_final_and_loss_best_without_touching_inputs(paired_runs, tmp_path):
    module, constant, cosine = paired_runs
    before = hashes(constant), hashes(cosine)
    output = tmp_path / 'report'
    report = module.compare_runs(constant, cosine, output)
    assert report['historical_frozen_runtime_verified'] is False
    assert report['best_selection'] == 'minimum post-update waveform MSE among logged points'
    assert (output / 'fixed_final.csv').is_file()
    assert (output / 'waveform_best.csv').is_file()
    assert (output / 'loss_grad_lr.png').stat().st_size > 1000
    assert (output / 'audit.json').is_file()
    assert before == (hashes(constant), hashes(cosine))
    with pytest.raises(ValueError,match='output'):
        module.compare_runs(constant, cosine, output)
    with pytest.raises(ValueError,match='output'):
        module.compare_runs(constant, cosine, constant / 'report')


@pytest.mark.parametrize('target,change,error',[
 ('training_summary.json',lambda x:x.update(resumed_from_updates=100),'fresh'),
 ('config.json',lambda x:x['training'].update(clip_grad=.25),'configuration'),
 ('status.json',lambda x:x.update(state='running'),'completed'),
 ('final_metrics.json',lambda x:x.update(full_rmse=1),'RMSE'),
 ('environment.json',lambda x:x['packages'].update(torch='different'),'environment'),
 ('comparison_contract.json',lambda x:x['sources'].update(**{'rnn_fd.py':'0'*64}),'source'),
])
def test_compare_rejects_unfair_or_corrupt_inputs(paired_runs,tmp_path,target,change,error):
    module, constant, cosine = paired_runs
    path = cosine / target
    value = json.loads(path.read_text()); change(value); write_json(path,value)
    output = tmp_path / 'bad-report'
    with pytest.raises(ValueError,match=error): module.compare_runs(constant,cosine,output)
    assert not output.exists()


def test_compare_rejects_wrong_lr_and_initial_model(paired_runs,tmp_path):
    module,constant,cosine = paired_runs
    history = cosine / 'loss_history.csv'
    original = history.read_text()
    history.write_text(original.replace('[0.0001]','[0.001]',1))
    with pytest.raises(ValueError,match='learning rate'): module.compare_runs(constant,cosine,tmp_path/'bad-lr')
    history.write_text(original)
    initial = np.load(cosine/'initial_velocity.npy'); initial[0,0] += 1; np.save(cosine/'initial_velocity.npy',initial)
    with pytest.raises(ValueError,match='initial'): module.compare_runs(constant,cosine,tmp_path/'bad-initial')


def test_cli_config_check_has_no_training_side_effects(tmp_path):
    before = list(tmp_path.iterdir())
    result = subprocess.run([sys.executable,str(SCRIPT),'check-configs'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['only_treatment'] == 'optimizer.use_scheduler'
    assert list(tmp_path.iterdir()) == before
