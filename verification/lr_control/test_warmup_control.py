"""Approved AMFMS-inspired W400/half-floor schedule; not a claimed IFWI gain."""
import copy
import csv
import json
import math
import subprocess
import sys

import numpy as np
import pytest
import torch
import yaml

from test_lr_control import ROOT, SCRIPT, configs, audit_module
from test_comparison import paired_runs, write_json
from test_runtime_contract import small_problem, make_model, assert_nested_equal

WARMUP_CONFIG = ROOT / 'experiments/configs/modern13_warmup400_cosine_half.yaml'


def warmup_config():
    assert WARMUP_CONFIG.is_file(), 'Missing approved W400/half-floor LR configuration'
    return yaml.safe_load(WARMUP_CONFIG.read_text())


def expected_rate(update):
    if update <= 400: return 1e-4 * update / 400
    return 5e-5 + 5e-5 * (1 + math.cos(math.pi * (update - 400) / 3601)) / 2


def warmup_comparator():
    module = audit_module()
    assert callable(getattr(module, 'validate_warmup_pair', None)), 'Missing independent warmup configuration validation'
    assert callable(getattr(module, 'compare_warmup_runs', None)), 'Missing independent constant/warmup comparison'
    return module


def test_registered_warmup_configuration_and_old_pair_strictness():
    constant, cosine = configs(); warmup = warmup_config(); module = warmup_comparator()
    assert warmup['optimizer']['scheduler_params'] == {'warmup_epochs':400,'max_epochs':4001,'eta_min':5e-5}
    stripped = copy.deepcopy(warmup)
    stripped['optimizer']['scheduler_params'] = copy.deepcopy(cosine['optimizer']['scheduler_params'])
    for config in (stripped, cosine):
        config.pop('description'); config.pop('experiment_name')
    assert stripped == cosine
    report = module.validate_warmup_pair(constant, warmup)
    assert report['comparison'] == 'constant_vs_warmup400_cosine_half'
    assert report['warmup_config_sha256'] == module.json_hash(warmup)
    assert set(report['changed_fields']) == {'optimizer.use_scheduler','optimizer.scheduler_params.warmup_epochs','optimizer.scheduler_params.eta_min'}
    with pytest.raises(ValueError,match='configuration'): module.validate_pair(constant,warmup)
    assert module.validate_pair(*configs())['only_treatment'] == 'optimizer.use_scheduler'


@pytest.mark.parametrize('section,key,value', [
    ('optimizer','learning_rate',.001), ('training','max_iterations',4000),
    ('training','clip_grad',.25), ('model','omega_0',50), ('loss','use_prior',True),
])
def test_warmup_rejects_other_factors(section,key,value):
    module = warmup_comparator(); constant,_ = configs(); warmup = warmup_config()
    warmup[section][key] = value
    with pytest.raises(ValueError,match='configuration'): module.validate_warmup_pair(constant,warmup)


@pytest.mark.parametrize('key,value', [('warmup_epochs',100),('warmup_epochs',399),('warmup_epochs',400.0),
                                       ('max_epochs',4000),('eta_min',1e-5),('unknown',1)])
def test_warmup_rejects_nonregistered_schedule(key,value):
    module = warmup_comparator(); constant,_ = configs(); warmup = warmup_config()
    warmup['optimizer']['scheduler_params'][key] = value
    with pytest.raises(ValueError,match='configuration'): module.validate_warmup_pair(constant,warmup)


def test_warmup_endpoints_monotonic_boundary_and_scheduler_resume():
    from improved_modules.optimizers import create_improved_optimizer
    config = warmup_config()
    opt,scheduler,_ = create_improved_optimizer(torch.nn.Linear(2,1),config['optimizer'])
    values=[]
    for index in range(4001):
        scheduler.step(index);values.append(opt.param_groups[0]['lr'])
    assert values[0] == pytest.approx(2.5e-7)
    assert values[399] == pytest.approx(1e-4)
    assert values[400] == pytest.approx(expected_rate(401),abs=1e-15)
    assert 0 < values[399]-values[400] < 1e-10
    assert values[-1] == pytest.approx(5e-5)
    assert all(a < b for a,b in zip(values[:399],values[1:400]))
    assert all(a > b for a,b in zip(values[399:-1],values[400:]))
    for completed in (399,400,401):
        scheduler.step(completed-1)
        other_opt,other,_ = create_improved_optimizer(torch.nn.Linear(2,1),config['optimizer'])
        other.load_state_dict(scheduler.state_dict());other.step(completed)
        assert other_opt.param_groups[0]['lr'] == values[completed]
    with pytest.raises(ValueError,match='horizon'): scheduler.step(4001)


def test_real_adam_resume_across_warmup_boundary():
    from improved_modules.optimizers import create_improved_optimizer
    config=warmup_config(); torch.manual_seed(3)
    initial=torch.nn.Linear(2,1).state_dict(); x=torch.tensor([[1.,2.],[-2.,1.]])
    def start():
        net=torch.nn.Linear(2,1);net.load_state_dict(initial)
        opt,scheduler,_=create_improved_optimizer(net,config['optimizer'])
        return net,opt,scheduler
    def steps(net,opt,scheduler,first,last):
        for step in range(first,last):
            scheduler.step(step);opt.zero_grad();net(x).square().mean().backward();opt.step()
    full,fo,fs=start();steps(full,fo,fs,0,403)
    for split in (399,400,401):
        net,opt,scheduler=start();steps(net,opt,scheduler,0,split)
        saved=copy.deepcopy((net.state_dict(),opt.state_dict(),scheduler.state_dict()))
        resumed,ro,rs=start();resumed.load_state_dict(saved[0]);ro.load_state_dict(saved[1]);rs.load_state_dict(saved[2])
        steps(resumed,ro,rs,split,403)
        assert_nested_equal(full.state_dict(),resumed.state_dict());assert_nested_equal(fo.state_dict(),ro.state_dict())
        assert fs.state_dict()==rs.state_dict()


def test_real_small_fd_warmup_and_resume(small_problem,tmp_path):
    from experiment_runtime import train_loop,evaluate_loss
    data,config=small_problem;config['optimizer']=warmup_config()['optimizer']
    full=make_model(config,data);result=train_loop(full,data,config,tmp_path/'full')
    train_loop(make_model(config,data),data,config,tmp_path/'first',stop_after=2)
    resumed=make_model(config,data)
    train_loop(resumed,data,config,tmp_path/'resumed',resume=tmp_path/'first/last.pth')
    a=torch.load(tmp_path/'full/last.pth',weights_only=False);b=torch.load(tmp_path/'resumed/last.pth',weights_only=False)
    for key in ('model','optimizer','scheduler','history','best_state','best_loss','best_update'): assert_nested_equal(a[key],b[key])
    for row in result['history']:
        assert json.loads(row['learning_rates'])[0]==expected_rate(row['completed_updates'])
        assert np.isfinite(row['gradient_norm']) and row['gradient_norm']>0
    for filename in ('last','best'):
        ck=torch.load(tmp_path/'full'/(filename+'.pth'),weights_only=False);full.vel_net.load_state_dict(ck['model'])
        value,_=evaluate_loss(full,data,0)
        assert value==pytest.approx(result['history'][ck['completed_updates']-1]['loss_after_update'],rel=1e-6)


def test_constant_warmup_report_needs_no_pure_cosine_run(paired_runs,tmp_path):
    _,constant,candidate=paired_runs;module=warmup_comparator()
    write_json(candidate/'config.json',warmup_config())
    path=candidate/'loss_history.csv'
    with path.open(newline='') as stream:
        reader=csv.DictReader(stream);fields=reader.fieldnames;rows=list(reader)
    for row in rows: row['learning_rates']=json.dumps([expected_rate(int(row['completed_updates']))])
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    output=tmp_path/'warmup_report';report=module.compare_warmup_runs(constant,candidate,output)
    assert report['comparison']=='constant_vs_warmup400_cosine_half'
    with (output/'fixed_final.csv').open() as stream:
        assert [row['arm'] for row in csv.DictReader(stream)]==['constant','warmup400_cosine_half']
    assert (output/'waveform_best.csv').is_file() and (output/'loss_grad_lr.png').is_file()
    with pytest.raises(ValueError,match='configuration'): module.compare_runs(constant,candidate,tmp_path/'wrong-old-pair')
    # The approved warmup trajectory has a different cosine phase and floor.
    rows[400]['learning_rates']='[0.0001]'
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError,match='learning rate'):
        module.compare_warmup_runs(constant,candidate,tmp_path/'wrong-boundary')


def test_warmup_cli_config_check_is_read_only(tmp_path):
    result=subprocess.run([sys.executable,str(SCRIPT),'check-warmup-configs'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['comparison']=='constant_vs_warmup400_cosine_half'
    assert list(tmp_path.iterdir())==[]
