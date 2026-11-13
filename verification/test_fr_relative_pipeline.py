import json
import pytest
from experiments.fr_relative_pipeline import validate_receipts, write_once, estimate
from experiments.fr_relative_suite import SOURCES, ROOT


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj))


def test_all_frozen_source_paths_exist():
    assert len(set(SOURCES))==len(SOURCES)
    assert all((ROOT/name).is_file() for name in SOURCES)


def test_same_receipt_and_process_handle_are_required(tmp_path):
    directory=tmp_path/'stage'
    start=dict(run_id='actual',pid=7,parent_pid=8)
    end=dict(start,status='COMPLETED',exit_code=0,cost_totals={})
    result=dict(status='COMPLETED',stage='timing',signature='sig',cost_totals={})
    for name,obj in [('control/LAUNCH.json',start),('control/EXIT.json',end),('result.json',result)]: save(directory/name,obj)
    assert validate_receipts(directory,'timing','sig',0,8)[1]==end
    for stage,sig,rc,launcher in [('formal','sig',0,8),('timing','other',0,8),('timing','sig',1,8),('timing','sig',0,9)]:
        with pytest.raises(ValueError): validate_receipts(directory,stage,sig,rc,launcher)
    save(directory/'control/EXIT.json',dict(end,pid=9))
    with pytest.raises(ValueError): validate_receipts(directory,'timing','sig',0,8)


def test_launch_records_are_never_overwritten(tmp_path):
    path=tmp_path/'receipt.json';write_once(path,dict(value=1))
    with pytest.raises(FileExistsError): write_once(path,dict(value=2))
    assert json.loads(path.read_text())==dict(value=1)


def test_estimate_uses_all_six_actual_medians_and_separates_validation(tmp_path):
    save(tmp_path/'timing/result.json',dict(cases=[dict(arm=str(i),wall_seconds_median=float(i+1)) for i in range(6)]))
    path=tmp_path/'preflight/ledger.jsonl';path.parent.mkdir(parents=True)
    path.write_text(json.dumps(dict(forward_fields_complete=13,wall_seconds=1.5))+'\n')
    report=estimate(tmp_path)
    assert report['training_seconds']==21000
    assert report['warmup_seconds']==84
    assert report['evaluation_seconds_estimate']==366
    assert 'equal-accuracy' in report['limitations']


def test_actual_supervisor_receipts_match_collector_identity(tmp_path,monkeypatch):
    from experiments import fr_relative_pipeline as pipeline
    from experiments import fr_relative_suite as suite
    from experiments.fr_relative_analysis import verify_identity
    root=tmp_path/'task';root.mkdir();plan=tmp_path/'plan.json';save(plan,dict(signature='sig'))
    monkeypatch.setattr(suite,'verify_plan',lambda _:None)
    monkeypatch.setattr(suite,'release_guard',lambda:None)
    monkeypatch.setattr(pipeline,'estimate',lambda _:dict(test=True))
    monkeypatch.setattr(pipeline,'run_analysis',lambda *a:0)
    class Child:
        pid=800;returncode=0
        def __init__(self,command,**kwargs):
            stage=command[command.index('--stage')+1];directory=root/stage
            counts=suite.budget(stage)
            totals={f'{direction}_fields_{state}':counts[f'{direction}_fields']
                    for direction in ('forward','backward') for state in ('attempt','complete')}
            launch=dict(run_id=stage,pid=900,parent_pid=self.pid)
            for name,value in [('control/LAUNCH.json',launch),('control/EXIT.json',dict(launch,status='COMPLETED',exit_code=0,cost_totals=totals)),
                               ('result.json',dict(status='COMPLETED',stage=stage,signature='sig',cost_totals=totals))]: save(directory/name,value)
        def poll(self):return 0
    monkeypatch.setattr(pipeline.subprocess,'Popen',Child)
    assert pipeline.main(['--execute','--output',str(root),'--plan',str(plan),'--dense-data','unused',
        '--dataset-index','unused','--reference-config','unused','--initial-checkpoint','unused',
        '--run-id','unique','--commit','recorded'])==0
    launch=json.loads((root/'PIPELINE_LAUNCH.json').read_text())
    end=json.loads((root/'PIPELINE_EXIT.json').read_text())
    verify_identity(launch,end)
    assert end['status']=='COMPLETED' and end['exit_code']==0
    assert [r['stage'] for r in end['completed_phases']]==['preflight','timing','formal']
    with pytest.raises(FileExistsError):
        pipeline.main(['--execute','--output',str(root),'--plan',str(plan),'--dense-data','unused',
            '--dataset-index','unused','--reference-config','unused','--initial-checkpoint','unused',
            '--run-id','duplicate','--commit','recorded'])
