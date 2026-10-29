from pathlib import Path
import json
import subprocess
import sys
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]


def test_demo_cli_creates_trace_and_safe_checkpoint(tmp_path):
    out=tmp_path/'demo'
    result=subprocess.run([sys.executable,'-m','ifwi_research.demo','--method','routing','--steps','3',
                           '--out',str(out)],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['updates']==3 and summary['is_geological_benchmark'] is False
    assert len((out/'history.jsonl').read_text().splitlines())==3
    payload=torch.load(out/'checkpoint.pt',weights_only=True)
    assert payload['completed']==3
    again=subprocess.run([sys.executable,'-m','ifwi_research.demo','--out',str(out)],cwd=ROOT,capture_output=True,text=True)
    assert again.returncode!=0


def test_run_cli_help_is_standalone():
    result=subprocess.run([sys.executable,'-m','ifwi_research.run','--help'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '--baseline' in result.stdout and '--research' in result.stdout


def test_research_json_unknown_fields_rejected(tmp_path):
    from ifwi_research.run import read_research_config
    p=tmp_path/'bad.json'; p.write_text(json.dumps({'method':'baseline','invented_flag':1}))
    with pytest.raises(ValueError,match='Unknown'): read_research_config(p)


def test_research_seed_default_lr_provenance(tmp_path):
    from ifwi_research.run import read_research_config
    p=tmp_path/'valid.json'; p.write_text('{"method":"wavelet"}')
    cfg,prior,interface=read_research_config(p,baseline_learning_rate=.0002,baseline_seed=5)
    assert cfg.seed==5 and cfg.learning_rate==.0002 and prior is None and interface is None
