from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]


def test_cpu_demo_reports_missing_prior_without_inventing_weights(tmp_path):
    out=tmp_path/'diagnostic'
    result=subprocess.run([sys.executable,'-m','ifwi_research.demo_illumination','--out',str(out)],
                          cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['is_geological_benchmark'] is False
    assert summary['prior_status']=='not_run_missing_qualified_checkpoint'
    assert summary['diagnostics']['source_forward_solves']==3
    assert summary['diagnostics']['source_backward_solves']==6
    data=np.load(out/'sensitivity.npz',allow_pickle=False)
    assert data['relative_sensitivity'].shape==(1,12,20)
    assert np.isfinite(data['raw_diagonal']).all() and data['raw_diagonal'].max()>0
    assert 'truth' not in data.files and not list(out.glob('*.pt'))


def test_cpu_demo_rejects_requested_missing_or_legacy_checkpoint(tmp_path):
    checkpoint=tmp_path/'legacy.pt'
    torch.save({'meta':{'schema':'ifwi_small_dfm_v1'},'model':{}},checkpoint)
    for path in (tmp_path/'missing.pt',checkpoint):
        out=tmp_path/path.stem
        result=subprocess.run([sys.executable,'-m','ifwi_research.demo_illumination','--out',str(out),
                               '--checkpoint',str(path)],cwd=ROOT,capture_output=True,text=True)
        assert result.returncode!=0
        expected='Incompatible schema/architecture' if path==checkpoint else 'Checkpoint missing'
        assert expected in result.stderr
        assert not out.exists()
