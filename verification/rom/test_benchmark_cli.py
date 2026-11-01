import json
import os
from pathlib import Path
import subprocess
import sys


def test_cpu_small_marmousi_cli_saves_complete_comparison(tmp_path):
    root=Path(__file__).resolve().parents[2];out=tmp_path/'small'
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'-1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    result=subprocess.run([sys.executable,'-B','-m','experiments.run_rom_benchmark','--small','--device','cpu','--output',str(out)],
        cwd=root,env=env,capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+'\n'+result.stderr
    summary=json.loads((out/'summary.json').read_text())
    assert summary['source_hashes_unchanged']
    assert summary['offline_truth_snapshots_used'] is False
    assert summary['device']=='cpu'
    assert len(summary['comparisons'])>=4
    assert all(r['waveform_relative_error']>=0 for r in summary['comparisons'])
    assert (out/'config.json').is_file() and (out/'true_velocity.npy').is_file()
    assert (out/'comparison.csv').is_file()
