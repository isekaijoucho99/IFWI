"""Compare completed, compatible experiments; preserve replicate identity."""
import argparse
import json
from pathlib import Path
import pandas as pd


def load_completed_results(root):
    records=[]; contracts={}
    for p in sorted(Path(root).iterdir()):
        if not p.is_dir() or not (p/'status.json').exists(): continue
        if json.loads((p/'status.json').read_text())['state']!='completed': continue
        config=json.loads((p/'config.json').read_text(encoding='utf-8'))
        contract=json.loads((p/'comparison_contract.json').read_text(encoding='utf-8'))
        seed=config['seed']
        if seed in contracts and contract!=contracts[seed]:
            raise ValueError('Incompatible experiment contracts for seed '+str(seed)+': '+p.name)
        contracts[seed]=contract
        metrics=json.loads((p/'metrics.json').read_text(encoding='utf-8'))
        history=pd.read_csv(p/'loss_history.csv')
        row={'run_id':p.name,'experiment':config['experiment_name'],'seed':seed,'updates':int(history.iloc[-1]['completed_updates'])}
        row.update({k:v for k,v in metrics.items() if not isinstance(v,(dict,list))})
        if (p/'final_metrics.json').exists():
            final=json.loads((p/'final_metrics.json').read_text(encoding='utf-8'))
            row.update({'final_'+k:v for k,v in final.items() if not isinstance(v,(dict,list))})
        summary=p/'training_summary.json'
        if summary.exists():
            s=json.loads(summary.read_text()); row.update(seconds=s['total_seconds'],peak_cuda_bytes=s['peak_cuda_bytes'],
                executed_updates=s['executed_updates'],resumed_from_updates=s['resumed_from_updates'],timing_scope=s['timing_scope'])
        records.append(row)
    # Across seeds allow stochastic observations/seed to differ, but hold protocol fixed.
    protocols=[]
    for c in contracts.values():
        protocols.append({k:v for k,v in c.items() if k not in ('seed','observed')})
    if protocols and any(c!=protocols[0] for c in protocols[1:]):
        raise ValueError('Incompatible protocols across seeds')
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',required=True)
    parser.add_argument('--output',default='comparison')
    args=parser.parse_args(); rows=load_completed_results(args.results_dir)
    if not rows: raise ValueError('No completed compatible runs')
    out=Path(args.results_dir)/args.output; out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows); df.to_csv(out/'runs.csv',index=False)
    numeric=[c for c in df.select_dtypes('number').columns if c not in ('seed',)]
    # Average repeats within seed before aggregating independent seeds.
    byseed=df.groupby(['experiment','seed'])[numeric].mean()
    byseed.to_csv(out/'by_seed.csv')
    byseed.groupby('experiment').agg(['mean','std','count']).to_csv(out/'summary.csv')
    print(df.to_string(index=False)); print('Saved comparison to',out)
    return 0

if __name__=='__main__': raise SystemExit(main())
