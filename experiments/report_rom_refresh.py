"""Compact, all-arm cost/accuracy report from completed bounded refresh pilots."""
from pathlib import Path
import argparse
import csv
import json
import shutil


def count_event(tree,name):
    if isinstance(tree,dict):return int(tree.get('event')==name)+sum(count_event(v,name) for v in tree.values())
    if isinstance(tree,list):return sum(count_event(v,name) for v in tree)
    return 0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True);table=[];audits=[];inventory=[]
    labels=dict(fdtd='原始 FDTD',frozen='固定基空间',fresh='每次更新完整重建',
        correction1='一次残差校正＋动态刷新',correction2='两次残差校正＋动态刷新',correction3='三次残差校正＋动态刷新')
    for index,path in enumerate(args.runs):
        root=Path(path);summary=json.loads((root/'summary.json').read_text())
        cfg=json.loads((root/'config.json').read_text());workers=cfg['refresh'].get('cache_workers',1)
        target=out/f'run_{index+1}_workers{workers}';target.mkdir(exist_ok=True)
        for file in ['summary.json','config.json','preparation.json','provenance.json']:
            shutil.copy2(root/file,target/file)
        inventory.append(dict(run=str(root),source_hashes_unchanged=summary['source_hashes_unchanged'],
            process_wall_seconds=summary['experiment_wall_seconds']))
        reference=next((arm for arm in summary['methods'] if arm['method']=='fdtd'),None)
        for method in summary['methods']:
            used=[a for a in method['audits'] if a['used_for_update']]
            final=method['audits'][-1]
            def maximum(key):
                valid=[a[key]['relative'] for a in used if key in a and a[key]['relative'] is not None]
                return max(valid) if valid else None
            row=dict(run=index+1,cache_workers=workers,method=method['method'],label=labels[method['method']],
                updates=method['completed_updates'],online_seconds=method['online_total_seconds'],
                steady_after_first_seconds=method['online_total_seconds']-method['steps'][0]['online_seconds'],
                steady_after_first_speedup=(reference['online_total_seconds']-reference['steps'][0]['online_seconds'])/
                    (method['online_total_seconds']-method['steps'][0]['online_seconds']) if reference else None,
                online_speedup=method['online_speedup'],all_used_gradients_pass=method['all_used_gradients_pass'],
                maximum_waveform_relative_error=maximum('waveform'),maximum_velocity_gradient_relative_error=maximum('velocity_gradient'),
                maximum_parameter_gradient_relative_error=maximum('parameter_gradient'),
                full_online_refactorizations=count_event(method['steps'],'full_refactorization'),
                native_update_count=sum(step['backend']=='fdtd' for step in method['steps']),
                source_consistency_passes=count_event(method['steps'],'adjoint_source_consistency_pass'),
                including_preparation_seconds=method['including_initial_preparation_seconds'],
                audit_seconds=method['audit_seconds'],audit_export_seconds=method['audit_artifact_export_seconds'],
                final_fdtd_mse=final.get('fdtd_mse'),final_rmse_mps=final.get('rmse_mps'),final_ssim=final.get('ssim'),
                host_process_peak_mib=method['memory_after_online']['host_process_peak_mib'],
                cuda_peak_allocated_mib=method['memory_after_online'].get('cuda_peak_allocated_mib'))
            table.append(row)
            for state in method['audits']:audits.append(dict(run=index+1,method=method['method'],**state))
    with (out/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    (out/'all_audits.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'LOCAL_ARTIFACTS.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 实际 IFWI 动态降阶小规模验证','','所有数值来自完成的 GPU 轨迹；全部方法保留。梯度误差取实际用于更新的5个状态之最大值；最终状态另外列出，未被用来更新。初始准备可以从在线主口径排除，每次动态刷新、失败尝试、在线指标、数据传输、保存实际基的内存快照、优化器及回退均在在线总时间内。后验核验不影响训练决策，实际耗时单列。','','| 运行/缓存并发 | 方法 | 更新数 | 在线总秒 | 相对 FDTD | 最大波形误差 % | 最大速度梯度误差 % | 最大参数梯度误差 % | 在线完整重建次数 | 精度通过 |','|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    def pct(v):return '未定义' if v is None else f'{100*v:.3f}'
    for row in table:
        lines.append(f'| {row["run"]}/{row["cache_workers"]} | {row["label"]} | {row["updates"]} | {row["online_seconds"]:.3f} | {row["online_speedup"]:.3f}× | {pct(row["maximum_waveform_relative_error"])} | {pct(row["maximum_velocity_gradient_relative_error"])} | {pct(row["maximum_parameter_gradient_relative_error"])} | {row["full_online_refactorizations"]} | {row["all_used_gradients_pass"]} |')
    lines+=['','精度门槛：波形相对L2<1%、速度和原 SIREN 参数梯度相对L2<5%。固定基即使很快也不能在失配后算作同精度加速。所有运行均单种子、五次更新；不是收敛或长期性能结论。','','| 运行 | 方法 | 含初始准备秒 | 后验核验秒 | 末态原 FDTD MSE | 末态速度 RMSE m/s | 末态 SSIM |','|---|---|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f'| {row["run"]} | {row["label"]} | {row["including_preparation_seconds"]:.3f} | {row["audit_seconds"]:.3f} | {row["final_fdtd_mse"]:.8f} | {row["final_rmse_mps"]:.5f} | {row["final_ssim"]:.6f} |')
    lines+=['','详细协议、失败路径及存储见 DESIGN.md 与每次运行的 config/preparation/summary。LU 存储为数据/索引估计，进程 RSS/peak 含共享初始缓存、成对实验及临时空间，不能当作独占部署内存。最终状态用前一步的基核验；在继续下一次更新之前，动态算法会再进行校正/刷新。','']
    (out/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8');print(out/'RESULTS.md')


if __name__=='__main__':main()
