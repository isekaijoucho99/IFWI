"""Build traceable diagnostic figures and compact tables from completed ROM runs.

Figure contract: compare accuracy and measured online cost together; a low cost
is not a valid FWI speedup unless gradient accuracy passes. Quantitative grid,
183 mm width, Python-only, editable PDF/SVG, 600-dpi preview. Timing intervals
are the range of three technical repeats, not statistical confidence intervals.
All completed ranks and states remain in Source Data; the plots show background
and the declared held-out 2% perturbation, whose gradients are nonzero.
"""
from pathlib import Path
import argparse
import csv
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--runs',nargs='+',required=True)
    ap.add_argument('--output',required=True)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
        'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,
        'legend.fontsize':7,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'svg.fonttype':'none'})
    runs=[];rows=[]
    for index,path in enumerate(args.runs):
        root=Path(path);result=json.loads((root/'summary.json').read_text(encoding='utf-8'))
        config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        label=('Primal-preserving + adjoint' if config['rom'].get('protected_primal_rank',0)
            else 'Primal + adjoint' if result.get('data_conditioned_adjoint_calibration') else 'Primal')
        name=f'run_{index+1}_{config["rom"].get("basis_mode","primal")}'
        target=out/name;target.mkdir(exist_ok=True)
        for file in ['summary.json','config.json','provenance.json','comparison.csv']:
            shutil.copy2(root/file,target/file)
        for file in root.glob('siren_gradient_*.json'):shutil.copy2(file,target/file.name)
        for file in root.glob('offline_*.json'):shutil.copy2(file,target/file.name)
        for file in root.glob('timing_*.json'):shutil.copy2(file,target/file.name)
        for row in result['comparisons']:
            rows.append(dict(method=label,**row))
        runs.append((root,result,label))
    if not rows:raise ValueError('No completed comparisons')
    with (out/'source_data.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(1,3,figsize=(7.204724,3.622047),layout='constrained')
    palette=['#44779D','#B46A40','#7C6BA4','#4A8E79']
    for ri,(root,result,label) in enumerate(runs):
        for state,marker,style in [('background','o','-'),('perturbed','s','--')]:
            selected=sorted([r for r in result['comparisons'] if r['state']==state],key=lambda r:r['rank'])
            if not selected:continue
            rank=[r['rank'] for r in selected]
            legend=label+(' (background)' if state=='background' else ' (+2% held out)')
            for ax,key in zip(axes[:2],['waveform_relative_error','velocity_gradient_relative_error']):
                values=np.array([r[key] for r in selected])*100
                if np.any(values<=0) or not np.isfinite(values).all():raise ValueError('Log error values must be positive')
                ax.plot(rank,values,style,marker=marker,ms=3,color=palette[ri],lw=1,label=legend)
            seconds=np.array([r['rom_forward_backward_seconds'] for r in selected])
            ranges=[]
            for row in selected:
                timing=json.loads((root/f'timing_{row["rank"]}_{state}.json').read_text())
                samples=[v['seconds'] for v in timing['forward_backward']['rom']['samples']]
                ranges.append([np.median(samples)-min(samples),max(samples)-np.median(samples)])
            axes[2].errorbar(rank,seconds,yerr=np.array(ranges).T,fmt=marker+style,ms=3,
                color=palette[ri],lw=1,capsize=2,label=legend)
    axes[0].set_ylabel('Waveform relative error (%)');axes[0].set_yscale('log')
    axes[1].set_ylabel('Velocity-gradient relative error (%)');axes[1].set_yscale('log')
    axes[0].axhline(1,color='#777777',ls=':',lw=.8)
    axes[1].axhline(5,color='#777777',ls=':',lw=.8)
    axes[2].set_ylabel('Online forward + backward (s)');axes[2].set_yscale('log')
    fdtd=np.median([r['fdtd_forward_backward_seconds'] for r in rows if r['state']!='truth'])
    axes[2].axhline(fdtd,color='#555555',lw=.8,label='Original FDTD (median)')
    for i,ax in enumerate(axes):
        ax.set_xlabel('Reduced pressure dimension');ax.text(-.13,1.06,chr(97+i),transform=ax.transAxes,weight='bold',fontsize=9)
    handles,labels=axes[2].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,fontsize=7,frameon=False)
    fig.savefig(out/'accuracy_cost.png',dpi=600)
    fig.savefig(out/'accuracy_cost.pdf')
    fig.savefig(out/'accuracy_cost.svg')
    plt.close(fig)
    # A transparent representative selection: largest completed joint rank,
    # or the largest primal rank if the joint run has no completed comparisons.
    complete=[run for run in runs if run[1]['comparisons']]
    root,result,label=complete[-1];rank=max(r['rank'] for r in result['comparisons'])
    fd=np.load(root/'fdtd_background_snapshots.npz');rr=np.load(root/f'rom_{rank}_background_snapshots.npz')
    fields_a,fields_b=fd['fields'][:,0],rr['fields'][:,0]
    fig,axes=plt.subplots(3,len(fields_a),figsize=(7.204724,4.409449),layout='constrained')
    max_abs=float(max(abs(fields_a).max(),abs(fields_b).max()))
    err_abs=float(abs(fields_b-fields_a).max())
    if max_abs<=0 or err_abs<=0:raise ValueError('Expected nonzero wavefields and errors')
    extent=[0,(fields_a.shape[-1]-1)*15/1000,(fields_a.shape[-2]-1)*15/1000,0]
    for j,(a,b,frame) in enumerate(zip(fields_a,fields_b,fd['indices'])):
        for i,array in enumerate([a,b,b-a]):
            scale=max_abs if i<2 else err_abs
            im=axes[i,j].imshow(array,extent=extent,cmap='RdBu_r',vmin=-scale,vmax=scale,aspect='auto')
            if i==0:axes[i,j].set_title(f'{(int(frame)+1)*.0019:.3f} s',fontsize=8)
            if j==0:axes[i,j].set_ylabel(['FDTD\nDepth (km)',f'ROM (r={rank})\nDepth (km)','ROM − FDTD\nDepth (km)'][i])
            else:axes[i,j].set_yticklabels([])
            if i==2:axes[i,j].set_xlabel('Distance (km)')
            else:axes[i,j].set_xticklabels([])
        axes[0,j].text(0,1.12,chr(97+j),transform=axes[0,j].transAxes,weight='bold',fontsize=9)
    fig.colorbar(axes[0,0].images[0],ax=axes[:2].ravel().tolist(),label='Pressure (original solver units)',shrink=.75)
    fig.colorbar(axes[2,0].images[0],ax=axes[2].tolist(),label='Pressure difference',shrink=.75)
    fig.savefig(out/'wavefields.png',dpi=600)
    fig.savefig(out/'wavefields.pdf')
    fig.savefig(out/'wavefields.svg')
    plt.close(fig)
    np.savez_compressed(out/'wavefield_source_data.npz',fdtd=fields_a,rom=fields_b,
        frame_indices=fd['indices'],rank=rank,dt=.0019,dx=15.)
    (out/'FIGURE_QA.md').write_text(
        '# Figure contract and QA\n\n'
        'Claim: waveform accuracy, inversion-gradient accuracy, and online cost must be assessed jointly.\n\n'
        'Accuracy-cost: quantitative grid; every completed rank is retained; background and held-out 2% '
        'perturbation are shown. Truth-state gradients are zero, so relative gradient errors are undefined; '
        'truth rows remain in source_data.csv. No random seeds are averaged. Timing marks are medians, '
        'with min–max ranges of three warmed, alternating-order technical repeats. No significance test. '
        '1% waveform and 5% gradient guides are declared engineering thresholds. S1, nt=1000 only; '
        'this is not the 49-shot baseline training protocol.\n\n'
        'Wavefield plate: largest completed joint rank (or primal fallback), background model, '
        'four equally spaced recorded frames. Whole physical grid; PML excluded for both. '
        'Common symmetric linear pressure scale for both methods and all times; separate symmetric '
        'difference scale. No smoothing, normalization, gamma, stitching, or local editing. '
        'Raw arrays and exact frame indices are saved.\n\n'
        'Python matplotlib; 183 mm width; editable PDF/SVG and 600-dpi PNG. Automated source and '
        'PDF glyph audits plus panel inspection are recorded alongside the report.\n',encoding='utf-8')
    print(out)


if __name__=='__main__':main()
