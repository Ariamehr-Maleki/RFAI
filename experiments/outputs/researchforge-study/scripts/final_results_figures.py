"""Publication figures from verified, frozen held-out reports; no model selection."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'work/plot-runtime'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'work/plot-config'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from final_evaluation_gate import verify_lock
from final_results_tables import checked_report,STATUSES
from frozen_ranker import digest
from supervisor_state import atomic_state

LABELS={'full_lambdamart':'Full LambdaMART','full_binary_lightgbm':'Full classifier',
    'no_content_lambdamart':'No-content LambdaMART','graph__lambdamart':'Graph LambdaMART',
    'graph__binary_lightgbm':'Graph classifier','lambdamart':'LambdaMART','binary_lightgbm':'Classifier',
    'semantic_standalone':'Semantic retrieval','fixed_selected':'Selected fixed scorer',
    'frequency45':'Frequency (45 papers)','rrf':'Reciprocal rank fusion','adamic_adar':'Adamic–Adar',
    'ncn':'Neural Common Neighbor','max_paper_bm25':'Maximum paper BM25'}

def label(name):
    if name.startswith('factorial_'):return name[10:].replace('explicit_time','time').replace('+',' + ')
    return LABELS.get(name,name.replace('_',' '))

def contrast_label(name):
    a,b=name.split('__minus__');return label(a)+' − '+label(b)

def save(fig,folder,name):
    fig.savefig(folder/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(folder/(name+'.png'),bbox_inches='tight',dpi=300)
    plt.close(fig)

def forest(ax,rows,names,title,xlabel):
    for i,(row,name) in enumerate(zip(rows,names)):
        d=row['delta'];lo,hi=row['ci95']
        if not np.all(np.isfinite([d,lo,hi])) or hi<lo:raise ValueError('Invalid contrast interval')
        # Draw interval endpoints directly; percentile CIs need not bracket point estimates.
        ax.plot([lo,hi],[i,i],color='#315c7c',lw=1.8)
        ax.scatter([d],[i],color='#153b56',s=24,zorder=3)
    ax.axvline(0,color='#666666',lw=.8,ls='--');ax.set_yticks(range(len(names)),names)
    ax.invert_yaxis();ax.set_title(title,loc='left',fontweight='bold');ax.set_xlabel(xlabel)
    ax.grid(axis='x',alpha=.2);ax.spines[['right','top']].set_visible(False)

def draw(reports,subgroups,cost,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
        'axes.titlesize':11,'axes.labelsize':9,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(3,1,figsize=(10,11),layout='constrained',height_ratios=[8,3,2])
    for ax,(dataset,metric) in zip(axes,[('prescience','nDCG@1000'),('ogbl-collab','Hits@50'),('exHarmony','Authors nDCG@10')]):
        contrasts=reports[dataset]['primary_comparisons']
        forest(ax,list(contrasts.values()),[contrast_label(n) for n in contrasts],dataset,'Difference in '+metric)
    save(fig,folder,'figure1_primary_contrasts')
    cells={n:v['ndcg1000'] for n,v in reports['prescience']['seed_summary'].items() if n.startswith('factorial_')}
    if len(cells)!=8:raise ValueError('All eight frozen factorial cells required')
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    values=list(cells.values());ys=np.arange(len(values))
    ax.errorbar([v['mean'] for v in values],ys,xerr=[v['sample_sd'] for v in values],fmt='o',color='#315c7c',capsize=3)
    ax.set_yticks(ys,[label(n) for n in cells]);ax.invert_yaxis();ax.set_xlabel('nDCG@1000 (mean ± model-seed SD)')
    ax.set_title('Fixed candidate pool: eight primitive-feature combinations',loc='left')
    ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    save(fig,folder,'figure2_factorial')
    primary='full_lambdamart__minus__full_binary_lightgbm'
    group=subgroups[primary]['groups'];fig,axes=plt.subplots(1,3,figsize=(11,3.7),layout='constrained')
    for ax,metric,title in zip(axes,['ndcg1000','new_recall10','repeat_recall10'],['nDCG@1000','New Recall@10','Repeat Recall@10']):
        rows=[];names=[]
        for g in ['all','1-3','4-10','11+']:
            row=group[metric+':'+g]
            if row['delta'] is None or row['ci95'] is None:continue
            rows.append(row);names.append(g+f" (n={row['queries']})"+(' *' if row['sparse_group'] else ''))
        forest(ax,rows,names,title,'LambdaMART − classifier')
    save(fig,folder,'figure3_secondary_subgroups')
    if cost['status']!='Completed isolated development pipeline cost; not final quality evaluation':raise ValueError('Unrecognized cost status')
    fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
    for mode,color in [('full','#315c7c'),('no_content','#a15f24')]:
        value=cost['results'][mode]
        for name,summary in value['end_to_end'].items():
            y=len(axes[0].lines)
            axes[0].plot([summary['median_seconds'],summary['p95_seconds']],[y,y],color=color,lw=2)
            axes[0].scatter(summary['median_seconds'],y,color=color,s=24)
        latencies=[v['median_seconds'] for v in value['end_to_end'].values()]
        axes[1].scatter(latencies[0],reports['prescience']['seed_summary'][mode+'_lambdamart']['ndcg1000']['mean'],label=mode,color=color,s=45)
    names=[mode+' / '+label(n) for mode in ['full','no_content'] for n in cost['results'][mode]['end_to_end']]
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis();axes[0].set_xlabel('Seconds/query: median point to p95 endpoint')
    axes[0].set_title('Warm development-query pipeline latency',loc='left')
    axes[1].set_xlabel('Development median seconds/query');axes[1].set_ylabel('Held-out nDCG@1000');axes[1].legend()
    axes[1].set_title('LambdaMART quality and measured cost',loc='left')
    for ax in axes:ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    save(fig,folder,'figure4_pipeline_cost')

CAPTIONS='''# Figure captions

1. Predeclared primary paired differences on three held-out tasks. Points average query/edge effects over model seeds before cluster inference; bars are marginal 95% cluster-bootstrap intervals. Dashed lines denote zero. Distinct axes reflect distinct metrics. Separate task-specific Holm-adjusted tests are in the contrast table; intervals are not multiplicity adjusted.
2. All eight primitive-feature combinations on the same full candidate pool. Bars show sample standard deviation across three fitted model seeds, not confidence intervals. The full practical model and end-to-end no-content retriever are separate experiments. The label “time” means explicit temporal columns; historical eligibility and windows remain in all cells.
3. Secondary history-stratum analysis for full LambdaMART minus its matched classifier. All means average model seeds before author-cluster resampling; bars are marginal 95% intervals. n is eligible queries. An asterisk marks fewer than20 author clusters. New/repeat Recall@10 uses one unchanged full ranking; empty relevance subsets are omitted only for that metric. These observational subgroups do not establish causal mechanisms. Interaction tests are supplied separately.
4. Pipeline cost measured on30 fixed development queries with three warm repeats, seed42 models,1000 candidates and one numerical thread. Left: median to95th percentile of joint retrieval, features and model ranking latency. These quantiles describe timed calls, not confidence intervals. Right: held-out LambdaMART quality paired with the corresponding development timing estimate; quality and latency use different query sets. No encoding/training cost is included in these points. Desktop contention and sampled-memory limitations apply.
'''

def export(root,lock_path,cost_path,output):
    root=Path(root).resolve();output=Path(output).resolve();cost_path=Path(cost_path).resolve()
    if not output.is_relative_to(root) or output.exists():raise ValueError('New workspace figure directory required')
    lock=verify_lock(root,lock_path);reports={};sources={}
    for dataset in STATUSES:
        path,report=checked_report(root,lock,dataset);reports[dataset]=report;sources[str(path.relative_to(root))]=digest(path)
    p=root/reports['prescience']['config']['output_directory']/'subgroups.json'
    if 'subgroups.json' not in reports['prescience']['output_sha256']:raise ValueError('Unbound subgroup report')
    if not cost_path.is_relative_to(root):raise ValueError('Cost outside workspace')
    relative=cost_path.relative_to(root).as_posix()
    if relative not in lock['payload']['artifacts']:raise ValueError('Cost evidence was not frozen')
    sources[relative]=digest(cost_path)
    draw(reports,json.loads(p.read_text()),json.loads(cost_path.read_text()),output)
    (output/'CAPTIONS.md').write_text(CAPTIONS,encoding='utf-8')
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(ROOT/'work/plot-runtime')])}
    atomic_state(output/'provenance.json',dict(status='Verified frozen-result figures',lock_sha256=lock['lock_sha256'],
        source_sha256=sources,script_sha256=digest(__file__),plot_packages=packages,
        files={p.name:digest(p) for p in output.iterdir() if p.is_file()}))
    return output

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--cost',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();print(export(ROOT,args.lock,args.cost,args.output))
