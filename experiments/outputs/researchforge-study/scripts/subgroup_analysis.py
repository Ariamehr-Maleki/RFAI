"""Paired DEVELOPMENT subgroup analysis using saved rankings and prior history.

Historical subgroup cutpoints are fixed. First/repeat labels evaluate the same
full ranking. Joint author-cluster resampling supports differences of effects.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from publication_ablation_design import history_group
from prescience_pilot import rank_metrics

METRICS=['ndcg1000','new_recall10','repeat_recall10']
GROUPS=['all','1-3','4-10','11+']

def measures(row):
    truth=set(row['truth']);new=set(row['new_truth']);pred=row['predicted']
    if not new<=truth or len(pred)!=len(set(pred)):raise ValueError('Invalid truth or duplicate ranking')
    repeated=truth-new
    result=rank_metrics(pred,truth,new)
    result['repeat_recall10']=len(set(pred[:10])&repeated)/len(repeated) if repeated else None
    return {m:result[m] for m in METRICS}

def compare(primary,baseline,metadata,repeats=5000,seed=20261026):
    def index(rows):
        result={str(r['qid']):r for r in rows}
        if len(result)!=len(rows):raise ValueError('Duplicate query ID')
        return result
    a,b,meta=index(primary),index(baseline),index(metadata)
    if set(a)!=set(b) or not set(a)<=set(meta):raise ValueError('Query sets differ')
    keys=[(m,g) for m in METRICS for g in GROUPS];authors=sorted({str(r['seed']) for r in a.values()})
    author_index={v:i for i,v in enumerate(authors)}
    sums=np.zeros((len(authors),len(keys)));counts=np.zeros_like(sums,dtype=np.int64)
    for qid,row in a.items():
        ref=b[qid];info=meta[qid]
        if row['seed']!=ref['seed'] or row['seed']!=info['seed']:raise ValueError('Seed author mismatch')
        if set(row['truth'])!=set(ref['truth']) or set(row['new_truth'])!=set(ref['new_truth']):raise ValueError('Relevant sets differ')
        if set(row['truth'])!=set(info['truth']) or set(row['new_truth'])!=set(info['new_truth']):raise ValueError('Metadata labels differ')
        ga=history_group(info['seed_history_count']);ma,mb=measures(row),measures(ref);i=author_index[str(row['seed'])]
        for j,(metric,group) in enumerate(keys):
            if group in ['all',ga] and ma[metric] is not None:
                if mb[metric] is None:raise ValueError('Inconsistent metric denominator')
                sums[i,j]+=ma[metric]-mb[metric];counts[i,j]+=1
    totals=counts.sum(0);deltas=np.divide(sums.sum(0),totals,out=np.full(len(keys),np.nan),where=totals>0)
    rng=np.random.default_rng(seed);boot=np.full((repeats,len(keys)),np.nan)
    # Chunked draws avoid allocating repeats x all authors for large samples.
    for start in range(0,repeats,64):
        sample=rng.integers(0,len(authors),size=(min(64,repeats-start),len(authors)))
        sc=sums[sample].sum(1);ct=counts[sample].sum(1)
        boot[start:start+len(sample)]=np.divide(sc,ct,out=np.full(sc.shape,np.nan),where=ct>0)
    def interval(values):
        good=values[np.isfinite(values)]
        return dict(ci95=np.quantile(good,[.025,.975]).tolist() if len(good) else None,valid_bootstrap_draws=len(good))
    groups={}
    for j,(metric,group) in enumerate(keys):
        groups[f'{metric}:{group}']=dict(delta=float(deltas[j]) if totals[j] else None,queries=int(totals[j]),
            author_clusters=int((counts[:,j]>0).sum()),**interval(boot[:,j]))
    interactions={}
    for metric in METRICS:
        for g1,g2 in itertools.combinations(GROUPS[1:],2):
            i,j=keys.index((metric,g1)),keys.index((metric,g2))
            interactions[f'{metric}:{g1}_minus_{g2}']=dict(
                difference_of_deltas=float(deltas[i]-deltas[j]) if totals[i] and totals[j] else None,
                **interval(boot[:,i]-boot[:,j]))
    return dict(status='Exploratory DEVELOPMENT paired subgroup differences; no multiplicity adjustment',
        bootstrap=dict(seed=seed,draws=repeats,unit='seed author; joint resampling across all groups'),
        groups=groups,interactions=interactions,
        limitations=['Validation-selected methods; intervals are exploratory.',
            'Repeated-author dependence is clustered; crossed network dependence is not fully captured.',
            'Empty first-time/repeat relevant sets excluded only from that metric; counts reported.',
            'History categories are observational, not causal effects.',
            'Sparse groups may have undefined resamples; valid draw and cluster counts must accompany interpretation.'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path)
    ap.add_argument('--metadata-run',type=Path);ap.add_argument('--method',default='lambdamart')
    ap.add_argument('--baseline',default='binary_lightgbm');args=ap.parse_args()
    report=json.loads((args.run/'report.json').read_text())
    if not str(report.get('status','')).lower().startswith('development'):raise ValueError('Development predictions required')
    source=args.metadata_run or args.run
    metadata=[json.loads(p.read_text()) for p in (source/'valid_queries').glob('*.json')]
    rows=lambda method:json.loads((args.run/(method+'.predictions.json')).read_text())
    result=compare(rows(args.method),rows(args.baseline),metadata)
    result.update(method=args.method,baseline=args.baseline)
    path=args.run/f'subgroups-{args.method}-vs-{args.baseline}.json'
    path.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
