"""Frozen secondary subgroup analyses; average model seeds before resampling."""
import itertools
import numpy as np
from subgroup_analysis import measures,METRICS,GROUPS
from publication_ablation_design import history_group


def compare(primary,baseline,repetitions=10000,seed=314159):
    if set(primary)!=set(baseline) or len(primary)<3:raise ValueError('At least three matching model seeds required')
    if repetitions<100:raise ValueError('At least100 bootstrap draws required')
    reference=None;arrays=[];ids=None
    for model_seed in sorted(primary):
        a={str(r['qid']):r for r in primary[model_seed]};b={str(r['qid']):r for r in baseline[model_seed]}
        if len(a)!=len(primary[model_seed]) or len(b)!=len(baseline[model_seed]) or set(a)!=set(b):raise ValueError('Duplicate or mismatched query IDs')
        if ids is not None and ids!=sorted(a):raise ValueError('Query sets differ across model seeds')
        ids=sorted(a);info=[];values=[]
        for qid in ids:
            row,ref=a[qid],b[qid]
            def identity(r):return (str(r['seed']),int(r['seed_history_count']),frozenset(r['truth']),frozenset(r['new_truth']))
            if identity(row)!=identity(ref):raise ValueError('Paired subgroup metadata differs')
            info.append(identity(row));x,y=measures(row),measures(ref);difference=[]
            for metric in METRICS:
                if (x[metric] is None)!=(y[metric] is None):raise ValueError('Metric eligibility differs')
                difference.append(np.nan if x[metric] is None else x[metric]-y[metric])
            values.append(difference)
        if reference is not None and info!=reference:raise ValueError('Subgroup metadata differs across model seeds')
        reference=info;arrays.append(values)
    matrix=np.asarray(arrays,dtype=float)
    if not np.all(np.isfinite(matrix)==np.isfinite(matrix[0])):raise ValueError('Eligibility changes across model seeds')
    # NaN denotes an empty new/repeat relevance set; never an extra observation.
    averaged=matrix.mean(0);authors=sorted({r[0] for r in reference});author_index={a:i for i,a in enumerate(authors)}
    keys=list(itertools.product(METRICS,GROUPS));sums=np.zeros((len(authors),len(keys)));counts=np.zeros_like(sums,dtype=int)
    for q,meta in enumerate(reference):
        author,history,_,_=meta;group=history_group(history)
        for j,(metric,g) in enumerate(keys):
            value=averaged[q,METRICS.index(metric)]
            if g in ['all',group] and np.isfinite(value):sums[author_index[author],j]+=value;counts[author_index[author],j]+=1
    totals=counts.sum(0);deltas=np.divide(sums.sum(0),totals,out=np.full(len(keys),np.nan),where=totals>0)
    rng=np.random.default_rng(seed);boot=np.full((repetitions,len(keys)),np.nan)
    for start in range(0,repetitions,64):
        sample=rng.integers(0,len(authors),size=(min(64,repetitions-start),len(authors)))
        a=sums[sample].sum(1);b=counts[sample].sum(1)
        boot[start:start+len(sample)]=np.divide(a,b,out=np.full(a.shape,np.nan),where=b>0)
    def interval(values):
        good=values[np.isfinite(values)]
        return dict(ci95=np.quantile(good,[.025,.975]).tolist() if len(good) else None,valid_bootstrap_draws=len(good))
    groups={}
    for j,(metric,group) in enumerate(keys):
        clusters=int((counts[:,j]>0).sum())
        groups[f'{metric}:{group}']=dict(delta=float(deltas[j]) if totals[j] else None,queries=int(totals[j]),author_clusters=clusters,
            sparse_group=clusters<20,**interval(boot[:,j]))
    interactions={}
    for metric in METRICS:
        for a,b in itertools.combinations(GROUPS[1:],2):
            i,j=keys.index((metric,a)),keys.index((metric,b))
            interactions[f'{metric}:{a}_minus_{b}']=dict(difference_of_deltas=float(deltas[i]-deltas[j]) if totals[i] and totals[j] else None,**interval(boot[:,i]-boot[:,j]))
    return dict(status='Frozen secondary subgroup analysis; marginal intervals, no multiplicity-adjusted claims',
        query_count=len(ids),model_seed_count=len(primary),bootstrap=dict(repetitions=repetitions,seed=seed,unit='seed author; joint draws across groups, after model-seed averaging'),
        groups=groups,interactions=interactions,limitations=['Subgroups observational, not causal.',
            'Sparse groups flagged below20 author clusters; no reliable subgroup superiority claim.',
            'Crossed network dependence not fully modeled; intervals marginal and secondary.',
            'Empty new/repeat truth excluded for that metric only; all use the same full ranking.',
            'A difference between significant and nonsignificant groups is not a significant interaction.'])
