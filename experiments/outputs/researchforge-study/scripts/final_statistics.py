"""Statistical primitives for frozen evaluations; no data access on import.

Seed-average query metrics before author-cluster inference. Sign-flip tests assume
symmetry/exchangeability of paired cluster effects; crossed dependence remains a
limitation. CI is marginal; Holm adjusts the declared family of p-values.
This module does not authorize reading any test labels.
"""
import numpy as np

def holm(pvalues):
    names=list(pvalues);values=np.asarray([pvalues[n] for n in names],dtype=float)
    if not len(names) or not np.isfinite(values).all() or np.any((values<0)|(values>1)):
        raise ValueError('A nonempty declared family of valid p-values is required')
    order=np.argsort(values,kind='stable');adjusted=np.empty_like(values);previous=0.
    for rank,i in enumerate(order):
        previous=max(previous,min(1.,float(values[i])*(len(names)-rank)));adjusted[i]=previous
    return {n:float(adjusted[i]) for i,n in enumerate(names)}

def seed_average_pairs(primary,baseline,metric):
    """Input maps model-seed IDs to lists of saved query metric rows."""
    if set(primary)!=set(baseline) or len(primary)<3:raise ValueError('At least3 matching model seeds required')
    seeds=sorted(primary);ids=None;authors=None;reference_truth=None;differences=[];seed_deltas={}
    for seed in seeds:
        def keyed(rows):
            result={str(r['qid']):r for r in rows}
            if len(result)!=len(rows):raise ValueError('Duplicate query ID')
            return result
        a,b=keyed(primary[seed]),keyed(baseline[seed])
        if set(a)!=set(b):raise ValueError('Paired query IDs differ')
        current=sorted(a)
        if ids is not None and current!=ids:raise ValueError('Query set changes across model seeds')
        ids=current;values=[];clusters=[];truths=[]
        for qid in ids:
            x,y=a[qid],b[qid]
            if x['seed']!=y['seed']:raise ValueError('Source author mismatch')
            if set(x['truth'])!=set(y['truth']) or set(x.get('new_truth',[]))!=set(y.get('new_truth',[])):
                raise ValueError('Relevance sets differ')
            truths.append((frozenset(x['truth']),frozenset(x.get('new_truth',[]))))
            av,bv=x[metric],y[metric]
            if (av is None)!=(bv is None):raise ValueError('Paired metric eligibility differs')
            if av is not None and not np.isfinite([float(av),float(bv)]).all():raise ValueError('Non-finite metric')
            values.append(np.nan if av is None else float(av)-float(bv));clusters.append(str(x['seed']))
        if authors is not None and authors!=clusters:raise ValueError('Author clusters change across seeds')
        if reference_truth is not None and truths!=reference_truth:raise ValueError('Relevance changes across model seeds')
        reference_truth=truths
        authors=clusters;array=np.asarray(values)
        if np.isinf(array).any():raise ValueError('Infinite metric')
        differences.append(array);seed_deltas[str(seed)]=float(np.nanmean(array)) if np.isfinite(array).any() else None
    matrix=np.stack(differences);eligible=np.isfinite(matrix)
    if not np.all(eligible==eligible[0]):raise ValueError('Metric eligibility changes across seeds')
    keep=eligible[0]
    if not keep.any():raise ValueError('No eligible paired queries')
    return dict(qids=[ids[i] for i in np.flatnonzero(keep)],
                authors=[authors[i] for i in np.flatnonzero(keep)],
                differences=matrix[:,keep].mean(0),per_seed_deltas=seed_deltas,
                total_queries=len(ids),eligible_queries=int(keep.sum()))

def cluster_inference(differences,authors,repetitions=10000,seed=314159):
    differences=np.asarray(differences,dtype=float)
    if not len(differences) or len(differences)!=len(authors) or not np.isfinite(differences).all():
        raise ValueError('Finite aligned paired differences required')
    _,inverse=np.unique(np.asarray(authors,dtype=str),return_inverse=True)
    counts=np.bincount(inverse);sums=np.bincount(inverse,weights=differences)
    if len(counts)<2 or repetitions<100:raise ValueError('Insufficient clusters or resamples')
    rng=np.random.default_rng(seed);boot=np.empty(repetitions);null=np.empty(repetitions)
    for start in range(0,repetitions,64):
        size=min(64,repetitions-start)
        draws=rng.integers(0,len(counts),size=(size,len(counts)))
        boot[start:start+size]=sums[draws].sum(1)/counts[draws].sum(1)
        signs=rng.integers(0,2,size=(size,len(counts)))*2-1
        null[start:start+size]=(signs*sums).sum(1)/counts.sum()
    delta=float(differences.mean())
    p=(1+int(np.sum(np.abs(null)>=abs(delta)-1e-15)))/(repetitions+1)
    return dict(delta=delta,ci95=np.quantile(boot,[.025,.975]).tolist(),
        p_two_sided_cluster_signflip=float(p),queries=len(differences),clusters=len(counts),
        repetitions=repetitions,random_seed=seed,
        assumptions='Paired cluster sign symmetry under null; cluster bootstrap; crossed network dependence not captured')
