"""Development-only precision worksheet for prospective final sampling review.

Does not choose a sample size, change a protocol, create a lock or read test data.
Normal-approximation projections assume unchanged cluster composition/variance;
they are not final confidence intervals or a power calculation.
"""
import json
import math
from pathlib import Path
import numpy as np
from final_statistics import seed_average_pairs
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'outputs/researchforge-study'
BASE=STUDY/'results/prescience-expanded-1200-1000-500-1000/full'


def planning_se(values,authors):
    values=np.asarray(values,dtype=float)
    if len(values)!=len(authors) or len(values)<2 or not np.isfinite(values).all():raise ValueError('Invalid planning observations')
    _,inverse=np.unique(np.asarray(authors),return_inverse=True)
    counts=np.bincount(inverse);sums=np.bincount(inverse,weights=values);g=len(counts)
    if g<2:raise ValueError('Need multiple author clusters')
    residual=sums-counts*values.mean()
    return float(np.sqrt(g/(g-1)*np.sum(residual**2)/len(values)**2)),g


def main():
    sources={}
    def load(relative,fixed_seed=None):
        result={}
        for seed in [42,43,44]:
            folder=BASE/f'seed-{seed if fixed_seed is None else fixed_seed}';path=folder/relative.format(seed=seed)
            verification=path.parent/'independent_metric_verification.json'
            if json.loads(verification.read_text())['status']!='PASS':raise ValueError('Unverified development output')
            rows=json.loads(path.read_text())
            result[seed]=[{k:r[k] for k in ['qid','seed','truth','new_truth','ndcg1000']} for r in rows]
            sources[path.relative_to(ROOT).as_posix()]=digest(path)
        return result
    # With one observation per author, the formula must equal ordinary mean SE.
    probe=np.array([-.2,.1,.4,.3]);se,g=planning_se(probe,['a','b','c','d'])
    if not math.isclose(se,float(probe.std(ddof=1)/2),rel_tol=1e-12):raise AssertionError('Planning SE sanity check failed')
    primary=load('lambdamart.predictions.json');classifier=load('binary_lightgbm.predictions.json')
    factorial=load('factorial-s{seed}/text+network+explicit_time.predictions.json')
    contrasts={'practical_ranker_minus_classifier':(primary,classifier)}
    for baseline in ['frequency45','semantic_standalone','rrf']:
        contrasts['practical_ranker_minus_'+baseline]=(primary,load(baseline+'.predictions.json'))
    sensitivity_path=BASE/'seed-42/fixed-scorer-sensitivity/report.json'
    sensitivity=json.loads(sensitivity_path.read_text())
    selected=max(sensitivity['metrics'],key=lambda name:sensitivity['metrics'][name]['shared']['ndcg1000'])
    sources[sensitivity_path.relative_to(ROOT).as_posix()]=digest(sensitivity_path)
    contrasts['practical_ranker_minus_selected_fixed']=(primary,load('fixed-scorer-sensitivity/'+selected+'.shared.predictions.json',fixed_seed=42))
    for omitted,remainder in [('text','network+explicit_time'),('network','text+explicit_time'),('explicit_time','text+network')]:
        contrasts['factorial_full_minus_without_'+omitted]=(factorial,load('factorial-s{seed}/'+remainder+'.predictions.json'))
    records={}
    for name,(a,b) in contrasts.items():
        pairs=seed_average_pairs(a,b,'ndcg1000');n=pairs['eligible_queries']
        se,g=planning_se(pairs['differences'],pairs['authors'])
        records[name]=dict(development_queries=n,development_author_clusters=g,cluster_robust_mean_se=se,
            projected_marginal95_normal_halfwidth_by_query_count={str(count):1.96*se*math.sqrt(n/count) for count in [500,1000,2000]},
            projection_assumption='Same author cluster composition and paired variance as development; no temporal-shift allowance.')
    output=dict(status='Development-only prospective precision worksheet; no final sample size selected',
        comparisons=records,selected_fixed_variant=selected,source_predictions_sha256=sources,script_sha256=digest(Path(__file__)),
        limitations=['Validation selection makes these planning variances uncertain.',
            'Projections are marginal normal approximations, not the final cluster-bootstrap intervals or Holm-adjusted tests.',
            'Not a power calculation, not evidence of final significance, and not a recommendation to tune using final labels.',
            'Final sample decision also requires measured pipeline cost, RAM/disk budget and reviewed global lock.'])
    atomic_state(STUDY/'data-audit/development-precision-planning.json',output)
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
