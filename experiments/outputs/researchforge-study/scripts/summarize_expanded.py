"""Summarize completed expanded development; never read held-out corpus labels."""
import argparse
import json
from pathlib import Path
import numpy as np
from final_statistics import seed_average_pairs,cluster_inference,holm
from fit_factorial import read,write,sha

SEEDS=[42,43,44]

def summarize(base):
    base=Path(base);summary={};sources={};queries=None;full_predictions={};no_content_predictions={}
    for mode in ['full','no_content']:
        reports={}
        for seed in SEEDS:
            folder=base/mode/f'seed-{seed}';r=read(folder/'report.json')
            if not r['status'].startswith('Development') or read(folder/'independent_metric_verification.json')['status']!='PASS':
                raise ValueError('Incomplete or unverified development results')
            sources[str((folder/'report.json').relative_to(base))]=sha(folder/'report.json');reports[seed]=r
            rows=read(folder/'lambdamart.predictions.json');ids=sorted(str(x['qid']) for x in rows)
            if queries is not None and ids!=queries:raise ValueError('Validation queries differ across modes/seeds')
            queries=ids
            (full_predictions if mode=='full' else no_content_predictions)[seed]=rows
        methods=set(reports[SEEDS[0]]['metrics'])
        if any(set(r['metrics'])!=methods for r in reports.values()):raise ValueError('Method inventories differ')
        summary[mode]={}
        for name in sorted(methods):
            summary[mode][name]={}
            for metric in reports[SEEDS[0]]['metrics'][name]:
                values=[reports[s]['metrics'][name][metric] for s in SEEDS]
                if any(v is None for v in values):
                    if not all(v is None for v in values):raise ValueError('Metric eligibility changes across seeds')
                    record=dict(mean=None,sample_sd=None,per_seed=values)
                else:record=dict(mean=float(np.mean(values)),sample_sd=float(np.std(values,ddof=1)),per_seed=values)
                summary[mode][name][metric]=record
    comparisons={}
    def compare(name,a,b):
        paired=seed_average_pairs(a,b,'ndcg1000')
        comparisons[name]=dict(**cluster_inference(paired['differences'],paired['authors'],repetitions=5000),
            per_seed_deltas=paired['per_seed_deltas'])
    compare('full_pipeline_minus_no_content_pipeline',full_predictions,no_content_predictions)
    classifiers={s:read(base/'full'/f'seed-{s}'/'binary_lightgbm.predictions.json') for s in SEEDS}
    compare('full_ranker_minus_matched_classifier',full_predictions,classifiers)
    for omitted,remaining in [('text','network+explicit_time'),('network','text+explicit_time'),('explicit_time','text+network')]:
        a={};b={}
        for seed in SEEDS:
            folder=base/'full'/f'seed-{seed}'/f'factorial-s{seed}'
            if read(folder/'independent_metric_verification.json')['status']!='PASS':raise ValueError('Unverified factorial')
            a[seed]=read(folder/'text+network+explicit_time.predictions.json');b[seed]=read(folder/(remaining+'.predictions.json'))
        compare('factorial_full_minus_without_'+omitted,a,b)
    adjusted=holm({n:v['p_two_sided_cluster_signflip'] for n,v in comparisons.items()})
    for n,v in comparisons.items():v['exploratory_family_holm_p']=adjusted[n]
    result=dict(status='Completed expanded DEVELOPMENT summary; adaptive validation selection; no final test',
        seeds=SEEDS,validation_queries=len(queries),seed_summary=summary,comparisons=comparisons,source_reports_sha256=sources,
        limitations=['Sample SD is not a confidence interval; shared query metrics averaged across model seeds.',
            'The five exploratory contrasts are not the eight frozen final primary comparisons.',
            'Holm does not correct adaptive validation model selection; no confirmatory significance.',
            'Training-group subsampling differs by seed; corpus/padding/validation remain common.',
            'Final fixed-scorer variant selection and final configuration lock remain pending.'])
    write(base/'report.json',result);return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True,type=Path)
    print(json.dumps(summarize(ap.parse_args().base),indent=2))
