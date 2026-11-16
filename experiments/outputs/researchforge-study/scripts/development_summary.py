"""Summarize saved predictions; exploratory author-cluster intervals, no new tuning."""
import collections
import argparse
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]
RUN=OUT/'results/text-graph-200-100-1000-s42'

def paired(a,b,metric,seed=42):
    by_id={r['qid']:r for r in b};clusters=collections.defaultdict(list)
    for row in a:
        other=by_id[row['qid']]
        assert row['seed']==other['seed']
        if row[metric] is not None and other[metric] is not None:
            clusters[row['seed']].append(row[metric]-other[metric])
    values=list(clusters.values());sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    rng=np.random.default_rng(seed);sample=rng.integers(0,len(values),size=(5000,len(values)))
    diffs=sums[sample].sum(axis=1)/counts[sample].sum(axis=1)
    return dict(delta=float(sums.sum()/counts.sum()),ci95=np.quantile(diffs,[.025,.975]).tolist(),
        clusters=len(values),queries=int(counts.sum()),bootstrap_repetitions=5000)

def main():
    global RUN
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=RUN)
    RUN=parser.parse_args().run
    report=json.loads((RUN/'report.json').read_text())
    with (RUN/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method']+list(next(iter(report['metrics'].values())).keys()))
        writer.writeheader()
        for method,row in report['metrics'].items():writer.writerow(dict(method=method,**row))
    primary=json.loads((RUN/'lambdamart.predictions.json').read_text())
    comparisons={}
    for method in ['frequency10','autoresearch_description','rrf','binary_lightgbm','lambdamart_without_text','lambdamart_without_recency']:
        other=json.loads((RUN/f'{method}.predictions.json').read_text())
        comparisons[method]={m:paired(primary,other,m) for m in ['ndcg10','ndcg1000','rprecision','new_recall10']}
    result=dict(status='Exploratory development intervals only; validation was used for method/configuration selection',
        warning='Not adjusted for adaptive selection or multiple comparisons; no confirmatory significance claim',comparisons=comparisons)
    (RUN/'exploratory_intervals.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
