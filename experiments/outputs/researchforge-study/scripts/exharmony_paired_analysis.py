"""Exploratory paired author/title-component intervals for saved validation runs.

Reads only validation predictions and query metadata; no held-out qrel parsing.
Components use all frozen validation queries, including unsampled bridges.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np
from exharmony_index import DATA,AUDIT
from exharmony_prepare import norm


def validation_components():
    split=json.loads((AUDIT/'exharmony_split.json').read_text())
    allowed=set(split['validation']);parent={q:q for q in allowed}
    def root(q):
        while q!=parent[q]:parent[q]=parent[parent[q]];q=parent[q]
        return q
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    first={}
    with (DATA/'papers_test_judgeable.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);qid=r['id']
            if qid not in allowed:continue
            for author in r.get('authorships',[]):
                aid=author.get('author',{}).get('id')
                if not aid:continue
                if aid in first:union(qid,first[aid])
                else:first[aid]=qid
    first={}
    with (DATA/'papers_test.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);qid=r['id']
            if qid not in allowed:continue
            title=norm(r.get('title'))
            if not title:continue
            if title in first:union(qid,first[title])
            else:first[title]=qid
    return {q:root(q) for q in allowed}


def paired(a,b,clusters):
    other={r['qid']:r for r in b}
    assert {r['qid'] for r in a}==set(other)
    groups=collections.defaultdict(list)
    for row in a:
        ref=other[row['qid']]
        assert row['truth']==ref['truth']
        groups[clusters[row['qid']]].append(row['ndcg10']-ref['ndcg10'])
    values=list(groups.values());sums=np.array([sum(v) for v in values]);counts=np.array([len(v) for v in values])
    draws=np.random.default_rng(42).integers(0,len(values),size=(5000,len(values)))
    diffs=sums[draws].sum(1)/counts[draws].sum(1)
    return dict(delta=float(sums.sum()/counts.sum()),ci95=np.quantile(diffs,[.025,.975]).tolist(),
                queries=int(counts.sum()),author_title_components=len(values),bootstrap_draws=5000)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);run=ap.parse_args().run
    report=json.loads((run/'report.json').read_text())
    assert report['status'].startswith('Development')
    clusters=validation_components();comparisons={}
    for judgment in ['authors','cite','simcite']:
        a=json.loads((run/f'{judgment}.lambdamart.predictions.json').read_text())
        comparisons[judgment]={}
        for name in ['max_bm25','binary_lightgbm']:
            b=json.loads((run/f'{judgment}.{name}.predictions.json').read_text())
            comparisons[judgment][name]=paired(a,b,clusters)
    result=dict(status='Exploratory validation-selected component-bootstrap intervals; not adjusted for multiple comparisons',
                comparisons=comparisons,limitations=['No confirmatory significance or best-seed claim.',
                'Cite/SimCite relevant authors can overlap across query-author components; this dependence is not fully captured.',
                'One sampled seed at a time; do not pool overlapping seeds as independent queries.'])
    (run/'paired_exploratory.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
