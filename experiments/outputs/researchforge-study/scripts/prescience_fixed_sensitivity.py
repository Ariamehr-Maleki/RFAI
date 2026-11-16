"""Six predeclared fixed-scorer interpretations on existing development queries."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from prescience_pilot import DATA,load_data,evaluate,rank_metrics
from prescience_fixed_scorer import FixedScorer,VARIANTS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');args=ap.parse_args();run=Path(args.run)
    assert json.loads((run/'report.json').read_text())['status'].startswith('Development')
    out=run/'fixed-scorer-sensitivity';out.mkdir(exist_ok=True);start=time.perf_counter()
    config=dict(variants=VARIANTS,source_sha256=hashlib.sha256(Path(__file__).with_name('prescience_fixed_scorer.py').read_bytes()).hexdigest(),
                protocol='Development shared-pool and unpadded standalone rankings; no test access, no parameter selection for a superiority claim.')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    papers,histories,dates=load_data()
    for batch in pq.ParquetFile(DATA/'train.parquet').iter_batches(batch_size=2048,columns=['corpus_id','key_references']):
        for p in batch.to_pylist():
            if p['corpus_id'] in papers:papers[p['corpus_id']]['refs']=tuple(r['corpus_id'] for r in p['key_references'] or [] if r['corpus_id'])
    meta=[json.loads(p.read_text()) for p in sorted((run/'valid_queries').glob('*.json'))]
    scores={name:[] for name in VARIANTS};standalone={name:[] for name in VARIANTS}
    for i,q in enumerate(meta):
        scorer=FixedScorer(papers,histories,dates,q['seed'],q['date'])
        for name,options in VARIANTS.items():
            values=scorer.score(**options)
            scores[name].extend(values.get(a,0) for a in q['candidates'])
            pred=sorted(values,key=lambda a:(-values[a],a))[:1000]
            standalone[name].append(dict(qid=q['qid'],seed=q['seed'],predicted=pred,truth=q['truth'],new_truth=q['new_truth'],
                candidate_recall=len(set(pred)&set(q['truth']))/len(q['truth']),**rank_metrics(pred,q['truth'],q['new_truth'])))
        if (i+1)%10==0:print(f'Scored {i+1}/{len(meta)} development queries',flush=True)
    groups=np.array([len(q['candidates']) for q in meta]);results={}
    for name in VARIANTS:
        agg,rows=evaluate(np.array(scores[name]),groups,meta,True)
        (out/f'{name}.shared.predictions.json').write_text(json.dumps(rows))
        (out/f'{name}.standalone.predictions.json').write_text(json.dumps(standalone[name]))
        keys=['ndcg10','ndcg1000','rprecision','recall10','recall50','new_recall10','candidate_recall']
        results[name]=dict(shared=agg,standalone={k:float(np.mean([r[k] for r in standalone[name] if r[k] is not None]))
            if any(r[k] is not None for r in standalone[name]) else None for k in keys})
    # Default must reproduce saved feature scores for every held-out development candidate.
    expected=[]
    for q in meta:
        with np.load(run/'valid_queries'/(q['qid']+'.npz')) as z:expected.extend(z['X'][:,17])
    error=float(np.max(np.abs(np.array(expected)-np.array(scores['default']))))
    np.testing.assert_allclose(scores['default'],expected,rtol=1e-5,atol=1e-7)
    report=dict(status='Description-based sensitivity; not exact official reproduction',config=config,metrics=results,
                default_feature_max_abs_error=error,wall_seconds=time.perf_counter()-start,
                limitations=['No arbitrary completion of standalone rankings with zero-score authors.',
                             'All six alternatives retained; best validation variant is not a confirmatory winner.',
                             'Cleaned per-paper author IDs; official raw-input frequency parity remains a separate check.'])
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
