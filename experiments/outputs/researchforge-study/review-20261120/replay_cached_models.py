"""Independently execute archived LightGBM models against archived feature caches.

This is stronger than recomputing metrics, but does not reconstruct features
from provider data. No source from the original pipeline is imported.
"""
from pathlib import Path
from zipfile import ZipFile
import json, hashlib, io, time
import numpy as np
from lightgbm import Booster

HERE=Path(__file__).resolve().parent
ARCHIVE=HERE.parent/'deliverables/researchforge-reproducibility.zip'

def rankhash(ids):
    return hashlib.sha256(json.dumps(ids,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()

def main():
    started=time.monotonic();results=[]
    with ZipFile(ARCHIVE) as z:
        for dataset,names in [('prescience',['full_lambdamart','full_binary_lightgbm']),('exHarmony',['lambdamart','binary_lightgbm'])]:
            base=f'outputs/researchforge-study/results/final/{dataset}/'
            report=json.loads(z.read(base+'report.json'));cfg=report['config'];models={};expected={}
            for method in names:
                for seed,spec in cfg['methods'][method]['models'].items():
                    payload=z.read(spec['path']);assert hashlib.sha256(payload).hexdigest()==spec['sha256']
                    models[method,seed]=(Booster(model_str=payload.decode('utf-8')), [cfg['feature_names'].index(f) for f in spec['features']])
                    stem=method if dataset=='prescience' else 'authors.'+method
                    records=json.loads(z.read(base+f'{stem}.s{seed}.predictions.json'))
                    expected[method,seed]={r['qid']:(rankhash(r['predicted']),len(r['predicted'])) for r in records}
                    del records
            queries=[n for n in z.namelist() if n.startswith(base+'queries/') and n.endswith('.json') and not n.endswith('.receipt.json')]
            counts={key:0 for key in models};failures=[]
            for i,n in enumerate(queries):
                record=json.loads(z.read(n))
                if dataset=='prescience':
                    qid=record['query']['qid'];ids=record['metadata']['full']['candidates'];npz=n[:-5]+'.full.npz'
                else:
                    qid=record['qid'];ids=record['candidates'];npz=n[:-5]+'.npz'
                with np.load(io.BytesIO(z.read(npz)),allow_pickle=False) as cache:
                    X=cache['X'];assert len(X)==len(ids) and np.isfinite(X).all()
                    for key,(model,cols) in models.items():
                        scores=model.predict(X[:,cols],num_threads=1)
                        ranked=[ids[j] for j in sorted(range(len(ids)),key=lambda j:(-scores[j],ids[j]))]
                        if qid not in expected[key]:raise ValueError('Unexpected cached query '+qid)
                        expected_hash,retained_count=expected[key][qid]
                        if rankhash(ranked[:retained_count])!=expected_hash:failures.append({'query':qid,'method':key[0],'seed':key[1]})
                        counts[key]+=1
                if i%100==0:print(dataset,'cached-model replay',i,'/',len(queries),flush=True)
            for key,count in counts.items():assert count==len(expected[key])
            result={'dataset':dataset,'query_count':len(queries),'rankings_reexecuted':sum(counts.values()),
                    'methods':names,'seeds':[42,43,44],'exact_ranking_mismatches':failures,
                    'status':'PASS' if not failures else 'FAIL',
                    'comparison_depth':1000 if dataset=='prescience' else 100,
                    'score_difference':'Not available: archived ranked-author files do not retain per-author scores; exact ordered identifier parity is checked.'}
            results.append(result)
            (HERE/'cached-model-replay.json').write_text(json.dumps({'results':results,'elapsed_seconds':time.monotonic()-started,
                'scope':'Saved LightGBM execution on complete saved feature caches; not provider-to-feature reconstruction, model refitting, or new test evaluation.',
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2),encoding='utf-8')
            print(result,flush=True)

if __name__=='__main__':main()
