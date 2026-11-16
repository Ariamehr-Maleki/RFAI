"""Independently recompute saved pilot predictions with the official TREC library."""
import argparse
import json
from pathlib import Path
import numpy as np
import pytrec_eval

def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');args=ap.parse_args()
    folder=Path(args.run);results={}
    requested={f'{m}.{k}' for m in ['ndcg_cut','map_cut','recall'] for k in [10,20,100]}
    for path in folder.glob('*.predictions.json'):
        rows=json.loads(path.read_text())
        qrels={r['qid']:{str(a):1 for a in r['truth']} for r in rows}
        run={r['qid']:{str(a):float(len(r['predicted'])-i) for i,a in enumerate(r['predicted'])} for r in rows}
        official=pytrec_eval.RelevanceEvaluator(qrels,requested).evaluate(run)
        errors=[]
        for k in [10,20,100]:
            cut={r['qid']:{str(a):float(len(r['predicted'])-i) for i,a in enumerate(r['predicted'][:k])} for r in rows}
            rr=pytrec_eval.RelevanceEvaluator(qrels,{'recip_rank'}).evaluate(cut)
            for r in rows:
                for a,b in [('ndcg','ndcg_cut'),('map','map_cut'),('recall','recall')]:
                    errors.append(abs(r[f'{a}{k}']-official[r['qid']][f'{b}_{k}']))
                errors.append(abs(r[f'rr{k}']-rr[r['qid']]['recip_rank']))
        maximum=max(errors,default=0)
        assert maximum<1e-10,(path.name,maximum)
        results[path.name]=dict(queries=len(rows),comparisons=len(errors),max_absolute_error=maximum)
    assert results,'No saved predictions to verify'
    report=dict(library='pytrec-eval-terrier 0.5.11 / pytrec_eval',status='PASS',results=results)
    (folder/'metric_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
