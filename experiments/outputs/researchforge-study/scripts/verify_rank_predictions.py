"""Independently verify saved development ranking metrics with pytrec_eval."""
import argparse
import json
from pathlib import Path
import numpy as np
import pytrec_eval

TRANSLATE={'ndcg10':'ndcg_cut_10','ndcg1000':'ndcg_cut_1000',
           'rprecision':'Rprec','recall10':'recall_10','recall50':'recall_50'}

def verify(rows,tolerance=1e-10):
    qrels={};runs={};new_qrels={}
    for row in rows:
        qid=str(row['qid']);pred=row['predicted'];truth=set(row['truth']);new=set(row['new_truth'])
        if qid in qrels or len(pred)!=len(set(pred)) or not truth or not new<=truth:
            raise ValueError('Invalid saved ranking/relevance set')
        qrels[qid]={str(a):1 for a in truth}
        runs[qid]={str(a):float(len(pred)-i) for i,a in enumerate(pred)}
        if new:new_qrels[qid]={str(a):1 for a in new}
    evaluator=pytrec_eval.RelevanceEvaluator(qrels,{'ndcg_cut.10,1000','recall.10,50','Rprec'})
    values=evaluator.evaluate(runs)
    new_values=pytrec_eval.RelevanceEvaluator(new_qrels,{'recall.10'}).evaluate(runs) if new_qrels else {}
    errors=[];metrics={k:[] for k in [*TRANSLATE,'candidate_recall','new_recall10']}
    for row in rows:
        qid=str(row['qid'])
        for local,official in TRANSLATE.items():
            value=values[qid][official];errors.append(abs(value-row[local]));metrics[local].append(value)
        coverage=len(set(row['predicted'])&set(row['truth']))/len(set(row['truth']))
        errors.append(abs(coverage-row['candidate_recall']));metrics['candidate_recall'].append(coverage)
        if qid in new_values:
            value=new_values[qid]['recall_10']
            if row['new_recall10'] is None:raise ValueError('Missing new-recall value')
            errors.append(abs(value-row['new_recall10']));metrics['new_recall10'].append(value)
        elif row['new_recall10'] is not None:raise ValueError('Undefined new recall reported as a value')
    maximum=max(errors,default=0.)
    if maximum>tolerance:raise ValueError(f'Saved metric differs from independent evaluator: {maximum}')
    return dict(status='PASS',queries=len(rows),metric_comparisons=len(errors),max_absolute_error=maximum,
                aggregate={k:float(np.mean(v)) if v else None for k,v in metrics.items()})

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);args=ap.parse_args()
    report=json.loads((args.run/'report.json').read_text())
    if not str(report.get('status','')).lower().startswith(('development','description-based sensitivity')):
        raise ValueError('Development report required')
    results={}
    for path in sorted(args.run.glob('*.predictions.json')):
        name=path.name.removesuffix('.predictions.json')
        result=verify(json.loads(path.read_text()))
        if name in report['metrics']:expected=report['metrics'][name]
        else:
            variant,mode=name.rsplit('.',1);expected=report['metrics'][variant][mode]
        for metric,value in result['aggregate'].items():
            if value is None:
                if expected[metric] is not None:raise ValueError('Aggregate eligibility differs')
            elif abs(expected[metric]-value)>1e-10:raise ValueError('Reported aggregate differs: '+name+'/'+metric)
        results[name]=result;print(json.dumps(dict(method=name,**result)),flush=True)
    if not results:raise ValueError('No saved prediction files')
    (args.run/'independent_metric_verification.json').write_text(json.dumps(dict(status='PASS',library='pytrec_eval',results=results),indent=2))

if __name__=='__main__':main()
