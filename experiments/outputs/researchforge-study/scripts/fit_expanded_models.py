"""Matched library ranker/classifier/logistic fits on expanded development views."""
import argparse
import json
import os
from pathlib import Path
import time
import joblib
import numpy as np
from lightgbm import LGBMRanker,LGBMClassifier,Booster
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
from fit_factorial import GRID,load_partition,read,write,sha
from prescience_pilot import evaluate,rank_metrics
from prescience_retrieval_ablation import model_columns

def fit_models(view,grid=GRID,logistic_cs=(.1,1,10)):
    view=Path(view);cache=read(view/'cache_report.json');mode=cache['config']['mode']
    if cache['status']!='Development feature cache complete; model fitting pending':raise ValueError('Invalid development cache')
    features=cache['features'];cols=model_columns(mode)
    tx,ty,tg,tm,ta=load_partition(view,'train',features);vx,vy,vg,vm,va=load_partition(view,'valid',features)
    if {r['qid'] for r in tm}&{r['qid'] for r in vm} or max(r['date'] for r in tm)>=min(r['date'] for r in vm):
        raise ValueError('Query overlap or nonchronological development split')
    if {**ta,**va}!=cache['cache_sha256']:raise ValueError('Cache view changed')
    config=dict(cache['config'],fit_code_sha256=sha(__file__),cache_report_sha256=sha(view/'cache_report.json'),
        columns=cols,grid=[list(v) for v in grid],logistic_cs=list(logistic_cs),
        selection='validation ndcg1000; first declared configuration wins ties')
    path=view/'fit_config.json'
    if path.exists() and read(path)!=config:raise ValueError('Model-fitting config changed')
    lock=view/'models.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid()),f)
    try:
        write(path,config);began=time.perf_counter();metrics={};tuning=[];models={}
        with threadpool_limits(limits=1):
            for family in ['lambdamart','binary_lightgbm','logistic']:
                cell_path=view/(family+'.model.json')
                if cell_path.exists():
                    cell=read(cell_path)
                    for name,digest in cell['artifacts'].items():
                        if sha(view/name)!=digest:raise ValueError('Model artifact changed')
                else:
                    best=None;trials=[]
                    for param in (logistic_cs if family=='logistic' else grid):
                        if family=='logistic':
                            model=make_pipeline(StandardScaler(),LogisticRegression(C=param,max_iter=500,random_state=config['seed']))
                            parameters=dict(C=param)
                        else:
                            leaves,trees=param;parameters=dict(num_leaves=leaves,n_estimators=trees)
                            model=(LGBMRanker if family=='lambdamart' else LGBMClassifier)(**parameters,
                                objective='lambdarank' if family=='lambdamart' else 'binary',learning_rate=.05,
                                reg_lambda=1,random_state=config['seed'],n_jobs=1,verbosity=-1,deterministic=True,force_col_wise=True)
                        tick=time.perf_counter();model.fit(tx[:,cols],ty,**({'group':tg} if family=='lambdamart' else {}))
                        scores=model.predict(vx[:,cols]) if family=='lambdamart' else model.predict_proba(vx[:,cols])[:,1]
                        agg,_=evaluate(scores,vg,vm);trial=dict(parameters=parameters,seconds=time.perf_counter()-tick,**agg);trials.append(trial)
                        if best is None or agg['ndcg1000']>best[0]:best=(agg['ndcg1000'],model,scores,trial)
                    _,model,scores,selected=best
                    model_path=view/(family+('.joblib' if family=='logistic' else '.txt'))
                    temporary=model_path.with_suffix('.part')
                    if family=='logistic':
                        joblib.dump(model,temporary);temporary.replace(model_path)
                        replay=joblib.load(model_path).predict_proba(vx[:,cols])[:,1]
                    else:
                        model.booster_.save_model(str(temporary));temporary.replace(model_path)
                        replay=Booster(model_file=str(model_path)).predict(vx[:,cols],num_threads=1)
                    np.testing.assert_allclose(replay,scores,rtol=1e-10,atol=1e-12)
                    agg,rows=evaluate(replay,vg,vm,True);predpath=view/(family+'.predictions.json');write(predpath,rows)
                    cell=dict(metrics=agg,tuning=trials,selected=selected,feature_names=[features[i] for i in cols],
                        roundtrip_max_error=float(np.max(np.abs(replay-scores))),
                        artifacts={p.name:sha(p) for p in [model_path,predpath]})
                    write(cell_path,cell)
                metrics[family]=cell['metrics'];models[family]=cell;tuning.extend(dict(method=family,**t) for t in cell['tuning'])
                print(json.dumps(dict(family=family,ndcg1000=cell['metrics']['ndcg1000'])),flush=True)
            baselines={'frequency10':0,'frequency45':1,'frequency_all':2,'common_neighbors10':4,
                'autoresearch_description':17,'rrf':19}
            if mode=='full':baselines.update(lexical_mean10=14,semantic_mean10=20,category_cosine=10)
            for name,col in baselines.items():
                metrics[name],rows=evaluate(vx[:,col],vg,vm,True);write(view/(name+'.predictions.json'),rows)
            for name in vm[0]['standalone']:
                rows=[]
                for q in vm:
                    pred=q['standalone'][name]
                    rows.append(dict(qid=q['qid'],seed=q['seed'],truth=q['truth'],new_truth=q['new_truth'],predicted=pred,
                        candidate_recall=len(set(pred)&set(q['truth']))/len(q['truth']),**rank_metrics(pred,q['truth'],q['new_truth'])))
                keys=['ndcg1000','ndcg10','rprecision','recall10','recall50','new_recall10','candidate_recall']
                metrics[name]={k:float(np.mean([r[k] for r in rows if r[k] is not None])) if any(r[k] is not None for r in rows) else None for k in keys}
                write(view/(name+'.predictions.json'),rows)
        result=dict(status='Development expanded comparison; official test sealed',config=config,features=features,
            metrics=metrics,models=models,tuning=tuning,wall_seconds_this_session=time.perf_counter()-began,
            limitations=['Validation-selected configurations; no confirmatory superiority.',
                'Seed variation includes recorded training-group subsampling; shared validation queries.',
                'Full/no-content pipelines require separate fits; no-content RRF contains graph channels only.',
                'Fixed Autoresearch sensitivity variants must additionally be evaluated before final freezing.'])
        write(view/'report.json',result);return result
    finally:lock.unlink()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--view',type=Path,required=True);args=ap.parse_args();fit_models(args.view)

if __name__=='__main__':main()
