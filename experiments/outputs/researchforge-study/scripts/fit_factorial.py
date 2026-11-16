"""Refit eight primitive-feature DEVELOPMENT ablations on a completed hybrid cache.

Only reads saved train/valid query features and labels. No raw benchmark loader
or official test access. Canonical date/qid order is recorded, shared pools fixed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from lightgbm import LGBMRanker,Booster
from threadpoolctl import threadpool_limits
from publication_ablation_design import factorial_columns
from prescience_pilot import evaluate

GRID=[(7,60),(15,100),(31,100)]

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,value):
    part=path.with_suffix('.part');part.write_text(json.dumps(value,indent=2));part.replace(path)

def load_partition(source,partition,features):
    folder=source/(partition+'_queries');items=[];artifacts={}
    for meta_path in folder.glob('*.json'):
        meta=read(meta_path);data_path=meta_path.with_suffix('.npz')
        if meta_path.stem!=str(meta['qid']):raise ValueError('Query filename/ID mismatch')
        with np.load(data_path,allow_pickle=False) as data:x,y=data['X'],data['y']
        candidates=meta['candidates']
        if len(set(candidates))!=len(candidates) or candidates!=sorted(candidates):
            raise ValueError('Candidates must be unique and ID-sorted for stable ties')
        if x.shape!=(len(candidates),len(features)) or y.shape!=(len(candidates),):raise ValueError('Feature/group dimensions disagree')
        if not np.isfinite(x).all() or not np.isin(y,[0,1]).all():raise ValueError('Invalid features or labels')
        truth=set(meta['truth'])
        if not truth or not set(meta['new_truth'])<=truth:raise ValueError('Invalid relevance metadata')
        if not np.array_equal(y,np.asarray([a in truth for a in candidates],dtype=np.int8)):
            raise ValueError('Cached labels do not match candidate IDs')
        if partition=='valid' and meta.get('training_positive_injections',0):raise ValueError('Validation positive injection forbidden')
        expected=len(set(candidates)&truth)/len(truth)
        if partition=='valid' and not np.isclose(meta['candidate_recall'],expected):raise ValueError('Candidate recall mismatch')
        for p in [meta_path,data_path]:artifacts[p.relative_to(source).as_posix()]=sha(p)
        items.append((meta,x,y))
    if not items:raise ValueError('No completed query cache for '+partition)
    items.sort(key=lambda item:(item[0]['date'],str(item[0]['qid'])))
    ids=[m['qid'] for m,_,_ in items]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate query IDs')
    return (np.concatenate([x for _,x,_ in items]),np.concatenate([y for _,_,y in items]),
            np.asarray([len(y) for _,_,y in items]),[m for m,_,_ in items],artifacts)

def fit(source,destination=None,seed=42,grid=GRID):
    source=Path(source).resolve();destination=Path(destination or source/f'factorial-s{seed}').resolve()
    report_path=source/'report.json'
    if not report_path.exists():raise RuntimeError('Completed DEVELOPMENT source report required')
    original=read(report_path)
    if not str(original.get('status','')).lower().startswith('development'):
        raise ValueError('Only completed development input is accepted')
    features=original['features'];settings=factorial_columns(features)
    tx,ty,tg,tm,ta=load_partition(source,'train',features)
    vx,vy,vg,vm,va=load_partition(source,'valid',features)
    if {m['qid'] for m in tm}&{m['qid'] for m in vm}:raise ValueError('Train/validation query overlap')
    if max(m['date'] for m in tm)>=min(m['date'] for m in vm):raise ValueError('Nonchronological train/validation split')
    for key,actual in [('train_queries',len(tm)),('valid_queries',len(vm))]:
        if original['config'][key]!=actual:raise ValueError('Incomplete source cache: '+key)
    config=dict(status='Development only; fixed joint candidate pool; no test labels',seed=seed,
        source=str(source),source_report_sha256=sha(report_path),cache_artifacts={**ta,**va},
        feature_names=features,columns=settings,grid=[list(v) for v in grid],
        selection='validation ndcg1000; ties retain first grid configuration',
        train_query_ids=[m['qid'] for m in tm],valid_query_ids=[m['qid'] for m in vm],
        canonical_order='date then string query ID; candidates sorted by ID',
        code_sha256={name:sha(Path(__file__).with_name(name)) for name in
            ['fit_factorial.py','publication_ablation_design.py','prescience_pilot.py']})
    destination.mkdir(parents=True,exist_ok=True)
    cfg_path=destination/'config.json'
    if cfg_path.exists() and read(cfg_path)!=config:raise ValueError('Factorial configuration changed; use a new output directory')
    lock=destination/'fit.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid()),f)
    began=time.perf_counter()
    try:
        write(cfg_path,config);metrics={};tuning=[];cells={}
        with threadpool_limits(limits=1):
            for name,cols in settings.items():
                cell_path=destination/(name+'.cell.json')
                if cell_path.exists():
                    cell=read(cell_path)
                    for artifact,digest in cell['artifact_sha256'].items():
                        if sha(destination/artifact)!=digest:raise ValueError('Saved cell artifact changed')
                else:
                    best=None;trials=[]
                    for leaves,trees in grid:
                        model=LGBMRanker(objective='lambdarank',n_estimators=trees,num_leaves=leaves,
                            learning_rate=.05,reg_lambda=1,random_state=seed,n_jobs=1,verbosity=-1,
                            deterministic=True,force_col_wise=True)
                        tick=time.perf_counter();model.fit(tx[:,cols],ty,group=tg)
                        scores=model.predict(vx[:,cols]);agg,_=evaluate(scores,vg,vm)
                        trial=dict(leaves=leaves,trees=trees,seconds=time.perf_counter()-tick,**agg);trials.append(trial)
                        if best is None or agg['ndcg1000']>best[0]:best=(agg['ndcg1000'],scores,model,trial)
                    _,scores,model,selected=best
                    path=destination/(name+'.txt');part=path.with_suffix('.part')
                    model.booster_.save_model(str(part));part.replace(path)
                    reloaded=Booster(model_file=str(path));roundtrip=reloaded.predict(vx[:,cols],num_threads=1)
                    np.testing.assert_allclose(roundtrip,scores,rtol=1e-12,atol=1e-12)
                    agg,predictions=evaluate(roundtrip,vg,vm,True)
                    predpath=destination/(name+'.predictions.json');write(predpath,predictions)
                    cell=dict(features=[features[i] for i in cols],columns=cols,selected=selected,
                        tuning=trials,metrics=agg,model_roundtrip_max_error=float(np.max(np.abs(roundtrip-scores))),
                        artifact_sha256={p.name:sha(p) for p in [path,predpath]})
                    write(cell_path,cell)
                metrics[name]=cell['metrics'];cells[name]=cell
                tuning.extend(dict(method=name,**row) for row in cell['tuning'])
                print(json.dumps(dict(cell=name,ndcg1000=cell['metrics']['ndcg1000'])),flush=True)
        result=dict(status='Development factorial; validation selected; no held-out results',config=config,
            metrics=metrics,tuning=tuning,cells=cells,wall_seconds_this_session=time.perf_counter()-began,
            limitations=['Joint candidate pool retains all retrieval channels in every cell.',
                'Category evidence belongs to text; mixed aggregate scores excluded from every cell.',
                'Explicit-time removal retains chronological masks and recent-publication windows.',
                'Fixed data and deterministic tree parameters may yield identical model-seed predictions.',
                'Validation selection makes scores exploratory; not confirmatory significance.'])
        write(destination/'report.json',result);return result
    finally:lock.unlink()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--output',type=Path);ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args();fit(args.source,args.output,args.seed)

if __name__=='__main__':main()
