"""Temporal OGB-Collab development run; reads TRAIN and VALID only.

Train features: graph through 2016, labels: 2017. Validation features: graph
through 2017, labels/negatives: official 2018 split. Never reads test.pt.
"""
import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from lightgbm import LGBMClassifier, LGBMRanker
from ogb.linkproppred import Evaluator

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parents[1]
DATA=ROOT/'work/data/ogbl-collab'
NAMES=['common_neighbors','adamic_adar','resource_allocation','jaccard',
       'log_degree_product','log_degree_min','log_degree_max','prior_weight']

def log(s):
    print(time.strftime('%H:%M:%S'),s,flush=True)

def unpack():
    dest=DATA/'extracted'
    if not (dest/'collab/raw/edge.csv.gz').exists():
        dest.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(DATA/'collab.zip') as z:
            for name in z.namelist():
                target=(dest/name).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ValueError('Unsafe archive member')
            z.extractall(dest)
    return dest/'collab'

def graph(edges,weights,n):
    a=sp.csr_matrix((np.concatenate([weights,weights]),
         (np.concatenate([edges[:,0],edges[:,1]]),np.concatenate([edges[:,1],edges[:,0]]))),shape=(n,n))
    a.sum_duplicates()
    binary=a.copy()
    binary.data[:]=1
    degree=np.asarray(binary.sum(axis=1)).ravel()
    return a,binary,degree

def features(edges,state):
    weight,a,d=state
    result=np.empty((len(edges),len(NAMES)),dtype=np.float32)
    aa=np.zeros_like(d); aa[d>1]=1/np.log(d[d>1])
    ra=1/np.maximum(d,1)
    for start in range(0,len(edges),2048):
        u,v=edges[start:start+2048].T
        shared=a[u].multiply(a[v])
        cn=np.asarray(shared.sum(axis=1)).ravel()
        result[start:start+len(u)]=np.column_stack([
            cn,np.asarray(shared@aa).ravel(),np.asarray(shared@ra).ravel(),
            cn/np.maximum(d[u]+d[v]-cn,1),np.log1p(d[u]*d[v]),
            np.log1p(np.minimum(d[u],d[v])),np.log1p(np.maximum(d[u],d[v])),
            np.asarray(weight[u,v]).ravel()])
    return result

def training_edges(pos,all_seen,n,seed,negative_ratio=5):
    rng=np.random.default_rng(seed)
    banned=set((np.minimum(all_seen[:,0],all_seen[:,1])*n+np.maximum(all_seen[:,0],all_seen[:,1])).tolist())
    grouped={}
    for u,v in pos:
        grouped.setdefault(int(u),set()).add(int(v))
    edges,labels,groups=[],[],[]
    for u,positive in sorted(grouped.items()):
        negatives=set()
        while len(negatives)<negative_ratio*len(positive):
            v=int(rng.integers(n))
            if v!=u and min(u,v)*n+max(u,v) not in banned:
                negatives.add(v)
        candidates=sorted(positive|negatives)
        edges.extend((u,v) for v in candidates)
        labels.extend(int(v in positive) for v in candidates)
        groups.append(len(candidates))
    return np.array(edges),np.array(labels),np.array(groups)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-positives',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=42)
    args=ap.parse_args()
    start=time.perf_counter()
    path=unpack()
    raw=path/'raw'
    edges=pd.read_csv(raw/'edge.csv.gz',header=None).to_numpy(dtype=np.int64)
    year=pd.read_csv(raw/'edge_year.csv.gz',header=None).to_numpy().ravel()
    weights=pd.read_csv(raw/'edge_weight.csv.gz',header=None).to_numpy(dtype=np.float32).ravel()
    n=int(pd.read_csv(raw/'num-node-list.csv.gz',header=None).iloc[0,0])
    assert year.max()==2017
    # Public trusted OGB release uses legacy torch serialization of numpy arrays.
    valid=torch.load(path/'split/time/valid.pt',map_location='cpu',weights_only=False)
    vp=np.asarray(valid['edge']); vn=np.asarray(valid['edge_neg'])
    assert len(vn)==100000
    train_pos=edges[year==2017]
    rng=np.random.default_rng(args.seed)
    if args.train_positives and len(train_pos)>args.train_positives:
        train_pos=train_pos[rng.choice(len(train_pos),args.train_positives,replace=False)]
    te,ty,tg=training_edges(train_pos,edges,n,args.seed)
    log(f'Nodes={n}, raw edges={len(edges)}, training rows={len(te)}, validation pos/neg={len(vp)}/{len(vn)}')
    tx=features(te,graph(edges[year<=2016],weights[year<=2016],n))
    val_state=graph(edges,weights,n)
    vx=features(np.concatenate([vp,vn]),val_state)
    evaluator=Evaluator(name='ogbl-collab')
    run=OUT/'results'/f'ogb-validation-{args.train_positives}-s{args.seed}'
    run.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(run/'features.npz',train_X=tx,train_y=ty,train_group=tg,valid_X=vx)
    scores={name:vx[:,i] for i,name in enumerate(NAMES) if name in ['common_neighbors','adamic_adar','resource_allocation','prior_weight']}
    tuning=[]
    for family in ['lambdamart','binary_lightgbm']:
        best=None
        for leaves,estimators in [(7,60),(15,100),(31,100)]:
            cls=LGBMRanker if family=='lambdamart' else LGBMClassifier
            m=cls(n_estimators=estimators,num_leaves=leaves,learning_rate=.05,
                  reg_lambda=1,n_jobs=2,verbosity=-1,random_state=args.seed,
                  deterministic=True,force_col_wise=True,
                  objective='lambdarank' if family=='lambdamart' else 'binary')
            m.fit(tx,ty,**({'group':tg} if family=='lambdamart' else {}))
            pred=m.predict(vx) if family=='lambdamart' else m.predict_proba(vx)[:,1]
            value=evaluator.eval({'y_pred_pos':pred[:len(vp)],'y_pred_neg':pred[len(vp):]})['hits@50']
            tuning.append(dict(family=family,num_leaves=leaves,n_estimators=estimators,validation_hits50=value))
            if best is None or value>best[0]: best=(value,pred,m)
        scores[family]=best[1]
        best[2].booster_.save_model(str(run/f'{family}.txt'))
    results={}
    for name,pred in scores.items():
        results[name]=evaluator.eval({'y_pred_pos':pred[:len(vp)],'y_pred_neg':pred[len(vp):]})
        np.savez_compressed(run/f'{name}.predictions.npz',positive=pred[:len(vp)],negative=pred[len(vp):])
    report=dict(status='development validation only, NOT official test result',config=vars(args),features=NAMES,
                nodes=n,training_graph_cutoff=2016,training_label_year=2017,validation_graph_cutoff=2017,
                validation_label_year=2018,validation_positives=len(vp),validation_negatives=len(vn),
                scores=results,tuning=tuning,wall_seconds=time.perf_counter()-start,
                limitations=['No node text embeddings in this graph-only development run.',
                             'Query-wise LambdaMART objective is not identical to global-negative OGB evaluation.',
                             'No confidence intervals or repeated seeds yet.',
                             'Test edges and scores have not been read.'])
    (run/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    log(json.dumps(report,indent=2))

if __name__=='__main__': main()
