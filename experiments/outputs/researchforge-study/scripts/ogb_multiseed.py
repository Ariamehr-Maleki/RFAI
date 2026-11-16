"""Expanded DEVELOPMENT comparison; never opens official test.pt.

Released node features are a supplementary track: their temporal provenance
is not certified. Every family gets the same training examples and 3 configs.
"""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, LGBMRanker
from ogb.linkproppred import Evaluator
import ogb_validation as base

def node_features(edges, nodes):
    result = np.empty((len(edges), 2 + 2 * nodes.shape[1]), dtype=np.float32)
    for start in range(0, len(edges), 2048):
        u, v = edges[start:start+2048].T
        left, right = nodes[u], nodes[v]
        product = left * right
        norm = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        delta = np.abs(left-right)
        result[start:start+len(u)] = np.column_stack([
            product.sum(1)/np.maximum(norm, 1e-12),
            np.linalg.norm(delta, axis=1), product, delta])
    return result

def summarize(pred, count, source, evaluator, bootstrap=2000):
    pos, neg = pred[:count], pred[count:]
    official = float(evaluator.eval({'y_pred_pos':pos,'y_pred_neg':neg})['hits@50'])
    threshold = np.partition(neg, -50)[-50]
    success = (pos > threshold).astype(float)
    assert abs(success.mean()-official) < 1e-12
    unique, inverse = np.unique(source, return_inverse=True)
    totals = np.bincount(inverse, weights=success)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(8841)
    samples = np.empty(bootstrap)
    for i in range(bootstrap):
        index = rng.integers(len(unique), size=len(unique))
        samples[i] = totals[index].sum()/sizes[index].sum()
    return dict(hits50=official, source_cluster_ci95=np.quantile(samples,[.025,.975]).tolist(),
                negative_threshold=float(threshold), source_clusters=len(unique))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-positives', type=int, default=10000)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,43,44])
    args=parser.parse_args()
    run=base.OUT/'results'/f'ogb-expanded-{args.train_positives}'
    run.mkdir(parents=True,exist_ok=True)
    config=dict(**vars(args), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                base_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                grid=[[7,60],[15,100],[31,100]], threads=1)
    lock=run/'config.json'
    if lock.exists(): assert json.loads(lock.read_text())==config, 'Do not overwrite a different experiment'
    else: lock.write_text(json.dumps(config,indent=2))
    path=base.unpack(); raw=path/'raw'
    edges=pd.read_csv(raw/'edge.csv.gz',header=None).to_numpy(dtype=np.int64)
    year=pd.read_csv(raw/'edge_year.csv.gz',header=None).to_numpy().ravel()
    weights=pd.read_csv(raw/'edge_weight.csv.gz',header=None).to_numpy(dtype=np.float32).ravel()
    n=int(pd.read_csv(raw/'num-node-list.csv.gz',header=None).iloc[0,0])
    assert year.max()==2017
    valid=torch.load(path/'split/time/valid.pt',map_location='cpu',weights_only=False)
    vp,vn=np.asarray(valid['edge']),np.asarray(valid['edge_neg'])
    assert len(vn)==100000
    ve=np.concatenate([vp,vn])
    nodes=pd.read_csv(raw/'node-feat.csv.gz',header=None,dtype=np.float32).to_numpy()
    assert nodes.shape==(n,128) and np.isfinite(nodes).all()
    vx=base.features(ve,base.graph(edges,weights,n))
    nx=node_features(ve,nodes)
    evaluator=Evaluator(name='ogbl-collab')
    report=dict(status='development validation; not final test',config=config,
                validation_positives=len(vp),validation_negatives=len(vn),runs={},
                limitations=['Node-feature track has unverified temporal provenance and is supplementary.',
                'Intervals condition on fixed shared negative pool; source-only clusters do not model all network dependence.',
                'Validation is used for tuning; intervals are exploratory, not confirmatory.',
                'LambdaMART query objective differs from global-negative OGB Hits@50.',
                'Seed variation includes positive subsampling and negative sampling, not only optimizer randomness.'])
    report['fixed_baselines']={}
    for name,col in [('common_neighbors',0),('adamic_adar',1),('resource_allocation',2),('prior_weight',7)]:
        pred=vx[:,col]
        report['fixed_baselines'][name]=summarize(pred,len(vp),vp[:,0],evaluator)
        np.savez_compressed(run/f'{name}.predictions.npz',positive=pred[:len(vp)],negative=pred[len(vp):])
    report['fixed_baselines']['released_node_cosine']=summarize(nx[:,0],len(vp),vp[:,0],evaluator)
    np.savez_compressed(run/'released_node_cosine.predictions.npz',positive=nx[:len(vp),0],negative=nx[len(vp):,0])
    for seed in args.seeds:
        complete=run/f'seed-{seed}.json'
        if complete.exists():
            report['runs'][str(seed)]=json.loads(complete.read_text()); continue
        began=time.perf_counter()
        positive=edges[year==2017]
        rng=np.random.default_rng(seed)
        if len(positive)>args.train_positives:
            positive=positive[rng.choice(len(positive),args.train_positives,replace=False)]
        te,ty,tg=base.training_edges(positive,edges,n,seed)
        tx=base.features(te,base.graph(edges[year<=2016],weights[year<=2016],n))
        tn=node_features(te,nodes)
        result=dict(seed=seed,training_rows=len(ty),training_groups=len(tg),methods={},tuning=[])
        for track in ['graph','graph_plus_released_nodes']:
            train=tx if track=='graph' else np.column_stack([tx,tn])
            validation=vx if track=='graph' else np.column_stack([vx,nx])
            for family in ['lambdamart','binary_lightgbm']:
                best=None
                for leaves,trees in config['grid']:
                    cls=LGBMRanker if family=='lambdamart' else LGBMClassifier
                    model=cls(n_estimators=trees,num_leaves=leaves,learning_rate=.05,
                        reg_lambda=1,n_jobs=1,verbosity=-1,random_state=seed,
                        deterministic=True,force_col_wise=True,
                        objective='lambdarank' if family=='lambdamart' else 'binary')
                    model.fit(train,ty,**({'group':tg} if family=='lambdamart' else {}))
                    pred=model.predict(validation) if family=='lambdamart' else model.predict_proba(validation)[:,1]
                    score=float(evaluator.eval({'y_pred_pos':pred[:len(vp)],'y_pred_neg':pred[len(vp):]})['hits@50'])
                    result['tuning'].append(dict(track=track,family=family,leaves=leaves,trees=trees,hits50=score))
                    if best is None or score>best[0]:best=(score,pred,model,leaves,trees)
                name=f'{track}__{family}'; score,pred,model,leaves,trees=best
                result['methods'][name]=dict(**summarize(pred,len(vp),vp[:,0],evaluator),leaves=leaves,trees=trees)
                model.booster_.save_model(str(run/f's{seed}-{name}.txt'))
                np.savez_compressed(run/f's{seed}-{name}.predictions.npz',positive=pred[:len(vp)],negative=pred[len(vp):])
                base.log(f'Seed={seed} method={name} validation Hits50={score:.6f}')
            del train,validation; gc.collect()
        result['wall_seconds']=time.perf_counter()-began
        complete.write_text(json.dumps(result,indent=2))
        report['runs'][str(seed)]=result
    names=next(iter(report['runs'].values()))['methods']
    report['seed_summary']={name:dict(mean=float(np.mean([r['methods'][name]['hits50'] for r in report['runs'].values()])),
        std=float(np.std([r['methods'][name]['hits50'] for r in report['runs'].values()],ddof=1))) for name in names}
    (run/'report.json').write_text(json.dumps(report,indent=2))
    base.log(json.dumps(report['seed_summary'],indent=2))

if __name__=='__main__':main()
