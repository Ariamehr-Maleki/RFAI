"""Resumable expanded DEVELOPMENT collection with common validation IDs.

No official test reader. Same query manifest and padding seed across full/no-
content modes. Per-seed training group subsets are recorded; caches reused with
NTFS hard links, never regenerated merely to vary a model seed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import shutil
import numpy as np
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from prescience_pilot import ROOT,OUT,DATA,load_data,make_queries,sample_queries,day
from prescience_hybrid_graph import QueryEngine,FEATURES
from prescience_retrieval_ablation import NoContentEngine
from text_index import TextIndex
from semantic_index import SemanticIndex
from fit_factorial import sha,read,write

SEEDS=[42,43,44]

def query_plan(queries,train_pool=1200,train_per_seed=1000,valid_count=500):
    split=day('2024-07-01')
    pool=sample_queries([q for q in queries if q['date']<split],train_pool,20261026)
    valid=sample_queries([q for q in queries if q['date']>=split],valid_count,20261027)
    if len(pool)<train_pool or len(valid)<valid_count or train_per_seed>train_pool:
        raise ValueError('Insufficient queries or invalid requested sizes')
    return dict(status='Development query manifest; official test untouched',split_day=split,
        train_pool=pool,validation=valid,seeds=SEEDS,
        train_ids_by_seed={str(s):[q['qid'] for q in sample_queries(pool,train_per_seed,s)] for s in SEEDS},
        sampling='Common validation; per-seed training-group subsets of one fixed pool',padding_seed=42)

def save_plan(path,plan):
    if path.exists() and read(path)!=plan:raise ValueError('Existing query plan differs; use a new experiment')
    if not path.exists():write(path,plan)

def collect_cache(engine,queries,folder,training,candidates,padding_seed):
    folder.mkdir(parents=True,exist_ok=True)
    for i,q in enumerate(queries):
        meta=folder/(str(q['qid'])+'.json');data=meta.with_suffix('.npz');receipt=meta.with_suffix('.receipt')
        if receipt.exists():
            record=read(receipt)
            if sha(meta)!=record['metadata_sha256'] or sha(data)!=record['features_sha256']:
                raise ValueError('Committed feature cache changed')
        else:
            tick=time.perf_counter();x,y,m=engine.run(q,candidates,padding_seed,training)
            m['seconds']=time.perf_counter()-tick
            part=data.with_suffix('.part')
            with part.open('wb') as f:np.savez_compressed(f,X=x,y=y)
            part.replace(data);write(meta,m)
            write(receipt,dict(metadata_sha256=sha(meta),features_sha256=sha(data)))
        print(json.dumps(dict(partition=folder.name,completed=i+1,total=len(queries),qid=q['qid'])),flush=True)

def make_view(cache,plan,seed,view,source_config):
    """Make immutable cache views; no copying large feature matrices per seed."""
    view.mkdir(parents=True,exist_ok=True)
    selected={'train':plan['train_ids_by_seed'][str(seed)],'valid':[q['qid'] for q in plan['validation']]}
    inventory={}
    for partition,ids in selected.items():
        folder=view/(partition+'_queries');folder.mkdir(exist_ok=True)
        existing={p.stem for p in folder.glob('*.json')}
        if existing-set(map(str,ids)):raise ValueError('Unexpected query already present in cache view')
        for qid in ids:
            for suffix in ['.json','.npz']:
                src=cache/(partition+'_queries')/(str(qid)+suffix);dst=folder/src.name
                if dst.exists():
                    if sha(dst)!=sha(src):raise ValueError('View differs from source cache')
                else:os.link(src,dst)
                inventory[dst.relative_to(view).as_posix()]=sha(src)
    config=dict(source_config,seed=seed,train_queries=len(selected['train']),valid_queries=len(selected['valid']))
    report=dict(status='Development feature cache complete; model fitting pending',config=config,features=FEATURES,
        cache_origin=str(cache.resolve()),cache_sha256=inventory,
        model_columns_policy=source_config['mode'])
    # This report identifies completed feature collection, not completed ranking.
    path=view/'cache_report.json'
    if path.exists() and read(path)!=report:raise ValueError('Cache-view provenance changed')
    write(path,report)
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['full','no_content'],default='full')
    ap.add_argument('--train-pool',type=int,default=1200);ap.add_argument('--train-per-seed',type=int,default=1000)
    ap.add_argument('--valid-queries',type=int,default=500);ap.add_argument('--candidates',type=int,default=1000)
    args=ap.parse_args();base=OUT/'results'/f'prescience-expanded-{args.train_pool}-{args.train_per_seed}-{args.valid_queries}-{args.candidates}'
    import psutil
    if psutil.virtual_memory().available<2500*1024**2 or shutil.disk_usage(ROOT).free<1024**3+500*1024**2:
        raise RuntimeError('Wait for2500MiB availableRAM and1GiB+500MiB disk reserve after exact-value loader compaction')
    run=base/args.mode;run.mkdir(parents=True,exist_ok=True)
    lock=run/'collect.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid()),f)
    try:
        if args.mode=='full':semantic=SemanticIndex()
        else:semantic=None
        config=dict(**vars(args),padding_seed=42,code_sha256={n:sha(Path(__file__).with_name(n)) for n in
            ['prescience_expanded.py','prescience_hybrid_graph.py','prescience_retrieval_ablation.py',
             'prescience_pilot.py','text_index.py','semantic_index.py']},
             train_source_sha256=read(DATA/'train.parquet.manifest.json')['sha256'])
        if semantic is not None:config['semantic_sha256']=semantic.manifest['sha256']
        save_plan(run/'config.json',config)
        papers,histories,dates=load_data()
        for batch in pq.ParquetFile(DATA/'train.parquet').iter_batches(batch_size=2048,columns=['corpus_id','key_references']):
            for row in batch.to_pylist():
                pid=str(row['corpus_id'])
                if pid in papers:papers[pid]['refs']=tuple(str(r['corpus_id']) for r in row['key_references'] or [] if r['corpus_id'])
        plan=query_plan(make_queries(papers,histories,dates),args.train_pool,args.train_per_seed,args.valid_queries)
        save_plan(base/'query_manifest.json',plan)
        index=TextIndex() if args.mode=='full' else None
        engine=QueryEngine(papers,histories,dates,index,semantic) if args.mode=='full' else NoContentEngine(papers,histories,dates)
        with threadpool_limits(limits=1):
            collect_cache(engine,plan['train_pool'],run/'train_queries',True,args.candidates,42)
            collect_cache(engine,plan['validation'],run/'valid_queries',False,args.candidates,42)
        del engine,index,semantic,papers,histories,dates;gc.collect()
        views=[]
        for seed in SEEDS:
            view=run/f'seed-{seed}';make_view(run,plan,seed,view,config);views.append(str(view))
        write(run/'collection_report.json',dict(status='Development collection complete; model fitting pending',
            config=config,query_manifest_sha256=sha(base/'query_manifest.json'),views=views))
    finally:lock.unlink()

if __name__=='__main__':main()
