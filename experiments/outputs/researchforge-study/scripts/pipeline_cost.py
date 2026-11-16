"""Isolated warm DEVELOPMENT pipeline cost, exact saved feature parity required.

Retrieval and feature construction are timed jointly. Index similarity calls are
subtimings; their remainder is NOT labeled pure graph time or pure feature time.
"""
import argparse
import gc
import json
from pathlib import Path
import platform
import threading
import time
import numpy as np
import psutil
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
from reranking_cost import require_idle
from frozen_ranker import FrozenRanker,model_spec,digest
from prescience_hybrid_graph import QueryEngine,FEATURES
from prescience_retrieval_ablation import NoContentEngine
from prescience_pilot import load_data,DATA
from semantic_index import SemanticIndex
from text_index import TextIndex
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
CODE=['pipeline_cost.py','reranking_cost.py','frozen_ranker.py','prescience_hybrid_graph.py',
      'prescience_retrieval_ablation.py','prescience_pilot.py','semantic_index.py','text_index.py','supervisor_state.py']


class TimedIndex:
    def __init__(self,index):self.index=index;self.seconds=0.
    def __getattr__(self,name):return getattr(self.index,name)
    def similarities(self,*args,**kwargs):
        start=time.perf_counter()
        try:return self.index.similarities(*args,**kwargs)
        finally:self.seconds+=time.perf_counter()-start


def summary(values):
    values=np.asarray(values,dtype=float)
    return dict(median_seconds=float(np.median(values)),p95_seconds=float(np.quantile(values,.95)))


def benchmark(base,output,count=30,repeats=3):
    require_idle();base=Path(base).resolve();output=Path(output).resolve()
    if not output.is_relative_to(ROOT) or output.exists():raise ValueError('New task-workspace output required')
    if count<1 or repeats<3:raise ValueError('Positive query count and at least3 repeats required')
    full=base/'full/seed-42';no_content=base/'no_content/seed-42'
    for folder in [full,no_content]:
        if not json.loads((folder/'report.json').read_text())['status'].startswith('Development'):raise ValueError('Development artifacts only')
    paths=sorted((full/'valid_queries').glob('*.json'))[:count]
    if len(paths)!=count:raise ValueError('Insufficient fixed cost queries')
    queries=[{k:row[k] for k in ['qid','seed','date','truth']} for p in paths for row in [json.loads(p.read_text())]]
    if any(not (no_content/'valid_queries'/p.name).exists() for p in paths):raise ValueError('Cost query sets differ')
    proc=psutil.Process();stop=threading.Event();peak=[proc.memory_info().rss];samples=[0]
    def monitor():
        while not stop.wait(.02):peak[0]=max(peak[0],proc.memory_info().rss);samples[0]+=1
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start();started=time.perf_counter()
    try:
        tick=time.perf_counter();papers,histories,dates=load_data()
        for batch in pq.ParquetFile(DATA/'train.parquet').iter_batches(batch_size=2048,columns=['corpus_id','key_references']):
            for row in batch.to_pylist():
                cid=str(row['corpus_id'])
                if cid in papers:papers[cid]['refs']=tuple(str(r['corpus_id']) for r in row['key_references'] or [] if r['corpus_id'])
        data_load=time.perf_counter()-tick
        tick=time.perf_counter();lexical=TimedIndex(TextIndex());semantic=TimedIndex(SemanticIndex())
        index_load=time.perf_counter()-tick
        result={};sources={str(p.relative_to(ROOT)):digest(p) for p in paths}
        with threadpool_limits(limits=1):
            for mode,folder in [('full',full),('no_content',no_content)]:
                engine=QueryEngine(papers,histories,dates,lexical,semantic) if mode=='full' else NoContentEngine(papers,histories,dates)
                models={name:FrozenRanker(model_spec(folder,name),FEATURES) for name in ['lambdamart','binary_lightgbm','logistic']}
                reference=[]
                for path,q in zip(paths,queries):
                    with np.load(folder/'valid_queries'/path.with_suffix('.npz').name,allow_pickle=False) as z:expected=z['X']
                    meta=json.loads((folder/'valid_queries'/path.name).read_text());x,_,actual=engine.run(q,1000,42,False)
                    np.testing.assert_allclose(x,expected,rtol=1e-5,atol=1e-7)
                    if actual['candidates']!=meta['candidates'] or actual['training_positive_injections']:raise ValueError('Cost run differs from saved development candidates')
                    reference.append((expected,actual['candidates']))
                    sources[str((folder/'valid_queries'/path.with_suffix('.npz').name).relative_to(ROOT))]=digest(folder/'valid_queries'/path.with_suffix('.npz').name)
                observations=[]
                for repeat in range(repeats):
                    for q,(expected,ids) in zip(queries,reference):
                        lexical.seconds=0.;semantic.seconds=0.;tick=time.perf_counter()
                        x,_,meta=engine.run(q,1000,42,False);pipeline=time.perf_counter()-tick
                        np.testing.assert_allclose(x,expected,rtol=1e-5,atol=1e-7)
                        if meta['candidates']!=ids:raise ValueError('Nondeterministic measured candidates')
                        timings={}
                        for name,model in models.items():
                            tick=time.perf_counter();values=model.predict(x);ranking=np.argsort(-values,kind='stable');timings[name]=time.perf_counter()-tick
                            if len(ranking)!=len(ids):raise ValueError('Incomplete measured ranking')
                        observations.append(dict(qid=q['qid'],repeat=repeat,retrieval_and_features_seconds=pipeline,
                            lexical_similarity_seconds=lexical.seconds if mode=='full' else 0.,semantic_similarity_seconds=semantic.seconds if mode=='full' else 0.,
                            remainder_seconds=pipeline-(lexical.seconds+semantic.seconds if mode=='full' else 0.),reranking_seconds=timings,candidates=len(ids)))
                result[mode]=dict(observations=observations,retrieval_and_features=summary([r['retrieval_and_features_seconds'] for r in observations]),
                    end_to_end={name:summary([r['retrieval_and_features_seconds']+r['reranking_seconds'][name] for r in observations]) for name in models},
                    models={name:dict(spec=model.spec,bytes=Path(model.spec['path']).stat().st_size) for name,model in models.items()})
                del engine,models,reference;gc.collect()
        require_idle()
        report=dict(status='Completed isolated development pipeline cost; not final quality evaluation',query_count=count,repeats=repeats,model_seed=42,padding_seed=42,candidates=1000,
            query_ids=[q['qid'] for q in queries],results=result,data_load_seconds=data_load,index_load_seconds=index_load,
            sampled_peak_process_rss_bytes=peak[0],rss_sample_interval_seconds=.02,rss_samples=samples[0],wall_seconds=time.perf_counter()-started,
            machine=dict(platform=platform.platform(),logical_cpus=psutil.cpu_count(),total_ram_bytes=psutil.virtual_memory().total,threadpool_limit=1),
            source_sha256=sources,code_sha256={n:digest(Path(__file__).with_name(n)) for n in CODE},
            limitations=['Warm full-query repetitions; initial parity passes warm indices and caches.',
                'Retrieval and feature construction are joint; similarity subtimings omit aggregation and do not identify causal component costs.',
                'RSS is sampled process memory including data/indices/models, not guaranteed OS peak or incremental model memory.',
                'Desktop processes may contend; study training/encoding workers prohibited before and after.',
                'Seed42 models only; timing repeats are not independent scientific observations.',
                'Historical preprocessing/training totals require separate logs; interrupted wall time is not active compute.'])
        atomic_state(output,report);return report
    finally:stop.set();watcher.join(timeout=2)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--queries',type=int,default=30);ap.add_argument('--repeats',type=int,default=3);args=ap.parse_args()
    print(json.dumps(benchmark(args.base,args.output,args.queries,args.repeats),indent=2))
