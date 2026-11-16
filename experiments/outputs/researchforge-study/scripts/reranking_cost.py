"""Measure saved-model reranking on development feature caches only.

Run after active training/encoding finishes. Does not measure retrieval or
encoding costs; elapsed times from concurrent experiments cannot replace it.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from frozen_ranker import FrozenRanker,model_spec,digest

ROOT=Path(__file__).resolve().parents[3]
BUSY=('semantic_encode.py','ncn_temporal.py','prescience_hybrid_graph.py','prescience_expanded.py',
      'fit_expanded_models.py','fit_factorial.py','prescience_test_text_cache.py',
      'prescience_final_lexical.py','compare_ncn_development.py','prescience_fixed_sensitivity.py',
      'prescience_final_run.py','ogb_final_run.py','exharmony_final_run.py')


def require_idle():
    busy=[];unknown=[]
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            if proc.pid==os.getpid():continue
            name=(proc.info['name'] or '').lower()
            if not name.startswith(('python','pythonw')):continue
            args=proc.info['cmdline']
            if args is None:unknown.append(proc.pid)
            elif any(Path(arg).name in BUSY for arg in args):busy.append(proc.pid)
        except (psutil.AccessDenied,psutil.ZombieProcess):unknown.append(proc.pid)
        except psutil.NoSuchProcess:continue
    if busy or unknown:raise RuntimeError(f'Isolated costs deferred: busy workers={busy}; unreadable Python processes={unknown}')
    if psutil.virtual_memory().available<2500*1024**2:raise RuntimeError('Need2500MiB availableRAM for isolated costs')


def measure(predictor,matrices,repeats=5):
    if not matrices or repeats<3:raise ValueError('Queries and at least3 timing repeats required')
    # Preload and warm every selected query; report warm-cache inference only.
    reference=[predictor.predict(x) for x in matrices]
    timings=[];start_cpu=time.process_time()
    for repeat in range(repeats):
        row=[]
        for matrix,expected in zip(matrices,reference):
            tick=time.perf_counter();actual=predictor.predict(matrix);row.append(time.perf_counter()-tick)
            np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
        timings.append(row)
    cpu=time.process_time()-start_cpu
    values=np.asarray(timings)
    return dict(repeats=repeats,query_count=len(matrices),candidate_rows=[len(x) for x in matrices],
        per_repeat_query_seconds=timings,median_seconds=float(np.median(values)),p95_seconds=float(np.quantile(values,.95)),
        process_cpu_seconds_including_parity_checks=cpu,
        measurement='Warm precomputed-feature inference, includes schema checks/projection; excludes loading/retrieval/encoding',
        sampling='Fixed query set, repeated timing observations are not independent query samples')


def benchmark(folder,method,output,count=100,repeats=5,metadata_folder=None):
    require_idle()
    folder=Path(folder).resolve();metadata=Path(metadata_folder or folder).resolve()
    report=json.loads((metadata/'report.json').read_text())
    if not str(report.get('status','')).lower().startswith('development'):
        raise ValueError('Development feature cache required; no final labels accepted')
    features=report['features'];paths=sorted((metadata/'valid_queries').glob('*.npz'))[:count]
    if count<1 or not paths:raise ValueError('No development feature caches')
    matrices=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:matrices.append(z['X'])
    spec=model_spec(folder,method);rss_before=psutil.Process().memory_info().rss
    with threadpool_limits(limits=1):
        load_start=time.perf_counter();predictor=FrozenRanker(spec,features);load_seconds=time.perf_counter()-load_start
        result=measure(predictor,matrices,repeats)
    require_idle()
    result.update(status='Development isolated reranking cost; no ranking-quality result',model=spec,
        model_bytes=Path(spec['path']).stat().st_size,model_load_seconds=load_seconds,
        rss_before_model_load=rss_before,rss_after_measurement=psutil.Process().memory_info().rss,
        memory_note='RSS snapshots only, not peak memory; matrices already loaded before first snapshot',
        machine=dict(platform=platform.platform(),logical_cpus=psutil.cpu_count(),total_ram=psutil.virtual_memory().total),
        feature_cache_sha256={str(p):digest(p) for p in paths},
        code_sha256={name:digest(Path(__file__).with_name(name)) for name in ['reranking_cost.py','frozen_ranker.py']},
        limitations=['No CPU isolation from unrelated desktop programs; task-worker checks before/after measurement.',
                     'All methods must use identical query IDs and timing repeat counts for cost comparison.',
                     'Separate end-to-end retrieval, preprocessing and training cost studies remain necessary.'])
    output=Path(output)
    with output.open('x') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--method',required=True);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--metadata-run',type=Path);parser.add_argument('--queries',type=int,default=100)
    parser.add_argument('--repeats',type=int,default=5);args=parser.parse_args()
    print(json.dumps(benchmark(args.run,args.method,args.output,args.queries,args.repeats,args.metadata_run),indent=2))
