"""One TRAIN-only resource probe; no model fitting or final relevance access."""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/researchforge-study/data-audit/hybrid-memory-probe.json'
proc=psutil.Process();start=time.perf_counter();snapshots=[];peak=[0];minimum=[psutil.virtual_memory().available]
initial=minimum[0];done=threading.Event()

def save(status):
    record=dict(status=status,pid=os.getpid(),initial_available_bytes=initial,
        sampled_peak_rss_bytes=peak[0],minimum_available_bytes=minimum[0],seconds=time.perf_counter()-start,
        sample_interval_seconds=.2,snapshots=snapshots,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='One historical query probes resource use, not benchmark quality or worst-case memory.')
    OUT.write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)

def monitor():
    while not done.wait(.2):
        peak[0]=max(peak[0],proc.memory_info().rss);available=psutil.virtual_memory().available
        minimum[0]=min(minimum[0],available)
        if available<640*1024**2:
            save('Stopped safely below640MiB available RAM');os._exit(75)

def mark(stage):
    row=dict(stage=stage,rss_bytes=proc.memory_info().rss,available_bytes=psutil.virtual_memory().available,seconds=time.perf_counter()-start)
    snapshots.append(row);print(json.dumps(row),flush=True)

if __name__=='__main__':
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        from prescience_hybrid_graph import load_data,QueryEngine,TextIndex,SemanticIndex,make_queries,DATA,pq
        from threadpoolctl import threadpool_limits
        mark('imports');papers,histories,dates=load_data();mark('loaded paper and author histories')
        for batch in pq.ParquetFile(DATA/'train.parquet').iter_batches(batch_size=2048,columns=['corpus_id','key_references']):
            for row in batch.to_pylist():
                cid=str(row['corpus_id'])
                if cid in papers:papers[cid]['refs']=tuple(str(r['corpus_id']) for r in row['key_references'] or [] if r['corpus_id'])
        mark('references');index=TextIndex();semantic=SemanticIndex();engine=QueryEngine(papers,histories,dates,index,semantic)
        mark('all indices and engine');queries=make_queries(papers,histories,dates)
        with threadpool_limits(limits=1):
            engine.run(queries[0],1000,42,False)
        mark('one historical query');save('PASS: full historical data and one query loaded under resource watchdog')
    finally:done.set();watcher.join(timeout=1)
