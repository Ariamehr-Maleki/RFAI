"""Pause at the existing RAM reserve, retaining loaded data and frozen semantics."""
from pathlib import Path
import json
import time
import psutil
import official_comparator as base
import official_comparator_bounded as bounded

original_memory=psutil.virtual_memory
waits=0
wait_seconds=0.

def wait_for_memory(minimum,phase):
    global waits,wait_seconds
    state=original_memory()
    while state.available<minimum:
        print(json.dumps({'resource_wait':phase,'available_GB':state.available/1e9,'required_GB':minimum/1e9}),flush=True)
        start=time.monotonic();time.sleep(15);wait_seconds+=time.monotonic()-start;waits+=1
        state=original_memory()
    return state

def main():
    addendum=dict(change='Wait in 15-second intervals at RAM reserve; no score/candidate/configuration changes',
        original_lock_sha256=base.sha(base.HERE/'supplementary-lock.json'),
        bounded_addendum_sha256=base.sha(base.HERE/'runtime-bounded-addendum.json'),
        wrapper_sha256=base.sha(Path(__file__)),before_load_bytes=1800*1024**2,
        runtime_reserve_bytes=450*1024**2)
    path=base.HERE/'runtime-memory-wait-addendum.json'
    if path.exists() and base.read(path)!=addendum:raise RuntimeError('Memory-wait wrapper changed')
    if not path.exists():base.write(path,addendum)
    wait_for_memory(addendum['before_load_bytes'],'before_load')
    psutil.virtual_memory=lambda:wait_for_memory(addendum['runtime_reserve_bytes'],'between_tracks')
    try:bounded.main()
    finally:
        psutil.virtual_memory=original_memory
        base.write(base.HERE/'runtime-memory-wait.json',dict(wait_intervals=waits,wait_seconds=wait_seconds,
            addendum_sha256=base.sha(path)))

if __name__=='__main__':main()
