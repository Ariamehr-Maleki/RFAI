"""Wait for the complete semantic index, then run one specified DEVELOPMENT pilot.

This is a dependency supervisor, not hyperparameter search or final test access.
The user revoked the former date limit. Every state transition is recorded.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]
SCRIPTS=Path(__file__).resolve().parent
STATE=ROOT/'work/hybrid-chain-state.json'
LOCK=ROOT/'work/hybrid-chain.lock'
INDEX=ROOT/'work/indices/prescience-minilm'
RUN=ROOT/'outputs/researchforge-study/results/hybrid-graph-dedup-200-100-1000-s42'


def state(stage,**kwargs):
    record=dict(stage=stage,supervisor_pid=os.getpid(),updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kwargs)
    atomic_state(STATE,record)


def main():
    expected=hashlib.sha256((SCRIPTS/'prescience_hybrid_graph.py').read_bytes()).hexdigest()
    while True:
        if (RUN/'report.json').exists():
            state('already complete; inspect results');return
        if not (INDEX/'manifest.json').exists():
            progress=json.loads((INDEX/'progress.json').read_text())
            worker=progress.get('worker_pid')
            state('waiting for complete semantic index',encoder_pid=worker,rows=progress['completed_rows'],total=progress['total_rows'])
            # Give a restart/checkpoint handover a grace period. Heartbeat can
            # diagnose a stopped encoder; never launch duplicate encoding here.
            if worker and not psutil.pid_exists(worker) and time.time()-(INDEX/'progress.json').stat().st_mtime>600:
                state('encoder stopped; inspect before restart',encoder_pid=worker);return
            time.sleep(30);continue
        manifest=json.loads((INDEX/'manifest.json').read_text())
        assert manifest['status']=='complete'
        # TRAIN-only measured full-loader/query peak 1.90GB, 27Oct2026.
        # Retain a runtime reserve guard; this changes scheduling, not the pilot.
        if psutil.virtual_memory().available<2500*1024**2 or shutil.disk_usage(ROOT).free<1024**3+150*1024**2:
            state('waiting for resources');time.sleep(30);continue
        assert hashlib.sha256((SCRIPTS/'prescience_hybrid_graph.py').read_bytes()).hexdigest()==expected,'Runner changed while queued; inspect and restart supervisor'
        command=[sys.executable,'-u',str(SCRIPTS/'prescience_hybrid_graph.py'),'--train-queries','200','--valid-queries','100','--candidates','1000','--seed','42']
        with (ROOT/'work/hybrid-pilot.log').open('a') as log:
            worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state('running development hybrid pilot',worker_pid=worker.pid,command=command,script_sha256=expected)
            peak=0;minimum=psutil.virtual_memory().available;last=0
            while worker.poll() is None:
                available=psutil.virtual_memory().available;minimum=min(minimum,available)
                try:
                    parent=psutil.Process(worker.pid)
                    peak=max(peak,sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)]))
                except psutil.NoSuchProcess:pass
                if available<640*1024**2:
                    worker.terminate();worker.wait()
                    state('resource reserve reached; query checkpoints retained',worker_pid=worker.pid,
                          sampled_peak_rss_bytes=peak,minimum_available_bytes=minimum)
                    return
                if time.time()-last>30:
                    state('running development hybrid pilot',worker_pid=worker.pid,command=command,script_sha256=expected,
                          sampled_peak_rss_bytes=peak,minimum_available_bytes=minimum)
                    last=time.time()
                time.sleep(.5)
            assert worker.returncode==0,'Hybrid pilot failed: inspect work/hybrid-pilot.log'
        subprocess.run([sys.executable,str(SCRIPTS/'development_summary.py'),'--run',str(RUN)],cwd=ROOT,check=True)
        state('complete',result=str(RUN/'report.json'));return


if __name__=='__main__':
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    try:main()
    except Exception as error:
        state('failed',error=repr(error));raise
    finally:LOCK.unlink()
