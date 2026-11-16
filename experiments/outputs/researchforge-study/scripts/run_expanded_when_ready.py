"""Sequential expanded development chain after pilot, factorial and NCN finish."""
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

ROOT=Path(__file__).resolve().parents[3];SCRIPTS=Path(__file__).resolve().parent
RESULTS=ROOT/'outputs/researchforge-study/results';WORK=ROOT/'work'
BASE=RESULTS/'prescience-expanded-1200-1000-500-1000'
STATE=WORK/'expanded-chain-state.json';LOCK=WORK/'expanded-chain.lock'
FILES=['prescience_expanded.py','prescience_retrieval_ablation.py','fit_expanded_models.py',
       'prescience_hybrid_graph.py','prescience_pilot.py','fit_factorial.py',
       'publication_ablation_design.py','verify_rank_predictions.py','subgroup_analysis.py']
FILES+=['prescience_fixed_sensitivity.py','prescience_fixed_scorer.py','summarize_expanded.py','final_statistics.py']

def state(stage,**extra):
    value=dict(stage=stage,supervisor_pid=os.getpid(),updated_epoch=time.time(),**extra)
    atomic_state(STATE,value)

class CollectionMemoryStop(RuntimeError):
    def __init__(self,owned_pids):
        super().__init__('Memory reserve reached during resumable collection')
        self.owned_pids=set(owned_pids)


def recover_collection(run,owned_pids):
    """Only an owned, dead collector's lock may be removed after receipt audit."""
    run=Path(run);lock=run/'collect.lock'
    if any(psutil.pid_exists(pid) for pid in owned_pids):raise RuntimeError('Owned collector still alive')
    if lock.exists() and json.loads(lock.read_text())['pid'] not in owned_pids:
        raise RuntimeError('Collection lock belongs to another process')
    count=0
    for partition in ['train_queries','valid_queries']:
        for receipt in (run/partition).glob('*.receipt'):
            record=json.loads(receipt.read_text())
            for suffix,key in [('.json','metadata_sha256'),('.npz','features_sha256')]:
                if hashlib.sha256(receipt.with_suffix(suffix).read_bytes()).hexdigest()!=record[key]:
                    raise ValueError('Committed cache changed: '+str(receipt))
            count+=1
    if lock.exists():lock.unlink()
    return count

def main():
    fd=os.open(LOCK,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid()),f)
    hashes={n:hashlib.sha256((SCRIPTS/n).read_bytes()).hexdigest() for n in FILES}
    def unchanged():
        for n,digest in hashes.items():
            if hashlib.sha256((SCRIPTS/n).read_bytes()).hexdigest()!=digest:raise ValueError('Queued code changed: '+n)
    def execute_once(label,script,args):
        while psutil.virtual_memory().available<3000*1024**2:
            state('waiting for sequential worker RAM',next_stage=label);time.sleep(30)
        unchanged();command=[sys.executable,'-u',str(SCRIPTS/script),*map(str,args)]
        with (WORK/(label+'.log')).open('a') as log:
            log_start=log.tell()
            worker=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state(label,worker_pid=worker.pid,command=command,code_sha256=hashes)
            peak=0;minimum=psutil.virtual_memory().available;last=0
            while worker.poll() is None:
                available=psutil.virtual_memory().available;minimum=min(minimum,available)
                try:
                    parent=psutil.Process(worker.pid)
                    peak=max(peak,sum(p.memory_info().rss for p in [parent,*parent.children(recursive=True)]))
                except psutil.NoSuchProcess:pass
                if available<640*1024**2:
                    # Terminate only this owned worker tree, deepest children first.
                    # Windows parent termination alone need not stop descendants.
                    owned={worker.pid};children=[]
                    try:
                        children=psutil.Process(worker.pid).children(recursive=True)
                        owned.update(p.pid for p in children)
                        for process in reversed(children):
                            try:process.terminate()
                            except psutil.NoSuchProcess:pass
                    except psutil.NoSuchProcess:pass
                    worker.terminate();worker.wait()
                    psutil.wait_procs(children,timeout=10)
                    if script=='prescience_expanded.py':raise CollectionMemoryStop(owned)
                    raise RuntimeError('Memory reserve reached; preserve per-query checkpoints and inspect before resuming')
                if time.time()-last>30:
                    state(label,worker_pid=worker.pid,command=command,code_sha256=hashes,
                          sampled_peak_process_tree_rss_bytes=peak,minimum_available_bytes=minimum)
                    last=time.time()
                time.sleep(.5)
            code=worker.returncode
        if code:
            # A desktop allocation can race admission before the worker creates
            # its lock. Retry only that exact resource refusal from THIS launch.
            with (WORK/(label+'.log')).open('r') as log:
                log.seek(log_start);new_output=log.read()
            if script=='prescience_expanded.py' and 'RuntimeError: Wait for2500MiB availableRAM and1GiB+500MiB disk reserve after exact-value loader compaction' in new_output:
                raise CollectionMemoryStop({worker.pid})
            raise RuntimeError(f'{label} exited{code}; inspect its log')
    def execute(label,script,args):
        while True:
            try:return execute_once(label,script,args)
            except CollectionMemoryStop as error:
                mode=str(args[1]);count=recover_collection(BASE/mode,error.owned_pids)
                state('collection checkpoint audit passed after memory stop; waiting to resume',mode=mode,verified_receipts=count)
                # Avoid immediate restart during transient desktop memory pressure.
                time.sleep(60)
                while psutil.virtual_memory().available<3000*1024**2 or shutil.disk_usage(ROOT).free<1024**3+800*1024**2:
                    state('waiting for3000MiB RAM to resume audited collection',mode=mode,verified_receipts=count);time.sleep(30)
    try:
        dependencies=[RESULTS/'hybrid-graph-dedup-200-100-1000-s42/report.json',
                      RESULTS/'hybrid-graph-dedup-200-100-1000-s42/factorial-s42/report.json',
                      RESULTS/'ncn-expanded-10000/report.json']
        while not all(p.exists() for p in dependencies):
            state('waiting for hybrid pilot, factorial and three-seed NCN completion',
                  missing=[str(p) for p in dependencies if not p.exists()])
            time.sleep(30)
        # A real overlap attempt hit the640MiB desktop reserve after117 queries.
        # Keep both configurations/checkpoints; serialize remaining development
        # behind the already-running label-free cache, not behind a model change.
        encoding_audit=ROOT/'outputs/researchforge-study/data-audit/final_semantic_index.json'
        while not encoding_audit.exists():
            state('waiting for final label-free encoding to finish; sequential memory policy',
                  preserved_collection=str(BASE/'full/train_queries'));time.sleep(30)
        if not json.loads(encoding_audit.read_text())['status'].startswith('PASS'):raise ValueError('Final preprocessing audit failed')
        unchanged()
        for mode in ['full','no_content']:
            run=BASE/mode
            # Exact-value TRAIN loader interning reduced retained RSS by~508MB.
            while psutil.virtual_memory().available<2500*1024**2 or shutil.disk_usage(ROOT).free<1024**3+800*1024**2:
                state('waiting for expanded collection resources',mode=mode);time.sleep(30)
            if not (run/'collection_report.json').exists():
                execute('expanded-collect-'+mode,'prescience_expanded.py',['--mode',mode])
            for seed in [42,43,44]:
                view=run/f'seed-{seed}'
                if not (view/'report.json').exists():
                    execute(f'expanded-fit-{mode}-{seed}','fit_expanded_models.py',['--view',view])
                if not (view/'independent_metric_verification.json').exists():
                    execute(f'expanded-verify-{mode}-{seed}','verify_rank_predictions.py',['--run',view])
                if not (view/'subgroups-lambdamart-vs-binary_lightgbm.json').exists():
                    execute(f'expanded-subgroups-{mode}-{seed}','subgroup_analysis.py',['--run',view])
                if mode=='full':
                    factorial=view/f'factorial-s{seed}'
                    if not (factorial/'report.json').exists():
                        execute(f'expanded-factorial-{seed}','fit_factorial.py',['--source',view,'--seed',seed])
                    if not (factorial/'independent_metric_verification.json').exists():
                        execute(f'expanded-factorial-verify-{seed}','verify_rank_predictions.py',['--run',factorial])
            sensitivity=run/'seed-42/fixed-scorer-sensitivity'
            if not (sensitivity/'report.json').exists():
                execute('expanded-fixed-'+mode,'prescience_fixed_sensitivity.py',[run/'seed-42'])
            if not (sensitivity/'independent_metric_verification.json').exists():
                execute('expanded-fixed-verify-'+mode,'verify_rank_predictions.py',['--run',sensitivity])
        execute('expanded-summary','summarize_expanded.py',['--base',BASE])
        state('complete development chain; inspect all results before final configuration freeze',base=str(BASE))
    except Exception as error:
        state('failed',error=repr(error));raise
    finally:LOCK.unlink()

if __name__=='__main__':main()
