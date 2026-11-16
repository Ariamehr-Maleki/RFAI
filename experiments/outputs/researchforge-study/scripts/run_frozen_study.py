"""Serial execution after an independently reviewed, existing global final lock.

Never assembles/freezes configurations, chooses a sample or fits a model. It can
wait for the lock while development runs. A missing lock cannot open test data.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import psutil
from final_evaluation_gate import verify_lock
from final_results_tables import checked_report
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3];SCRIPTS=Path(__file__).resolve().parent
STUDY=ROOT/'outputs/researchforge-study';WORK=ROOT/'work'
RUNNERS=[('ogbl-collab','ogb_final_run.py'),('exHarmony','exharmony_final_run.py'),('prescience','prescience_final_run.py')]
STATE=WORK/'frozen-study-state.json';WRITER=WORK/'frozen-study-supervisor.lock'


def state(stage,**extra):
    atomic_state(STATE,dict(stage=stage,supervisor_pid=os.getpid(),updated_epoch=time.time(),**extra))


def code_inventory():
    names={'run_frozen_study.py','final_results_tables.py','final_results_figures.py'}
    for _,name in RUNNERS:
        names.add(name)
        for node in ast.parse((SCRIPTS/name).read_text()).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CODE' for t in node.targets):names.update(ast.literal_eval(node.value))
    names.add('ncn_temporal.py')
    return {name:digest(SCRIPTS/name) for name in sorted(names)}


def unchanged(pinned):
    for name,expected in pinned.items():
        if digest(SCRIPTS/name)!=expected:raise ValueError('Queued final source changed; inspect before restart: '+name)


def local_output(relative):
    p=Path(relative)
    if p.is_absolute() or not (ROOT/p).resolve().is_relative_to(ROOT):raise ValueError('Workspace-relative output required')
    return (ROOT/p).resolve()


def workload_busy():
    # Waiting supervisors are harmless; real collection/training/encoding is not.
    scripts={'semantic_encode.py','prescience_test_text_cache.py','prescience_expanded.py','fit_expanded_models.py',
             'fit_factorial.py','prescience_fixed_sensitivity.py','pipeline_cost.py','ncn_temporal.py',
             'ogb_final_run.py','exharmony_final_run.py','prescience_final_run.py'}
    for process in psutil.process_iter(['pid','cmdline']):
        try:
            argv=process.info['cmdline'] or []
            if any(Path(arg).name in scripts for arg in argv):return True
        except (psutil.AccessDenied,psutil.NoSuchProcess):continue
    return False


def execute(label,command,pinned,min_ram=3500,min_disk=2048):
    while workload_busy() or psutil.virtual_memory().available<min_ram*1024**2 or shutil.disk_usage(ROOT).free<min_disk*1024**2:
        unchanged(pinned);state('waiting for isolated final execution resources',next_stage=label);time.sleep(30)
    unchanged(pinned)
    logpath=WORK/(label+'.log')
    with logpath.open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        state(label,worker_pid=process.pid,command=command,code_sha256=pinned)
        peak=0;last=0
        while process.poll() is None:
            available=psutil.virtual_memory().available;disk=shutil.disk_usage(ROOT).free
            try:
                parent=psutil.Process(process.pid);tree=[parent,*parent.children(recursive=True)]
                peak=max(peak,sum(p.memory_info().rss for p in tree))
            except (psutil.NoSuchProcess,psutil.AccessDenied):tree=[]
            if available<640*1024**2 or disk<1024**3:
                for p in reversed(tree):
                    try:p.terminate()
                    except psutil.NoSuchProcess:pass
                process.wait()
                raise RuntimeError('Final worker reserve reached. Preserve checkpoints; inspect owned stale writer lock before resume.')
            if time.time()-last>30:
                state(label,worker_pid=process.pid,command=command,code_sha256=pinned,sampled_peak_tree_rss_bytes=peak,available_ram_bytes=available,free_disk_bytes=disk)
                last=time.time()
            time.sleep(.5)
        if process.returncode:raise RuntimeError(f'{label} exited {process.returncode}; inspect {logpath}')


def verify_export(folder,lock,status):
    p=folder/'provenance.json'
    if not p.exists():raise ValueError('Partial export retained; inspect before retry: '+str(folder))
    report=json.loads(p.read_text())
    if report.get('lock_sha256')!=lock['lock_sha256'] or report.get('status')!=status:raise ValueError('Export provenance mismatch')
    for name,sha in report['files'].items():
        path=(folder/name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path)!=sha:raise ValueError('Changed export: '+name)
    return report


def run(lock_path,run_id,wait_for_lock=False):
    lock_path=Path(lock_path).resolve()
    if not lock_path.is_relative_to(ROOT) or not run_id:raise ValueError('Explicit local lock and run ID required')
    fd=os.open(WRITER,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as f:json.dump(dict(pid=os.getpid(),lock=str(lock_path),run_id=run_id),f)
    pinned=code_inventory()
    try:
        while not lock_path.exists():
            if not wait_for_lock:raise FileNotFoundError(lock_path)
            unchanged(pinned);state('waiting for reviewed global freeze; development labels only',expected_lock=str(lock_path),run_id=run_id,code_sha256=pinned);time.sleep(30)
        unchanged(pinned);lock=verify_lock(ROOT,lock_path)
        ledger=STUDY/'results/final/ledger'
        for dataset,script in RUNNERS:
            out=local_output(lock['payload']['plan']['datasets'][dataset]['output_directory'])
            if not (out/'report.json').exists():
                execute('final-'+dataset,[sys.executable,'-u',str(SCRIPTS/script),'--lock',str(lock_path),'--run-id',run_id,'--ledger',str(ledger)],pinned)
            lock=verify_lock(ROOT,lock_path);_,report=checked_report(ROOT,lock,dataset)
            if report.get('run_id')!=run_id:raise ValueError('Existing dataset belongs to another run ID')
            state('verified held-out dataset',dataset=dataset,run_id=run_id,lock_sha256=lock['lock_sha256'])
        tables=STUDY/'results/final/tables';figures=STUDY/'results/final/figures'
        if not tables.exists():execute('final-tables',[sys.executable,'-u',str(SCRIPTS/'final_results_tables.py'),'--lock',str(lock_path),'--output',str(tables)],pinned,min_ram=1500)
        verify_export(tables,lock,'Verified frozen-result table export')
        cost=STUDY/'results/prescience-expanded-1200-1000-500-1000/isolated_pipeline_cost.json'
        if not figures.exists():execute('final-figures',[sys.executable,'-u',str(SCRIPTS/'final_results_figures.py'),'--lock',str(lock_path),'--cost',str(cost),'--output',str(figures)],pinned,min_ram=1500)
        verify_export(figures,lock,'Verified frozen-result figures')
        state('final experiments and numerical exports complete; manuscript and visual review still required',run_id=run_id,lock_sha256=lock['lock_sha256'],tables=str(tables),figures=str(figures))
    except Exception as error:state('failed; preserve outputs and inspect before resume',error=repr(error));raise
    finally:WRITER.unlink()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lock',required=True,type=Path);ap.add_argument('--run-id',required=True);ap.add_argument('--wait-for-lock',action='store_true');args=ap.parse_args()
    run(args.lock,args.run_id,args.wait_for_lock)
