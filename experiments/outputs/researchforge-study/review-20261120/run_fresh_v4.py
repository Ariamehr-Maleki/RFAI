"""One-writer continuation after verified predecessor processes exit."""
from pathlib import Path
import subprocess,os,sys,time,json,uuid,hashlib
import psutil
R=Path(__file__).resolve().parent;GUARD=R/'fresh-v4-exclusive.json'
def write(p,o):
    t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.partial');t.write_text(json.dumps(o,indent=2),encoding='utf-8')
    for i in range(40):
        try:t.replace(p);return
        except PermissionError:
            if i==39:raise
            time.sleep(.5)
def alive_predecessors():
    found=[]
    for p in psutil.process_iter(['pid','name','cmdline']):
        try:
            a=p.info['cmdline'] or []
            if 'python' in (p.info['name'] or '').lower() and any(Path(x).name in ['run_fresh_v3.py','fresh_io_v3.py','run_fresh_stages.py'] for x in a):found.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return found
def main():
    with GUARD.open('x',encoding='utf-8') as h:json.dump({'pid':os.getpid(),'started':time.time(),'script':str(Path(__file__))},h)
    try:
        while ids:=alive_predecessors():
            write(R/'fresh-v4-state.json',{'status':'waiting for predecessor to exit; no concurrent scientific worker','predecessor_pids':ids,'supervisor_pid':os.getpid()});time.sleep(15)
        env=os.environ.copy();env.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        steps=[]
        if not (R/'fresh-cohort/rankers-complete.json').exists():steps.append(('rankers',['fresh_io_v4.py','rankers']))
        steps += [('audit-rankers',['audit_fresh_rankers.py']),('audit-overlap',['audit_fresh_overlap_features.py']),('official',['fresh_io_v4.py','official']),('summarize',['fresh_io_v4.py','summarize'])]
        for stage,args in steps:
            with (R/('fresh-'+stage+'-v4.log')).open('a',encoding='utf-8') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(R/args[0]),*args[1:]],stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
                state={'stage':stage,'pid':proc.pid,'supervisor_pid':os.getpid(),'status':'running','runtime_version':4};write(R/'fresh-v4-state.json',state)
                code=proc.wait();state.update(exit_code=code,status='completed' if code==0 else 'failed');write(R/'fresh-v4-state.json',state)
                if code:raise SystemExit(code)
        write(R/'fresh-v4-state.json',{'status':'stages completed; final independent official/statistics audit and manuscript integration still required','supervisor_pid':os.getpid(),'runtime_version':4})
    finally:
        # Only remove the exclusive token this process created.
        if GUARD.exists() and json.loads(GUARD.read_text())['pid']==os.getpid():GUARD.unlink()
if __name__=='__main__':main()
