from pathlib import Path
import subprocess,os,json,sys
R=Path(__file__).resolve().parent
env=os.environ.copy();env.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for stage in ['rankers','official','summarize']:
 with (R/('fresh-'+stage+'.log')).open('a',encoding='utf-8') as log:
  proc=subprocess.Popen([sys.executable,'-u',str(R/'fresh_cohort.py'),stage],stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
  (R/'fresh-supervisor-state.json').write_text(json.dumps({'stage':stage,'pid':proc.pid,'supervisor_pid':os.getpid(),'status':'running'},indent=2))
  code=proc.wait()
  (R/'fresh-supervisor-state.json').write_text(json.dumps({'stage':stage,'pid':proc.pid,'supervisor_pid':os.getpid(),'exit_code':code,'status':'completed' if code==0 else 'failed'},indent=2))
  if code:raise SystemExit(code)
