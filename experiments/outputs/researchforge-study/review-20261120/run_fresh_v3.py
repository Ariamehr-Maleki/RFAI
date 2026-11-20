from pathlib import Path
import subprocess,os,json,sys
from fresh_io_v3 import durable_write
R=Path(__file__).resolve().parent
env=os.environ.copy();env.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for stage in ['rankers','official','summarize']:
 with (R/('fresh-'+stage+'-v3.log')).open('a',encoding='utf-8') as log:
  proc=subprocess.Popen([sys.executable,'-u',str(R/'fresh_io_v3.py'),stage],stdout=log,stderr=subprocess.STDOUT,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
  durable_write(R/'fresh-supervisor-state.json',{'stage':stage,'pid':proc.pid,'supervisor_pid':os.getpid(),'status':'running','runtime_version':3})
  code=proc.wait()
  durable_write(R/'fresh-supervisor-state.json',{'stage':stage,'pid':proc.pid,'supervisor_pid':os.getpid(),'exit_code':code,'status':'completed' if code==0 else 'failed','runtime_version':3})
  if code:raise SystemExit(code)
