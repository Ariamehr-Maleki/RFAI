"""Operational-only atomic-write retries; original frozen runner stays unchanged."""
from pathlib import Path
import os,sys,time,json,uuid,hashlib
import fresh_cohort as f
R=Path(__file__).resolve().parent

def durable_write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
 temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
 for attempt in range(40):
  try:temp.replace(path);return
  except PermissionError:
   if attempt==39:raise
   time.sleep(min(.1*2**min(attempt,4),1.0))

def setup():
 identity=f.sha(f.OUT/'sample-manifest.json');checked=0
 for p in (f.OUT/'ranker-queries').glob('*.json'):
  if p.name.endswith('.receipt.json'):continue
  obj=f.read(p);assert obj['sample_sha256']==identity
  assert f.sha(p)==f.read(p.with_suffix('.receipt.json'))['sha256']
  assert f.sha(p.with_suffix('.npz'))==obj['cache_sha256'];checked+=1
 lock={'change':'Only unique temporary filenames and bounded PermissionError retries for identical JSON serialization; no model/data/sampling/statistical change',
  'prior_prelock_sha256':f.sha(f.OUT/'pre-inference-lock-v2.json'),'sample_sha256':identity,
  'frozen_runner_sha256':f.sha(R/'fresh_cohort.py'),'wrapper_sha256':f.sha(Path(__file__)),
  'retry_attempts':40,'backoff_seconds':'0.1, 0.2, 0.4, 0.8, then <=1.0','reason':'WinError5 replacing progress.json after 84 fully committed queries; cause not established'}
 target=f.OUT/'runtime-write-lock-v3.json'
 if target.exists():assert f.read(target)==lock
 else:durable_write(target,lock)
 durable_write(R/'fresh-resume-verification.json',{'status':'PASS','committed_ranker_queries_checked':checked,'sample_sha256':identity,'runtime_lock_sha256':f.sha(target),'checks':'JSON receipt, feature NPZ hash, sample identity; no rescoring'})
 f.write=durable_write

if __name__=='__main__':
 setup();stage=sys.argv[1]
 if stage not in ['rankers','official','summarize']:raise ValueError(stage)
 getattr(f,stage)()
