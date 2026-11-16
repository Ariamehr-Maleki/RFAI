"""Recover checkpointed elapsed encoding time, never call it isolated CPU time."""
import hashlib
import json
from pathlib import Path
import math
from supervisor_state import atomic_state
ROOT=Path(__file__).resolve().parents[3]
LOGS=['minilm-full.log','minilm-resume-1809.log','minilm-resume-20261027.log']

def account(sessions,total,config):
    results=[];pids=set()
    for name,rows in sessions:
        if not rows:raise ValueError('Missing encoding checkpoints: '+name)
        first,last=rows[0],rows[-1];pid=first['worker_pid']
        if pid in pids:raise ValueError('Repeated process/session; do not double-count elapsed time')
        pids.add(pid);start=int(round(first['completed_rows']-first['session_seconds']*first['session_docs_per_second']))
        previous=start;elapsed=0.
        for row in rows:
            if row['worker_pid']!=pid or row['config']!=config or row['total_rows']!=total:raise ValueError('Session identity differs')
            end=row['completed_rows'];seconds=row['session_seconds'];rate=row['session_docs_per_second']
            if end<previous or not math.isfinite(seconds) or seconds<elapsed:raise ValueError('Nonmonotone session')
            if abs((end-start)-seconds*rate)>.01:raise ValueError('Inconsistent checkpoint throughput')
            previous=end;elapsed=seconds
        results.append(dict(log=name,pid=pid,start_row=start,end_row=last['completed_rows'],rows=last['completed_rows']-start,recorded_elapsed_seconds=last['session_seconds']))
    results.sort(key=lambda row:row['start_row']);end=0
    for row in results:
        if row['start_row']!=end or row['rows']<=0:raise ValueError('Overlapping or missing committed encoding rows')
        end=row['end_row']
    if end!=total:raise ValueError('Incomplete encoding accounting')
    elapsed=sum(row['recorded_elapsed_seconds'] for row in results)
    return dict(status='Complete checkpointed TRAIN encoding elapsed-time accounting',sessions=results,rows=total,
        recorded_elapsed_seconds=elapsed,recorded_elapsed_hours=elapsed/3600,rows_per_recorded_second=total/elapsed,
        interpretation='Elapsed time inside encoding sessions through last committed checkpoints; not active CPU time or isolated throughput.',
        exclusions=['Model download/load and post-loop final hashing are excluded from this timer.',
                    'Uncommitted work after the last checkpoint is not included.',
                    'OS suspension, contention and waiting inside a live session may be included.',
                    'The gaps between separate sessions are excluded; no end-to-end calendar duration is claimed.'])

if __name__=='__main__':
    manifest=json.loads((ROOT/'work/indices/prescience-minilm/manifest.json').read_text());sessions=[];hashes={}
    for name in LOGS:
        path=ROOT/'work'/name;rows=[]
        for line in path.read_text().splitlines():
            if line.startswith('{'):
                row=json.loads(line)
                if 'completed_rows' in row and 'session_docs_per_second' in row:rows.append(row)
        sessions.append((name,rows));hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=account(sessions,manifest['total_rows'],manifest['config']);report.update(log_sha256=hashes,embedding_bytes=manifest['bytes'],embedding_sha256=manifest['sha256'])
    atomic_state(ROOT/'outputs/researchforge-study/data-audit/train_encoding_cost.json',report);print(json.dumps(report,indent=2))
