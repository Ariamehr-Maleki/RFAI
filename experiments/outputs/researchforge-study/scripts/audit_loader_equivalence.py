"""Fingerprint the complete TRAIN loader output, preserving every order/value."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import time
import psutil
ROOT=Path(__file__).resolve().parents[3]

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    module=runpy.run_path(str(args.source));module['load_data'].__globals__['DATA']=ROOT/'work/data/prescience'
    start=time.perf_counter();papers,histories,dates=module['load_data']();fingerprints={}
    for name,table in [('papers',papers),('histories',histories),('dates',dates)]:
        h=hashlib.sha256()
        # Hash insertion order too: it may affect query construction or deterministic ties.
        for key,value in table.items():h.update(json.dumps([key,value],sort_keys=True,separators=(',',':')).encode());h.update(b'\n')
        fingerprints[name]=dict(rows=len(table),sha256=h.hexdigest())
    out=dict(status='Complete TRAIN loader fingerprint',source=str(args.source),source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        fingerprints=fingerprints,rss_bytes=psutil.Process().memory_info().rss,seconds=time.perf_counter()-start)
    args.output.write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
