"""Verified resumable public wheel download using bounded HTTP byte ranges."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'work/ncn-wheels'
NAME='torch-2.5.1-cp312-cp312-win_amd64.whl'

def main():
    DEST.mkdir(exist_ok=True)
    session=requests.Session()
    response=session.get('https://pypi.org/pypi/torch/2.5.1/json',timeout=30);response.raise_for_status()
    entry=next(x for x in response.json()['urls'] if x['filename']==NAME)
    total=entry['size'];expected=entry['digests']['sha256'];target=DEST/NAME
    part=target.with_suffix('.whl.part')
    if not target.exists():
        offset=part.stat().st_size if part.exists() else 0
        assert offset<=total
        if shutil.disk_usage(DEST).free-(total-offset)<1024**3:raise OSError('Download would violate1GiB reserve')
        with part.open('ab') as out:
            while offset<total:
                end=min(offset+1024**2,total)-1
                for attempt in range(3):
                    try:
                        r=session.get(entry['url'],headers={'Range':f'bytes={offset}-{end}'},timeout=(15,30))
                        r.raise_for_status()
                        assert r.status_code==206 and r.headers.get('Content-Range')==f'bytes {offset}-{end}/{total}'
                        payload=r.content;assert len(payload)==end-offset+1
                        break
                    except (requests.RequestException,AssertionError):
                        if attempt==2:raise
                        time.sleep(1+attempt)
                out.write(payload);out.flush();os.fsync(out.fileno());offset=end+1
                if offset%(10*1024**2)==0 or offset==total:print(f'Downloaded {offset}/{total}',flush=True)
        with part.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        assert actual==expected,'SHA256 does not match PyPI metadata'
        part.replace(target)
    with target.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
    assert actual==expected
    with zipfile.ZipFile(target) as z:uncompressed=sum(x.file_size for x in z.infolist())
    record=dict(url=entry['url'],filename=NAME,bytes=total,sha256=actual,uncompressed_bytes=uncompressed,
                free_disk_bytes=shutil.disk_usage(DEST).free,status='verified against official PyPI metadata')
    target.with_suffix('.whl.manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
