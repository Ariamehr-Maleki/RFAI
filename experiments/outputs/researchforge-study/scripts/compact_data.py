"""Lossless, verified compression of task-owned downloaded JSONL artifacts."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = (ROOT / 'work/data/exHarmony').resolve()

def main():
    for name in ['papers_collection.jsonl', 'authors_info.jsonl', 'authors_works_collection_ids.jsonl']:
        src = (DATA / name).resolve()
        assert src.is_relative_to(DATA) and DATA.is_relative_to((ROOT / 'work').resolve())
        dst = src.with_suffix(src.suffix + '.gz')
        if not src.exists():
            print(f'Already compacted: {dst}', flush=True)
            continue
        manifest = json.loads(src.with_suffix(src.suffix+'.manifest.json').read_text())
        with src.open('rb') as f:
            original = hashlib.file_digest(f, 'sha256').hexdigest()
        assert original == manifest['sha256']
        partial = dst.with_suffix(dst.suffix+'.part')
        with src.open('rb') as f, gzip.open(partial, 'wb', compresslevel=3) as out:
            shutil.copyfileobj(f, out, length=1024*1024)
        with gzip.open(partial, 'rb') as f:
            restored = hashlib.file_digest(f, 'sha256').hexdigest()
        assert restored == original, 'Round-trip checksum mismatch; original retained'
        partial.replace(dst)
        with dst.open('rb') as f:
            compressed_hash = hashlib.file_digest(f, 'sha256').hexdigest()
        record = dict(original=manifest, compression='gzip level 3', compressed_bytes=dst.stat().st_size,
                      compressed_sha256=compressed_hash, round_trip_sha256=restored)
        dst.with_suffix(dst.suffix+'.manifest.json').write_text(json.dumps(record,indent=2))
        src.unlink()  # exact task-owned path, only after verified lossless round trip
        print(json.dumps(record), flush=True)

if __name__ == '__main__':
    main()
