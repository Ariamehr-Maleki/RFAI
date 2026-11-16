"""Download public benchmark artifacts with checksums and atomic completion."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'work' / 'data'
SOURCES = {
    'prescience': {
        name: f'https://huggingface.co/datasets/allenai/prescience/resolve/main/{name}'
        for name in ['README.md', 'author_publications.jsonl', 'author_disambiguation.jsonl', 'train.parquet', 'test.parquet']
    },
    'ogbl-collab': {'collab.zip': 'https://snap.stanford.edu/ogb/data/linkproppred/collab.zip'},
    'exHarmony': {
        name: f'https://huggingface.co/datasets/Reviewerly/exHarmony/resolve/main/{name}'
        for name in ['README.md', 'qrels/qrels.test.authors.tsv', 'qrels/qrels.test.cite.tsv',
                     'qrels/qrels.test.simcite.tsv', 'papers_test.jsonl', 'papers_test_judgeable.jsonl',
                     'authors_works_collection_ids.jsonl', 'authors_info.jsonl', 'papers_collection.jsonl']
    },
}

def fetch(item):
    dataset, name, url = item
    path = DATA / dataset / name
    path.parent.mkdir(parents=True, exist_ok=True)
    extracted_record = ROOT / 'outputs/researchforge-study/data-audit/ogb-archive-compaction.json'
    if dataset == 'ogbl-collab' and name == 'collab.zip' and not path.exists() and extracted_record.exists():
        # Verify raw bytes only, including the sealed test file. Never parse its
        # tensors/labels here. A verified extracted corpus replaces redundant ZIP.
        record = json.loads(extracted_record.read_text())
        for entry in record['files']:
            target = path.parent / 'extracted' / entry['path']
            assert target.resolve().is_relative_to((path.parent / 'extracted').resolve())
            assert target.stat().st_size == entry['bytes']
            with target.open('rb') as f:
                assert hashlib.file_digest(f, 'sha256').hexdigest() == entry['sha256']
        print('Using verified extracted OGB corpus; redundant ZIP was removed', flush=True)
        return record
    compressed = path.with_suffix(path.suffix+'.gz')
    compressed_manifest = compressed.with_suffix(compressed.suffix+'.manifest.json')
    if not path.exists() and compressed.exists() and compressed_manifest.exists():
        record = json.loads(compressed_manifest.read_text())
        with compressed.open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest() == record['compressed_sha256']
        print(f'Using verified compressed copy: {compressed}', flush=True)
        return record['original']
    if not path.exists():
        partial = path.with_suffix(path.suffix+'.part')
        for attempt in range(8):
            try:
                start = partial.stat().st_size if partial.exists() else 0
                headers={'User-Agent': 'ResearchForge-reproducible-study/0.1'}
                if start:
                    headers['Range']=f'bytes={start}-'
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as r:
                    append = start > 0 and r.status == 206
                    if append and not r.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                        raise IOError('Server returned an unexpected byte range')
                    expected = r.headers.get('Content-Length')
                    count, progress = 0, 0
                    with partial.open('ab' if append else 'wb') as f:
                        while chunk := r.read1(256 * 1024):
                            if count % (16*1024*1024) < 256*1024 and shutil.disk_usage(path.parent).free < 1024**3:
                                raise OSError('Less than 1 GiB free; stopping this download before exhausting the drive')
                            f.write(chunk)
                            count += len(chunk)
                            if count-progress >= 16*1024*1024:
                                print(f'{dataset}/{name}: {(start if append else 0)+count:,} bytes',flush=True)
                                progress=count
                        if expected and count != int(expected):
                            raise IOError(f'Truncated download: {count}/{expected}')
                partial.replace(path)
                break
            except Exception as e:
                print(f'{dataset}/{name}: attempt {attempt+1}: {e}', flush=True)
                if attempt == 7:
                    raise
                time.sleep(min(2 ** attempt,30))
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    result = dict(dataset=dataset, filename=name, source=url, bytes=path.stat().st_size, sha256=digest)
    print(json.dumps(result), flush=True)
    path.with_suffix(path.suffix+'.manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=SOURCES)
    args = parser.parse_args()
    items = [(d, n, u) for d in args.datasets for n,u in SOURCES[d].items()]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(fetch, items))
