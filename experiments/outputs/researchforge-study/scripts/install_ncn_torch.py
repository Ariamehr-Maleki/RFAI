"""Extract verified Torch wheel into task-only NTFS-compressed isolated runtime.

No changes to active encoder environment. Static development libraries (.lib)
are omitted: only binary-wheel inference/training is supported, not C++ builds.
Disk guard
reserves 1 GiB plus all remaining embedding bytes. This is a wheel extraction,
not a global Python installation. Run NCN with the explicit runtime sys.path.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[3]
DEST = ROOT / 'work/ncn-runtime'
WHEEL = ROOT / 'work/ncn-wheels/torch-2.5.1-cp312-cp312-win_amd64.whl'


def reserve():
    p = json.loads((ROOT/'work/indices/prescience-minilm/progress.json').read_text())
    return 1024**3 + (p['total_rows'] - p['completed_rows']) * 384 * 2


def main():
    manifest = json.loads(WHEEL.with_suffix('.whl.manifest.json').read_text())
    with WHEEL.open('rb') as f:
        assert hashlib.file_digest(f, 'sha256').hexdigest() == manifest['sha256']
    DEST.mkdir(exist_ok=True)
    assert DEST.resolve().is_relative_to((ROOT/'work').resolve())
    subprocess.run(['compact.exe', '/C', str(DEST)], check=True)
    before = shutil.disk_usage(DEST).free
    records = []
    omitted = []
    with zipfile.ZipFile(WHEEL) as z:
        for info in z.infolist():
            if info.filename.endswith('.lib'):
                omitted.append(dict(path=info.filename, bytes=info.file_size, reason='Static link library for source compilation; binary runtime only'))
                continue
            target = DEST / info.filename
            assert target.resolve().is_relative_to(DEST.resolve())
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                crc = 0
                existing_hash = hashlib.sha256()
                with target.open('rb') as existing:
                    while block := existing.read(4 * 1024**2):
                        crc = zlib.crc32(block, crc)
                        existing_hash.update(block)
                if crc == info.CRC:
                    records.append(dict(path=info.filename, bytes=info.file_size, sha256=existing_hash.hexdigest()))
                    continue
            digest = hashlib.sha256()
            with z.open(info) as source, target.open('wb') as out:
                while block := source.read(4 * 1024**2):
                    if shutil.disk_usage(DEST).free - len(block) < reserve():
                        raise OSError('Disk reserve reached; partial install is not usable')
                    out.write(block)
                    out.flush()
                    digest.update(block)
            records.append(dict(path=info.filename, bytes=info.file_size, sha256=digest.hexdigest()))
    record = dict(status='runtime files extracted; import/smoke validation pending',
                  wheel_sha256=manifest['sha256'], uncompressed_bytes=manifest['uncompressed_bytes'],
                  free_before=before, free_after=shutil.disk_usage(DEST).free,
                  compression='Inherited NTFS compression', omitted_static_libraries=omitted, files=records)
    (DEST/'torch-extraction-manifest.json').write_text(json.dumps(record, indent=2))
    print(json.dumps({k:v for k,v in record.items() if k != 'files'}), flush=True)


if __name__ == '__main__':
    main()
