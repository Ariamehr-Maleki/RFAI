"""NTFS compression of committed query JSON; paths and logical bytes unchanged.

No scores are inspected. Every target is bound to an existing receipt and its
SHA256 is checked before and after compression. No artifact is deleted.
"""
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'outputs/researchforge-study'
QUERIES = STUDY / 'results/final/prescience/queries'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def allocated(path):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    function = kernel.GetCompressedFileSizeW
    function.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.DWORD)]
    function.restype = wintypes.DWORD
    high = wintypes.DWORD()
    ctypes.set_last_error(0)
    low = function(str(path), ctypes.byref(high))
    if low == 0xFFFFFFFF and ctypes.get_last_error():
        raise ctypes.WinError(ctypes.get_last_error())
    return (high.value << 32) | low


def main():
    if os.name != 'nt':
        raise RuntimeError('NTFS compression requires Windows')
    audit = STUDY / 'data-audit' / f'ntfs-checkpoints-{time.time_ns()}.jsonl'
    disk_before = shutil.disk_usage(ROOT).free
    total_before = total_after = count = 0
    # Snapshot only receipts already committed. Later checkpoints are untouched.
    receipts = sorted(QUERIES.glob('*.receipt'))
    with audit.open('x', encoding='utf-8') as out:
        for receipt in receipts:
            manifest = json.loads(receipt.read_text(encoding='utf-8'))
            name = receipt.stem + '.json'
            expected = manifest[name]
            path = (QUERIES / name).resolve()
            if path.parent != QUERIES.resolve():
                raise RuntimeError('Target outside query directory')
            if sha(path) != expected:
                raise RuntimeError(f'Pre-compression receipt mismatch: {path.name}')
            before = allocated(path)
            result = subprocess.run(['compact.exe', '/C', '/I', '/Q', str(path)],
                                    capture_output=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f'compact failed: {path.name}: {result.stderr!r}')
            if sha(path) != expected:
                raise RuntimeError(f'Post-compression receipt mismatch: {path.name}')
            after = allocated(path)
            out.write(json.dumps(dict(file=name, sha256=expected,
                       allocated_before=before, allocated_after=after,
                       logical_bytes=path.stat().st_size, verified=True)) + '\n')
            out.flush()
            total_before += before
            total_after += after
            count += 1
            if count % 100 == 0:
                print(f'{count}/{len(receipts)} committed files verified', flush=True)
        summary = dict(status='PASS', files=count, allocated_before=total_before,
                       allocated_after=total_after, saved_bytes=total_before-total_after,
                       free_disk_before=disk_before, free_disk_after=shutil.disk_usage(ROOT).free,
                       content_and_paths='unchanged; receipt SHA256 verified before and after')
        out.write(json.dumps(summary) + '\n')
    print(json.dumps(dict(audit=str(audit), **summary)), flush=True)


if __name__ == '__main__':
    main()
