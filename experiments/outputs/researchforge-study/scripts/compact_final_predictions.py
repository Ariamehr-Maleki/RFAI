"""Transparent NTFS compression of completed report-bound prediction JSON files."""
import json
from pathlib import Path
import shutil
import subprocess
import time
from compact_final_checkpoints import ROOT, STUDY, sha, allocated


def main():
    folder = STUDY / 'results/final/prescience'
    report_path = folder / 'report.json'
    report_hash = sha(report_path)
    report = json.loads(report_path.read_text())
    if report['status'] != 'Completed frozen PreScience evaluation':
        raise ValueError('Completed report required')
    targets = [(name, value) for name, value in report['output_sha256'].items()
               if name.endswith('.predictions.json')]
    audit = STUDY / 'data-audit' / f'ntfs-final-predictions-{time.time_ns()}.jsonl'
    saved = 0
    with audit.open('x', encoding='utf-8') as stream:
        for name, expected in targets:
            path = (folder / name).resolve()
            if path.parent != folder.resolve() or sha(path) != expected:
                raise ValueError('Unverified prediction: ' + name)
            before = allocated(path)
            subprocess.run(['compact.exe', '/C', '/I', '/Q', str(path)],
                           check=True, capture_output=True, timeout=60)
            if sha(path) != expected:
                raise ValueError('Prediction bytes changed: ' + name)
            after = allocated(path)
            saved += before - after
            stream.write(json.dumps(dict(file=name, sha256=expected,
                         allocated_before=before, allocated_after=after)) + '\n')
            stream.flush()
        if sha(report_path) != report_hash:
            raise ValueError('Report changed during storage operation')
        result = dict(status='PASS', files=len(targets), saved_bytes=saved,
                      report_sha256=report_hash, free_disk_bytes=shutil.disk_usage(ROOT).free)
        stream.write(json.dumps(result) + '\n')
    print(json.dumps(dict(audit=str(audit), **result)), flush=True)


if __name__ == '__main__':
    main()
