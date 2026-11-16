"""Verified progress snapshot of completed final datasets, not a full export."""
import json
from pathlib import Path
from final_evaluation_gate import verify_lock
from final_results_tables import checked_report, table_rows, csv_text, STATUSES
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'outputs/researchforge-study'


def main():
    lock = verify_lock(ROOT, STUDY / 'final/global-lock.json')
    reports, sources, pending = {}, {}, []
    for dataset in STATUSES:
        cfg = lock['payload']['plan']['datasets'][dataset]
        if not (ROOT / cfg['output_directory'] / 'report.json').exists():
            pending.append(dataset)
            continue
        path, report = checked_report(ROOT, lock, dataset)
        reports[dataset] = report
        sources[dataset] = dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                                checked_output_files=len(report['output_sha256']))
    if not reports:
        raise ValueError('No completed final dataset')
    out = STUDY / 'results/final/completed-snapshot'
    out.mkdir(exist_ok=True)
    performance, contrasts = table_rows(reports)
    (out / 'performance.csv').write_text(csv_text(performance), encoding='utf-8')
    (out / 'primary_contrasts.csv').write_text(csv_text(contrasts), encoding='utf-8')
    atomic_state(out / 'provenance.json', dict(
        status='Verified completed-dataset snapshot; not the full study deliverable',
        lock_sha256=lock['lock_sha256'], sources=sources, pending_datasets=pending,
        script_sha256=digest(__file__),
        files={name:digest(out / name) for name in ['performance.csv', 'primary_contrasts.csv']}))
    print(json.dumps(dict(completed=list(reports), pending=pending, sources=sources), indent=2))


if __name__ == '__main__':
    main()
