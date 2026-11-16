"""Report integrity checks use disposable synthetic files, never real test scores."""
from pathlib import Path
import tempfile
import unittest
from build_final_professor_report import ROOT, STATUSES, verify_export_sources, prerequisites
from frozen_ranker import digest


class ProfessorReportIntegrity(unittest.TestCase):
    def test_stale_or_incomplete_export_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temporary:
            root = Path(temporary).resolve()
            records = {}
            for dataset in STATUSES:
                path = root / (dataset + '.json')
                path.write_text('synthetic fixture', encoding='utf-8')
                records[dataset] = dict(path=path.name, sha256=digest(path))
            expected = {v['path']: v['sha256'] for v in records.values()}
            table = dict(source_reports=records)
            figures = dict(source_sha256=dict(expected))
            verify_export_sources(root, table, 'tables', expected)
            verify_export_sources(root, figures, 'figures', expected)
            figures['source_sha256'].pop('prescience.json')
            with self.assertRaisesRegex(ValueError, 'does not bind current report'):
                verify_export_sources(root, figures, 'figures', expected)
            (root / 'prescience.json').write_text('revised synthetic fixture', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Export source differs'):
                verify_export_sources(root, table, 'tables', expected)

    def test_missing_final_evidence_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as temporary:
            with self.assertRaisesRegex(RuntimeError, 'requires completed evidence'):
                prerequisites(Path(temporary))


if __name__ == '__main__':
    unittest.main()
