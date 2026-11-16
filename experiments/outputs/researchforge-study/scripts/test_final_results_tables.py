import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from final_results_tables import ROOT,STATUSES,checked_report,table_rows,csv_text
from final_evaluation_gate import canonical
from frozen_ranker import digest


class TableTests(unittest.TestCase):
    def test_negative_results_retained_and_metrics_not_pooled(self):
        contrast=dict(delta=-.02,ci95=[-.03,-.01],holm_p=.01,queries=30,clusters=20)
        cell=dict(mean=.1,sample_sd=.02)
        reports={
            'prescience':dict(seed_summary={'ranker':{'ndcg1000':cell,'new_recall10':None}},primary_comparisons={'ranker__minus__baseline':contrast}),
            'ogbl-collab':dict(seed_summary={'ranker':cell},primary_comparisons={}),
            'exHarmony':dict(seed_summary={'authors':{'ranker':{'ndcg10':cell}},'cite':{'ranker':{'ndcg10':cell}}},primary_comparisons={})}
        rows,pairs=table_rows(reports)
        self.assertEqual(len(rows),4);self.assertEqual(len(pairs),1)
        self.assertEqual(pairs[0]['delta'],-.02);self.assertIn('-0.02',csv_text(pairs))
        self.assertEqual({r['metric'] for r in rows},{'ndcg1000','hits50','ndcg10'})
    def test_development_report_and_changed_prediction_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temporary:
            folder=Path(temporary);cfg=dict(output_directory=str(folder.relative_to(ROOT)))
            lock=dict(lock_sha256='fixture',payload=dict(plan=dict(datasets={'ogbl-collab':cfg})))
            data=folder/'prediction.json';data.write_text('[0.1]')
            report=dict(status=STATUSES['ogbl-collab'],lock_sha256='fixture',config=cfg,
                config_sha256=hashlib.sha256(canonical(cfg)).hexdigest(),output_sha256={data.name:digest(data)})
            path=folder/'report.json';path.write_text(json.dumps(report))
            self.assertEqual(checked_report(ROOT,lock,'ogbl-collab')[1],report)
            data.write_text('[0.2]')
            with self.assertRaisesRegex(ValueError,'Changed final output'):checked_report(ROOT,lock,'ogbl-collab')
            report['status']='development validation';path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'Wrong final report'):checked_report(ROOT,lock,'ogbl-collab')


if __name__=='__main__':unittest.main()
