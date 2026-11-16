"""Synthetic gate fixtures only; never touch real test data or create real locks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from final_evaluation_gate import DATASETS,ROLES,freeze_plan,authorize_evaluation,verify_lock


class GateTests(unittest.TestCase):
    def setUp(self):
        work=Path(__file__).resolve().parents[3]/'work'
        self.tmp=tempfile.TemporaryDirectory(dir=work)
        self.root=Path(self.tmp.name);self.lock=self.root/'lock.json';self.ledger=self.root/'ledger'
        artifacts=[]
        for role in ROLES:
            name=role+'.json';(self.root/name).write_text(json.dumps({'status':'complete synthetic fixture','role':role,'metrics':{'synthetic':0.}}))
            artifacts.append(dict(role=role,path=name))
        self.plan=dict(schema_version=1,stage='ready_for_final_test',datasets={d:dict(
            primary_metric=m,seeds=[42,43,44],methods={'ranker':{'trees':100},'baseline':{'mode':'fixed'}},
            candidate_policy='Synthetic shared pool',training_policy='Synthetic development only',
            test_split='Synthetic fixture',uncertainty_policy='Synthetic paired clusters',multiplicity='Holm',
            primary_comparisons=[['ranker','baseline']],artifacts=copy.deepcopy(artifacts)) for d,m in DATASETS.items()})

    def tearDown(self):self.tmp.cleanup()

    def test_missing_or_draft_plan_never_creates_lock(self):
        with self.assertRaises(FileNotFoundError):authorize_evaluation(self.root,self.lock,'prescience','run1',self.ledger)
        self.plan['stage']='development'
        with self.assertRaises(ValueError):freeze_plan(self.root,self.plan,self.lock)
        self.assertFalse(self.lock.exists())

    def test_missing_artifact_rejected_before_lock(self):
        (self.root/'model.json').unlink()
        with self.assertRaises(ValueError):freeze_plan(self.root,self.plan,self.lock)
        self.assertFalse(self.lock.exists())

    def test_modified_code_rejected_before_test_authorization(self):
        freeze_plan(self.root,self.plan,self.lock)
        (self.root/'code.json').write_text('tampered')
        with self.assertRaises(ValueError):authorize_evaluation(self.root,self.lock,'prescience','run1',self.ledger)
        self.assertFalse(self.ledger.exists())

    def test_running_report_cannot_finalize(self):
        (self.root/'development_report.json').write_text(json.dumps({'status':'running','metrics':{'synthetic':1.}}))
        with self.assertRaises(ValueError):freeze_plan(self.root,self.plan,self.lock)
        self.assertFalse(self.lock.exists())

    def test_complete_ncn_schema_accepted_but_partial_history_rejected(self):
        report=dict(status='Expanded temporal CPU DEVELOPMENT; not final test',config=dict(epochs=2),
                    best_validation_hits50=.6,history=[dict(epoch=1,validation_hits50=.5),dict(epoch=2,validation_hits50=.6)])
        path=self.root/'development_report.json';path.write_text(json.dumps(report))
        report['history'].pop();path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'not a completed'):freeze_plan(self.root,self.plan,self.lock)
        report['history'].append(dict(epoch=2,validation_hits50=.6));path.write_text(json.dumps(report))
        freeze_plan(self.root,self.plan,self.lock)
        self.assertTrue(self.lock.exists())

    def test_ncn_best_score_must_match_complete_history(self):
        from final_evaluation_gate import has_completed_scores
        report=dict(config=dict(epochs=1),best_validation_hits50=.7,history=[dict(epoch=1,validation_hits50=.6)])
        self.assertFalse(has_completed_scores(report))
        report['best_validation_hits50']=float('nan');self.assertFalse(has_completed_scores(report))

    def test_immutable_lock_and_only_identical_run_resume(self):
        freeze_plan(self.root,self.plan,self.lock)
        with self.assertRaises(FileExistsError):freeze_plan(self.root,self.plan,self.lock)
        first=authorize_evaluation(self.root,self.lock,'prescience','run1',self.ledger)
        self.assertEqual(first,authorize_evaluation(self.root,self.lock,'prescience','run1',self.ledger))
        with self.assertRaises(ValueError):authorize_evaluation(self.root,self.lock,'prescience','run2',self.ledger)

    def test_changed_lock_and_invalid_comparison_rejected(self):
        self.plan['datasets']['prescience']['primary_comparisons']=[['ranker','unknown']]
        with self.assertRaises(ValueError):freeze_plan(self.root,self.plan,self.lock)
        self.plan['datasets']['prescience']['primary_comparisons']=[['ranker','baseline']]
        freeze_plan(self.root,self.plan,self.lock)
        obj=json.loads(self.lock.read_text());obj['payload']['plan']['datasets']['prescience']['seeds']=[9,10,11]
        self.lock.write_text(json.dumps(obj))
        with self.assertRaises(ValueError):verify_lock(self.root,self.lock)


if __name__=='__main__':unittest.main()
