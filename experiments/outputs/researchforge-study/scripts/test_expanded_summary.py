"""Synthetic contract tests for the shared-query expanded summary."""
import json
from pathlib import Path
import tempfile
import unittest
from summarize_expanded import summarize, SEEDS


class SummaryTests(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[3]/'work'
        self.temp=tempfile.TemporaryDirectory(prefix='summary-test-',dir=root)
        self.base=Path(self.temp.name)
        def save(path,value):
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value))
        self.save=save
        def rows(score):
            return [dict(qid=str(i),seed='author'+str(i//2),truth=['x'],new_truth=['x'],ndcg1000=score)
                    for i in range(6)]
        for mode in ['full','no_content']:
            for seed in SEEDS:
                folder=self.base/mode/f'seed-{seed}'
                score=(.6 if mode=='full' else .4)+.01*(seed-42)
                save(folder/'report.json',dict(status='Development fixture',metrics={
                    'lambdamart':dict(ndcg1000=score), 'binary_lightgbm':dict(ndcg1000=score-.05)}))
                save(folder/'independent_metric_verification.json',dict(status='PASS'))
                save(folder/'lambdamart.predictions.json',rows(score))
                save(folder/'binary_lightgbm.predictions.json',rows(score-.05))
                if mode=='full':
                    factorial=folder/f'factorial-s{seed}'
                    save(factorial/'independent_metric_verification.json',dict(status='PASS'))
                    for name in ['text+network+explicit_time','network+explicit_time','text+explicit_time','text+network']:
                        save(factorial/(name+'.predictions.json'),rows(.55 if name=='text+network+explicit_time' else .5))
    def tearDown(self):
        self.temp.cleanup()
    def test_seed_averaging_and_exploratory_family(self):
        result=summarize(self.base)
        self.assertEqual(result['validation_queries'],6)
        self.assertEqual(len(result['comparisons']),5)
        self.assertAlmostEqual(result['seed_summary']['full']['lambdamart']['ndcg1000']['sample_sd'],.01)
        comparison=result['comparisons']['full_pipeline_minus_no_content_pipeline']
        self.assertAlmostEqual(comparison['delta'],.2)
        self.assertEqual(comparison['queries'],6)
        self.assertEqual(comparison['clusters'],3)
    def test_changed_validation_queries_rejected(self):
        path=self.base/'no_content/seed-44/lambdamart.predictions.json'
        rows=json.loads(path.read_text());rows[0]['qid']='changed';self.save(path,rows)
        with self.assertRaisesRegex(ValueError,'Validation queries differ'):summarize(self.base)
    def test_changed_labels_rejected(self):
        path=self.base/'no_content/seed-44/lambdamart.predictions.json'
        rows=json.loads(path.read_text());rows[0]['truth']=['different'];self.save(path,rows)
        with self.assertRaisesRegex(ValueError,'Relevance sets differ'):summarize(self.base)
    def test_unverified_input_rejected(self):
        self.save(self.base/'full/seed-42/independent_metric_verification.json',dict(status='FAIL'))
        with self.assertRaisesRegex(ValueError,'unverified'):summarize(self.base)


if __name__=='__main__':unittest.main()
