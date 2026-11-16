import contextlib
import io
import json
import unittest
import numpy as np
from prescience_expanded import query_plan,make_view,collect_cache
from prescience_pilot import day
from prescience_retrieval_ablation import NoContentEngine,model_columns
from prescience_hybrid_graph import FEATURES
from publication_ablation_design import COMPONENTS
from fit_expanded_models import fit_models
from fit_factorial import read,sha
from test_factorial_runner import FactorialTests
from test_hybrid_graph import HybridGraphTests

class ExpandedTests(unittest.TestCase):
    def test_manifest_has_common_validation_and_recorded_training_subsets(self):
        queries=[dict(qid=str(i),date=day('2024-01-01')+i,seed='a',truth=['b']) for i in range(30)]
        queries += [dict(qid=f'v{i}',date=day('2024-08-01')+i,seed='a',truth=['b']) for i in range(10)]
        plan=query_plan(queries,20,15,5)
        self.assertEqual(plan,query_plan(queries,20,15,5))
        self.assertEqual(len(plan['validation']),5)
        self.assertEqual(len(set(plan['train_ids_by_seed']['42'])),15)
        self.assertNotEqual(plan['train_ids_by_seed']['42'],plan['train_ids_by_seed']['43'])
        self.assertFalse({q['qid'] for q in plan['train_pool']}&{q['qid'] for q in plan['validation']})
    def test_content_removed_from_candidates_and_features(self):
        fixture=HybridGraphTests();fixture.setUp()
        try:
            engine=NoContentEngine(fixture.papers,fixture.h,fixture.d)
            q=dict(qid='q',date=5,seed='a',truth=['b'])
            x,y,m=engine.run(q,3,42,False)
            for p in fixture.papers.values():p['categories']=('arbitrary changed content',)
            xx,yy,mm=engine.run(dict(q,truth=['d']),3,42,False)
            np.testing.assert_array_equal(x,xx);self.assertEqual(m['candidates'],mm['candidates'])
            self.assertFalse(np.array_equal(y,yy));self.assertNotIn('z',m['candidates'])
            self.assertNotIn('semantic_standalone',m['standalone'])
            text=[i for i,f in enumerate(FEATURES) if f in COMPONENTS['text']]
            self.assertTrue(np.all(x[:,text]==0));self.assertFalse(set(text)&set(model_columns('no_content')))
        finally:fixture.tearDown()
    def test_hardlink_views_and_all_model_roundtrips(self):
        fixture=FactorialTests();fixture.setUp()
        try:
            source=fixture.source
            for path in (source/'valid_queries').glob('*.json'):
                meta=read(path);meta['standalone']={'rrf_standalone':meta['candidates']};path.write_text(json.dumps(meta))
            train_ids=sorted(p.stem for p in (source/'train_queries').glob('*.json'))
            valid_ids=sorted(p.stem for p in (source/'valid_queries').glob('*.json'))
            plan=dict(train_ids_by_seed={'42':train_ids},validation=[dict(qid=q) for q in valid_ids])
            view=source/'view';make_view(source,plan,42,view,dict(mode='full'))
            self.assertEqual(sha(view/'train_queries'/f'{train_ids[0]}.npz'),sha(source/'train_queries'/f'{train_ids[0]}.npz'))
            with contextlib.redirect_stdout(io.StringIO()):
                report=fit_models(view,grid=[(3,4)],logistic_cs=(1.,))
                replay=fit_models(view,grid=[(3,4)],logistic_cs=(1.,))
            self.assertEqual(report['metrics'],replay['metrics'])
            self.assertEqual(set(report['models']),{'lambdamart','binary_lightgbm','logistic'})
            for cell in report['models'].values():self.assertLess(cell['roundtrip_max_error'],1e-10)
            self.assertTrue((view/'logistic.joblib').exists())
        finally:fixture.tearDown()

if __name__=='__main__':unittest.main()
