import contextlib
import io
import json
import unittest
from unittest.mock import patch
import numpy as np
import test_factorial_runner as fixtures
from prescience_expanded import make_view
from fit_expanded_models import fit_models
from fit_factorial import fit,read,load_partition
from frozen_ranker import FrozenRanker,model_spec
from reranking_cost import measure,require_idle


class FrozenInferenceTests(unittest.TestCase):
    def test_saved_families_and_factorial_schema_roundtrip(self):
        fixture=fixtures.FactorialTests();fixture.setUp()
        try:
            source=fixture.source
            for path in (source/'valid_queries').glob('*.json'):
                row=read(path);row['standalone']={'rrf_standalone':row['candidates']};path.write_text(json.dumps(row))
            ids=sorted(p.stem for p in (source/'train_queries').glob('*.json'))
            valid=sorted(p.stem for p in (source/'valid_queries').glob('*.json'))
            plan=dict(train_ids_by_seed={'42':ids},validation=[dict(qid=q) for q in valid])
            view=source/'inference-view';make_view(source,plan,42,view,dict(mode='full'))
            with contextlib.redirect_stdout(io.StringIO()):
                report=fit_models(view,grid=[(3,4)],logistic_cs=(1.,))
                factorial=fit(view,grid=[(3,4)])
            features=report['features'];x,_,groups,metadata,_=load_partition(view,'valid',features)
            for folder,names in [(view,report['models']), (view/'factorial-s42',factorial['cells'])]:
                for name in names:
                    spec=model_spec(folder,name);model=FrozenRanker(spec,features);scores=model.predict(x)
                    # Input columns may be reordered only with an exact corresponding schema.
                    permutation=list(reversed(range(len(features))))
                    other=FrozenRanker(spec,[features[i] for i in permutation])
                    np.testing.assert_allclose(scores,other.predict(x[:,permutation]),rtol=1e-12,atol=1e-12)
                    expected=read(folder/(name+'.predictions.json'));offset=0
                    for size,meta,row in zip(groups,metadata,expected):
                        ranked=sorted(range(size),key=lambda i:(-scores[offset+i],meta['candidates'][i]))
                        self.assertEqual([meta['candidates'][i] for i in ranked],row['predicted']);offset+=size
            spec=model_spec(view,'lambdamart')
            with self.assertRaisesRegex(ValueError,'Missing or duplicate'):FrozenRanker(spec,features[:-1])
            spec['sha256']='bad'
            with self.assertRaisesRegex(ValueError,'Frozen model changed'):FrozenRanker(spec,features)
        finally:fixture.tearDown()

    def test_timing_measures_fixed_query_set_without_labels(self):
        class Predictor:
            def predict(self,matrix):return matrix[:,0]
        result=measure(Predictor(),[np.ones((3,2)),np.zeros((5,2))],3)
        self.assertEqual(result['query_count'],2)
        self.assertEqual(result['candidate_rows'],[3,5])
        self.assertEqual(np.asarray(result['per_repeat_query_seconds']).shape,(3,2))

    def test_active_worker_rejects_isolated_cost_claim(self):
        class Worker:
            pid=-1
            info=dict(name='python.exe',cmdline=['python.exe','semantic_encode.py'])
        with patch('reranking_cost.psutil.process_iter',return_value=[Worker()]):
            with self.assertRaisesRegex(RuntimeError,'busy workers'):require_idle()


if __name__=='__main__':unittest.main()
