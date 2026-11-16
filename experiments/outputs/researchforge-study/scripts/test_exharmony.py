import unittest
from unittest.mock import patch
import tempfile
from pathlib import Path
import numpy as np
from scipy import sparse
import exharmony_pilot as e
import exharmony_bm25_pilot as bm
import pytrec_eval

class ExHarmonyTests(unittest.TestCase):
    def test_metrics_against_official_trec_library(self):
        rng=np.random.default_rng(19)
        for _ in range(30):
            pred=rng.permutation(300)[:100].tolist()
            truth=set(rng.choice(400,int(rng.integers(1,250)),replace=False).tolist())
            qrels={'q':{str(a):1 for a in truth}}
            run={'q':{str(a):float(100-i) for i,a in enumerate(pred)}}
            requested={f'{m}.{k}' for m in ['ndcg_cut','map_cut','recall'] for k in [10,20,100]}
            result=pytrec_eval.RelevanceEvaluator(qrels,requested).evaluate(run)['q']
            ours=e.metrics(pred,truth)
            for k in [10,20,100]:
                for a,b in [('ndcg','ndcg_cut'),('map','map_cut'),('recall','recall')]:
                    self.assertAlmostEqual(ours[f'{a}{k}'],result[f'{b}_{k}'])
                cut={'q':{str(a):float(100-i) for i,a in enumerate(pred[:k])}}
                rr=pytrec_eval.RelevanceEvaluator(qrels,{'recip_rank'}).evaluate(cut)['q']['recip_rank']
                self.assertAlmostEqual(ours[f'rr{k}'],rr)
    def test_map_cut_denominator_is_all_relevant(self):
        # Matches NIST trec_eval m_map_cut.c: num_rel, not min(num_rel,k).
        m=e.metrics(list(range(100)),set(range(200)))
        self.assertAlmostEqual(m['map10'],.05)
        self.assertAlmostEqual(m['ndcg10'],1)
        self.assertAlmostEqual(m['recall100'],.5)
        m=e.metrics([7,3,9],{3,9})
        self.assertAlmostEqual(m['map10'],(.5+2/3)/2)
        self.assertAlmostEqual(m['rr10'],.5)
    def test_qrels_reader_filters_test_before_relevance_parse(self):
        with tempfile.TemporaryDirectory(dir=e.ROOT/'work') as folder:
            root=Path(folder);(root/'qrels').mkdir()
            (root/'qrels/qrels.test.authors.tsv').write_text('dev 0 https://openalex.org/A123 1\ntest deliberately invalid protected fields\n')
            for module in [e,bm]:
                with patch.object(module,'DATA',root):
                    self.assertEqual(dict(module.load_qrels('authors',{'dev'})),{'dev':{123}})
    def test_validation_candidates_and_features_do_not_use_truth(self):
        r=object.__new__(e.Retriever)
        r.docs=sparse.csr_matrix([[1.,0],[.8,.6],[0,1]],dtype=np.float32)
        r.pa=sparse.csr_matrix([[1.,0,0],[0,1.,0],[0,1.,1]],dtype=np.float32)
        r.ap=r.pa.T.tocsr();r.ids=np.array([11,22,33]);r.order=np.argsort(r.ids);r.sorted_ids=r.ids[r.order]
        r.counts=np.diff(r.ap.indptr)
        class V:
            def transform(self,_):return sparse.csr_matrix([[1.,0]],dtype=np.float32)
        r.vectorizer=V()
        x,y,m=r.query('q',{11},2,False)
        xx,yy,mm=r.query('q',{22},2,False)
        np.testing.assert_array_equal(x,xx)
        self.assertEqual(m['candidates'],mm['candidates'])
        self.assertFalse(np.array_equal(y,yy))

if __name__=='__main__':unittest.main()
