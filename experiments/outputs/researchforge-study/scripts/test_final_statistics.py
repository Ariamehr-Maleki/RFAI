import unittest
import numpy as np
from final_statistics import holm,seed_average_pairs,cluster_inference

class StatisticsTests(unittest.TestCase):
    def test_known_holm_family(self):
        actual=holm(dict(a=.01,b=.04,c=.03))
        np.testing.assert_allclose([actual[k] for k in ['a','b','c']],[.03,.06,.06])
        with self.assertRaises(ValueError):holm(dict(a=float('nan')))
    def test_seed_average_does_not_triple_sample_count(self):
        primary={};baseline={}
        for seed in [42,43,44]:
            base=[dict(qid=str(i),seed=f'a{i//2}',truth=['yes'],new_truth=[],score=.1) for i in range(6)]
            baseline[seed]=base;primary[seed]=[dict(r,score=.3) for r in base]
        pairs=seed_average_pairs(primary,baseline,'score')
        self.assertEqual(pairs['eligible_queries'],6)
        result=cluster_inference(pairs['differences'],pairs['authors'],repetitions=200)
        self.assertEqual(result['clusters'],3)
        self.assertAlmostEqual(result['delta'],.2)
        primary[44]=primary[44][:-1]
        with self.assertRaises(ValueError):seed_average_pairs(primary,baseline,'score')
    def test_identical_predictions_and_reproducibility(self):
        x=cluster_inference([0.,0.,0.],['a','b','c'],repetitions=200)
        self.assertEqual(x['ci95'],[0.,0.]);self.assertEqual(x['p_two_sided_cluster_signflip'],1.)
        y=cluster_inference([.1,-.2,.3],['a','b','c'],repetitions=200)
        self.assertEqual(y,cluster_inference([.1,-.2,.3],['a','b','c'],repetitions=200))
    def test_cross_seed_label_mutation_rejected(self):
        a={seed:[dict(qid='q',seed='author',truth=['a'],score=.2)] for seed in [42,43,44]}
        b={seed:[dict(qid='q',seed='author',truth=['a'],score=.1)] for seed in [42,43,44]}
        a[44][0]['truth']=['different'];b[44][0]['truth']=['different']
        with self.assertRaisesRegex(ValueError,'Relevance changes'):seed_average_pairs(a,b,'score')

if __name__=='__main__':unittest.main()
