import copy
import unittest
from final_subgroups import compare


class FinalSubgroupsTests(unittest.TestCase):
    def fixture(self):
        baseline=[dict(qid=str(i),seed=str(i),seed_history_count=n,truth=[1,2],new_truth=[1] if i!=2 else [],predicted=[8,9]) for i,n in enumerate([1,5,12])]
        a={s:copy.deepcopy(baseline) for s in [42,43,44]};b=copy.deepcopy(a)
        a[42][0]['predicted']=[1,2];a[43][0]['predicted']=[1,2]
        for s in a:a[s][1]['predicted']=[1,2]
        return a,b
    def test_seed_average_not_query_pool_and_direct_interaction(self):
        a,b=self.fixture();r=compare(a,b,repetitions=100)
        self.assertEqual(r['query_count'],3)
        self.assertEqual(r['groups']['new_recall10:all']['queries'],2)
        self.assertEqual(r['groups']['repeat_recall10:all']['queries'],3)
        self.assertAlmostEqual(r['groups']['ndcg1000:1-3']['delta'],2/3)
        self.assertAlmostEqual(r['interactions']['ndcg1000:1-3_minus_4-10']['difference_of_deltas'],-1/3)
        self.assertEqual(r['groups']['ndcg1000:all']['author_clusters'],3)
        self.assertTrue(r['groups']['ndcg1000:1-3']['sparse_group'])
    def test_changed_history_or_truth_rejected(self):
        a,b=self.fixture();a[43][0]['seed_history_count']=8
        with self.assertRaisesRegex(ValueError,'metadata'):compare(a,b,repetitions=100)
        a,b=self.fixture();a[44][0]['new_truth']=[]
        with self.assertRaisesRegex(ValueError,'metadata'):compare(a,b,repetitions=100)


if __name__=='__main__':unittest.main()
