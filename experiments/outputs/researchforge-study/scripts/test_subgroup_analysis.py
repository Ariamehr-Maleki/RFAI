import unittest
from subgroup_analysis import measures,compare

class SubgroupTests(unittest.TestCase):
    def fixture(self):
        a=[];b=[];meta=[]
        for i,history in enumerate([1,3,4,10,11,20]):
            base=dict(qid=str(i),seed=f'author-{i//2}',truth=['new','old'],new_truth=['new'] if i else [])
            a.append(dict(base,predicted=['new','old']+[f'x{j}' for j in range(10)]))
            b.append(dict(base,predicted=[f'x{j}' for j in range(10)]+['new','old']))
            meta.append(dict(base,seed_history_count=history))
        return a,b,meta
    def test_original_ranking_and_empty_denominators(self):
        a,b,meta=self.fixture();r=compare(a,b,meta,repeats=200)
        self.assertEqual(r['groups']['new_recall10:all']['queries'],5)
        self.assertEqual(r['groups']['repeat_recall10:all']['queries'],6)
        self.assertEqual(r['groups']['new_recall10:all']['delta'],1.)
        self.assertEqual(r['groups']['ndcg1000:all']['author_clusters'],3)
        self.assertEqual(r['interactions']['new_recall10:1-3_minus_11+']['difference_of_deltas'],0.)
    def test_identical_models_have_zero_effect(self):
        a,_,meta=self.fixture();r=compare(a,a,meta,repeats=200)
        self.assertEqual(r['groups']['ndcg1000:all']['ci95'],[0.,0.])
    def test_misaligned_labels_rejected(self):
        a,b,meta=self.fixture();b[0]['truth']=['other']
        with self.assertRaisesRegex(ValueError,'Relevant sets differ'):compare(a,b,meta,repeats=10)

if __name__=='__main__':unittest.main()
