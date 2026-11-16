"""Small synthetic fixtures test leakage/metric mechanics, not model quality."""
import ast
import unittest
import numpy as np
import prescience_pilot as p

class ProtocolTests(unittest.TestCase):
    def test_authors_deduplicated_without_reordering_seed(self):
        authors=[{'author_id':'b'},{'author_id':'a'},{'author_id':'b'},{'author_id':None}]
        self.assertEqual(p.unique_author_ids(authors),('b','a'))
        self.assertEqual(p.unique_author_ids(None),())

    def test_metrics_against_upstream(self):
        tree=ast.parse((p.ROOT/'work/prescience/utils.py').read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='calculate_ndcg')
        scope={'np':np,'log':lambda _:None}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'upstream_metric','exec'),scope)
        rng=np.random.default_rng(4)
        for _ in range(20):
            pred=rng.permutation(30).tolist()
            truth=rng.choice(40,size=6,replace=False).tolist()
            self.assertAlmostEqual(p.rank_metrics(pred,truth)['ndcg1000'],scope['calculate_ndcg'](pred,truth))
        self.assertEqual(p.rank_metrics(['x','y'],['x','y'])['rprecision'],1)
        self.assertEqual(p.rank_metrics(['x'],['y'])['ndcg10'],0)

    def test_candidates_do_not_see_labels_or_future(self):
        papers={
            '1':dict(date=1,authors=('a','b'),categories=('AI',),target=False),
            '2':dict(date=2,authors=('b','c'),categories=('AI',),target=False),
            '3':dict(date=3,authors=('d',),categories=('IR',),target=False),
            'same_day':dict(date=5,authors=('a','d'),categories=('IR',),target=False),
            'future':dict(date=6,authors=('a','d'),categories=('IR',),target=False)}
        h={'a':['1','same_day','future'],'b':['1','2'],'c':['2'],'d':['3','same_day','future']}
        d={a:[papers[i]['date'] for i in ids] for a,ids in h.items()}
        q=dict(qid='q',date=5,seed='a',truth=['b'])
        x,y,g,m,_=p.build_features([q],papers,h,d,3,42,False)
        q2=dict(q,truth=['d'])
        xx,yy,gg,mm,_=p.build_features([q2],papers,h,d,3,42,False)
        np.testing.assert_array_equal(x,xx)
        self.assertEqual(m[0]['candidates'],mm[0]['candidates'])
        idx=m[0]['candidates'].index('d')
        self.assertEqual(x[idx,1],0)  # same-day and future collaboration excluded
        self.assertEqual(int(sum(g)),len(x))
        self.assertNotIn('a',m[0]['candidates'])
        self.assertFalse(np.array_equal(y,yy))

    def test_seed_is_first_eligible_author(self):
        papers={'q':dict(target=True,date=5,authors=('new','a','b'))}
        dates={'new':[5],'a':[1],'b':[2]}
        q=p.make_queries(papers,{},dates)[0]
        self.assertEqual(q['seed'],'a')
        self.assertEqual(q['truth'],['b'])

if __name__=='__main__':
    unittest.main()
