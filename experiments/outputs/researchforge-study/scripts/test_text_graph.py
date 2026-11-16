"""Leakage checks for real text retrieval, citations and multi-hop expansion."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import sparse
from text_index import TextIndex, ROOT
from prescience_text_graph import QueryEngine

class TextGraphTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'work')
        path=Path(self.temp.name)
        self.papers={
            'p1':dict(date=1,authors=('a','b'),categories=('X',),refs=('p3','future')),
            'p2':dict(date=2,authors=('b','c'),categories=('Y',),refs=()),
            'p3':dict(date=0,authors=('d',),categories=('Z',),refs=()),
            'same':dict(date=5,authors=('a','d'),categories=('X',),refs=('p2',)),
            'future':dict(date=6,authors=('a','z'),categories=('X',),refs=())}
        self.h={'a':['p1','same','future'],'b':['p1','p2'],'c':['p2'],'d':['p3','same'],'z':['future']}
        self.d={a:[self.papers[p]['date'] for p in ids] for a,ids in self.h.items()}
        x=sparse.csr_matrix(np.array([[1,0],[0,1],[.8,.6],[1,0],[1,0]],dtype=np.float32))
        x.data.tofile(path/'data.bin');x.indices.astype(np.int32).tofile(path/'indices.bin')
        np.save(path/'indptr.npy',x.indptr)
        np.save(path/'dates.npy',np.array([1,2,0,5,6]))
        (path/'ids.json').write_text(json.dumps(list(self.papers)))
        (path/'manifest.json').write_text(json.dumps(dict(rows=5,dimensions=2)))
        self.index=TextIndex(path)
        self.engine=QueryEngine(self.papers,self.h,self.d,self.index)
    def tearDown(self):
        # Windows mapped files must be closed before cleanup.
        del self.engine,self.index
        import gc
        gc.collect()
        self.temp.cleanup()
    def test_text_filters_same_day_and_future(self):
        s=self.index.similarities(['p1','same','future'],5)
        np.testing.assert_allclose(s,[1,0,.8,0,0],atol=1e-6)
    def test_features_and_candidates_independent_of_labels(self):
        q=dict(qid='q',date=5,seed='a',truth=['b'])
        x,y,m=self.engine.run(q,3,42,False)
        xx,yy,mm=self.engine.run(dict(q,truth=['d']),3,42,False)
        np.testing.assert_array_equal(x,xx)
        self.assertEqual(m['candidates'],mm['candidates'])
        self.assertEqual(m['standalone'],mm['standalone'])
        self.assertNotIn('z',m['candidates'])
        self.assertFalse(np.array_equal(y,yy))
        d=m['candidates'].index('d')
        self.assertEqual(x[d,2],0) # no historical collaboration with d
        self.assertEqual(x[d,18],1) # eligible cited author only
        self.assertEqual(x[m['candidates'].index('c'),18],0) # same-day seed citation hidden
    def test_new_reference_cannot_change_past_prediction(self):
        q=dict(qid='q',date=5,seed='a',truth=['b'])
        x,y,m=self.engine.run(q,3,42,False)
        self.papers['future']['authors']=('a','d','c','b')
        self.papers['same']['refs']=('p1','p2','p3')
        xx,yy,mm=self.engine.run(q,3,42,False)
        np.testing.assert_array_equal(x,xx)
        self.assertEqual(m['standalone'],mm['standalone'])

if __name__=='__main__':unittest.main()
