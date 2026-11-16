import math
import unittest
import numpy as np
from scipy import sparse
from hashed_bm25 import HashedBM25

class HashedBM25Tests(unittest.TestCase):
    def test_candidate_features_do_not_use_validation_labels(self):
        from exharmony_bm25_pilot import Retriever
        r=object.__new__(Retriever)
        tf=np.array([[2,0],[1,1],[0,1]],dtype=np.float32)
        norms=np.linalg.norm(tf,axis=1)
        r.bm25=HashedBM25(sparse.csr_matrix(tf/norms[:,None]),norms,tf.sum(1))
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

    def test_against_direct_document_loop(self):
        tf=np.array([[2,0,1,0],[0,1,1,0],[1,2,0,1],[0,0,0,0]],dtype=np.float32)
        lengths=tf.sum(1);norms=np.linalg.norm(tf,axis=1)
        normalized=tf/np.maximum(norms[:,None],1e-12)
        bm=HashedBM25(sparse.csr_matrix(normalized),norms,lengths)
        actual=bm.score([0,1,1])
        expected=[]
        for d in range(len(tf)):
            total=0
            for term in [0,1]:
                df=int((tf[:,term]>0).sum());freq=float(tf[d,term])
                if freq:
                    total+=math.log(1+(len(tf)-df+.5)/(df+.5))*freq*2.2/(freq+1.2*(.25+.75*float(lengths[d])/float(lengths.mean())))
            expected.append(total)
        np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(bm.score([]),np.zeros(4))

if __name__=='__main__':unittest.main()
