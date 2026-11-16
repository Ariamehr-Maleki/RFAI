import unittest
import numpy as np
from ogb.linkproppred import Evaluator
from ogb_multiseed import node_features, summarize

class OGBExtendedTests(unittest.TestCase):
    def test_undirected_node_features(self):
        nodes=np.array([[1,0],[0,1],[1,1],[0,0]],dtype=np.float32)
        edges=np.array([[0,2],[1,2],[0,3]])
        x=node_features(edges,nodes)
        np.testing.assert_array_equal(x,node_features(edges[:,::-1],nodes))
        self.assertTrue(np.isfinite(x).all())
        self.assertAlmostEqual(float(x[0,0]),2**-.5,places=6)
        self.assertEqual(float(x[2,0]),0)

    def test_hits_strict_ties_against_official(self):
        # All negatives tie at 1: only the positive scored 2 succeeds.
        pred=np.r_[2.,1.,0.,np.ones(100)]
        result=summarize(pred,3,np.array([0,0,1]),Evaluator(name='ogbl-collab'),bootstrap=100)
        self.assertAlmostEqual(result['hits50'],1/3)

if __name__=='__main__':unittest.main()
