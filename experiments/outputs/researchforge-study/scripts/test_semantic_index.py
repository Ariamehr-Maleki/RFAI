import tempfile
import unittest
import numpy as np
from semantic_index import historical_scores, SemanticIndex, ROOT

class SemanticIndexTests(unittest.TestCase):
    def test_future_profile_and_retrieval_mask(self):
        x=np.array([[1,0],[0,1],[1,1],[0,1]],dtype=np.float16)
        dates=np.array([1,2,5,6])
        score=historical_scores(x,dates,[0,2,3],5,block_size=2)
        np.testing.assert_allclose(score,[1,0,0,0],atol=1e-6)
        np.testing.assert_array_equal(score,historical_scores(x,dates,[0],5))

    def test_blockwise_cosine_and_empty_history(self):
        rng=np.random.default_rng(51)
        x=rng.normal(size=(23,7)).astype(np.float16)
        dates=np.arange(23)
        actual=historical_scores(x,dates,[0,1,2],20,block_size=4)
        xf=x.astype(np.float32);q=xf[:3].mean(0);q/=np.linalg.norm(q)
        expected=xf@q/np.linalg.norm(xf,axis=1);expected[20:]=0
        np.testing.assert_allclose(actual,expected,atol=1e-6)
        np.testing.assert_array_equal(historical_scores(x,dates,[],20),np.zeros(23))

    def test_partial_embeddings_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as tmp:
            with self.assertRaisesRegex(RuntimeError,'incomplete'):
                SemanticIndex(path=tmp)

if __name__=='__main__':unittest.main()
