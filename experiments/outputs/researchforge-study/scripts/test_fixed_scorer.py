import unittest
import numpy as np
from test_text_graph import TextGraphTests
from prescience_fixed_scorer import FixedScorer,VARIANTS


class FixedTests(TextGraphTests):
    def test_default_parity_with_existing_features(self):
        query=dict(qid='q',date=5,seed='a',truth=['b'])
        x,_,meta=self.engine.run(query,3,42,False)
        scorer=FixedScorer(self.papers,self.h,self.d,'a',5)
        scores=scorer.score()
        np.testing.assert_allclose([scores.get(a,0) for a in meta['candidates']],x[:,17],rtol=1e-6)

    def test_all_variants_temporal_and_source_exclusion(self):
        for config in VARIANTS.values():
            scores=FixedScorer(self.papers,self.h,self.d,'a',5).score(**config)
            self.assertNotIn('a',scores);self.assertNotIn('z',scores)
            self.assertTrue(all(np.isfinite(v) and v>=0 for v in scores.values()))
            self.papers['same']['refs']=('p1','p2','p3')
            after=FixedScorer(self.papers,self.h,self.d,'a',5).score(**config)
            self.assertEqual(scores,after)


if __name__=='__main__':unittest.main()
