import numpy as np
import test_text_graph as fixtures
from prescience_hybrid_graph import QueryEngine, FEATURES
from semantic_index import SemanticIndex

class HybridGraphTests(fixtures.TextGraphTests):
    def setUp(self):
        super().setUp()
        self.semantic=SemanticIndex.__new__(SemanticIndex)
        self.semantic.ids=self.index.ids
        self.semantic.row=self.index.row
        self.semantic.dates=np.array(self.index.dates)
        self.semantic.matrix=self.index.matrix.toarray().astype(np.float16)
        self.engine=QueryEngine(self.papers,self.h,self.d,self.index,self.semantic)

    def test_semantic_features_and_retrieval_exclude_future(self):
        q=dict(qid='q',date=5,seed='a',truth=['d'])
        x,y,m=self.engine.run(q,3,42,False)
        self.assertEqual(x.shape[1],len(FEATURES))
        self.assertEqual(len(FEATURES),23)
        self.assertNotIn('z',m['standalone']['semantic_standalone'])
        self.assertNotIn('a',m['standalone']['semantic_standalone'])
        row=m['candidates'].index('d')
        self.assertGreater(float(x[row,20]),.79)
        # Same-day evidence for d matches perfectly but must not alter mean/max.
        self.assertLess(float(x[row,21]),.81)

    def tearDown(self):
        del self.semantic
        super().tearDown()
