import unittest
import test_indexed_helper as fixtures
import official_comparator_bounded as bounded
import official_comparator as base

class BoundedParity(fixtures.IndexedHistoryParity):
    def setUp(self):
        super().setUp()
        bounded.cache.clear();bounded.cached_publications=0;bounded.checks=0
        bounded.evictions=0;bounded.peak_authors=0;bounded.peak_publications=0
        self.optimized=bounded.factory()['get_preexisting_publications_for_author']

    def test_eviction_preserves_values(self):
        previous=(bounded.MAX_CACHED_AUTHORS,bounded.MAX_CACHED_PUBLICATIONS)
        bounded.MAX_CACHED_AUTHORS=2;bounded.MAX_CACHED_PUBLICATIONS=80
        histories={str(i):list(self.papers) for i in range(12)}
        view=base.PaperView(self.papers,'2020-01-15',historical=True)
        try:
            for author in list(histories)*2:
                self.assertEqual(self.optimized(author,'2020-01-15',histories,view),
                    self.reference(author,'2020-01-15',histories,view))
                self.assertLessEqual(len(bounded.cache),2)
                self.assertLessEqual(bounded.cached_publications,80)
            self.assertGreater(bounded.evictions,0)
        finally:bounded.MAX_CACHED_AUTHORS,bounded.MAX_CACHED_PUBLICATIONS=previous

if __name__=='__main__':unittest.main()
