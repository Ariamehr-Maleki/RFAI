"""Boundary/parity checks for the runtime optimization, not new score tuning."""
import unittest
from datetime import date,timedelta
import official_comparator as base
import official_comparator_fast as fast

class IndexedHistoryParity(unittest.TestCase):
    def setUp(self):
        fast.cache.clear();fast.checks=0
        self.reference=fast.original_factory()['get_preexisting_publications_for_author']
        self.optimized=fast.factory()['get_preexisting_publications_for_author']
        self.papers={}
        for i in range(40):
            day=date(2020,1,1)+timedelta(days=i//2)
            self.papers[str(i)]=dict(date=day.isoformat(),_ordinal=day.toordinal(),
                authors=[],citation_trajectory=[],key_references=[])

    def test_missing_none_duplicate_and_same_day(self):
        histories={'a':['missing','0','0','1','2'],'b':None,'c':[]}
        for cutoff in ['2019-12-31','2020-01-01','2020-01-02','2020-01-03']:
            for historical in [False,True]:
                view=base.PaperView(self.papers,cutoff,historical=historical)
                for author in histories:
                    self.assertEqual(self.optimized(author,cutoff,histories,view),
                                     self.reference(author,cutoff,histories,view))

    def test_reused_cache_across_reverse_cutoffs_and_history_objects(self):
        raw={'a':list(self.papers)}
        reordered={'a':sum(([str(i+1),str(i)] for i in range(0,40,2)),[])}
        for i in list(range(25))+list(range(24,-1,-1)):
            cutoff=(date(2020,1,1)+timedelta(days=i)).isoformat()
            for histories in [raw,reordered]:
                for historical in [False,True]:
                    view=base.PaperView(self.papers,cutoff,historical=historical)
                    self.assertEqual(self.optimized('a',cutoff,histories,view),
                                     self.reference('a',cutoff,histories,view))

if __name__=='__main__':unittest.main()
