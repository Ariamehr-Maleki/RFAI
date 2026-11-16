import unittest
from datetime import date
from official_comparator import PaperView, official_functions, metrics

def paper(day, authors, trajectory=None, refs=None):
    return dict(date=day,_ordinal=date.fromisoformat(day).toordinal(),
        authors=[{'author_id':a} for a in authors],citation_trajectory=trajectory or [],
        key_references=[{'corpus_id':r} for r in refs or []])

class ComparatorTests(unittest.TestCase):
    def test_cumulative_bin_boundary(self):
        papers={'p':paper('2024-01-01',['a'],[1,3,7])}
        before=PaperView(papers,'2024-01-30',True)
        exact=PaperView(papers,'2024-01-31',True)
        later=PaperView(papers,'2024-03-01',True)
        self.assertEqual(before['p']['citation_trajectory'],[])
        self.assertEqual(exact['p']['citation_trajectory'],[1])
        self.assertEqual(later['p']['citation_trajectory'],[1,3])
    def test_future_and_same_day_references_hidden(self):
        papers={'p':paper('2024-02-01',['a']),'f':paper('2024-02-02',['b'])}
        view=PaperView(papers,'2024-02-01',True)
        self.assertNotIn('p',view);self.assertIsNone(view.get('f'))
        with self.assertRaises(KeyError):_ = view['p']
    def test_dedup_preserves_first_order(self):
        papers={'p':paper('2024-01-01',['b','a','b',None])}
        view=PaperView(papers,'2024-02-01',True,True)
        self.assertEqual([a['author_id'] for a in view['p']['authors']],['b','a'])
    def test_exact_function_equal_when_inputs_unchanged(self):
        env=official_functions()
        self.assertEqual(env['EXPAND_NUM_RECENT'],240)
        self.assertEqual(env['HOP3_NUM_RECENT'],160)
        papers={'p1':paper('2024-01-01',['a','b']), 'p2':paper('2024-01-02',['b','c'])}
        histories={'a':['p1'],'b':['p1','p2'],'c':['p2']}
        candidates={'a','b','c'}
        args=('a','2024-04-01',2,candidates,histories)
        expected=env['predict_coauthors'](*args,papers)
        for historical,dedup in [(False,False),(True,False),(True,True)]:
            actual=env['predict_coauthors'](*args,PaperView(papers,'2024-04-01',historical,dedup))
            self.assertEqual(actual,expected)
    def test_future_citation_mutation_cannot_change_historical_predictions(self):
        env=official_functions()
        papers={'p1':paper('2024-01-01',['a','b']), 'p2':paper('2024-01-02',['b','c'],[1,5,10000])}
        histories={'a':['p1'],'b':['p1','p2'],'c':['p2']}
        args=('a','2024-02-03',2,{'a','b','c'},histories)
        first=env['predict_coauthors'](*args,PaperView(papers,'2024-02-03',True))
        papers['p2']['citation_trajectory']=[1,100000,900000]
        second=env['predict_coauthors'](*args,PaperView(papers,'2024-02-03',True))
        self.assertEqual(first,second)
    def test_audit_only_records_used_citation_field(self):
        view=PaperView({'p':paper('2024-01-01',['a'],[1,4,8])},'2024-02-01')
        self.assertEqual(view['p']['date'],'2024-01-01')
        self.assertEqual(view.unavailable_bins,{})
        view['p'].get('citation_trajectory')
        self.assertEqual(view.unavailable_bins,{'p':2})
    def test_metrics_penalize_missing_truth(self):
        m=metrics(['a','x'],['a','b'],['b'])
        self.assertEqual(m['recall10'],.5);self.assertEqual(m['new_recall10'],0)
        self.assertLess(m['ndcg1000'],1)

if __name__=='__main__':unittest.main()
