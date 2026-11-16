"""Synthetic render smoke check; these numbers are never manuscript evidence."""
import tempfile
import unittest
from pathlib import Path
from final_results_figures import draw,ROOT

class FigureTests(unittest.TestCase):
    def test_all_figures_with_negative_effects_and_empty_subgroup(self):
        cell=dict(mean=.2,sample_sd=.01);contrast=dict(delta=-.02,ci95=[-.04,-.001])
        names=['none','text','network','explicit_time','text+network','text+explicit_time','network+explicit_time','text+network+explicit_time']
        summary={'factorial_'+n:dict(ndcg1000=cell) for n in names}
        summary.update({m+'_lambdamart':dict(ndcg1000=cell) for m in ['full','no_content']})
        reports={d:dict(seed_summary=summary,primary_comparisons={'full_lambdamart__minus__full_binary_lightgbm':contrast}) for d in ['prescience','ogbl-collab','exHarmony']}
        groups={m+':'+g:dict(**contrast,queries=50,sparse_group=False) for m in ['ndcg1000','new_recall10','repeat_recall10'] for g in ['all','1-3','4-10','11+']}
        groups['new_recall10:1-3']=dict(delta=None,ci95=None,queries=0,sparse_group=True)
        cost=dict(status='Completed isolated development pipeline cost; not final quality evaluation',results={m:dict(end_to_end={n:dict(median_seconds=.2,p95_seconds=.4) for n in ['lambdamart','binary_lightgbm','logistic']}) for m in ['full','no_content']})
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temporary:
            target=Path(temporary)/'SYNTHETIC-NOT-RESULTS';draw(reports,{'full_lambdamart__minus__full_binary_lightgbm':dict(groups=groups)},cost,target)
            self.assertEqual(len(list(target.glob('*.pdf'))),4)
            self.assertEqual(len(list(target.glob('*.png'))),4)
            for p in target.glob('*.pdf'):self.assertTrue(p.read_bytes().startswith(b'%PDF'))
            for p in target.glob('*.png'):self.assertGreater(p.stat().st_size,10000)

if __name__=='__main__':unittest.main()
