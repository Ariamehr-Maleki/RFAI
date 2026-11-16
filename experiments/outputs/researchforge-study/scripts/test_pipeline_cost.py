"""Exercise cost instrumentation on real tiny retrieval engines and saved model."""
import gc
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import numpy as np
import pipeline_cost as cost
from prescience_pilot import day
from test_prescience_final_run import FinalRunTests
from supervisor_state import atomic_state


class PipelineCostTests(unittest.TestCase):
    def test_real_tiny_pipeline_parity_and_both_modes(self):
        fixture=FinalRunTests();fixture.setUp()
        try:
            folder=fixture.folder;base=folder/'expanded';output=folder/'cost.json'
            cfg=fixture.plan['datasets']['prescience'];spec=dict(cfg['methods']['ranker']['models']['42']);spec['path']=str(cost.ROOT/spec['path'])
            records=[('h1','2024-01-01',['a','b']),('h2','2024-01-02',['c','d']),('q1','2025-01-01',['a','c']),('q2','2025-01-02',['c','b'])]
            papers={i:dict(date=day(d),authors=a,categories=('science',),refs=('h1',)) for i,d,a in records}
            histories={a:sorted([i for i,p in papers.items() if a in p['authors']],key=lambda i:(papers[i]['date'],i)) for a in 'abcd'}
            dates={a:[papers[i]['date'] for i in ids] for a,ids in histories.items()}
            lexical=lambda:cost.TextIndex(folder/'lexical')
            semantic=lambda:cost.SemanticIndex(folder/'semantic',folder/'lexical')
            # Capture constructors before the module symbols are patched below.
            real_lexical=cost.TextIndex;real_semantic=cost.SemanticIndex
            queries=[dict(qid='q1',seed='a',date=day('2025-01-01'),truth=['c']),dict(qid='q2',seed='c',date=day('2025-01-02'),truth=['b'])]
            for mode in ['full','no_content']:
                view=base/mode/'seed-42';(view/'valid_queries').mkdir(parents=True)
                atomic_state(view/'report.json',dict(status='Development synthetic fixture'))
                engine=cost.QueryEngine(papers,histories,dates,lexical(),semantic()) if mode=='full' else cost.NoContentEngine(papers,histories,dates)
                for q in queries:
                    x,_,meta=engine.run(q,1000,42,False)
                    atomic_state(view/'valid_queries'/(q['qid']+'.json'),meta)
                    np.savez_compressed(view/'valid_queries'/(q['qid']+'.npz'),X=x)
                del engine
            gc.collect();shutil.copyfile(folder/'corpus.parquet',folder/'train.parquet')
            with patch.object(cost,'require_idle'),patch.object(cost,'load_data',return_value=(papers,histories,dates)),patch.object(cost,'DATA',folder),\
                 patch.object(cost,'TextIndex',side_effect=lambda:real_lexical(folder/'lexical')),\
                 patch.object(cost,'SemanticIndex',side_effect=lambda:real_semantic(folder/'semantic',folder/'lexical')),\
                 patch.object(cost,'model_spec',return_value=spec):
                report=cost.benchmark(base,output,count=2,repeats=3)
            self.assertEqual(set(report['results']),{'full','no_content'})
            for mode in report['results']:
                self.assertEqual(len(report['results'][mode]['observations']),6)
                self.assertEqual(set(report['results'][mode]['models']),{'lambdamart','binary_lightgbm','logistic'})
                self.assertGreater(report['results'][mode]['retrieval_and_features']['median_seconds'],0)
            self.assertGreater(report['sampled_peak_process_rss_bytes'],0)
            self.assertEqual(json.loads(output.read_text())['query_ids'],['q1','q2'])
        finally:gc.collect();fixture.tearDown()


if __name__=='__main__':unittest.main(defaultTest='PipelineCostTests')
