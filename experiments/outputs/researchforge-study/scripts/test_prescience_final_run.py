"""End-to-end fixture run with real tiny models/indices and a temporary global lock."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lightgbm import LGBMRanker
from final_evaluation_gate import freeze_plan,DATASETS,ROLES
from prescience_final_run import ROOT,CODE,FEATURES,run
from prescience_final_lexical import build
from frozen_ranker import digest
from supervisor_state import atomic_state


class FinalRunTests(unittest.TestCase):
    def test_resumed_identifiers_share_memory_without_changing_values(self):
        from prescience_final_run import intern_prediction_rows
        result=json.loads('{"predictions":{"a":{"42":{"qid":"query-123456","seed":"author-123456","truth":["author-654321"],"new_truth":[],"predicted":["author-654321"]}},"b":{"43":{"qid":"query-123456","seed":"author-123456","truth":["author-654321"],"new_truth":[],"predicted":["author-654321"]}}}}')
        before=json.dumps(result,sort_keys=True);intern_prediction_rows(result)
        self.assertEqual(before,json.dumps(result,sort_keys=True))
        a=result['predictions']['a']['42'];b=result['predictions']['b']['43']
        self.assertIs(a['predicted'][0],b['predicted'][0]);self.assertIs(a['truth'][0],a['predicted'][0])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='final-run-fixture-',dir=ROOT/'work');self.folder=Path(self.tmp.name)
        self.lock=self.folder/'freeze.json';self.ledger=self.folder/'ledger'
        self.relative=lambda path:str(Path(path).relative_to(ROOT))
        self.plan=dict(schema_version=1,stage='ready_for_final_test',datasets={})
        shared=[]
        for role in ROLES:
            path=self.folder/(role+'.json');atomic_state(path,dict(status='complete synthetic fixture',metrics={'synthetic':0.}))
            shared.append(dict(role=role,path=self.relative(path)))
        for dataset,metric in DATASETS.items():
            self.plan['datasets'][dataset]=dict(primary_metric=metric,seeds=[42,43,44],methods={'ranker':{'kind':'placeholder'},'baseline':{'kind':'placeholder'}},
                primary_comparisons=[['ranker','baseline']],multiplicity='Holm',candidate_policy='fixture',training_policy='fixture',
                test_split='temporary synthetic corpus',uncertainty_policy='fixture',artifacts=list(shared))
        rows=[]
        for cid,date,authors,target in [('h1','2024-01-01',['a','b'],False),('h2','2024-01-02',['c','d'],False),
                                       ('q1','2025-01-01',['a','c'],True),('q2','2025-01-02',['c','b'],True)]:
            rows.append(dict(corpus_id=cid,date=date,authors=[dict(author_id=a) for a in authors],roles=['target'] if target else ['history'],
                             categories=['science'],key_references=[dict(corpus_id='h1')],title='alpha science',abstract='beta gamma'))
        source=self.folder/'corpus.parquet';pq.write_table(pa.Table.from_pylist(rows),source)
        history=self.folder/'history.jsonl';history.write_text('\n'.join(json.dumps(dict(key=a,value=[r['corpus_id'] for r in rows if a in [x['author_id'] for x in r['authors']]])) for a in 'abcd'))
        lexical=self.folder/'lexical';build(source,lexical,2)
        semantic=self.folder/'semantic';semantic.mkdir();vectors=np.eye(4,dtype=np.float16)
        vectors.tofile(semantic/'embeddings.f16')
        atomic_state(semantic/'manifest.json',dict(status='complete',total_rows=4,bytes=32,config=dict(source_sha256=digest(source),dimensions=4)))
        rng=np.random.default_rng(7);x=rng.normal(size=(40,len(FEATURES)));y=np.tile([0,1],20)
        model=LGBMRanker(n_estimators=3,num_leaves=3,min_child_samples=1,verbosity=-1,n_jobs=1)
        model.fit(x,y,group=[2]*20);model_path=self.folder/'ranker.txt';model.booster_.save_model(str(model_path))
        provenance=self.folder/'development_report.json'
        spec=dict(kind='lightgbm',path=self.relative(model_path),sha256=digest(model_path),features=FEATURES,
                  source_report=self.relative(provenance),source_report_sha256=digest(provenance))
        cfg=self.plan['datasets']['prescience'];cfg.update(runner_schema=1,feature_names=FEATURES,candidates=3,padding_seed=42,
            inference=dict(repetitions=100,seed=314159),subgroup_comparisons=[['ranker','baseline']],output_directory=self.relative(self.folder/'result'),
            corpus_inputs=dict(corpus=self.relative(source),author_history=self.relative(history)),query_sampling=dict(count=0,seed=42),
            index_inputs=dict(lexical=self.relative(lexical),semantic=self.relative(semantic)),
            methods=dict(ranker=dict(kind='learned',candidate_mode='full',models={str(s):spec for s in [42,43,44]}),
                         baseline=dict(kind='feature',candidate_mode='full',feature='frequency45'),
                         graph_only=dict(kind='feature',candidate_mode='no_content',feature='frequency45'),
                         semantic=dict(kind='standalone',candidate_mode='full',ranking='semantic_standalone'),
                         fixed=dict(kind='fixed',candidate_mode='full',variant='default')))
        cfg['artifacts'] += [dict(role='code',path=self.relative(Path(__file__).with_name(name))) for name in CODE]
        cfg['artifacts'] += [dict(role='model',path=self.relative(model_path))]
        cfg['artifacts'] += [dict(role='data',path=self.relative(path)) for path in [source,history,semantic/'manifest.json',semantic/'embeddings.f16',lexical/'manifest.json']+
                             [lexical/name for name in json.loads((lexical/'manifest.json').read_text())['files']]]
    def tearDown(self):self.tmp.cleanup()
    def test_full_inference_resume_independent_metrics_and_corruption_guard(self):
        freeze_plan(ROOT,self.plan,self.lock)
        first=run(ROOT,self.lock,'fixture',self.ledger)
        self.assertEqual(first['queries'],2);self.assertEqual(first['primary_comparisons']['ranker__minus__baseline']['queries'],2)
        self.assertEqual(first,run(ROOT,self.lock,'fixture',self.ledger))
        out=self.folder/'result';verification=json.loads((out/'independent_metric_verification.json').read_text())
        self.assertEqual(len(verification['results']),15)
        for path in (out/'queries').glob('*.json'):
            q=json.loads(path.read_text())
            self.assertEqual(q['metadata']['full']['training_positive_injections'],0)
            self.assertEqual(q['metadata']['no_content']['training_positive_injections'],0)
        path=next((out/'queries').glob('*.npz'))
        with path.open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Completed final output changed'):run(ROOT,self.lock,'fixture',self.ledger)
    def test_no_lock_never_reads_corpus(self):
        with patch('prescience_final_run.load_authorized_corpus') as loader:
            with self.assertRaises(FileNotFoundError):run(ROOT,self.lock,'fixture',self.ledger)
            loader.assert_not_called()
    def test_missing_model_binding_blocks_before_corpus(self):
        cfg=self.plan['datasets']['prescience'];cfg['artifacts']=[entry for entry in cfg['artifacts'] if not entry['path'].endswith('ranker.txt')]
        freeze_plan(ROOT,self.plan,self.lock)
        with patch('prescience_final_run.load_authorized_corpus') as loader:
            with self.assertRaisesRegex(ValueError,'not locked'):run(ROOT,self.lock,'fixture',self.ledger)
            loader.assert_not_called()
    def test_committed_query_survives_interruption_and_is_not_recomputed(self):
        freeze_plan(ROOT,self.plan,self.lock)
        import prescience_final_run as runner
        real_atomic=runner.atomic_state;interrupted=[]
        def interrupt_after_commit(path,value):
            real_atomic(path,value)
            if Path(path).suffix=='.receipt' and not interrupted:
                interrupted.append(True);raise RuntimeError('fixture interruption after query commit')
        with patch('prescience_final_run.atomic_state',side_effect=interrupt_after_commit):
            with self.assertRaisesRegex(RuntimeError,'fixture interruption'):run(ROOT,self.lock,'fixture',self.ledger)
        original=runner.QueryEngine.run;recomputed=[]
        def spy(engine,query,*args,**kwargs):
            recomputed.append(query['qid'])
            return original(engine,query,*args,**kwargs)
        with patch.object(runner.QueryEngine,'run',new=spy):
            result=run(ROOT,self.lock,'fixture',self.ledger)
            # Both full and NoContentEngine call the common QueryEngine method.
            self.assertEqual(len(recomputed),2)
            self.assertEqual(len(set(recomputed)),1)
        self.assertEqual(result['queries'],2)


if __name__=='__main__':unittest.main()
