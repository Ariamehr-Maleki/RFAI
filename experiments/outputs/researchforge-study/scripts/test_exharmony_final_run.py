"""Temporary synthetic corpus/index/models/locks only; no benchmark labels."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from lightgbm import LGBMRanker
import exharmony_final_run as final
from final_evaluation_gate import freeze_plan,DATASETS,ROLES
from supervisor_state import atomic_state
from frozen_ranker import digest


class ExpertFinalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='expert-final-fixture-',dir=final.ROOT/'work');self.folder=Path(self.tmp.name)
        self.relative=lambda path:str(Path(path).relative_to(final.ROOT));self.lock=self.folder/'freeze.json';self.ledger=self.folder/'ledger'
        self.plan=dict(schema_version=1,stage='ready_for_final_test',datasets={});shared=[]
        for role in ROLES:
            path=self.folder/(role+'.json');atomic_state(path,dict(status='complete synthetic development fixture',metrics={'synthetic':0.}))
            shared.append(dict(role=role,path=self.relative(path)))
        for dataset,metric in DATASETS.items():
            self.plan['datasets'][dataset]=dict(primary_metric=metric,seeds=[42,43,44],methods={'ranker':{'kind':'placeholder'},'baseline':{'kind':'placeholder'}},
                primary_comparisons=[['ranker','baseline']],multiplicity='Holm',candidate_policy='fixture',training_policy='fixture',test_split='temporary synthetic split',uncertainty_policy='fixture',artifacts=list(shared))
        self.index=self.folder/'index';self.index.mkdir()
        texts=['alpha science','beta science','alpha beta','gamma science']
        vectorizer=HashingVectorizer(n_features=2**18,alternate_sign=False,norm=None,stop_words='english',dtype=np.float32)
        counts=vectorizer.transform(texts).tocsr();counts.sort_indices()
        norms=np.sqrt(np.asarray(counts.multiply(counts).sum(1)).ravel());lengths=np.asarray(counts.sum(1)).ravel()
        matrix=sparse.diags(1/norms).dot(counts).tocsr();matrix.sort_indices()
        matrix.data.astype(np.float32).tofile(self.index/'data.bin');matrix.indices.astype(np.int32).tofile(self.index/'indices.bin');np.save(self.index/'indptr.npy',matrix.indptr)
        pa=sparse.csr_matrix([[1,0,0],[0,1,0],[1,1,0],[0,0,1]],dtype=np.float32)
        sparse.save_npz(self.index/'paper_authors.npz',pa);sparse.save_npz(self.index/'author_papers.npz',pa.T.tocsr());np.save(self.index/'author_ids.npy',np.array([11,22,33]))
        atomic_state(self.index/'manifest.json',dict(rows=4,dimensions=2**18,nnz=matrix.nnz))
        np.savez_compressed(self.index/'bm25_stats.npz',norms=norms,lengths=lengths)
        atomic_state(self.index/'bm25_stats.json',dict(index_manifest_sha256=digest(self.index/'manifest.json'),stats_sha256=digest(self.index/'bm25_stats.npz')))
        queries=[dict(id='q1',title='alpha',abstract=''),dict(id='bridge',title='beta',abstract=''),dict(id='q2',title='gamma',abstract=''),dict(id='empty',title='the and',abstract='')]
        query_path=self.folder/'queries.jsonl';query_path.write_text('\n'.join(map(json.dumps,queries)))
        author_map={'q1':['a'],'bridge':['a','b'],'q2':['b'],'empty':['c']}
        judgeable=self.folder/'judgeable.jsonl';judgeable.write_text('\n'.join(json.dumps(dict(id=q,authorships=[dict(author=dict(id=a)) for a in authors])) for q,authors in author_map.items()))
        split=self.folder/'query_split.json';atomic_state(split,dict(train=['train'],validation=['valid'],test=[r['id'] for r in queries]))
        inputs=dict(queries=self.relative(query_path),judgeable=self.relative(judgeable),split=self.relative(split))
        for judgment in final.JUDGMENTS:
            path=self.folder/(judgment+'.tsv');path.write_text('train deliberately unparsed invalid fields\n'+'\n'.join(f'{q} 0 https://openalex.org/A{a} 1' for q in author_map for a in [11,999]))
            inputs[judgment]=self.relative(path)
        rng=np.random.default_rng(42);model=LGBMRanker(n_estimators=3,num_leaves=3,min_child_samples=1,verbosity=-1,n_jobs=1)
        model.fit(rng.normal(size=(40,7)),np.tile([0,1],20),group=[2]*20)
        model_path=self.folder/'ranker.txt';model.booster_.save_model(str(model_path));source=self.folder/'development_report.json'
        spec=dict(kind='lightgbm',features=final.bm.FEATURES,path=self.relative(model_path),sha256=digest(model_path),source_report=self.relative(source),source_report_sha256=digest(source))
        cfg=self.plan['datasets']['exHarmony'];cfg.update(runner_schema=1,feature_names=final.bm.FEATURES,candidates=3,primary_judgment='authors',
            query_sampling=dict(count=0,seed=42),inference=dict(repetitions=100,seed=314159),output_directory=self.relative(self.folder/'result'),
            inputs=inputs,index_directory=self.relative(self.index),methods=dict(ranker=dict(kind='learned',models={str(s):spec for s in [42,43,44]}),baseline=dict(kind='feature',feature='max_paper_bm25')))
        cfg['artifacts'] += [dict(role='model',path=self.relative(model_path)),dict(role='split',path=self.relative(split))]
        cfg['artifacts'] += [dict(role='data',path=v) for k,v in inputs.items() if k!='split']
        cfg['artifacts'] += [dict(role='data',path=self.relative(self.index/n)) for n in final.INDEX_FILES]
        cfg['artifacts'] += [dict(role='code',path=self.relative(Path(__file__).with_name(n))) for n in final.CODE]
    def tearDown(self):self.tmp.cleanup()
    def test_real_index_inference_missing_profiles_empty_retrieval_and_resume(self):
        freeze_plan(final.ROOT,self.plan,self.lock)
        original=final.bm.Retriever.query;calls=[]
        def spy(engine,text,truth,k,training):
            calls.append((truth,training));return original(engine,text,truth,k,training)
        with patch.object(final.bm.Retriever,'query',new=spy):result=final.run(final.ROOT,self.lock,'fixture',self.ledger)
        self.assertEqual(len(calls),4);self.assertTrue(all(t==set() and train is False for t,train in calls))
        self.assertEqual(result['queries'],4);self.assertEqual(result['test_components'],2)
        self.assertEqual(result['primary_comparisons']['ranker__minus__baseline']['clusters'],2)
        self.assertEqual(result,final.run(final.ROOT,self.lock,'fixture',self.ledger))
        rows=final.read(self.folder/'result/authors.baseline.s42.predictions.json')
        empty=next(r for r in rows if r['qid']=='empty');self.assertEqual(empty['predicted'],[]);self.assertEqual(empty['ndcg10'],0.)
        self.assertTrue(all(999 in r['truth'] and r['candidate_recall']<=.5 for r in rows))
        self.assertEqual(len(final.read(self.folder/'result/metric_verification.json')['results']),18)
        path=next((self.folder/'result/queries').glob('*.npz'))
        with path.open('ab') as stream:stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError,'Completed expert output changed'):final.run(final.ROOT,self.lock,'fixture',self.ledger)
    def test_gate_and_model_binding_before_label_reader(self):
        with patch.object(final,'labels') as loader:
            with self.assertRaises(FileNotFoundError):final.run(final.ROOT,self.lock,'fixture',self.ledger)
            loader.assert_not_called()
        cfg=self.plan['datasets']['exHarmony'];cfg['artifacts']=[a for a in cfg['artifacts'] if not a['path'].endswith('ranker.txt')]
        freeze_plan(final.ROOT,self.plan,self.lock)
        with patch.object(final,'labels') as loader:
            with self.assertRaisesRegex(ValueError,'Unbound model'):final.run(final.ROOT,self.lock,'fixture',self.ledger)
            loader.assert_not_called()
    def test_interrupted_query_reuses_commit(self):
        freeze_plan(final.ROOT,self.plan,self.lock);real=final.atomic_state;stopped=[]
        def interrupt(path,value):
            real(path,value)
            if Path(path).suffix=='.receipt' and not stopped:stopped.append(True);raise RuntimeError('fixture interrupt')
        with patch.object(final,'atomic_state',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'fixture interrupt'):final.run(final.ROOT,self.lock,'fixture',self.ledger)
        committed=next((self.folder/'result/queries').glob('*.json'));stamp=committed.stat().st_mtime_ns
        result=final.run(final.ROOT,self.lock,'fixture',self.ledger)
        self.assertEqual(result['queries'],4);self.assertEqual(committed.stat().st_mtime_ns,stamp)
    def test_all_test_metadata_keeps_unsampled_bridge(self):
        cfg=self.plan['datasets']['exHarmony'];paths={k:final.ROOT/v for k,v in cfg['inputs'].items()}
        _,components=final.query_metadata(paths,{'q1','q2','bridge','empty'})
        self.assertEqual(components['q1'],components['q2']);self.assertNotEqual(components['q1'],components['empty'])


if __name__=='__main__':unittest.main()
