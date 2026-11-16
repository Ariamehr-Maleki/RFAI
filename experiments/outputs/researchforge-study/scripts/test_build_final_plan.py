"""Assemble all three configurations using tiny opaque fixture artifacts only."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import build_final_plan as builder
from final_evaluation_gate import freeze_plan,verify_lock
from frozen_ranker import digest


class PlanAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='plan-assembly-fixture-',dir=builder.ROOT/'work')
        self.root=Path(self.tmp.name);self.study=self.root/'outputs/researchforge-study'
        self.results=self.study/'results';self.base=self.results/'prescience-expanded-1200-1000-500-1000'
        self.scripts=self.study/'scripts';self.scripts.mkdir(parents=True)
        # Copy source, not data. AST literal extraction runs on the actual schemas.
        for path in builder.SCRIPTS.glob('*.py'):shutil.copyfile(path,self.scripts/path.name)
        self.patch=patch.multiple(builder,ROOT=self.root,STUDY=self.study,RESULTS=self.results,BASE=self.base,SCRIPTS=self.scripts)
        self.patch.start()
        self.env=self.study/'environment.json';self.write(self.env,{'fixture':True})
        for name in ['PROTOCOL.md','SCIENTIFIC_REPORTS_PLAN.md','ncn-runtime.json']:self.write(self.study/name,{'fixture':True})
        self.write(self.base/'report.json',dict(status='Completed expanded DEVELOPMENT fixture',seed_summary={'synthetic':0.}))
        self.write(self.base/'query_manifest.json',{'fixture':True})
        for name in ['final_semantic_index','final_lexical_index','train_semantic_index','train_encoding_cost','loader-equivalence']:
            self.write(self.study/'data-audit'/f'{name}.json',dict(status='PASS synthetic fixture'))
        self.cost=self.base/'isolated_pipeline_cost.json';self.write(self.cost,self.cost_report())
        features=builder.literal('prescience_hybrid_graph.py','FEATURES')
        cells=['controls_only','text','network','explicit_time','text+network','text+explicit_time','network+explicit_time','text+network+explicit_time']
        for mode in ['full','no_content']:
            for seed in [42,43,44]:
                folder=self.base/mode/f'seed-{seed}';models={}
                for family in ['lambdamart','binary_lightgbm','logistic']:
                    path=folder/(family+('.joblib' if family=='logistic' else '.txt'));self.opaque(path)
                    models[family]=dict(feature_names=features,artifacts={path.name:digest(path)})
                self.write(folder/'report.json',dict(status='Development complete synthetic fixture',metrics={'fixture':0.},models=models))
                if mode=='full':
                    factorial=folder/f'factorial-s{seed}';records={}
                    for cell in cells:
                        path=factorial/(cell+'.txt');self.opaque(path)
                        records[cell]=dict(features=features[:3],artifact_sha256={path.name:digest(path)})
                    self.write(factorial/'report.json',dict(status='Development factorial fixture',metrics={'fixture':0.},cells=records))
        self.write(self.base/'full/seed-42/fixed-scorer-sensitivity/report.json',dict(status='Development synthetic fixture',metrics={
            'first':{'shared':{'ndcg1000':.5}},'second':{'shared':{'ndcg1000':.5}},'third':{'shared':{'ndcg1000':.4}}}))
        for rel in ['work/data/prescience/test.parquet','work/data/prescience/author_publications.jsonl']:self.opaque(self.root/rel)
        lexical=self.root/'work/indices/prescience-test-hash18';self.write(lexical/'manifest.json',dict(files={'data.bin':{}}));self.opaque(lexical/'data.bin')
        semantic=self.root/'work/indices/prescience-test-minilm';self.write(semantic/'manifest.json',dict(status='complete'));self.opaque(semantic/'embeddings.f16')
        for name in ['ogb-expanded-10000','ncn-expanded-10000','exharmony-expanded-1000-500']:
            self.write(self.results/name/'report.json',dict(status='Completed development fixture',metrics={'synthetic':0.}))
        ogb=self.results/'ogb-expanded-10000';self.write(ogb/'config.json',dict(fixture=True))
        raw=self.root/'work/data/ogbl-collab/extracted/collab/raw'
        for name in ['edge.csv.gz','edge_year.csv.gz','edge_weight.csv.gz','node-feat.csv.gz']:self.opaque(raw/name)
        self.opaque(raw.parent/'split/time/test.pt')
        for seed in [42,43,44]:
            for track in ['graph','graph_plus_released_nodes']:
                for family in ['lambdamart','binary_lightgbm']:self.opaque(ogb/f's{seed}-{track}__{family}.txt')
            ncn=self.results/f'ncn-expanded-10000-e100-s{seed}'
            self.opaque(ncn/'best.pt');self.write(ncn/'report.json',dict(status='Development complete synthetic fixture',config={'epochs':1},best_validation_hits50=.5,history=[dict(epoch=1,validation_hits50=.5)]))
            ex=self.results/f'exharmony-bm25-1000-500-1000-s{seed}'
            self.write(ex/'report.json',dict(status='Development complete synthetic fixture',metrics={'synthetic':0.}))
            for family in ['lambdamart','binary_lightgbm']:self.opaque(ex/(family+'.txt'))
        for name in ['model.py','utils.py']:self.opaque(self.root/'work/NeuralCommonNeighbor'/name)
        for name in ['papers_test.jsonl','papers_test_judgeable.jsonl','qrels/qrels.test.authors.tsv','qrels/qrels.test.cite.tsv','qrels/qrels.test.simcite.tsv']:
            self.opaque(self.root/'work/data/exHarmony'/name)
        self.write(self.study/'data-audit/exharmony_split.json',{'fixture':True})
        for name in builder.literal('exharmony_final_run.py','INDEX_FILES'):self.opaque(self.root/'work/indices/exharmony-hash18'/name)

    def tearDown(self):self.patch.stop();self.tmp.cleanup()

    def write(self,path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')

    def opaque(self,path):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'SYNTHETIC OPAQUE FIXTURE - NOT A BENCHMARK FILE')

    def cost_report(self):
        ids=[str(i) for i in range(30)]
        rows=[dict(qid=q,repeat=r,retrieval_and_features_seconds=.1,reranking_seconds={n:.01 for n in ['lambdamart','binary_lightgbm','logistic']}) for q in ids for r in range(3)]
        return dict(status='Completed isolated development pipeline cost; not final quality evaluation',query_count=30,repeats=3,model_seed=42,padding_seed=42,candidates=1000,query_ids=ids,results={m:dict(observations=copy.deepcopy(rows)) for m in ['full','no_content']})

    def assemble(self):return builder.assemble(100,'Synthetic fixture sample rationale only; no real final sampling decision.',self.env)

    def test_all_three_assemble_and_draft_cannot_freeze(self):
        draft=self.assemble();self.assertEqual(draft['stage'],'draft_pending_development_review')
        self.assertEqual(set(draft['datasets']),{'prescience','ogbl-collab','exHarmony'})
        p=draft['datasets']['prescience'];self.assertEqual(len(p['primary_comparisons']),8)
        self.assertEqual(p['fixed_selection']['variant'],'first')
        self.assertEqual(sum(k.startswith('factorial_') for k in p['methods']),8)
        self.assertEqual(len(draft['datasets']['ogbl-collab']['primary_comparisons']),3)
        self.assertEqual(len(draft['datasets']['exHarmony']['primary_comparisons']),2)
        lock=self.root/'synthetic-lock.json'
        with self.assertRaisesRegex(ValueError,'draft'):freeze_plan(self.root,draft,lock)
        self.assertFalse(lock.exists())
        # Only this temporary opaque fixture is marked ready. No real labels read.
        draft['stage']='ready_for_final_test';freeze_plan(self.root,draft,lock);verify_lock(self.root,lock)

    def test_incomplete_or_duplicate_cost_observations_rejected(self):
        for modification in ['missing','duplicate','nonfinite']:
            report=self.cost_report();rows=report['results']['full']['observations']
            if modification=='missing':rows.pop()
            elif modification=='duplicate':rows[-1]=dict(rows[0])
            else:rows[0]['retrieval_and_features_seconds']=float('nan')
            self.write(self.cost,report)
            with self.subTest(modification=modification),self.assertRaises(ValueError):self.assemble()

    def test_failed_lexical_audit_and_incomplete_development_rejected(self):
        path=self.study/'data-audit/final_lexical_index.json';self.write(path,dict(status='FAIL'))
        with self.assertRaisesRegex(ValueError,'Unverified final index'):self.assemble()
        self.write(path,dict(status='PASS fixture'))
        self.write(self.results/'ogb-expanded-10000/report.json',dict(status='running',metrics={'synthetic':.5}))
        with self.assertRaisesRegex(ValueError,'Incomplete development'):self.assemble()

    def test_corrupted_model_and_missing_prerequisite_rejected(self):
        model=self.base/'full/seed-42/lambdamart.txt';model.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'model artifact changed'):self.assemble()
        self.cost.unlink()
        with self.assertRaisesRegex(RuntimeError,'prerequisites incomplete'):self.assemble()


if __name__=='__main__':unittest.main()
