"""Only temporary synthetic graph/splits; never opens the real test split."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRanker
from final_evaluation_gate import freeze_plan,DATASETS,ROLES
from ogb_final_run import ROOT,CODE,run,success
from supervisor_state import atomic_state


class OGBFinalTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='ogb-final-fixture-',dir=ROOT/'work');self.folder=Path(self.tmp.name)
        self.relative=lambda path:str(Path(path).relative_to(ROOT))
        self.lock=self.folder/'freeze.json';self.ledger=self.folder/'ledger'
        shared=[]
        for role in ROLES:
            path=self.folder/(role+'.json');atomic_state(path,dict(status='complete synthetic fixture',metrics={'synthetic':0.}))
            shared.append(dict(role=role,path=self.relative(path)))
        self.plan=dict(schema_version=1,stage='ready_for_final_test',datasets={})
        for dataset,metric in DATASETS.items():
            self.plan['datasets'][dataset]=dict(primary_metric=metric,seeds=[42,43,44],methods={'ranker':{'kind':'placeholder'},'baseline':{'kind':'placeholder'}},
                primary_comparisons=[['ranker','baseline']],multiplicity='Holm',candidate_policy='fixture',training_policy='fixture',
                test_split='temporary synthetic split',uncertainty_policy='fixture',artifacts=list(shared))
        rng=np.random.default_rng(12)
        arrays=dict(edges=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[1,4]]),years=np.full(6,2017),weights=np.arange(1,7),nodes=rng.normal(size=(12,128)))
        inputs={}
        for name,array in arrays.items():
            path=self.folder/(name+'.csv');pd.DataFrame(array).to_csv(path,index=False,header=False);inputs[name]=self.relative(path)
        split=self.folder/'synthetic.pt'
        torch.save(dict(edge=torch.tensor([[0,2],[1,3],[2,4],[3,5]]),edge_neg=torch.tensor(rng.integers(0,12,size=(100,2)))),split)
        inputs['split']=self.relative(split)
        methods={'baseline':dict(kind='feature',feature='adamic_adar'),'cosine':dict(kind='feature',feature='released_node_cosine')}
        artifacts=[]
        for name,track,width in [('ranker','graph',8),('node_ranker','graph_plus_released_nodes',266)]:
            model=LGBMRanker(n_estimators=4,num_leaves=3,min_child_samples=1,verbosity=-1,n_jobs=1)
            model.fit(rng.normal(size=(40,width)),np.tile([0,1],20),group=[2]*20)
            path=self.folder/(name+'.txt');model.booster_.save_model(str(path))
            methods[name]=dict(kind='tree',track=track,models={str(s):self.relative(path) for s in [42,43,44]})
            artifacts.append(dict(role='model',path=self.relative(path)))
        cfg=self.plan['datasets']['ogbl-collab'];cfg.update(runner_schema=1,graph_cutoff_year=2017,negative_count=100,batch_size=32,
            inference=dict(repetitions=100,seed=314159),output_directory=self.relative(self.folder/'result'),inputs=inputs,methods=methods)
        cfg['artifacts']+=artifacts+[dict(role='data',path=p) for p in inputs.values()]
        cfg['artifacts']+=[dict(role='code',path=self.relative(Path(__file__).with_name(name))) for name in CODE]
    def tearDown(self):self.tmp.cleanup()
    def test_inference_metrics_resume_and_corruption(self):
        freeze_plan(ROOT,self.plan,self.lock);result=run(ROOT,self.lock,'fixture',self.ledger)
        self.assertEqual(result['positive_count'],4);self.assertEqual(result['negative_count'],100)
        self.assertEqual(result['primary_comparisons']['ranker__minus__baseline']['queries'],4)
        self.assertEqual(len(list((self.folder/'result').glob('*.npz'))),12)
        self.assertEqual(result,run(ROOT,self.lock,'fixture',self.ledger))
        path=next((self.folder/'result').glob('*.npz'))
        with path.open('ab') as stream:stream.write(b'changed')
        with self.assertRaisesRegex(ValueError,'Completed output changed'):run(ROOT,self.lock,'fixture',self.ledger)
    def test_no_lock_and_unbound_split_never_load_labels(self):
        with patch('torch.load') as loader:
            with self.assertRaises(FileNotFoundError):run(ROOT,self.lock,'fixture',self.ledger)
            loader.assert_not_called()
        cfg=self.plan['datasets']['ogbl-collab'];cfg['artifacts']=[a for a in cfg['artifacts'] if a['path']!=cfg['inputs']['split']]
        freeze_plan(ROOT,self.plan,self.lock)
        with patch('torch.load') as loader:
            with self.assertRaisesRegex(ValueError,'locked data'):run(ROOT,self.lock,'fixture',self.ledger)
            loader.assert_not_called()
    def test_interrupted_commit_is_reused(self):
        freeze_plan(ROOT,self.plan,self.lock)
        import ogb_final_run as runner
        real=runner.atomic_state;stopped=[]
        def interrupt(path,value):
            real(path,value)
            if Path(path).suffix=='.receipt' and not stopped:stopped.append(True);raise RuntimeError('fixture interruption')
        with patch('ogb_final_run.atomic_state',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'fixture interruption'):run(ROOT,self.lock,'fixture',self.ledger)
        committed=next((self.folder/'result').glob('*.npz'));stamp=committed.stat().st_mtime_ns
        result=run(ROOT,self.lock,'fixture',self.ledger)
        self.assertEqual(committed.stat().st_mtime_ns,stamp);self.assertEqual(len(result['per_seed']),4)
    def test_ties_fail_strict_threshold(self):
        from ogb.linkproppred import Evaluator
        passed,metric=success(np.concatenate([[1.,2.],np.ones(100)]),2,100,Evaluator(name='ogbl-collab'))
        np.testing.assert_array_equal(passed,[0.,1.]);self.assertEqual(metric,.5)


if __name__=='__main__':unittest.main()
