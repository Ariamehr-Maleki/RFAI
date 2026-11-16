"""Real library fits on synthetic grouped data; no benchmark outcomes implied."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from fit_factorial import fit,read,sha

ROOT=Path(__file__).resolve().parents[3]

class FactorialTests(unittest.TestCase):
    def setUp(self):
        temp=ROOT/'work/test-temp';temp.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=temp);self.source=Path(self.temp.name)/'source';self.source.mkdir()
        tree=ast.parse(Path(__file__).with_name('prescience_hybrid_graph.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FEATURES' for t in n.targets))
        self.features=ast.literal_eval(node.value);rng=np.random.default_rng(123)
        for part,nq,offset in [('train',12,0),('valid',5,100)]:
            folder=self.source/(part+'_queries');folder.mkdir()
            for i in range(nq):
                qid=f'{part}-{i:03d}';candidates=[f'a-{j:02d}' for j in range(20)]
                x=rng.normal(size=(20,len(self.features))).astype(np.float32)
                y=np.zeros(20,np.int8);y[np.argsort(x[:,14])[-3:]]=1
                truth=[candidates[j] for j in np.flatnonzero(y)]
                meta=dict(qid=qid,date=offset+i,seed=f'author-{i//2}',candidates=candidates,truth=truth,
                          new_truth=truth[:1],candidate_recall=1.,training_positive_injections=0)
                (folder/(qid+'.json')).write_text(json.dumps(meta))
                np.savez_compressed(folder/(qid+'.npz'),X=x,y=y)
        (self.source/'report.json').write_text(json.dumps(dict(status='Development only; synthetic fixture',
            features=self.features,config=dict(train_queries=12,valid_queries=5))))
    def tearDown(self):self.temp.cleanup()
    def run_fit(self):
        with contextlib.redirect_stdout(io.StringIO()):return fit(self.source,grid=[(3,4),(7,5)])
    def test_eight_cells_model_roundtrip_and_reuse(self):
        result=self.run_fit();self.assertEqual(len(result['metrics']),8);self.assertEqual(len(result['tuning']),16)
        dest=self.source/'factorial-s42';before={p.name:sha(p) for p in dest.glob('*.txt')}
        for name,cell in result['cells'].items():
            self.assertLessEqual(cell['model_roundtrip_max_error'],1e-12)
            rows=read(dest/(name+'.predictions.json'));self.assertEqual(len(rows),5)
            self.assertEqual({r['qid'] for r in rows},{f'valid-{i:03d}' for i in range(5)})
            self.assertEqual(set(cell['features'])&{'rrf','autoresearch_description'},set())
        self.run_fit();self.assertEqual(before,{p.name:sha(p) for p in dest.glob('*.txt')})
        victim=dest/'controls_only.predictions.json';victim.write_text('[]')
        with self.assertRaisesRegex(ValueError,'artifact changed'):self.run_fit()
    def test_misaligned_labels_rejected(self):
        path=self.source/'valid_queries/valid-000.npz'
        with np.load(path) as d:x,y=d['X'],d['y']
        y[0]=1-y[0];np.savez_compressed(path,X=x,y=y)
        with self.assertRaisesRegex(ValueError,'labels do not match'):self.run_fit()
    def test_wrong_split_or_injection_rejected(self):
        path=self.source/'valid_queries/valid-000.json';meta=read(path);meta['date']=-1;path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'Nonchronological'):self.run_fit()
        meta['date']=100;meta['training_positive_injections']=1;path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError,'injection forbidden'):self.run_fit()
    def test_test_report_rejected(self):
        path=self.source/'report.json';report=read(path);report['status']='Final test';path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError,'development input'):self.run_fit()

if __name__=='__main__':unittest.main()
