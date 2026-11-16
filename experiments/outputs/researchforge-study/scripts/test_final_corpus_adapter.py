"""Synthetic data only; no official held-out file is opened."""
import json
import unittest
from unittest.mock import patch
import pyarrow as pa
import pyarrow.parquet as pq
import test_final_evaluation_gate as fixtures
from final_evaluation_gate import freeze_plan
from prescience_final_data import load_authorized_corpus


class CorpusAdapterTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.GateTests();self.fixture.setUp();self.root=self.fixture.root
        rows=[]
        for cid,date,authors,target in [('old','2024-01-01',['a','b'],False),('same','2025-01-01',['z'],False),
                                       ('target','2025-01-01',['a','b','b','z'],True),('future','2025-02-01',['z'],False)]:
            rows.append(dict(corpus_id=cid,date=date,authors=[dict(author_id=a) for a in authors],
                             roles=['target'] if target else ['history'],categories=['test'],key_references=[dict(corpus_id='old')]))
        pq.write_table(pa.Table.from_pylist(rows),self.root/'corpus.parquet')
        (self.root/'authors.jsonl').write_text('\n'.join(json.dumps(dict(key=a,value=ids)) for a,ids in
            [('a',['target','old','old']),('b',['old','target']),('z',['future','target','same'])]))
        cfg=self.fixture.plan['datasets']['prescience'];cfg['corpus_inputs']=dict(corpus='corpus.parquet',author_history='authors.jsonl')
        cfg['query_sampling']=dict(count=0,seed=42)
        cfg['artifacts'] += [dict(role='data',path=p) for p in ['corpus.parquet','authors.jsonl']]
    def tearDown(self):self.fixture.tearDown()
    def run_adapter(self):return load_authorized_corpus(self.root,self.fixture.lock,'synthetic',self.fixture.ledger)
    def test_missing_lock_blocks_parquet_reader(self):
        with patch('prescience_final_data.pq.ParquetFile') as reader:
            with self.assertRaises(FileNotFoundError):self.run_adapter()
            reader.assert_not_called()
    def test_frozen_synthetic_corpus_dedup_and_strict_prior_eligibility(self):
        freeze_plan(self.root,self.fixture.plan,self.fixture.lock)
        _,papers,histories,dates,queries,audit=self.run_adapter()
        self.assertEqual(histories['a'],['old','target'])
        self.assertEqual(papers['target']['authors'],('a','b','z'))
        self.assertEqual(queries[0]['seed'],'a');self.assertEqual(queries[0]['truth'],['b'])
        self.assertEqual(audit['selected_queries'],1)
        self.assertEqual(dates['z'][0],papers['target']['date'])
    def test_changed_data_rejected_before_parquet_read(self):
        freeze_plan(self.root,self.fixture.plan,self.fixture.lock)
        with (self.root/'corpus.parquet').open('ab') as stream:stream.write(b'changed')
        with patch('prescience_final_data.pq.ParquetFile') as reader:
            with self.assertRaisesRegex(ValueError,'Locked artifact changed'):self.run_adapter()
            reader.assert_not_called()
    def test_unlocked_input_rejected_before_parquet_read(self):
        cfg=self.fixture.plan['datasets']['prescience']
        cfg['artifacts']=[a for a in cfg['artifacts'] if a['path']!='corpus.parquet']
        freeze_plan(self.root,self.fixture.plan,self.fixture.lock)
        with patch('prescience_final_data.pq.ParquetFile') as reader:
            with self.assertRaisesRegex(ValueError,'globally locked'):self.run_adapter()
            reader.assert_not_called()


if __name__=='__main__':unittest.main()
