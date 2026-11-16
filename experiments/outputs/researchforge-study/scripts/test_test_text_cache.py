"""Synthetic integration checks: reuse, sealed columns, interruption and corruption."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import prescience_test_text_cache as cache
from prescience_text_reuse_plan import text_hash

class FakeEncoder:
    def __init__(self,fail=False):self.fail=fail
    def encode(self,texts):
        if self.fail and any(int(t.split('-')[1].strip())>=512 for t in texts):
            raise RuntimeError('simulated interrupted encoding')
        return np.stack([np.full(384,int(t.split('-')[1].strip())/1000,dtype=np.float32) for t in texts])

class CacheTests(unittest.TestCase):
    def setUp(self):
        temp=cache.ROOT/'work/test-temp';temp.mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=temp);self.root=Path(self.tmp.name)
        self.plan=self.root/'plan';self.train=self.root/'train';self.model=self.root/'model';self.dest=self.root/'dest'
        for p in [self.plan,self.train,self.model]:p.mkdir()
        self.source=self.root/'test.parquet'
        self.rows=[dict(corpus_id=str(i),title=f'new-{i}',abstract='',authors=['SEALED']) for i in range(520)]
        pq.write_table(pa.Table.from_pylist(self.rows),self.source)
        self.vectors=np.stack([np.full(384,.25,np.float16),np.full(384,.75,np.float16)])
        self.vectors.tofile(self.train/'embeddings.f16')
        cfg=dict(repository='fixture',revision='pinned',dimensions=384,stored_dtype='float16',
                 max_length=256,pooling='attention-mask mean + L2 normalization',batch_size=8,threads=1,source_sha256='train-fixture')
        self.write(self.train/'manifest.json',dict(status='complete',completed_rows=2,total_rows=2,config=cfg,sha256=cache.digest(self.train/'embeddings.f16')))
        self.write(self.model/'download_manifest.json',dict(repository='fixture',revision='pinned',files={}))
        mapping=np.full(520,-1,np.int32);mapping[0]=0;mapping[2]=1
        np.save(self.plan/'train_row_for_test_row.npy',mapping)
        np.save(self.plan/'test_text_sha256.npy',np.asarray([text_hash(r) for r in self.rows],dtype='S64'))
        self.write(self.plan/'test_ids.json',[str(i) for i in range(520)])
        files={p.name:dict(sha256=cache.digest(p)) for p in self.plan.iterdir()}
        self.write(self.plan/'manifest.json',dict(train_rows=2,test_rows=520,source_sha256=dict(train='train-fixture',test=cache.digest(self.source)),plan_files=files))
    def tearDown(self):self.tmp.cleanup()
    def write(self,path,obj):path.write_text(json.dumps(obj))
    def run_build(self,fail=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return cache.build(self.source,self.plan,self.train,self.model,self.dest,lambda _:FakeEncoder(fail),reserve_bytes=0)
    def test_resume_and_reuse_with_only_allowed_columns(self):
        original=pq.ParquetFile;columns=[]
        class Reader:
            def __init__(self,path):self.file=original(path)
            def iter_batches(self,*,batch_size,columns):
                if columns!=['corpus_id','title','abstract']:raise AssertionError('Forbidden label column')
                return self.file.iter_batches(batch_size=batch_size,columns=columns)
        with patch.object(cache.pq,'ParquetFile',Reader):
            with self.assertRaisesRegex(RuntimeError,'interrupted'):self.run_build(fail=True)
            self.assertEqual(cache.read_json(self.dest/'progress.json')['completed_rows'],512)
            with (self.dest/'embeddings.f16').open('ab') as f:f.write(b'uncommitted tail')
            result=self.run_build()
        self.assertEqual((result['copied_rows'],result['encoded_rows']),(2,518))
        got=np.fromfile(self.dest/'embeddings.f16',dtype=np.float16).reshape(-1,384)
        np.testing.assert_array_equal(got[[0,2]],self.vectors)
        self.assertEqual(float(got[519,0]),float(np.float16(.519)))
        self.assertEqual(self.run_build()['sha256'],result['sha256'])
    def test_resume_rejects_corrupt_committed_prefix(self):
        with self.assertRaises(RuntimeError):self.run_build(fail=True)
        with (self.dest/'embeddings.f16').open('r+b') as f:f.write(b'XX')
        with self.assertRaisesRegex(ValueError,'prefix corrupted'):self.run_build()
    def test_missing_train_manifest_rejected_before_output_creation(self):
        (self.train/'manifest.json').unlink()
        with self.assertRaisesRegex(RuntimeError,'TRAIN cache incomplete'):self.run_build()
        self.assertFalse(self.dest.exists())
    def test_resume_after_final_chunk_before_manifest(self):
        original=cache.atomic_json
        def fail_manifest(path,value):
            if path.name=='manifest.json':raise RuntimeError('simulated manifest interruption')
            original(path,value)
        with patch.object(cache,'atomic_json',fail_manifest):
            with self.assertRaisesRegex(RuntimeError,'manifest interruption'):self.run_build()
        result=self.run_build()
        self.assertEqual(result['completed_rows'],520)
        self.assertEqual(result['sha256'],cache.digest(self.dest/'embeddings.f16'))

if __name__=='__main__':unittest.main()
