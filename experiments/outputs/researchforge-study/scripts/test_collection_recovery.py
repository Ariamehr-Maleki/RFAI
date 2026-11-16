import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from run_expanded_when_ready import ROOT,recover_collection


class RecoveryTests(unittest.TestCase):
    def test_committed_files_retained_and_only_dead_owned_lock_removed(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
            run=Path(directory);folder=run/'train_queries';folder.mkdir()
            meta=folder/'one.json';data=folder/'one.npz'
            meta.write_text('{}');data.write_bytes(b'synthetic feature payload')
            receipt={'metadata_sha256':hashlib.sha256(meta.read_bytes()).hexdigest(),
                     'features_sha256':hashlib.sha256(data.read_bytes()).hexdigest()}
            (folder/'one.receipt').write_text(json.dumps(receipt))
            lock=run/'collect.lock';lock.write_text('{"pid":123}')
            with patch('run_expanded_when_ready.psutil.pid_exists',return_value=True):
                with self.assertRaisesRegex(RuntimeError,'still alive'):recover_collection(run,{123})
            with patch('run_expanded_when_ready.psutil.pid_exists',return_value=False):
                with self.assertRaisesRegex(RuntimeError,'another process'):recover_collection(run,{456})
                data.write_bytes(b'corrupted')
                with self.assertRaisesRegex(ValueError,'cache changed'):recover_collection(run,{123})
                self.assertTrue(lock.exists())
                data.write_bytes(b'synthetic feature payload')
                self.assertEqual(recover_collection(run,{123}),1)
            self.assertFalse(lock.exists());self.assertTrue(meta.exists());self.assertTrue(data.exists())


if __name__=='__main__':unittest.main()
