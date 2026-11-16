import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_final_with_recovery as recovery


class ResourceWrapperTests(unittest.TestCase):
    def run_fixture(self, available, owner=999, interrupt_wait=False):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            work = root / 'work'
            work.mkdir()
            (work / 'frozen-study-state.json').write_text(json.dumps(dict(
                stage='final experiments and numerical exports complete', supervisor_pid=owner)))
            locations = dict(ROOT=root, WORK=work, STUDY=root / 'study',
                STATE=work / 'resource-state.json', WRITER=work / 'resource.lock')
            child = SimpleNamespace(pid=999, poll=lambda: 0)
            with patch.multiple(recovery, **locations), \
                 patch.object(recovery.psutil, 'virtual_memory', return_value=SimpleNamespace(available=available)), \
                 patch.object(recovery.psutil, 'disk_usage', return_value=SimpleNamespace(free=10*1024**3)), \
                 patch.object(recovery.time, 'monotonic', side_effect=[0, 0, 31]), \
                 patch.object(recovery.time, 'sleep', side_effect=RuntimeError('test wait interruption') if interrupt_wait else None), \
                 patch.object(recovery.subprocess, 'Popen', return_value=child) as launch:
                error = None
                try:
                    recovery.main()
                except RuntimeError as exc:
                    error = str(exc)
                state = json.loads(locations['STATE'].read_text())
                self.assertFalse(locations['WRITER'].exists())
                return error, launch.call_count, state

    def test_low_ram_does_not_launch(self):
        error, launches, _ = self.run_fixture(3000*1024**2, interrupt_wait=True)
        self.assertEqual(error, 'test wait interruption')
        self.assertEqual(launches, 0)

    def test_stable_admission_recognizes_owned_completion(self):
        error, launches, state = self.run_fixture(5000*1024**2)
        self.assertIsNone(error)
        self.assertEqual(launches, 1)
        self.assertTrue(state['stage'].startswith('complete numerical chain'))

    def test_unrelated_supervisor_state_is_rejected(self):
        error, launches, state = self.run_fixture(5000*1024**2, owner=888)
        self.assertEqual(launches, 1)
        self.assertIn('ownership mismatch', error)
        self.assertEqual(state['stage'], 'failed; manual inspection needed')


if __name__ == '__main__':
    unittest.main()
