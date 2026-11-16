"""Resource-only wrapper around the unchanged, globally locked final supervisor.

Requires 4500MiB available RAM for 30s before launch; retries only known resource
stops. No model, sample, source in the scientific lock or evaluation is changed.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import psutil
from recover_final_writer import recover
from supervisor_state import atomic_state

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / 'work'
STUDY = ROOT / 'outputs/researchforge-study'
STATE = WORK / 'final-resource-recovery-state.json'
WRITER = WORK / 'final-resource-recovery.lock'
MIN_RAM = 4500 * 1024**2
MIN_DISK = 2048 * 1024**2


def state(stage, **extra):
    atomic_state(STATE, dict(stage=stage, pid=os.getpid(), updated_epoch=time.time(), **extra))


def resource_failure(stage):
    error = stage.get('error', '')
    if 'Final worker reserve reached.' in error:
        return 'reserve'
    if 'final-prescience exited 1;' in error:
        return 'check_admission_log'
    return None


def main():
    fd = os.open(WRITER, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, 'w') as stream:
        json.dump(dict(pid=os.getpid()), stream)
    try:
        stable = None
        attempt = 0
        while True:
            available = psutil.virtual_memory().available
            free_disk = psutil.disk_usage(str(ROOT)).free
            if available < MIN_RAM or free_disk < MIN_DISK:
                stable = None
                state('waiting for4500MiB available RAM and2048MiB disk',
                      available_ram_bytes=available, free_disk_bytes=free_disk)
                time.sleep(15)
                continue
            if stable is None:
                stable = time.monotonic()
            if time.monotonic() - stable < 30:
                state('checking30second stable RAM admission', available_ram_bytes=available)
                time.sleep(5)
                continue
            if (WORK / 'frozen-study-supervisor.lock').exists():
                raise RuntimeError('Another frozen supervisor lock exists; no duplicate launch')
            attempt += 1
            log = WORK / f'final-resource-attempt-{os.getpid()}-{attempt}.log'
            worker_log = WORK / 'final-prescience.log'
            previous_size = worker_log.stat().st_size if worker_log.exists() else 0
            command = [str(ROOT / 'work/venv/Scripts/python.exe'), '-u',
                       str(STUDY / 'scripts/run_frozen_study.py'), '--lock',
                       str(STUDY / 'final/global-lock.json'), '--run-id', 'researchforge-final-v1']
            owned = set()
            with log.open('x', encoding='utf-8') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                owned.add(child.pid)
                while child.poll() is None:
                    try:
                        owned.update(p.pid for p in psutil.Process(child.pid).children(recursive=True))
                    except psutil.NoSuchProcess:
                        pass
                    state('unchanged frozen supervisor active', child_pid=child.pid,
                          attempt=attempt, owned_pids=sorted(owned), log=str(log))
                    time.sleep(2)
            last = json.loads((WORK / 'frozen-study-state.json').read_text())
            if last.get('supervisor_pid') not in owned:
                raise RuntimeError('Supervisor state ownership mismatch')
            if last.get('stage', '').startswith('final experiments and numerical exports complete'):
                state('complete numerical chain; manuscript/package still require review', attempt=attempt)
                return
            failure = resource_failure(last)
            if failure == 'check_admission_log':
                with worker_log.open('rb') as stream:
                    stream.seek(previous_size)
                    recent = stream.read().decode('utf-8', errors='replace')
                if 'RuntimeError: Wait for3500MiB availableRAM before final inference' not in recent:
                    raise RuntimeError('Non-resource worker failure; inspect without automatic retry')
            elif failure != 'reserve':
                raise RuntimeError('Unexpected frozen supervisor failure; inspect without automatic retry')
            writer = STUDY / 'results/final/prescience/writer.lock'
            if writer.exists():
                owner = json.loads(writer.read_text())['pid']
                if owner not in owned:
                    raise RuntimeError('Writer PID was not an observed owned descendant')
                recover(owner)
            state('resource stop audited; waiting before retry', attempt=attempt)
            stable = None
            time.sleep(30)
    except Exception as error:
        state('failed; manual inspection needed', error=repr(error))
        raise
    finally:
        WRITER.unlink()


if __name__ == '__main__':
    main()
