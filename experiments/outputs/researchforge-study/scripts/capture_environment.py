"""Read-only runtime inventory; writes a dated reproducibility artifact locally."""
import datetime as dt
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import psutil
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3];STUDY=ROOT/'outputs/researchforge-study'


def main():
    packages={d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata.get('Name')}
    record=dict(captured_utc=dt.datetime.now(dt.timezone.utc).isoformat(),python=sys.version,executable=sys.executable,
        platform=platform.platform(),machine=platform.machine(),logical_cpus=psutil.cpu_count(),total_ram_bytes=psutil.virtual_memory().total,
        base_packages=dict(sorted(packages.items(),key=lambda pair:pair[0].lower())),
        isolated_ncn_runtime=json.loads((STUDY/'ncn-runtime.json').read_text()),
        isolated_ncn_manifest_sha256=digest(STUDY/'ncn-runtime.json'),
        isolated_plot_packages={d.metadata['Name']:d.version for d in metadata.distributions(path=[str(ROOT/'work/plot-runtime')]) if d.metadata.get('Name')},
        ncn_verification_log_sha256=digest(ROOT/'work/test-ogb-final-ncn-readable.log'),
        notes=['Base interpreter and isolated NCN overlay differ; NCN inference selects overlay before importing Torch.',
               'Installed sparse extensions required approved local read execution; exact synthetic integration test passed.',
               'Plotting packages are an isolated workspace overlay and are never added to experiment sys.path.',
               'Snapshot before final freeze; recapture if packages change. No environment variables or credentials included.'])
    destination=STUDY/'environment-current.json';atomic_state(destination,record)
    print(json.dumps(dict(path=str(destination),package_count=len(packages),sha256=digest(destination))))


if __name__=='__main__':main()
