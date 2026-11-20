"""Operational V4: original V2 science, explicit mixed-V3 history, retry writes."""
from pathlib import Path
import sys,json,hashlib,ast
import fresh_cohort as f
from fresh_io_v3 import durable_write
R=Path(__file__).resolve().parent

def setup():
    assert f.sha(R/'fresh_cohort.py')=='a5df51a12c588c8f34867ad4eb7a870d0f06d1a6a18cd20675287f81bf6a2923'
    assert f.sha(R/'fresh_cohort_v3_imported_archive.py')=='437691d2f50b7705149f1afed0ff1ec55777787bdc38eb56474519fb2699fd77'
    # Confirm all scientific function bodies are identical, excluding the two
    # explicitly documented operational functions and added uuid import.
    def science(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return {n.name:ast.dump(n,include_attributes=False) for n in tree.body
                if isinstance(n,ast.FunctionDef) and n.name not in ['write','prelock']}
    assert science(R/'fresh_cohort.py')==science(R/'fresh_cohort_v3_imported_archive.py')
    spec={'version':4,'scope':'Operational resume; no scientific settings changed',
          'runner_sha256':f.sha(R/'fresh_cohort.py'),'wrapper_sha256':f.sha(Path(__file__)),
          'sample_sha256':f.sha(f.OUT/'sample-manifest.json'),
          'v2_lock_sha256':f.sha(f.OUT/'pre-inference-lock-v2.json'),
          'v3_runtime_lock_sha256':f.sha(f.OUT/'runtime-write-lock-v3.json'),
          'imported_v3_runner_sha256':f.sha(R/'fresh_cohort_v3_imported_archive.py'),
          'incident_sha256':f.sha(R/'OVERLAP_INCIDENT.md'),
          'scientific_function_AST_parity':'PASS except explicitly operational write/prelock'}
    target=f.OUT/'runtime-write-lock-v4.json'
    if target.exists():assert f.read(target)==spec
    else:durable_write(target,spec)
    f.write=durable_write

if __name__=='__main__':
    setup();stage=sys.argv[1]
    assert stage in ['rankers','official','summarize']
    getattr(f,stage)()
