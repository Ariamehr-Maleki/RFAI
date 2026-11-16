"""Build a local, audited ZIP after all three frozen evaluations complete.

Raw corpora/indices are referenced by checksums, not redistributed. Does not
publish/upload, infer a software license, or declare a manuscript accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from final_evaluation_gate import verify_lock
from final_results_tables import checked_report,STATUSES
from frozen_ranker import digest
from supervisor_state import atomic_state

ROOT=Path(__file__).resolve().parents[3]

def archive_files(root,files,destination,metadata,reserve=1024**3):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if not destination.is_relative_to(root) or destination.exists():raise ValueError('New workspace ZIP required')
    partial=destination.with_suffix('.partial')
    if partial.exists():raise ValueError('Inspect earlier partial archive before retrying')
    inventory={}
    for path in sorted(set(Path(p).resolve() for p in files)):
        if not path.is_relative_to(root) or not path.is_file():raise ValueError('Nonlocal/missing package artifact')
        inventory[path.relative_to(root).as_posix()]=dict(bytes=path.stat().st_size,sha256=digest(path))
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(**metadata,files=inventory)
    with zipfile.ZipFile(partial,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as archive:
        for name,expected in inventory.items():
            h=hashlib.sha256();size=0
            with (root/name).open('rb') as source,archive.open(name,'w',force_zip64=True) as sink:
                while chunk:=source.read(1024**2):
                    if shutil.disk_usage(root).free<reserve:raise OSError('Archive disk reserve reached; partial retained')
                    h.update(chunk);size+=len(chunk);sink.write(chunk)
            if h.hexdigest()!=expected['sha256'] or size!=expected['bytes']:raise ValueError('Source changed during archiving: '+name)
        archive.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    verify_archive(partial)
    partial.replace(destination)
    atomic_state(destination.with_suffix('.receipt.json'),dict(status='PASS archive read-back and every payload SHA256',
        sha256=digest(destination),bytes=destination.stat().st_size,file_count=len(inventory),metadata=metadata))
    return destination

def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate archive entries')
        manifest=json.loads(archive.read('PACKAGE_MANIFEST.json'))
        if set(names)!=set(manifest['files'])|{'PACKAGE_MANIFEST.json'}:raise ValueError('Archive inventory differs')
        for name,expected in manifest['files'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe archive path')
            h=hashlib.sha256();size=0
            with archive.open(name) as source:
                while chunk:=source.read(1024**2):h.update(chunk);size+=len(chunk)
            if size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:raise ValueError('Archive payload differs: '+name)
    return manifest

def build(root,lock_path,destination):
    root=Path(root).resolve();study=root/'outputs/researchforge-study';lock_path=Path(lock_path).resolve()
    lock=verify_lock(root,lock_path);files={lock_path};excluded={}
    for dataset in STATUSES:
        path,report=checked_report(root,lock,dataset);files.add(path)
        files.update(path.parent/name for name in report['output_sha256'])
    for cfg in lock['payload']['plan']['datasets'].values():
        for artifact in cfg['artifacts']:
            path=root/artifact['path']
            if artifact['role'] in ['code','model','protocol','environment','development_report'] or path.is_relative_to(study/'data-audit') or path.name=='isolated_pipeline_cost.json':files.add(path)
            else:excluded[artifact['path']]=lock['payload']['artifacts'][Path(artifact['path']).as_posix()]
    files.update((study/'scripts').glob('*.py'))
    files.update(study.glob('*.md'));files.update(study.glob('*.txt'));files.update(study.glob('*runtime.json'))
    # Explicit third-party notices accompany vendored source; no blanket relicensing.
    ncn=root/'work/NeuralCommonNeighbor'
    notices=list(ncn.glob('LICENSE*'))+list(ncn.glob('COPYING*'))+list(ncn.glob('README*'))
    if not notices:raise ValueError('Missing upstream NCN attribution/notice')
    files.update(notices)
    for directory,status in [
        (study/'results/final/tables','Verified frozen-result table export'),
        (study/'results/final/figures','Verified frozen-result figures'),
    ]:
        provenance=directory/'provenance.json'
        if not provenance.exists():raise ValueError('Final table/figure export required')
        record=json.loads(provenance.read_text())
        if record['lock_sha256']!=lock['lock_sha256']:raise ValueError('Export uses another lock')
        if record.get('status')!=status or not record.get('files'):raise ValueError('Incomplete or wrong export type')
        for name,sha in record['files'].items():
            path=(directory/name).resolve()
            if not path.is_relative_to(directory.resolve()):raise ValueError('Export payload outside its directory')
            if digest(path)!=sha:raise ValueError('Export changed: '+name)
            files.add(path)
        files.add(provenance)
    metadata=dict(status='Local research reproducibility package; submission and author declarations are separate',
        lock_sha256=lock['lock_sha256'],excluded_source_artifacts=excluded,
        raw_data_policy='Download upstream data/model assets separately under their licenses; verify listed checksums.',
        scope='Three frozen evaluations, all declared comparisons including negatives, source/models/protocol and final exports.',
        license_note='No new license is assigned. Retain original provider and upstream code licenses.',builder_sha256=digest(__file__))
    return archive_files(root,files,destination,metadata)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lock',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();print(build(ROOT,args.lock,args.output))
