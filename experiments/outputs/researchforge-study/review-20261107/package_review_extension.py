"""Package this dated additive extension; verify every ZIP payload by read-back."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
ROOT=STUDY.parents[1]
DEST=STUDY/'deliverables/review-20261108'

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def package(path,files,relative_to):
    files=sorted(files)
    entries={str(p.relative_to(relative_to)).replace('\\','/'):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}
    if len(entries)!=len(files):raise ValueError('Duplicate archive path')
    manifest=dict(description='November review extension; original October package is unchanged',files=entries)
    temporary=path.with_suffix(path.suffix+'.partial')
    with ZipFile(temporary,'w',compression=ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in files:z.write(p,str(p.relative_to(relative_to)).replace('\\','/'))
        z.writestr('EXTENSION_MANIFEST.json',json.dumps(manifest,indent=2))
    with ZipFile(temporary) as z:
        if len(z.namelist())!=len(entries)+1:raise ValueError('Archive entry count mismatch')
        for name,expected in entries.items():
            with z.open(name) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected['sha256']:raise ValueError('Read-back failed: '+name)
    temporary.replace(path)
    receipt=dict(status='Every archived payload read back and SHA256 verified',archive=path.name,
        sha256=sha(path),bytes=path.stat().st_size,payloads=len(entries),manifest=manifest,
        original_reproducibility_archive_sha256='94dc644363e1a62ae619664af5359c56135ab76e4b9156a389cef2180845dedb')
    path.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='manifest'}),flush=True)

def allowed(p):
    return p.is_file() and '__pycache__' not in p.parts and 'plot-cache' not in p.parts and not p.name.endswith(('.part','.partial'))

def main():
    r=json.loads((HERE/'official-comparator-results.json').read_text(encoding='utf-8'))
    if not r['status'].startswith('Complete'):raise ValueError('Complete experiment required')
    qa=json.loads((DEST/'VISUAL_QA.json').read_text(encoding='utf-8'))
    if qa.get('status')!='passed':raise ValueError('Visual inspection required')
    # Compact supervisor bundle contains the three PDFs and supporting review tables/text.
    files=[p for p in DEST.rglob('*') if allowed(p) and 'qa' not in p.relative_to(DEST).parts]
    package(STUDY/'deliverables/ResearchForge_Revised_Professor_Review_20261108.zip',files,DEST)
    # Extension retains workspace paths and all scorer checkpoints for direct resumption.
    files=[p for p in HERE.rglob('*') if allowed(p)]
    files += [STUDY/'scripts/render_final_documents.py',STUDY/'scripts/final_statistics.py',
        STUDY/'MANUSCRIPT_DRAFT.md',STUDY/'SUPPLEMENTARY_INFORMATION.md']
    files += [STUDY/'deliverables/final-professor-report/figures'/('figure2_factorial.'+ext) for ext in ['png','pdf']]
    files += [p for p in DEST.rglob('*') if allowed(p) and 'qa' not in p.relative_to(DEST).parts]
    files += sorted((ROOT/'work').glob('review-official*.log'))
    package(STUDY/'deliverables/ResearchForge_Review_Extension_20261108.zip',files,ROOT)

if __name__=='__main__':main()
