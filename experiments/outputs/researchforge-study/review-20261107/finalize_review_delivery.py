"""Record the visual inspection actually performed on these exact PDF bytes."""
from pathlib import Path
import hashlib
import json
import shutil

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
DEST=STUDY/'deliverables/review-20261108'
TMP=STUDY/'tmp/pdfs/review-20261108'
OBSERVED={
 'ResearchForge_Revised_Professor_Report.pdf':'4e1ee0d51335a7c1a8be52702e93dd57add78d9d64d4353adacc8645366d31d6',
 'ResearchForge_Revised_Manuscript.pdf':'2694e9785660a39c9f87c059e95efedd6847c62de2f858aefa705c56a0744735',
 'ResearchForge_Revised_Supplement.pdf':'b4e0ece3b5d225f46a39c1c62536d4fccad6337f2ed18a47445f3d03d35cc7cb'}

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    for name,value in OBSERVED.items():
        if sha(DEST/name)!=value:raise ValueError('PDF differs from visually inspected version: '+name)
    manifest=json.loads((DEST/'pdf_manifest.json').read_text(encoding='utf-8'))
    inventory=json.loads((TMP/'render_inventory.json').read_text(encoding='utf-8'))
    if sum(x['pages'] for x in inventory.values())!=23:raise ValueError('Unexpected page count')
    if any(inventory[n]['pdf_sha256']!=h for n,h in OBSERVED.items()):raise ValueError('Raster provenance mismatch')
    manifest['status']='Rendered; numerical anchors, text extraction and 23-page visual inspection passed'
    (DEST/'pdf_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    qa=dict(status='passed',date='2026-11-08',pdf_sha256=OBSERVED,pages_reviewed=23,
        review_method='All 23 final pages inspected in contact sheets; four critical pages also inspected at native 105 dpi; original factorial plot inspected separately.',
        critical_full_pages=['report-2.png','manuscript-08.png','manuscript-13.png','supplement-2.png'],
        corrected=['Removed redundant last report paragraph that created a nearly empty fifth page.',
                   'Disabled body-paragraph widows/orphans and re-rendered every final page.',
                   'Preserved jaccard10 as a literal feature identifier and normalized Unicode minus in PDF prose.'],
        checks=['No observed clipping, overlap or missing glyphs.',
                'Table headers, numeric cells, confidence intervals and both figures are legible.',
                'Headers, footers, numbering, section transitions and reference wrapping inspected.',
                'Text extraction contains no unresolved figure markup, replacement characters or template placeholders.',
                'Named-author declarations intentionally remain pending; no scientific or author approval is fabricated.'],
        pdf_manifest_sha256=sha(DEST/'pdf_manifest.json'),
        contact_sheets={p.name:sha(p) for p in TMP.glob('contact-*.png')})
    (DEST/'VISUAL_QA.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
    readme='''# ResearchForge: revised supervisor-review package

8 November 2026. Read this revision for the current scientific interpretation.
The original October results and archives remain preserved.

## The three PDFs

1. ResearchForge_Revised_Professor_Report.pdf — four-page English overview,
   benchmark results, two figures and a response to all eleven internal-review findings.
2. ResearchForge_Revised_Manuscript.pdf — thirteen-page revised author-review
   manuscript, including the official comparator, limitations and AI-use disclosure.
3. ResearchForge_Revised_Supplement.pdf — six pages of experimental details,
   supplementary analyses, uncertainty assumptions and reproducibility boundaries.

The title has 10 words and the abstract 170. format_audit.json records the
main-text count against Scientific Reports guidance. These are author-review
documents, not an accepted paper or an independent professor's endorsement.

## The finding that changes the earlier interpretation

The exact official scorer was found on non-default PreScience repository branches.
Across the same 1,000 queries, its harmonized historical-input nDCG@1000 is
0.507121 versus 0.488033 for the frozen practical ranker. Ranker minus comparator
is -0.019088, with marginal 95% author-cluster interval [-0.027010, -0.011302]
and Holm-adjusted p=0.000300 in the separate three-comparison family.
Its advantage remains after historical filtering; the temporal audit does not
explain away that result. The previous prose approximation remains separately labeled.

The ranker's advantage over its matched classifier and the negative fixed-pool
text effect remain under the additional shared-team dependence diagnostic.
There is no claim of a new ranking algorithm, general superiority, or proven
efficacy of the original eight-agent application. The three benchmarks are
different tasks. All new comparator and dependence analyses are post-result.

## Supporting evidence and reproduction

REVIEW_RESOLUTION.md responds to the initial REFEREE_REPORT.md. The CSV and JSON
files retain full-precision results. final-output-audit.json binds all 3,000
query/track checkpoints and verifies exact cohort/relevance identity with the
original three model seeds. Independent pytrec_eval disagreement is at most
3.33e-16. VISUAL_QA.json records inspection of all 23 final PDF pages.

The compact professor ZIP contains these review documents. The separate dated
ResearchForge_Review_Extension_20261108.zip contains the added executable sources,
upstream snapshots and licenses, protocols, predictions, checkpoints and logs.
Use it together with the preserved original researchforge-reproducibility.zip.
The extension includes the exact earlier Markdown, statistics/renderer and
factorial-figure dependencies needed to rebuild this revision. ZIP receipts
record full payload read-back and SHA256 verification.

REPRODUCTION.md gives the frozen experiment commands. After a fresh completed
comparator run, run audit_review_outputs.py before build_review_revision.py.
render_review_documents.py uses the document Python/ReportLab runtime. A newly
rendered PDF must receive a fresh visual review; the existing visual sign-off
is bound to the delivered PDF hashes and must not be reused for changed files.

## Before journal submission

Human authors must decide whether the bounded empirical contribution is sufficient,
review and approve the manuscript, provide authorship/affiliations/contributions,
funding and conflict statements, approve AI-use disclosure and arrange actual
reviewer/public access to code and data. No repository push, public deposit,
external message, fee payment or journal submission has occurred. Acceptance by
Scientific Reports is unknown, and an external reviewer may request further work.
'''
    (DEST/'READ_ME_FIRST.md').write_text(readme,encoding='utf-8')
    print(json.dumps({'visual_QA':'passed','pages':23,'PDFs':list(OBSERVED)}),flush=True)

if __name__=='__main__':main()
