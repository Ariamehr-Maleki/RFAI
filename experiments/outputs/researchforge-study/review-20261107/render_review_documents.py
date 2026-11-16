"""Render the dated scientific revision without modifying October deliverables."""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil
import sys
from reportlab.platypus import SimpleDocTemplate, Spacer, Image, KeepTogether, Table, TableStyle
from reportlab.graphics.shapes import Drawing, String, Line, Circle, Rect
from reportlab.graphics import renderPDF
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
DEST=STUDY/'deliverables/review-20261108'
sys.path.insert(0,str(STUDY/'scripts'))
import render_final_documents as old

original_clean=old.clean
old.clean=lambda text:original_clean(text).replace('\u2212','-')

original_table=old.table
def adjusted_table(rows):
    if rows[0][0]=='Finding':ratios=[.18,.45,.37]
    elif rows[0][-1]=='Feature identifiers':ratios=[.24,.09,.67]
    elif rows[0][0]=='Evaluation':ratios=[.23,.23,.27,.27]
    else:return original_table(rows)
    values=[[old.para(cell,'th' if i==0 else 'cell') for cell in row] for i,row in enumerate(rows)]
    t=Table(values,colWidths=[old.WIDTH*x for x in ratios],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),old.INK),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f6f8')]),
        ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#dce5eb')),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    return t
old.table=adjusted_table

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def official_figure(r):
    """Vector forest plot of the three declared, paired supplementary contrasts."""
    d=Drawing(old.WIDTH,225)
    d.add(String(0,207,'FROZEN RANKER MINUS OFFICIAL SCORER',fontName='Arial-Bold',fontSize=10,fillColor=old.INK))
    d.add(String(0,190,'Same 1,000 queries; marginal 95% author-cluster intervals',fontName='Arial',fontSize=8,fillColor=old.INK))
    tracks=['published_input','historical_input','harmonized_historical_input']
    labels=['Published inputs*','Historical inputs','Harmonized historical inputs']
    data=[r['contrasts'][t] for t in tracks]
    lo=min(0.,min(x['ci95'][0] for x in data));hi=max(0.,max(x['ci95'][1] for x in data))
    span=max(.02,hi-lo);lo-=span*.15;hi+=span*.15
    left,right=218,old.WIDTH-62
    fx=lambda x:left+(x-lo)/(hi-lo)*(right-left)
    d.add(Line(fx(0),52,fx(0),168,strokeColor=colors.HexColor('#9baab6'),strokeDashArray=[3,3]))
    for i,(label,x) in enumerate(zip(labels,data)):
        y=153-i*40;col=colors.HexColor('#a66f2b') if i==0 else old.TEAL
        d.add(String(0,y-3,label,fontName='Arial',fontSize=9,fillColor=old.INK))
        a,b=map(fx,x['ci95']);v=fx(x['delta'])
        d.add(Line(a,y,b,y,strokeWidth=2,strokeColor=col))
        for z in [a,b]:d.add(Line(z,y-4,z,y+4,strokeColor=col))
        d.add(Circle(v,y,3.5,fillColor=col,strokeColor=col))
        d.add(String(old.WIDTH,y-3,f"{x['delta']:+.4f}",textAnchor='end',fontName='Arial-Bold',fontSize=9,fillColor=col))
    d.add(Line(left,48,right,48,strokeColor=old.INK,strokeWidth=.5))
    for i in range(5):
        value=lo+(hi-lo)*i/4;x=fx(value)
        d.add(Line(x,45,x,48,strokeColor=old.INK))
        d.add(String(x,33,f'{value:+.3f}',textAnchor='middle',fontName='Arial',fontSize=7,fillColor=old.INK))
    d.add(String((left+right)/2,17,'Difference in nDCG@1000',textAnchor='middle',fontName='Arial',fontSize=8,fillColor=old.INK))
    d.add(String(0,0,'*Provenance sensitivity: released inputs can contain future citation information.',fontName='Arial-Italic',fontSize=7.5,fillColor=old.INK))
    return d

def story(text,r,kind):
    blocks=[];cursor=0
    for m in re.finditer(r'^!\[([^\]]+)\]\(([^)]+)\)$',text,re.M):
        blocks.extend(old.md_story(text[cursor:m.start()]))
        caption,filename=m.groups()
        if filename=='figures/official_comparator_contrasts.pdf':graphic=official_figure(r)
        else:
            path=(DEST/filename).resolve()
            if not path.is_relative_to(DEST.resolve()):raise ValueError('Figure outside dated delivery')
            graphic=Image(str(path));scale=min(old.WIDTH/graphic.imageWidth,380/graphic.imageHeight)
            graphic.drawWidth=graphic.imageWidth*scale;graphic.drawHeight=graphic.imageHeight*scale
        blocks.append(KeepTogether([graphic,Spacer(1,10),old.para(old.md_inline(caption),'small')]))
        cursor=m.end()
    blocks.extend(old.md_story(text[cursor:]))
    return blocks

def render(path,content,title):
    def page(canvas,doc):
        canvas.setStrokeColor(old.TEAL);canvas.setLineWidth(1.5)
        canvas.line(50,A4[1]-34,A4[0]-50,A4[1]-34)
        canvas.setFillColor(old.INK);canvas.setFont('Arial',8)
        canvas.drawString(50,27,'ResearchForge | Revised author review | 8 November 2026')
        canvas.drawRightString(A4[0]-50,27,str(doc.page))
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=50,bottomMargin=47,
        title=title,author='ResearchForge research team; author approval pending')
    doc.build(content,onFirstPage=page,onLaterPages=page)

def main():
    old.setup()
    old.STYLES['body'].allowWidows=0
    old.STYLES['body'].allowOrphans=0
    r=read(HERE/'official-comparator-results.json')
    if not r['status'].startswith('Complete'):raise ValueError('Incomplete experiment')
    source=read(DEST/'source_manifest.json')
    for name,value in source['files'].items():
        if sha(DEST/name)!=value:raise ValueError('Source changed: '+name)
    (DEST/'figures').mkdir(exist_ok=True)
    originals=STUDY/'deliverables/final-professor-report/figures'
    for ext in ['png','pdf']:
        shutil.copy2(originals/('figure2_factorial.'+ext),DEST/'figures'/('factorial_original.'+ext))
    renderPDF.drawToFile(official_figure(r),str(DEST/'figures/official_comparator_contrasts.pdf'))
    with (DEST/'official_comparator_results.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['method','ndcg1000','recall10','new_recall10','candidate_recall','ranker_minus_method','ci_low','ci_high','holm_p'])
        m=r['frozen_ranker_metrics'];w.writerow(['frozen_practical_ranker']+[m[k] for k in ['ndcg1000','recall10','new_recall10','candidate_recall']]+['','','',''])
        for name,v in r['results'].items():
            c=r['contrasts'][name];m=v['metrics']
            w.writerow([name]+[m[k] for k in ['ndcg1000','recall10','new_recall10','candidate_recall']]+[c['delta'],*c['ci95'],c['holm_p_three_supplementary_contrasts']])
    files={}
    for src,dest,kind,title in [
        ('PROFESSOR_REVIEW_REPORT.md','ResearchForge_Revised_Professor_Report.pdf','report','ResearchForge: Scientific Review and Updated Evidence'),
        ('MANUSCRIPT_REVISED.md','ResearchForge_Revised_Manuscript.pdf','manuscript','Evaluating Ranking Objectives and Temporal Information in Scientific Collaborator Forecasting'),
        ('SUPPLEMENTARY_REVISED.md','ResearchForge_Revised_Supplement.pdf','supplement','ResearchForge: Revised Supplementary Information')]:
        content=(DEST/src).read_text(encoding='utf-8')
        render(DEST/dest,story(content,r,kind),title)
        reader=PdfReader(DEST/dest);texts=[p.extract_text() or '' for p in reader.pages]
        if any(not t.strip() for t in texts):raise ValueError('Blank PDF page')
        joined=' '.join(texts)
        if '0.4880' not in joined:raise ValueError('Missing original numeric anchor')
        if f"{r['results']['harmonized_historical_input']['metrics']['ndcg1000']:.6f}" not in joined:raise ValueError('Missing new numeric anchor')
        files[dest]=dict(pages=len(texts),bytes=(DEST/dest).stat().st_size,sha256=sha(DEST/dest))
    manifest=dict(status='Rendered; text and numeric anchors pass; visual inspection pending',
        source_manifest_sha256=sha(DEST/'source_manifest.json'),renderer_sha256=sha(Path(__file__)),files=files,
        base_renderer_sha256=sha(STUDY/'scripts/render_final_documents.py'),python=sys.version,
        figure_files={str(p.relative_to(DEST)).replace('\\','/'):sha(p) for p in (DEST/'figures').iterdir() if p.is_file()})
    (DEST/'pdf_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)

if __name__=='__main__':main()
