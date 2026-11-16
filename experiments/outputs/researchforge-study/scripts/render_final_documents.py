"""Render the evidence-gated report and author-review manuscript with ReportLab.

Use the bundled document Python, not the frozen experiment environment.
Rendering does not modify numerical evidence. Visual review is a separate step.
"""
from pathlib import Path
import base64
import hashlib
import html
import io
import json
import re
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[3]
FOLDER = ROOT / 'outputs/researchforge-study/deliverables/final-professor-report'
INK, TEAL = colors.HexColor('#173047'), colors.HexColor('#087f8c')
WIDTH = A4[0] - 100


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def setup():
    for name, filename in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf'), ('Arial-Italic', 'ariali.ttf'), ('Arial-BoldItalic', 'arialbi.ttf')]:
        pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + filename))
    pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic', boldItalic='Arial-BoldItalic')


STYLES = {
    'body': ParagraphStyle('body', fontName='Arial', fontSize=10, leading=14.3, spaceAfter=9, textColor=INK),
    'h1': ParagraphStyle('title', fontName='Arial-Bold', fontSize=27, leading=32, spaceAfter=18, textColor=INK),
    'h2': ParagraphStyle('heading', fontName='Arial-Bold', fontSize=17, leading=21, spaceBefore=17, spaceAfter=10, keepWithNext=True, textColor=TEAL),
    'h3': ParagraphStyle('subheading', fontName='Arial-Bold', fontSize=11.5, leading=15, spaceBefore=10, spaceAfter=7, keepWithNext=True, textColor=INK),
    'cell': ParagraphStyle('cell', fontName='Arial', fontSize=8, leading=10.5, textColor=INK),
    'th': ParagraphStyle('th', fontName='Arial-Bold', fontSize=8, leading=10.5, textColor=colors.white),
    'small': ParagraphStyle('small', fontName='Arial', fontSize=8, leading=11, spaceAfter=8, textColor=colors.HexColor('#516778')),
}


def clean(text):
    return text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', ' - ').replace('→', ' / ')


def inline(text):
    text = clean(text)
    text = re.sub(r'</?strong>', lambda m: m.group().replace('strong', 'b'), text)
    text = text.replace('<code>', '<font size="7">').replace('</code>', '</font>')
    return text


def md_inline(text):
    text = html.escape(clean(text))
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`', r'<font size="8">\1</font>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<link href="\2" color="#087f8c">\1</link>', text)
    return text


def para(text, style='body'):
    return Paragraph(inline(text), STYLES[style])


def table(rows):
    n = len(rows[0])
    # Method/contrast labels get more width than numerical cells.
    ratios = [2.5] + [1] * (n - 1) if n >= 3 else [1] * n
    widths = [WIDTH * x / sum(ratios) for x in ratios]
    values = [[para(cell, 'th' if i == 0 else 'cell') for cell in row] for i, row in enumerate(rows)]
    result = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
    result.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), INK), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f1f6f8')]),
        ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#dce5eb')),
        ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    return result


def html_story(source):
    story = [para('RESEARCHFORGE AI  /  FINAL BENCHMARK REVIEW', 'small')]
    for match in re.finditer(r'<(h1|h2|h3|p|table|figure)\b[^>]*>(.*?)</\1>', source, re.S):
        tag, content = match.groups()
        if tag == 'table':
            rows = [re.findall(r'<t[hd]\b[^>]*>(.*?)</t[hd]>', row, re.S)
                    for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', content, re.S)]
            story.extend([table(rows), Spacer(1, 12)])
        elif tag == 'figure':
            data = base64.b64decode(re.search(r'base64,([^\"]+)', content).group(1))
            graphic = Image(io.BytesIO(data))
            scale = min(WIDTH / graphic.imageWidth, 610 / graphic.imageHeight)
            graphic.drawWidth, graphic.drawHeight = graphic.imageWidth * scale, graphic.imageHeight * scale
            caption = re.search(r'<figcaption>(.*?)</figcaption>', content, re.S).group(1)
            story.append(KeepTogether([graphic, Spacer(1, 8), para(caption, 'small')]))
        else:
            story.append(para(content, tag if tag.startswith('h') else 'body'))
    return story


def md_story(text):
    # Presentation-only spacing repair; code, URLs and checksums remain literal.
    protected = {}
    def protect(match):
        key = 'PLACEHOLDER' + chr(65 + len(protected) // 26) + chr(65 + len(protected) % 26) + 'END'
        protected[key] = match.group()
        return key
    text = re.sub(r'`[^`]+`|https?://\S+|\b(?:float16|float32|L2|SHA256|k1|v2)\b|\b[0-9a-f]{32,}\b', protect, text)
    text = re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[A-Za-z])', ' ', text)
    text = re.sub(r',(?=\S)(?!\d{3}(?:\D|$))', ', ', text)
    for key, value in protected.items():
        text = text.replace(key, value)
    story, buffer, rows = [], [], []
    def flush():
        if buffer:
            story.append(para(md_inline(' '.join(buffer))))
            buffer.clear()
        if rows:
            story.extend([table(rows), Spacer(1, 10)])
            rows.clear()
    for line in text.splitlines():
        if line.startswith('|'):
            if buffer:
                flush()
            if not re.match(r'^\|[\s:|\-]+\|$', line):
                rows.append([md_inline(cell.strip()) for cell in line.strip('|').split('|')])
        elif line.startswith('#'):
            flush()
            level = min(len(line) - len(line.lstrip('#')), 3)
            story.append(para(md_inline(line.lstrip('#').strip()), 'h' + str(level)))
        elif not line.strip():
            flush()
        elif re.match(r'^\d+\. ', line):
            flush()
            story.append(para(md_inline(line)))
        else:
            buffer.append(line)
    flush()
    return story


def render(path, story, title):
    def page(canvas, doc):
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1.5)
        canvas.line(50, A4[1] - 34, A4[0] - 50, A4[1] - 34)
        canvas.setFont('Arial', 8)
        canvas.setFillColor(INK)
        canvas.drawString(50, 27, 'ResearchForge AI | Author review | 29 October 2026')
        canvas.drawRightString(A4[0] - 50, 27, str(doc.page))
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=50, rightMargin=50,
                            topMargin=50, bottomMargin=47, title=title, author='ResearchForge research team')
    doc.build(story, onFirstPage=page, onLaterPages=page)


def main():
    setup()
    source = json.loads((FOLDER / 'source_manifest.json').read_text())
    for name, checksum in source['files'].items():
        if digest(FOLDER / name) != checksum:
            raise ValueError('Report source changed: ' + name)
    outputs = []
    report = FOLDER / 'ResearchForge_Final_Professor_Report.pdf'
    render(report, html_story((FOLDER / 'ResearchForge_Final_Professor_Report.html').read_text(encoding='utf-8')), 'ResearchForge: Final Benchmark Review')
    outputs.append(report)
    for name, dest in [('MANUSCRIPT_DRAFT.md', 'ResearchForge_Manuscript_Author_Review.pdf'),
                       ('SUPPLEMENTARY_INFORMATION.md', 'ResearchForge_Supplementary_Information.pdf')]:
        path = FOLDER / dest
        render(path, md_story((FOLDER / name).read_text(encoding='utf-8')), dest.removesuffix('.pdf').replace('_', ' '))
        outputs.append(path)
    records = {}
    for path in outputs:
        reader = PdfReader(path)
        texts = [p.extract_text() or '' for p in reader.pages]
        if any(not t.strip() for t in texts):
            raise ValueError('Empty PDF page: ' + path.name)
        records[path.name] = dict(sha256=digest(path), pages=len(texts), bytes=path.stat().st_size)
    manifest = dict(status='Rendered; text extraction passed; visual review pending',
                    source_manifest_sha256=digest(FOLDER / 'source_manifest.json'),
                    renderer_sha256=digest(__file__), files=records)
    (FOLDER / 'pdf_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
