"""Build page contact sheets after Poppler rendering; does not assert visual QA."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
DEST=STUDY/'deliverables/review-20261108'
TMP=STUDY/'tmp/pdfs/review-20261108'

def main():
    manifest=json.loads((DEST/'pdf_manifest.json').read_text(encoding='utf-8'))
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',16)
    inventory={}
    for filename,prefix in [('ResearchForge_Revised_Professor_Report.pdf','report'),
                            ('ResearchForge_Revised_Manuscript.pdf','manuscript'),
                            ('ResearchForge_Revised_Supplement.pdf','supplement')]:
        reader=PdfReader(DEST/filename)
        images=sorted(TMP.glob(prefix+'-*.png'))
        if len(images)!=len(reader.pages):raise ValueError('Rendered page count mismatch')
        text='\n'.join(p.extract_text() or '' for p in reader.pages)
        if any(s in text for s in ['PLACEHOLDER','![Figure','\ufffd']):raise ValueError('Unrendered markup or missing character')
        sheets=[]
        for offset in range(0,len(images),4):
            group=images[offset:offset+4]
            contact=Image.new('RGB',(940,1390),'#dce5eb');draw=ImageDraw.Draw(contact)
            for j,path in enumerate(group):
                im=Image.open(path).convert('RGB');im.thumbnail((450,650))
                x=10+(j%2)*470;y=30+(j//2)*690
                contact.paste(im,(x,y))
                draw.text((x,y-23),f'{prefix} - page {offset+j+1}',font=font,fill='#173047')
            path=TMP/f'contact-{prefix}-{offset//4+1}.png'
            contact.save(path);sheets.append(str(path))
        inventory[filename]=dict(pages=len(reader.pages),sheets=sheets,pdf_sha256=manifest['files'][filename]['sha256'])
    (TMP/'render_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(json.dumps(inventory),flush=True)

if __name__=='__main__':main()
