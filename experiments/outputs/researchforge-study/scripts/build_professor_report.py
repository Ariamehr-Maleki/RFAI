"""Dated professor report. Reads saved DEVELOPMENT evidence; never opens test labels."""
from pathlib import Path
import json, hashlib, re, html, csv
from datetime import datetime, timezone, timedelta
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Flowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs/researchforge-study'
OUT = BASE / 'deliverables/professor-report-20261027'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {}
def read(rel):
    p = BASE / rel
    raw = p.read_bytes()
    SOURCES[rel] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw)

hy = 'results/hybrid-graph-dedup-200-100-1000-s42/'
H = read(hy+'report.json')
F = read(hy+'factorial-s42/report.json')
G = read(hy+'subgroups-lambdamart-vs-binary_lightgbm.json')
O = read('results/ogb-expanded-10000/report.json')
N = read('results/ncn-expanded-10000/report.json')
E = read('results/exharmony-expanded-1000-500/report.json')
L = read('data-audit/loader-equivalence.json')
read('data-audit/train_semantic_index.json')
read('data-audit/final_lexical_index.json')
read('data-audit/expanded-memory-recovery.json')
read(hy+'independent_metric_verification.json')
read(hy+'factorial-s42/independent_metric_verification.json')
for seed in (42,43,44):
    read(f'results/ncn-expanded-10000-e100-s{seed}/independent_metric_verification.json')
for rel in ['AUDIT.md','PROTOCOL.md','MANUSCRIPT_DRAFT.md','RESEARCH_UPDATE_20261027.md','REPRODUCIBILITY.md','PUBLICATION_READINESS.md']:
    b=(BASE/rel).read_bytes(); SOURCES[rel]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
progress = json.loads((ROOT/'work/indices/prescience-test-minilm/progress.json').read_text())
stamp = datetime.now(timezone(timedelta(hours=3,minutes=30))).strftime('%d %B %Y at %H:%M Tehran time')
if (OUT/'source_manifest.json').exists():
    snapshot=json.loads((OUT/'source_manifest.json').read_text(encoding='utf-8'))
    stamp=snapshot['created'];progress=snapshot['label_free_progress_snapshot']
for name, file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf'),('Arial-BoldItalic','arialbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-BoldItalic')
INK=colors.HexColor('#162b40'); TEAL=colors.HexColor('#087f8c'); BLUE=colors.HexColor('#376eab'); GRAY=colors.HexColor('#667789'); LIGHT=colors.HexColor('#f2f6f9'); ORANGE=colors.HexColor('#ae672b')
W=512
ST=getSampleStyleSheet()
for name in ['Normal','BodyText']:
    ST[name].fontName='Arial'; ST[name].fontSize=10.8; ST[name].leading=15.1; ST[name].spaceAfter=9
ST.add(ParagraphStyle('TitleRF',fontName='Arial-Bold',fontSize=31,leading=35,textColor=colors.black,spaceAfter=18))
ST.add(ParagraphStyle('Deck',fontName='Arial',fontSize=14,leading=20,spaceAfter=18,textColor=INK))
ST.add(ParagraphStyle('H',fontName='Arial-Bold',fontSize=21,leading=25,spaceAfter=16,textColor=colors.black,keepWithNext=True))
ST.add(ParagraphStyle('Sub',fontName='Arial-Bold',fontSize=12,leading=16,spaceBefore=9,spaceAfter=7,textColor=colors.black,keepWithNext=True))
ST.add(ParagraphStyle('SmallRF',fontName='Arial',fontSize=9,leading=12,spaceAfter=8,textColor=GRAY))
ST.add(ParagraphStyle('Cell',fontName='Arial',fontSize=9.3,leading=12))
ST.add(ParagraphStyle('CellHead',fontName='Arial-Bold',fontSize=9.3,leading=12,textColor=colors.white))
ST.add(ParagraphStyle('Refs',fontName='Arial',fontSize=9.5,leading=13,spaceAfter=10))
ST.add(ParagraphStyle('ManH',fontName='Arial-Bold',fontSize=15,leading=19,spaceBefore=13,spaceAfter=9,keepWithNext=True))

def clean(s):
    return str(s).replace('\u2013','-').replace('\u2014','-').replace('\u2011','-').replace('\u2212','-').replace('\u00a0',' ')
def markup(s):
    s=html.escape(clean(s))
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'\1',s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<a href="\2" color="#376eab">\1</a>',s)
    return s
def par(s,style='Normal'): return Paragraph(markup(s),ST[style])
MD=[]; story=[]
def p(s,style='Normal'):
    story.append(par(s,style)); MD.append(s+'\n')
def h(s): story.append(par(s,'Sub'));MD.append('### '+s+'\n')
def page(num,title):
    if story: story.append(PageBreak())
    p(f'RESEARCHFORGE AI   /   {num:02d}', 'SmallRF')
    story.append(par(title,'H'));MD.append('## '+title+'\n')
def table(headers,rows,widths,padding=6):
    data=[[par(x,'CellHead') for x in headers]]+[[par(x,'Cell') for x in row] for row in rows]
    t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
    ts=[('BACKGROUND',(0,0),(-1,0),INK),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#d9d9d9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),padding),('BOTTOMPADDING',(0,0),(-1,-1),padding)]
    for i in range(1,len(data)):
        if i%2==0:ts.append(('BACKGROUND',(0,i),(-1,i),LIGHT))
    t.setStyle(TableStyle(ts));story.extend([t,Spacer(1,9)])
    MD.append('| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n')
def cap(s):p(s,'SmallRF')
def fmt(x):return f'{x:.4f}'
def mean_sd(d,sd='sample_sd'):return f"{d['mean']:.4f} ± {d[sd]:.4f}"

class Bars(Flowable):
    def __init__(self,rows,maxv=.7,labelwidth=190,height=None,xlabel='Score',errors=False):
        Flowable.__init__(self);self.rows=rows;self.maxv=maxv;self.labelwidth=labelwidth;self.width=W;self.height=height or (len(rows)*25+40);self.xlabel=xlabel;self.errors=errors
    def draw(self):
        c=self.canv;left=self.labelwidth;right=W-58;scale=(right-left)/self.maxv;step=(self.height-42)/len(self.rows)
        c.setFont('Arial',8.5)
        for j in range(5):
            v=self.maxv*j/4;x=left+v*scale;c.setStrokeColor(colors.HexColor('#e2e9ef'));c.setLineWidth(.5);c.line(x,31,x,self.height-3);c.setFillColor(GRAY);c.drawCentredString(x,18,f'{v:.2f}')
        for i,row in enumerate(self.rows):
            label,value,col,*err=row;y=self.height-12-i*step
            c.setFillColor(colors.black);c.setFont('Arial',9.2);c.drawRightString(left-9,y-3,label)
            c.setFillColor(col);c.rect(left,y-6,max(value*scale,.3),12,stroke=0,fill=1)
            if self.errors and err:
                sd=err[0];c.setStrokeColor(INK);c.setLineWidth(1);c.line(left+(value-sd)*scale,y,left+(value+sd)*scale,y)
                for v in [value-sd,value+sd]: c.line(left+v*scale,y-3,left+v*scale,y+3)
            c.setFillColor(INK);c.setFont('Arial',9);c.drawRightString(W,y-3,f'{value:.4f}')
        c.setFont('Arial',9);c.setFillColor(GRAY);c.drawCentredString((left+right)/2,2,self.xlabel)

class Forest(Flowable):
    def __init__(self,rows):Flowable.__init__(self);self.rows=rows;self.width=W;self.height=142
    def draw(self):
        c=self.canv;left=177;right=400;lo=-.07;hi=.13
        xx=lambda v:left+(v-lo)/(hi-lo)*(right-left)
        c.setStrokeColor(GRAY);c.setDash(2,3);c.line(xx(0),30,xx(0),138);c.setDash()
        for i,(label,g) in enumerate(self.rows):
            y=123-36*i;c.setFont('Arial',9.2);c.setFillColor(colors.black);c.drawRightString(left-10,y-3,label)
            a,b=g['ci95'];c.setStrokeColor(TEAL);c.setLineWidth(1.6);c.line(xx(a),y,xx(b),y);c.setFillColor(TEAL);c.circle(xx(g['delta']),y,3,fill=1,stroke=0)
            c.setFont('Arial',9);c.drawRightString(W,y-3,f"{g['delta']:+.4f}")
        for v in [-.05,0,.05,.10]:c.setFillColor(GRAY);c.drawCentredString(xx(v),17,f'{v:+.2f}')
        c.drawCentredString((left+right)/2,2,'LambdaMART minus matched classifier')

class Pipeline(Flowable):
    def __init__(self):Flowable.__init__(self);self.width=W;self.height=96
    def draw(self):
        c=self.canv;labels=[('1','Historical records','Strict date cutoff'),('2','Candidate retrieval','Text and network'),('3','Learned ranking','23 measured features'),('4','Offline evaluation','Coverage and ranking')]
        for i,(n,a,b) in enumerate(labels):
            x=60+i*130;c.setFillColor(TEAL);c.circle(x,71,13,stroke=0,fill=1);c.setFillColor(colors.white);c.setFont('Arial-Bold',12);c.drawCentredString(x,67,n)
            c.setFillColor(INK);c.setFont('Arial-Bold',9.5);c.drawCentredString(x,42,a);c.setFont('Arial',8.5);c.setFillColor(GRAY);c.drawCentredString(x,26,b)
            if i<3:c.setStrokeColor(GRAY);c.setLineWidth(.8);c.line(x+22,71,x+104,71);c.line(x+99,74,x+104,71);c.line(x+99,68,x+104,71)

class NumberedCanvas(canvas.Canvas):
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.saved=[]
    def showPage(self):self.saved.append(dict(self.__dict__));self._startPage()
    def save(self):
        total=len(self.saved)
        for st in self.saved:
            self.__dict__.update(st);self.setFont('Arial',8);self.setFillColor(GRAY);self.drawString(50,31,'ResearchForge AI  |  Research evidence and manuscript preparation');self.drawRightString(562,31,f'{self._pageNumber} / {total}');canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)
def build(path,items,title):
    SimpleDocTemplate(str(path),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=42,bottomMargin=53,title=title,author='ResearchForge AI study',pageCompression=1).build(items,canvasmaker=NumberedCanvas)

# 1 Executive review
p('RESEARCH REPORT   |   27 OCTOBER 2026','SmallRF')
story.append(Spacer(1,14));p('ResearchForge AI\nResearch and Benchmark Report','TitleRF')
p('Scientific collaborator discovery with text, collaboration networks and learning to rank','Deck')
p('Prepared for academic supervision and manuscript development toward Scientific Reports. Evidence snapshot: '+stamp+'.','SmallRF')
p('We have converted the original demonstration into a reproducible offline evaluation study. Real-data development experiments now cover collaborator forecasting, temporal link prediction and expertise retrieval. The central finding so far is conditional: learned ranking helps in the small PreScience pilot, but it does not consistently outperform strong alternatives across tasks.')
table(['Evidence','Verified development finding'],[
['PreScience','LambdaMART nDCG@1000 '+fmt(H['metrics']['lambdamart']['ndcg1000'])+' versus '+fmt(H['metrics']['binary_lightgbm']['ndcg1000'])+' for the matched classifier.'],
['ogbl-collab','NCN has the highest reported mean Hits@50 among the compared models: '+fmt(N['mean_hits50'])+'.'],
['Adapted exHarmony','Maximum BM25 leads Authors nDCG@10; learned models have higher means on citation-derived judgments.'],
['Final evaluation','Held-out labels remain sealed. Expanded PreScience development, isolated timing and the global configuration freeze are still pending.']],[117,395])
h('What the professor can assess now')
p('The research question, implemented method, comparison design, data audits and development evidence are ready for substantive review. The manuscript is a structured draft with detailed Methods. Final Abstract, Results and evidence-specific Discussion require the planned held-out runs. No submission or public release of the benchmark package has occurred.')
cap('Reading guide: project audit 2; related work 3; benchmark protocol 4; method 5; results 6-10; verification and costs 11; completion plan 12; references 13; extended results 14; evidence inventory 15.')

# 2
page(2,'From the original project to a research study')
p('The original repository and all supplied materials were reviewed: the two-page abstract, four-page paper, ten-slide presentation, demonstration infographic, README and principal implementation files. The code audit was anchored to repository commit 57243e190904f6ca7e6796734c32878ba8cd6c45 [12].')
p('ResearchForge organizes researcher discovery, profile building, matching, explanations, proposal generation, outreach drafting, memory and orchestration into eight named roles. These provide an application setting. The current study evaluates the matching component against observed records.')
table(['Audit finding','Research response'],[
['The inspected matching path did not populate the project embedding.','Implemented a separately versioned content retrieval and ranking pipeline.'],
['Matching was limited to the last ten researcher entries.','Introduced corpus retrieval, fixed candidate budgets and candidate-recall measurement.'],
['Skills, career and geography fields were partly constant or unavailable.','Used observed publication histories and IDs; did not invent missing benchmark metadata.'],
['Profiles included synthetic or placeholder metadata.','Excluded those fields from scientific experiments.'],
['The diagram implied A2A and indexed retrieval beyond the verified execution path.','Described implemented behavior; made no protocol-compliance or multi-agent efficiency claim.'],
['Time-saving, quality and some bibliography claims lacked support.','Replaced them with measured offline metrics and verified source identifiers.']],[240,272])
h('Evaluation boundary')
p('No human evaluation, survey or LLM judge is used. Historical coauthorship and expertise labels provide reproducible proxies. They do not establish collaboration willingness, compatibility, productivity, invitation acceptance, explanation usefulness or generated-proposal quality. The original interface can present ranked evidence, but its downstream drafting functions remain outside the effectiveness claim.')

# 3
page(3,'Research question and relation to prior work')
p('The study asks where publication text, network history and explicit recency improve collaborator ranking, how candidate coverage affects that result, and what computational cost is required. We separately test whether a ranking objective offers value over classification with identical inputs.')
table(['Prior work','Relationship to this study'],[
['CoBERT 2021 [1]','Sequential collaboration prediction; motivates distinguishing new from repeat partners.'],
['TEDRec 2025 [2]','Textual and dynamic network modeling. Its accessible abstract establishes overlap; full implementation was not verified.'],
['PreScience v2 2026 [3]','Forecasting benchmark and strong fixed history, propagation, citation and activity scorer. Implemented with disclosed ambiguities.'],
['exHarmony 2025 [4]','Expertise/reviewer-assignment proxies from Authors, Cite and SimCite judgments; used as a complementary adapted task.'],
['NCN 2024 [5]','Neural link-prediction comparator. Three 100-epoch development runs completed; the completion extension was not run.'],
['I-CAR 2026 [8]','Combines temporal, semantic and diffusion signals with supervised classifiers; reinforces the matched-classifier comparison.'],
['Hypergraph approaches 2026 [9,10]','Richer collaboration structures with different data or task protocols; reviewed but not reproduced.']],[155,357])
p('The intended contribution is a reproducible empirical account of conditional benefits and failures. Text/network/time fusion and library-based LambdaMART are established ideas. Neither adding agents nor using a ranking library is treated as algorithmic novelty.')
p('The review is targeted, not a systematic literature review. For the SCH repository [9], inspected split/evaluation behavior would require adaptation to preserve this study\'s sealed-test policy. This is an implementation compatibility observation, not a conclusion about the validity of its published experiments. The systematic-review paper [11] was examined through its publisher abstract only.')

# 4
page(4,'Benchmarks and information available at prediction')
table(['Task','Development design','Primary metric'],[
['PreScience\nCollaborator forecasting','200 training and 100 validation queries in the completed semantic pilot. Expanded run planned for 1,000 training groups per seed and 500 common validation queries.','nDCG@1000'],
['ogbl-collab\nFuture coauthor edges','10,000 positive 2017 training edges per seed; five negatives per positive. 60,084 validation positives and 100,000 shared negatives.','Hits@50'],
['Adapted exHarmony\nExpert retrieval','1,000 training and 500 validation queries per seed. Author/title-component partition of 4,766 / 1,589 / 1,589 queries.','Authors nDCG@10']],[125,289,98])
h('PreScience historical forecasting')
p('The query is the first eligible author of a target paper; relevance consists of its other eligible authors. Eligibility requires earlier publications. All evidence must precede the target date, with same-day records excluded. Target title, abstract and references are hidden. Development is divided chronologically at 1 July 2024. The downloaded TRAIN release contains 373,716 records, nine more than the paper reports; that discrepancy remains unresolved.')
h('OGB temporal graph prediction')
p('Training features use the graph through 2016 for 2017 labels. Validation features use the graph through 2017 for 2018 labels. Official Hits@50 compares positive edges with a shared negative threshold. Final inference is planned to retain the graph through 2017. Released node-vector provenance is not temporally certified, including for the NCN track.')
h('exHarmony adaptation')
p('Whole components of shared query authors or normalized titles determine the split. Of 1,204,150 collection records, 776 title matches were conservatively excluded, leaving 1,203,374 indexed records. Exact query-ID overlap was zero. Collection dates are unavailable, so temporal construction cannot be independently revalidated. Missing-profile relevant authors remain in denominators; empty retrieval queries remain in evaluation.')
cap('These tasks use different units, relevance proxies and metric definitions. Their scores cannot be pooled into one overall accuracy number. All reported effectiveness numbers in this report are development validation results.')

# 5
page(5,'Implemented retrieval and learning to rank')
story.append(Pipeline());MD.append('Pipeline: historical records -> candidate retrieval -> learned ranking -> offline evaluation.\n')
p('For each PreScience query, up to ten earlier papers define the seed profile. Lexical retrieval uses nonnegative unigram hashing into 262,144 dimensions and L2 normalization, without a fitted vocabulary or IDF. Semantic retrieval uses the pinned all-MiniLM-L6-v2 checkpoint with 256-token truncation, masked mean pooling and normalized 384-dimensional vectors.')
p('Each text channel retains up to 2,000 positive-cosine historical papers and aggregates evidence to authors. Reciprocal-rank fusion with constant 60 combines lexical, semantic, recent-frequency and approximate fixed-scoring channels. The common pool contains up to 1,000 eligible authors with deterministic padding. Positive injection is permitted only in training and is recorded: the completed pilot injected 251 missing relevant candidates.')
table(['Feature family','Count','Examples'],[
['History controls','3','Seed and candidate publication counts; history balance'],
['Content','7','Category cosine; lexical and semantic profile/retrieval evidence'],
['Network','9','Recent/lifetime coauthorship, neighborhood and degree signals, reverse frequency, citation history'],
['Explicit time','2','Last-collaboration recency; candidate activity decay'],
['Composite scores','2','Description-based fixed scorer; reciprocal-rank fusion']],[133,45,334])
h('Matched training and parameter selection')
p('LightGBM LGBMRanker with the lambdarank objective uses one group per query. LGBMClassifier receives the same examples and features. Both select among leaves/trees of 7/60, 15/100 and 31/100, with learning rate 0.05 and L2 regularization 1. Standardized logistic regression uses C values 0.1, 1 and 10. Validation nDCG@1000 chooses each configuration. Saved-model replay checks export equivalence.')
p('Eight factorial models remove both composite scores and test every combination of content, network and explicit-time features, while retaining three history controls. A separate planned end-to-end no-content pipeline also removes lexical and semantic candidate generation. This separates reranking effects from candidate coverage.')

# 6
page(6,'PreScience pilot ranking results')
p('The completed semantic pilot uses 200 training queries, 100 validation queries, 1,000 candidates and seed 42. All 23 saved prediction files passed independent metric verification. The principal comparison holds the candidate pool fixed; its macro candidate recall is '+fmt(H['metrics']['lambdamart']['candidate_recall'])+'.')
labels=[('LambdaMART','lambdamart'),('Matched classifier','binary_lightgbm'),('Logistic regression','logistic'),('Frequency last 10','frequency10'),('Frequency last 45','frequency45'),('Approximate fixed v2','autoresearch_description'),('RRF','rrf'),('Semantic mean similarity','semantic_mean10'),('Common neighbors','common_neighbors10'),('Random','random')]
table(['Method','nDCG\n@1000','nDCG\n@10','Recall\n@10','New recall\n@10'],[[a]+[fmt(H['metrics'][k][m]) for m in ['ndcg1000','ndcg10','recall10','new_recall10']] for a,k in labels],[190,80,80,80,82])
h('What the comparison supports')
p('LambdaMART has the highest nDCG@1000 in this practical pilot comparison. Its difference from the matched classifier is +0.0368. The exploratory author-cluster bootstrap interval is [0.0129, 0.0619]. These observations come from the validation data used to select configurations; they are not a confirmatory held-out effect.')
p('New-collaborator Recall@10 is only 0.0485 for LambdaMART and 0.0636 for the classifier. Thus, a higher aggregate ranking score does not establish better discovery of previously unseen partners. Pure frequency channels return no new relevant collaborators in the top ten of this pilot.')
cap('nDCG discounts lower ranks; candidate recall measures the fraction of relevant authors available to a reranker. New recall excludes queries without new relevant collaborators and uses the unchanged full ranking. Fixed v2 is a description-based approximation, not an exact official reproduction. Extended per-method results appear on page 14.')

# 7
page(7,'Which feature combinations helped in the pilot')
p('The factorial analysis trains eight separate rankers on the same candidate pool. Every cell retains three history controls; the fixed-score and RRF composites are removed throughout. Each cell is selected with the same three-configuration tree grid.')
flabels={'controls_only':'Controls only','text':'Content','network':'Network','explicit_time':'Explicit time','text+network':'Content + network','text+explicit_time':'Content + time','network+explicit_time':'Network + time','text+network+explicit_time':'Content + network + time'}
story.append(Bars([(flabels[k],v['ndcg1000'],TEAL if k=='network+explicit_time' else BLUE if k=='text+network+explicit_time' else GRAY) for k,v in F['metrics'].items()],maxv=.6,labelwidth=193,height=248,xlabel='Development nDCG@1000'))
cap('Figure 1. Eight feature combinations on 100 validation queries, seed 42. Bars begin at zero. Numerical values are direct exports from the saved factorial report. No uncertainty bars are implied by a single model seed.')
h('The full combination did not lead this comparison')
p('Network plus explicit time reaches 0.5121, compared with 0.5060 for content plus network plus explicit time. The full combination is therefore lower by 0.0061 in this pilot. The network-only model scores 0.5013. These results motivate testing conditional effects rather than assuming every information source helps.')
p('The practical full model on page 6 has 23 features; the full factorial model has 21. The practical model retains the two composite scores. Its 0.5245 result and the factorial model\'s 0.5060 result refer to different fitted models and should not be merged.')
p('Removing explicit-time columns does not eliminate all temporal information: strict historical eligibility, the recent-paper windows and the retrieved candidate pool still encode time. Similarly, a feature ablation within a shared candidate pool cannot by itself estimate the effect of removing text from retrieval. Expanded development includes a separate no-content pipeline for that reason.')

# 8
page(8,'New partners and candidate coverage')
groups=G['groups']
story.append(Forest([('nDCG@1000',groups['ndcg1000:all']),('New Recall@10',groups['new_recall10:all']),('Repeat Recall@10',groups['repeat_recall10:all'])]))
cap('Figure 2. Exploratory LambdaMART minus classifier differences with marginal 95% author-cluster bootstrap intervals from 5,000 draws. Eligible queries: 100 overall, 66 for new partners, 80 for repeat partners. Validation-selected; no multiplicity adjustment.')
p('The estimated repeat-partner recall difference is +0.0618, whereas the new-partner estimate is -0.0152 with an interval crossing zero. This pattern suggests that the aggregate advantage may reflect established collaborations. It does not establish a causal explanation or a statistically verified difference between the two effects.')
table(['Prior publications','Queries','nDCG@1000 difference','Exploratory 95% interval'],[[label,str(groups[key]['queries']),f"{groups[key]['delta']:+.4f}",f"[{groups[key]['ci95'][0]:.4f}, {groups[key]['ci95'][1]:.4f}]"] for label,key in [('1 to 3','ndcg1000:1-3'),('4 to 10','ndcg1000:4-10'),('11 or more','ndcg1000:11+')]], [119,61,150,182])
h('Retrieval coverage is a separate constraint')
stand=[('Frequency last 10','frequency10_standalone'),('Approximate fixed v2','autoresearch_description_standalone'),('Lexical retrieval','lexical_standalone'),('Semantic retrieval','semantic_standalone'),('Combined RRF pool','rrf_standalone')]
table(['Standalone pipeline','nDCG@1000','Candidate recall'],[[a,fmt(H['metrics'][k]['ndcg1000']),fmt(H['metrics'][k]['candidate_recall'])] for a,k in stand],[270,121,121])
p('The current combined pool covers 0.7157 of relevant authors on average. The earlier corrected lexical/network pool covered 0.7221 on the same 100 queries. Because retrieval and features changed together, this is not a controlled causal comparison; it provides no evidence that adding semantic retrieval improved coverage.')

# 9
page(9,'OGB comparison with a neural baseline')
p('Expanded development uses seeds 42, 43 and 44, each with 10,000 positive training edges and five negatives per positive. All methods are evaluated on 60,084 positive validation edges against the official pool of 100,000 shared negatives. Reported variation is the sample standard deviation across three selected runs, not a confidence interval.')
ogbrows=[('NCN with released node vectors',N['mean_hits50'],N['sample_sd']),('Graph matched classifier',O['seed_summary']['graph__binary_lightgbm']['mean'],O['seed_summary']['graph__binary_lightgbm']['std']),('Graph LambdaMART',O['seed_summary']['graph__lambdamart']['mean'],O['seed_summary']['graph__lambdamart']['std'])]
aa=O['fixed_baselines']['adamic_adar']; aa=aa['hits50'] if isinstance(aa,dict) else aa
ogbrows.append(('Adamic Adar',aa,None))
for a,k in [('Graph plus node LambdaMART','graph_plus_released_nodes__lambdamart'),('Graph plus node classifier','graph_plus_released_nodes__binary_lightgbm')]:ogbrows.append((a,O['seed_summary'][k]['mean'],O['seed_summary'][k]['std']))
story.append(Bars([(a,v,TEAL if i==0 else BLUE,vv or 0) for i,(a,v,vv) in enumerate(ogbrows)],maxv=.7,labelwidth=213,height=198,xlabel='Development Hits@50',errors=True))
cap('Figure 3. Hits@50 means and one sample standard deviation for learned models. Adamic Adar is fixed. The zero-based axis avoids exaggerating small gaps; the exact table gives their magnitude.')
table(['Method','Mean Hits@50','Seed SD'],[[a,fmt(v),fmt(sd) if sd is not None else 'Fixed'] for a,v,sd in ogbrows],[310,101,101])
p('NCN has the highest observed mean, about 0.0090 above graph LambdaMART. Its best checkpoints occurred at epochs 29, 9 and 36. Each NCN run selected from 100 validation epochs; each tree family selected among three configurations. This unequal selection budget and the uncertified temporal provenance of the released vectors limit direct interpretation.')
cap('The implemented NCN adaptation is not an exact upstream training-protocol reproduction. Independent saved-prediction verification passed for all three NCN runs. The official test split remains sealed; validation differences do not establish final superiority.')

# 10
page(10,'exHarmony results depend on the relevance proxy')
p('The complementary task retrieves experts from title/abstract queries. Models train and select on Authors judgments. Cite and SimCite assess transfer to citation-derived relevance. Each seed uses 1,000 training and 500 validation queries; samples can overlap across seeds. Their standard deviation therefore mixes training and query-sample variation.')
enames=[('Maximum BM25','max_bm25'),('LambdaMART','lambdamart'),('Matched classifier','binary_lightgbm')]
table(['Judgments','Maximum BM25','LambdaMART','Classifier'],[[lab]+[mean_sd(E['metrics'][key][method]['ndcg10']) for _,method in enames] for lab,key in [('Authors','authors'),('Cite','cite'),('SimCite','simcite')]],[85,142,142,143])
cap('Table. Development nDCG@10, mean ± sample SD across seeds 42-44. The same Authors-trained learners are assessed under the other relevance definitions. Values are not comparable to PreScience nDCG@1000 or OGB Hits@50.')
rows=[]
for lab,key in [('Authors','authors'),('Cite','cite'),('SimCite','simcite')]:
    for label,method in enames:
        d=E['metrics'][key][method]['ndcg10'];rows.append((lab+' / '+label,d['mean'],{'max_bm25':GRAY,'lambdamart':TEAL,'binary_lightgbm':BLUE}[method],d['sample_sd']))
story.append(Bars(rows,maxv=.3,labelwidth=193,height=215,xlabel='Development nDCG@10',errors=True))
cap('Figure 4. Error bars show sample SD, not inferential confidence intervals. All 63 expanded development prediction files were independently checked.')
p('Maximum BM25 has the highest Authors mean. LambdaMART has higher means for Cite and SimCite, but this does not establish an advantage on the primary Authors task or a general ranking-objective benefit. The retrieval implementation uses hashed BM25, not exact Lucene BM25.')
p('Missing relevant profiles impose a limitation before ranking. The audited Authors validation macro coverage ceiling is 0.71381; it is not the achieved candidate recall and is not a measurement of the untouched test split. Such missing-profile labels remain in evaluation denominators.')

# 11
page(11,'Verification engineering and measured costs')
p('Experiment outputs include saved predictions, configurations, selected models and SHA256 receipts. Independent evaluation checks the reported metrics from predictions. Tests of final runners use small synthetic fixtures; those tests validate software behavior and are not evidence of predictive quality on final data.')
table(['Completed check','Evidence'],[
['PreScience semantic pilot and factorial','23 practical and 8 factorial prediction outputs passed independent checks.'],
['OGB and exHarmony development','Three NCN runs verified on full validation arrays; 63 exHarmony prediction files checked.'],
['TRAIN semantic index','373,716 rows; 287,013,888 vector bytes. Source/order/hash, finite values and norms audited.'],
['Final lexical preprocessing','464,942 text records and 38,501,080 nonzero entries audited. This is label-free preprocessing.'],
['Duplicate author correction','3,147 TRAIN records with repeated author entries corrected; old results retained as superseded.'],
['Interrupted-run recovery','117 committed expanded training-query receipts passed hashes after the memory guard stopped the collector.'],
['Final inference software','All three runners have fixture tests for frozen-model use, independent metrics, resume and corruption guards.']],[180,332])
h('A verified memory improvement')
p(f"Exact-value string interning reduced retained TRAIN-loader RSS from {L['before']['rss_bytes']/1e6:.1f} MB to {L['after']['rss_bytes']/1e6:.1f} MB, a reduction of {L['rss_reduction_bytes']/1e6:.1f} MB. Complete fingerprints, including dictionary insertion order, were identical. This is an engineering result; it does not alter the scientific inputs.")
h('Cost evidence and its limits')
p('TRAIN embedding sessions account for 29.62 recorded elapsed hours over committed rows. This is not active CPU time or isolated throughput: suspension and contention may be included, while inter-session gaps and other phases are excluded. It must not be used as a fair algorithm runtime comparison.')
p('The isolated cost harness is implemented but has not run. It will measure 30 fixed development queries with three warm repeats, separating joint retrieval/feature construction from scoring/sorting, and sampling process RSS. The desktop has four logical CPUs and approximately 12.77 GB RAM. Heavy stages are now scheduled sequentially after a guarded concurrent attempt exhausted the reserve.')

# 12
page(12,'Manuscript status and the path to completion')
p('The companion manuscript contains the Introduction, related work, task definitions, detailed Methods, interpretation boundaries and reproducibility statements. Its final quantitative Abstract, Results and Discussion are pending. The study currently supports a mixed development narrative, not a claim that the full combination wins everywhere.')
table(['Stage','Current status','Completion condition'],[
['Expanded PreScience','Queued after label-free encoding; 117 training-query caches retained.','Complete 3 seeds, practical and no-content pipelines, 8 factorial cells and fixed-score sensitivity.'],
['Final semantic preprocessing',f"Running at snapshot: {progress['completed_rows']:,} / {progress['total_rows']:,} rows committed.",'Complete and independently verify index provenance, integrity and vector norms.'],
['Isolated cost measurement','Harness and scheduling implemented; measurements pending.','Complete fixed-query timing and memory report without competing study workers.'],
['Global configuration freeze','Assembler and guards implemented; real lock absent.','Review all models, source hashes, sample rationale, comparisons and inputs before test labels.'],
['Held-out evaluation','Not started for any of the three tasks.','Execute frozen models, verify outputs and calculate paired uncertainty and adjusted tests.'],
['Manuscript and release package','Detailed draft and export scripts exist.','Insert final evidence, audit every claim, complete supplement and verify archive contents.']],[120,186,206])
p('Final inference is planned to average paired effects across model seeds before 10,000 cluster-bootstrap draws and cluster sign-flip tests. Clusters are seed authors, OGB source nodes and exHarmony author/title components. Holm correction is applied within each task\'s prespecified primary family. Crossed network dependence remains a limitation.')
p('For Scientific Reports, retain a descriptive title of at most 20 words, an unstructured abstract of at most 200 words, clear data/code availability and AI-use disclosure [13]. The report\'s illustrations are for supervision; the eventual article will use its own concise display set. Scientific soundness and editorial assessment determine suitability; acceptance cannot be guaranteed.')
cap('Before submission, the authors must confirm names/order, affiliations, contact information, contributions, funding and competing interests, and approve the claims. Public deposit, submission and fees remain separate actions. No journal submission or public benchmark-package upload has occurred.')

# 13
page(13,'Selected references and source records')
refs=[
'[1] Koopmann T, Kobs K, Herud K, Hotho A. CoBERT: Scientific Collaboration Prediction via Sequential Recommendation. ICDMW (2021). [DOI 10.1109/ICDMW53433.2021.00013](https://doi.org/10.1109/ICDMW53433.2021.00013).',
'[2] Guan K, Huang W, Chen T, Chan WK. TEDRec: Transformer-based scientific collaborator recommendation via textual-edge dynamic network modeling. Information Processing & Management 62, 104283 (2025). [DOI 10.1016/j.ipm.2025.104283](https://doi.org/10.1016/j.ipm.2025.104283). Abstract examined.',
'[3] Ajith A et al. PreScience: A Dataset and Benchmark for Scientific Forecasting. [arXiv 2602.20459v2](https://arxiv.org/html/2602.20459v2) (2026).',
'[4] Ebrahimi S, Salamat S, Arabzadeh N, Bashari M, Bagheri E. exHarmony: Authorship and Citations for Benchmarking the Reviewer Assignment Problem. ECIR (2025). [DOI 10.1007/978-3-031-88714-7_1](https://doi.org/10.1007/978-3-031-88714-7_1).',
'[5] Wang X, Yang H, Zhang M. Neural Common Neighbor with Completion for Link Prediction. [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/3efb4bdc6bfe13e1ff95b4407c37961d-Abstract-Conference.html) (2024).',
'[6] Open Graph Benchmark. [Link Property Prediction documentation](https://ogb.stanford.edu/docs/linkprop/), ogbl-collab.',
'[7] LightGBM. [LGBMRanker documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html). Library implementation used for LambdaMART.',
'[8] Zhong Y, Liu X. Interdisciplinary scientific collaborator recommendation via interest diffusion and collaborative affinity. Discover Artificial Intelligence 6, 1367 (2026). [DOI 10.1007/s44163-026-01809-9](https://doi.org/10.1007/s44163-026-01809-9).',
'[9] Wang X, Huang W, Zhao B, Li S. Scientific collaborator recommendation via hypergraph embedding. Information Processing & Management 63, 104423 (2026). [DOI 10.1016/j.ipm.2025.104423](https://doi.org/10.1016/j.ipm.2025.104423).',
'[10] Li M, Xie X. Identifying scientific collaboration opportunities based on heterogeneous hypergraph link prediction. Journal of King Saud University Computer and Information Sciences 38, 82 (2026). [DOI 10.1007/s44443-026-00474-3](https://doi.org/10.1007/s44443-026-00474-3).',
'[11] Moreira LS et al. Scientific collaborator recommendation: A systematic literature review. Engineering Applications of Artificial Intelligence 181, 115279 (2026). [DOI 10.1016/j.engappai.2026.115279](https://doi.org/10.1016/j.engappai.2026.115279). Publisher abstract examined; full text unavailable.',
'[12] [ResearchForge AI repository](https://github.com/tuba89/ResearchForge-AI). Original application, audited commit recorded on page 2.',
'[13] Scientific Reports. [Submission guidelines](https://www.nature.com/srep/author-instructions/submission-guidelines). Official guidance checked 27 October 2026.'
]
for ref in refs:p(ref,'Refs')

# 14
page(14,'Extended PreScience development results')
p('The following table preserves all 23 outputs from the corrected semantic pilot. Feature-removal models use the shared full candidate pool. Standalone channels use their own retrieval. These are selected validation results from one seed, not final test estimates.','SmallRF')
name_map=dict((k,a) for a,k in labels+stand)
name_map.update({'frequency_all':'Lifetime frequency','category_cosine':'Category cosine','lexical_mean10':'Lexical mean similarity','lambdamart_without_text':'Ranker without lexical and semantic features','lambdamart_without_semantic':'Ranker without semantic features','lambdamart_without_lexical':'Ranker without lexical features','lambdamart_without_recency':'Ranker without explicit recency','lambdamart_text_only':'Ranker content features only','rrf_standalone':'RRF standalone'})
for a,k in stand:name_map[k]=a+' standalone' if k!='rrf_standalone' else 'RRF standalone'
table(['Method','nDCG@1000','nDCG@10','Candidate recall'],[[name_map.get(k,k),fmt(v['ndcg1000']),fmt(v['ndcg10']),fmt(v['candidate_recall'])] for k,v in H['metrics'].items()],[239,91,91,91],padding=4)
cap('The legacy "without text" row removes lexical/semantic columns but retains category information. It is not the expanded no-content pipeline. The category-only adapter of the original prototype and superseded early pilots are retained in historical files rather than mixed into this current 23-output table.')

# 15
page(15,'Evidence inventory and reproducibility handoff')
p('All numerical tables and charts in this report are generated from saved JSON reports. The accompanying source manifest records exact SHA256 values for the inputs used to prepare this dated snapshot. Current development results remain separate from future frozen final outputs.')
table(['Evidence set','Local study artifact'],[
['Original materials and code audit','AUDIT.md'],
['Information policy and comparisons','PROTOCOL.md'],
['Corrected semantic pilot','results/hybrid-graph-dedup-200-100-1000-s42/report.json'],
['Eight feature combinations','Same run / factorial-s42/report.json'],
['Exploratory paired subgroups','Same run / subgroups-lambdamart-vs-binary_lightgbm.json'],
['OGB tree and fixed methods','results/ogb-expanded-10000/report.json'],
['Three NCN runs','results/ncn-expanded-10000/report.json'],
['Expanded expert retrieval','results/exharmony-expanded-1000-500/report.json'],
['Index and memory integrity','data-audit/train_semantic_index.json and loader-equivalence.json'],
['Manuscript source and reproduction','MANUSCRIPT_DRAFT.md and REPRODUCIBILITY.md']],[175,337])
h('What is implemented for the final study')
p('Final runners replay saved models, bind dataset and feature identities, reject mismatched configurations, preserve interrupted progress, and verify metrics independently. Table, figure and package exporters require the actual verified final reports. A passing fixture test does not replace an executed benchmark; the final exports have not yet been generated from held-out data.')
h('Responsible interpretation')
p('The report preserves negative findings, approximate reproductions, missing-profile relevance, unequal tuning budgets and unresolved provenance. Publication text plus network and time may help in some conditions, but current results do not support universal superiority or a new ranking algorithm. The final manuscript should explain the observed conditions and costs even if the held-out outcomes remain mixed.')
p('OpenAI Codex assisted the literature search, software implementation, experiment execution, analysis and writing. Human authors remain responsible for reviewing the evidence and approving the manuscript. The benchmark implementation is local; the original public repository does not yet contain this study package.')

build(OUT/'ResearchForge_Professor_Report.pdf',story,'ResearchForge AI Research and Benchmark Report')
(OUT/'ResearchForge_Professor_Report.md').write_text('# ResearchForge AI Research and Benchmark Report\n\n'+'\n'.join(MD),encoding='utf-8')

# Faithful, typeset companion manuscript; no held-out results invented.
man=(BASE/'MANUSCRIPT_DRAFT.md').read_text(encoding='utf-8')
man=clean(man)
# Repair spacing artifacts in the presentation copy only.
man=man.split('## References')[0]
protected={}
def protect(m):
    key='TOKEN'+chr(65+len(protected)//26)+chr(65+len(protected)%26)+'TOKEN'
    protected[key]=m.group(0);return key
man=re.sub(r'`[^`]+`|https?://\S+|\b(?:float16|float32|L2|SHA256|k1|v2)\b',protect,man)
man=re.sub(r'(?<=[a-z])(?=\d)', ' ', man)
man=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', man)
man=re.sub(r',(?=\S)(?!\d{3}(?:\D|$))', ', ', man)
man=re.sub(r'(?<!\d),(?=\S)', ', ', man)
man=re.sub(r':(?=\d)', ': ', man)
for key,value in protected.items():man=man.replace(key,value)
man+='## References\n\n'+'\n\n'.join(refs[:11])+'\n'
parts=[]
parts.append(par('RESEARCHFORGE AI   |   MANUSCRIPT DRAFT','SmallRF'))
parts.append(par('Temporal Learning to Rank for Scientific Collaborator Discovery','TitleRF'))
parts.append(par('A Reproducible Offline Study','Deck'))
parts.append(par('27 October 2026. Detailed methods and development context. Held-out Results and final conclusions are pending.','Normal'))
parts.append(par('The companion professor report contains the current numerical development results. This manuscript is not a final submission version.','Normal'))
parts.append(Spacer(1,12))
buffer=[]
def flush():
    if buffer:parts.append(par(' '.join(buffer)));buffer.clear()
for line in man.splitlines()[1:]:
    if line.startswith('**Working draft'):continue
    if line.startswith('## '):
        flush();title=line[3:].replace('—',' ').replace('-',' ')
        title=re.sub(r'^\d+[.\d]*\s*','',title)
        if title=='References':parts.append(PageBreak())
        parts.append(par(re.sub(r'[^\w\s]',' ',title),'ManH'))
    elif line.startswith('### '):
        flush();title=re.sub(r'^\d+[.\d]*\s*','',line[4:]);parts.append(par(re.sub(r'[^\w\s]',' ',title),'Sub'))
    elif not line.strip():flush()
    elif re.match(r'^\d+\. |^\[\d+\]',line):
        flush();parts.append(par(line,'Refs'))
    else:buffer.append(line)
flush()
build(OUT/'ResearchForge_Manuscript_Draft.pdf',parts,'ResearchForge AI Manuscript Draft')
(OUT/'ResearchForge_Manuscript_Draft.md').write_text(man,encoding='utf-8')
with (OUT/'benchmark_results.csv').open('w',newline='',encoding='utf-8') as f:
    wr=csv.writer(f);wr.writerow(['dataset','stage','method','metric','value','sample_sd'])
    for k,v in H['metrics'].items():wr.writerow(['PreScience','development pilot',k,'ndcg1000',v['ndcg1000'],''])
    for k,v in F['metrics'].items():wr.writerow(['PreScience','development factorial',k,'ndcg1000',v['ndcg1000'],''])
    for a,v,sd in ogbrows:wr.writerow(['ogbl-collab','development',a,'hits50',v,sd])
    for key,methods in E['metrics'].items():
        for _,method in enames:
            d=methods[method]['ndcg10'];wr.writerow(['exHarmony '+key,'development',method,'ndcg10',d['mean'],d['sample_sd']])
(OUT/'source_manifest.json').write_text(json.dumps({'created':stamp,'evidence_stage':'development only','inputs':SOURCES,'label_free_progress_snapshot':progress},indent=2),encoding='utf-8')
print(json.dumps({'output_directory':str(OUT),'sources':len(SOURCES),'snapshot':stamp,'reports':[str(p) for p in OUT.glob('*.pdf')]},indent=2))
