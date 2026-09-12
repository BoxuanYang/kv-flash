from pathlib import Path
import sys, ast, json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/pdfs/deps'))
from pypdf import PdfReader
import fitz
from PIL import Image,ImageOps
f=ROOT/'output/pdf/sglang_scheduler_initialization_guide_zh.pdf'
r=PdfReader(f)
text='\n'.join(p.extract_text() for p in r.pages)
assert len(r.pages)==14
assert '\ufffd' not in text
names=['init_running_status','init_model_worker','init_cache_with_memory_pool','init_schedule_policy','init_overlap']
tree=ast.parse((ROOT/'tmp/pdfs/source_snapshot/managers/scheduler.py').read_text(encoding='utf-8-sig'))
checks={n.name:sorted(set(x.attr for x in ast.walk(n) if isinstance(x,ast.Attribute) and isinstance(x.ctx,ast.Store) and isinstance(x.value,ast.Name) and x.value.id=='self')) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name in names}
missing={k:[m for m in v if m not in text] for k,v in checks.items()}
assert not any(missing.values()),missing
d=fitz.open(f)
bad=[(i+1,tuple(s['bbox'])) for i,p in enumerate(d) for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans'] if s['bbox'][0]<42 or s['bbox'][2]>554 or s['bbox'][1]<10 or s['bbox'][3]>824]
assert not bad,bad
out=ROOT/'tmp/pdfs'
for i,p in enumerate(d):
    p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(str(out/f'page-{i+1:02}.png'))
thumbs=[ImageOps.contain(Image.open(out/f'page-{i+1:02}.png').convert('RGB'),(447,632)) for i in range(len(d))]
sheets=[Image.new('RGB',(914,1320),'#dbe3e9') for _ in range(4)]
for i,t in enumerate(thumbs): sheets[i//4].paste(t,(10+(i%2)*452,24+((i%4)//2)*654))
for i,s in enumerate(sheets): s.save(out/f'contact-{i+1}.png')
(out/'qa.json').write_text(json.dumps({'pages':len(r.pages),'member_coverage':checks,'missing':missing,'bounds_issues':bad,'pdf_bytes':f.stat().st_size},ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS: 14 pages, all direct members covered, text and bounds valid; all pages rendered.')
