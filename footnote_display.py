"""Conservative PDF-backed footnote and standalone heading cleanup for display."""
import re


def clean_display_artifacts(text):
    """Narrow presentation fixes; never strip unproven numeric citations."""
    text = re.sub(r'(?<!\w)(?:1âwphi1|1awphi1|1\^wphi1)(?!\w)', '', text)
    text = re.sub(r'^(?:THE RULING OF THE COURT|The Antecedents)\s+', '', text)
    text = re.sub(r'\b(REINSTATED|REVERSED|AFFIRMED|GRANTED|DENIED)(?=only\b)', r'\1 ', text)
    return re.sub(r'\s+', ' ', text).strip()


def pdf_markers(page):
    runs=[]
    color=None
    def operand(op,args,cm,tm):
        nonlocal color
        if op==b'rg': color=tuple(float(x) for x in args)
        elif op in (b'g',b'k'): color=None
    def visitor(text,cm,tm,font,size):
        if not text.strip(): return
        y=tm[4]*cm[1]+tm[5]*cm[3]+cm[5]
        x=tm[4]*cm[0]+tm[5]*cm[2]+cm[4]
        runs.append({'text':text.strip(),'x':x,'y':y,'size':abs(size*cm[3]),'color':color})
    page.extract_text(visitor_operand_before=operand,visitor_text=visitor)
    markers=[]
    for i,r in enumerate(runs):
        if not i: continue
        before=' '.join(x['text'] for x in runs[max(0,i-3):i])
        after=' '.join(x['text'] for x in runs[i+1:i+4])
        # A centered, isolated Roman numeral with vertical space on both sides
        # is a section heading, not an inline number or a name suffix.
        if (i<len(runs)-1 and re.fullmatch(r'I|II|III|IV|V|VI|VII|VIII|IX|X',r['text'])
                and not page.get('/Rotate',0)
                and abs(r['x']+len(r['text'])*r['size']*.25-float(page.mediabox.width)/2)<r['size']*2
                and abs(r['y']-runs[i-1]['y'])>r['size']*1.2
                and abs(r['y']-runs[i+1]['y'])>r['size']*1.2
                and sum(abs(other['y']-r['y'])<2 for other in runs)==1):
            markers.append({'number':r['text'],'before':before,'after':after,'kind':'section_heading'})
            continue
        if not re.fullmatch(r'\d{1,3}',r['text']): continue
        # This PDF style uses red raised numeric references. Ordinary numbers,
        # baseline footnote definitions and unrecognized styles stay unchanged.
        color=r['color']; prev=runs[i-1]
        if not (color and len(color)==3 and color[0]>.8 and color[1]<.2 and color[2]<.2): continue
        if not 2 < r['y']-prev['y'] < max(14,prev['size']): continue
        markers.append({'number':r['text'],'before':before,'after':after,'kind':'footnote'})
    return markers


def display_sentences(rows,markers):
    normalized=lambda t:re.sub(r'\s+',' ',t).strip()
    text='';spans=[]
    for row in rows:
        start=len(text);text+=normalized(row['text']);spans.append((row['id'],start,len(text)));text+=' '
    removed=set()
    for marker in markers:
        before=normalized(marker['before']);after=normalized(marker['after'])
        # A footnote can end the page. Require a substantial, unique preceding
        # context rather than dropping all occurrences of the same number.
        if not before or (not after and len(before.split())<3): continue
        pattern=re.escape(before)+r'\s+('+re.escape(marker['number'])+r')'
        pattern += r'\s+'+re.escape(after) if after else r'(?=\s|$)'
        matches=list(re.finditer(pattern,text))
        if len(matches)==1: removed.update(range(*matches[0].span(1)))
    return {sid:clean_display_artifacts(''.join(text[i] for i in range(start,end) if i not in removed))
            for sid,start,end in spans}
