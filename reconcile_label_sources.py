"""Match annotations to cached HTML after removing identified citation links only."""
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from lxml import html
from audit_label_context import normalized, OUT


def strip_citation_links(tree):
    removed = []
    for a in tree.xpath('//a'):
        citation = (re.fullmatch(r'#f(?:nt)?\d+', a.get('href', ''), re.I) or
                    (a.get('class') == 'nt' and re.fullmatch(r'fnt\d+\.\d+', a.get('name', ''), re.I)))
        if a.text_content().strip().isdigit() and citation:
            removed.append({'href': a.get('href'), 'text': a.text_content()})
            a.drop_tree()
    return removed


def normalize_offsets(text):
    chars, positions = [], []
    for i, c in enumerate(text):
        for n in unicodedata.normalize('NFKD', c.replace('₱', 'P')).casefold():
            if n.isalnum():
                chars.append(n)
                positions.append(i)
    return ''.join(chars), positions


def main():
    audit = json.loads((OUT/'context_audit.json').read_text(encoding='utf-8'))
    cache, records = {}, []
    for r in audit:
        if r['source_match'] == 'normalized_substring':
            continue
        source = Path(r['source_file'])
        doc = json.loads(source.read_text(encoding='utf-8'))
        path = Path(doc.get('raw_html_file', source.with_suffix('.html')))
        if not path.exists():
            records.append({'sentence_id':r['sentence_id'], 'status':'html_unavailable'})
            continue
        if str(path) not in cache:
            tree = html.fromstring(path.read_bytes())
            removed = strip_citation_links(tree)
            text = tree.text_content()
            clean, positions = normalize_offsets(text)
            cache[str(path)] = (text,clean,positions,removed)
        text,clean,positions,removed = cache[str(path)]
        target = normalized(r['text'])
        start = clean.find(target)
        entry = {'sentence_id':r['sentence_id'], 'dataset_index':r['dataset_index'],
                 'html_source':str(path), 'html_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'status':'matched_after_citation_cleanup' if start >= 0 else 'unresolved',
                 'method':'Normalized whole-source substring after removing numeric #f/#fnt links or class=nt named fntN.N citation markers; no general number deletion',
                 'citation_links_removed':len(removed), 'semantic_label_verified':False}
        if start >= 0:
            begin,end = positions[start],positions[start+len(target)-1]+1
            entry.update(source_text_span=[begin,end], source_excerpt=text[begin:end],
                         surrounding_context=text[max(0,begin-250):min(len(text),end+250)])
        else:
            first = clean.find(target[:35])
            last = clean.find(target[-35:], max(0, first)) if first >= 0 else -1
            if first >= 0 and first <= last <= first + len(target) * 2 + 300:
                segment = clean[first:last+35]
                entry['diagnostic_differences'] = [
                    {'operation':op, 'annotation':target[a:b], 'source':segment[c:d]}
                    for op,a,b,c,d in SequenceMatcher(None,target,segment,autojunk=False).get_opcodes() if op != 'equal']
                entry['diagnostic_source_excerpt'] = text[positions[first]:positions[last+34]+1]
        records.append(entry)
    (OUT/'source_reconciliation.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    from collections import Counter
    print(dict(Counter(r['status'] for r in records)))
    print('Unresolved indices:',[r.get('dataset_index') for r in records if r['status']!='matched_after_citation_cleanup'])
    for r in records:
        if r['status'] == 'unresolved':
            print(r['dataset_index'], json.dumps(r.get('diagnostic_differences','anchors_not_found'), ensure_ascii=True))


if __name__ == '__main__':
    main()
