"""Attach source context to development annotations; never infer correct labels."""
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher
from urllib.request import Request, urlopen
from core import read_jsonl
from import_additional_digests import extract_source

OUT = Path('data/private/label_audit_20261025')
DATA = Path('data/revision_20261020_metadata/sentences.jsonl')
FETCH = {
    'GR159097_20100705': 'https://lawphil.net/judjuris/juri2010/jul2010/gr_159097_2010.html',
    'GRL29330_19871214': 'https://lawphil.net/judjuris/juri1987/dec1987/gr_l-29330_1987.html',
    'GRL74027_19891207': 'https://lawphil.net/judjuris/juri1989/dec1989/gr_74027_1989.html',
    'GR107112_19940224': 'https://lawphil.net/judjuris/juri1994/feb1994/gr_107112_1994.html',
}


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text.replace('₱', 'P')).casefold() if c.isalnum())


def source_path(case):
    if case in FETCH:
        p = OUT / 'sources' / (case + '.json')
        if not p.exists():
            raw = urlopen(Request(FETCH[case], headers={'User-Agent': 'FIRmSUM research source audit'}), timeout=45).read()
            docket = re.search(r'\d+', case).group()
            try:
                text = extract_source(raw, docket)
            except ValueError:
                from lxml import html
                block = html.fromstring(raw).xpath('//blockquote')[0]
                text = block.text_content()
                if docket not in text[:2500] or not re.search(r'SO\s+ORDERED|WHEREFORE', text, re.I):
                    raise ValueError('Fallback source identity/disposition check failed: ' + case)
                text = '\n\n'.join(' '.join(s.split()) for s in re.split(r'[\r\n]{2,}', text) if s.strip())
            p.parent.mkdir(parents=True, exist_ok=True)
            p.with_suffix('.html').write_bytes(raw)
            p.write_text(json.dumps({'decision_url': FETCH[case], 'raw_text': text,
                'raw_html_sha256': hashlib.sha256(raw).hexdigest()}, ensure_ascii=False, indent=2), encoding='utf-8')
        return p
    if case.startswith('GRL12219'):
        return Path('data/private/attorney_references_20261022/picart_source.json')
    return Path('data/private/additional_digest_development_20261024_v3/sources') / (re.search(r'\d+', case).group() + '.json')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(DATA)
    sources = {}
    for case in sorted({r['case_id'] for r in rows if r['split'] != 'test'}):
        p = source_path(case)
        doc = json.loads(p.read_text(encoding='utf-8'))
        text = doc.get('raw_text') or doc.get('majority_text') or doc['text']
        paragraphs = [s for s in text.split('\n\n') if s.strip()]
        sources[case] = (p, doc, paragraphs, [normalized(s) for s in paragraphs])
    reviewed = []
    for index, row in enumerate(rows):
        if row['split'] == 'test':
            continue
        p, doc, paras, norms = sources[row['case_id']]
        target = normalized(row['text'])
        matches = [i for i, s in enumerate(norms) if target in s]
        best = matches[0] if matches else max(range(len(paras)), key=lambda i: SequenceMatcher(None, target, norms[i], autojunk=False).ratio())
        reviewed.append({'dataset_index': index, 'sentence_id': row['sentence_id'],
            'case_id': row['case_id'], 'split': row['split'], 'text': row['text'],
            'original_label': row['label'], 'source_file': str(p),
            'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
            'source_url': doc.get('decision_url') or doc.get('source_url'),
            'source_match': 'normalized_substring' if matches else 'approximate_candidate_not_verified',
            'source_matches': matches, 'context_paragraph_index': best,
            'source_context': [{'paragraph_index': j, 'text': paras[j]} for j in range(max(0, best-1), min(len(paras), best+2))],
            'text_quality_flags': ['replacement_character'] if '\ufffd' in row['text'] else [],
            'reviewer_type': 'AI_assistant', 'attorney_validated': False,
            'final_label': None})
    (OUT / 'context_audit.json').write_text(json.dumps(reviewed, ensure_ascii=False, indent=2), encoding='utf-8')
    from collections import Counter
    print(json.dumps({'development_rows': len(reviewed), 'source_matches': dict(Counter(r['source_match'] for r in reviewed)),
                      'encoding_damage_rows': sum(bool(r['text_quality_flags']) for r in reviewed)}, indent=2))


if __name__ == '__main__':
    main()
