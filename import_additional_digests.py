"""Import supplied digest references without fabricating review or sentence labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen

from import_attorney_references import split_blocks


def extract_source(raw, docket):
    from lxml import html
    tree = html.fromstring(raw)
    nodes = tree.xpath('//blockquote')
    if not nodes:
        raise ValueError('No LawPhil decision block found')
    node = nodes[0]
    if not re.search(r'SO\s+ORDERED', ' '.join(node.itertext()), re.I):
        parent = node.getparent()
        if parent.tag == 'td' and re.search(r'SO\s+ORDERED', ' '.join(parent.itertext()), re.I):
            node = parent
    internal_tables = set(node.xpath('.//table'))
    # LawPhil wraps the entire page in a layout table. Only tables INSIDE
    # the decision should suppress their nested paragraphs.
    parts = [' '.join(n.itertext()).strip() for n in node.xpath('.//p | .//table')
             if not any(a in internal_tables for a in n.iterancestors('table'))]
    text = '\n\n'.join(' '.join(t.split()) for t in parts if t)
    if docket not in text[:2500] or not re.search(r'SO\s+ORDERED', text, re.I):
        raise ValueError('Docket/disposition extraction check failed')
    return text


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cached-sources', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new to preserve prior artifacts')
    intake = Path('data/private/additional_digest_intake_20261024')
    inventory = json.loads((intake / 'inventory.json').read_text(encoding='utf-8'))
    pages = json.loads((intake / 'extracted_pages.json').read_text(encoding='utf-8'))
    # Allocate before generation. Existing classifier splits are preserved.
    validation = {'127206', '176697'}
    links = [u for page in pages for u in page['links']]
    records = []
    for item in inventory['records']:
        gr = item['docket']
        urls = sorted(set(u for u in links if re.search(r'/gr_' + gr + r'_', u)))
        if len(urls) != 1:
            raise ValueError('Ambiguous or missing exact-docket link: ' + gr)
        start, end = item['pages']
        text = ' '.join(page['text'] for page in pages[start-1:end])
        # A link for the next case sometimes sits at the prior page's foot.
        text = re.sub(r'\s+G\.R\.\s+No\.\s+\d+\s*$', '', text)
        blocks = split_blocks([text])
        overlap = item['existing_exposure_records']
        old_splits = {reason.removeprefix('existing_sentence_dataset_')
                      for record in overlap for reason in record['reasons']
                      if reason.startswith('existing_sentence_dataset_')}
        if len(old_splits) > 1:
            raise ValueError('Conflicting existing allocation')
        split = next(iter(old_splits)) if old_splits else ('validation' if gr in validation else 'train')
        records.append({
            'case_id': overlap[0]['case_id'] if overlap else 'GR' + gr,
            'docket': gr, 'additional_dockets': item['additional_dockets'],
            'title': item['title_as_identified_from_digest'], 'digest_pages': [start, end],
            'decision_url': urls[0], 'blocks': blocks, 'split': split,
            'allocation_status': 'existing_split_preserved' if old_splits else 'prospective_development_assignment',
            'reference_use': 'quarantined_existing_test' if split == 'test' else 'development_reference',
            'approval_basis': 'user_instruction_to_assume_approval',
            'attorney_accepted': None, 'reviewer': None, 'review_date': None,
            'independent_final_evaluation_eligible': False,
            'source_digest_pdf_sha256': inventory['source_sha256'],
            'training_eligible': False,
            'notes': ['Reference digest only; not source sentence annotations.',
                      'Full-source alignment remains unchecked; no automatic model training.',
                      'Assumed approval is not a documented attorney review.']})
    args.output.mkdir(parents=True)
    (args.output / 'references.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    sources = args.output / 'sources'
    sources.mkdir()

    def fetch(record):
        if record['split'] == 'test':
            return {'docket': record['docket'], 'status': 'not_fetched_existing_test'}
        url = record['decision_url']
        try:
            if args.cached_sources:
                raw = (args.cached_sources / (record['docket'] + '.html')).read_bytes()
            else:
                with urlopen(Request(url, headers={'User-Agent': 'Mozilla/5.0'}), timeout=45) as response:
                    raw = response.read()
            (sources / (record['docket'] + '.html')).write_bytes(raw)
            text = extract_source(raw, record['docket'])
            doc = {'decision_url': url, 'raw_html_sha256': hashlib.sha256(raw).hexdigest(),
                   'raw_text': text, 'source_sha256': hashlib.sha256(text.encode()).hexdigest(),
                   'extraction_status': 'machine_extracted_needs_boundary_review',
                   'source_complete_verified': False}
            (sources / (record['docket'] + '.json')).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding='utf-8')
            return {'docket': record['docket'], 'status': 'downloaded', 'characters': len(text)}
        except Exception as exc:
            return {'docket': record['docket'], 'status': 'error', 'error': str(exc)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, records))
    (args.output / 'retrieval_report.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
