"""Preserve user-supplied attorney digests as references, never as sentence labels.

DOCX paragraph positions and headings are checked against an explicit manifest.
The original document and imported text stay under gitignored data/private/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from core import ROLES, read_jsonl

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def paragraphs(path):
    with ZipFile(path) as z:
        root = ET.fromstring(z.read('word/document.xml'))
    if any(root.iter(W + 'ins')) or any(root.iter(W + 'del')):
        raise ValueError('Resolve tracked changes before importing a reference')
    if any(root.iter(W + 'tbl')):
        raise ValueError('Table-based references need an explicit table import mapping')
    body = root.find(W + 'body')
    return [''.join(node.text or '' if node.tag == W + 't' else '\n' if node.tag == W + 'br' else '\t' if node.tag == W + 'tab' else ''
                    for node in p.iter()) for p in body.findall(W + 'p')]


def split_blocks(items):
    blocks = {role: [] for role in ROLES}
    role = None
    for text in items:
        parts = re.split(r'\b(FACTS?|ISSUES?|RULING)\s*:', text, flags=re.I)
        if role and parts[0].strip():
            blocks[role].append(parts[0].strip())
        for i in range(1, len(parts), 2):
            role = {'FACTS': 'FACT', 'ISSUES': 'ISSUE'}.get(parts[i].upper(), parts[i].upper())
            if parts[i + 1].strip():
                blocks[role].append(parts[i + 1].strip())
    if any(not blocks[role] for role in ROLES):
        raise ValueError('Every reference needs explicit, nonempty Facts, Issue and Ruling blocks')
    return {role: '\n\n'.join(blocks[role]) for role in ROLES}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docx', required=True)
    parser.add_argument('--manifest', default='data/reference_import_manifest.json')
    parser.add_argument('--dataset', default='data/revision_20261020_metadata/sentences.jsonl')
    parser.add_argument('--output', default='data/private/attorney_references_20261022')
    args = parser.parse_args()
    document = Path(args.docx)
    paras = paragraphs(document)
    cases = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['cases']
    rows = read_jsonl(args.dataset)
    sha = hashlib.sha256(document.read_bytes()).hexdigest()
    records = []
    for i, case in enumerate(cases):
        start = case['start_paragraph']
        end = cases[i + 1]['start_paragraph'] if i + 1 < len(cases) else len(paras)
        if paras[start].strip() != case['title']:
            raise ValueError('Document heading no longer matches the manifest: ' + case['name'])
        overlaps = sorted({r['split'] for r in rows if r['case_id'] == case['case_id']})
        records.append({**case, 'blocks': split_blocks(paras[start:end]),
                        'source_docx_sha256': sha, 'source_paragraphs_1_based': [start + 1, end],
                        'authorship': 'attorney_created_as_reported_by_user',
                        'use': 'development_reference_only', 'reference_scope': 'digest_selected_issues',
                        'classifier_split_overlap': overlaps,
                        'independent_accuracy_benchmark': False,
                        'notes': ['Preserved wording; no silent corrections to artifacts or citations.',
                                  'Digest headings do not label original decision sentences.',
                                  'Reference never supplied to the model during comparison.']})
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / document.name
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != sha:
        raise ValueError('Existing source document differs; use a new output directory')
    shutil.copyfile(document, target)
    out = {'source_document': str(document.resolve()), 'source_sha256': sha,
           'paragraph_count': len(paras), 'cases': records}
    (folder / 'references.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps([{k: r[k] for k in ('name', 'case_id', 'classifier_split_overlap', 'use')} for r in records], indent=2))


if __name__ == '__main__':
    main()
