"""Read the supplied workbook using bundled Python/openpyxl; preserve source annotations."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import openpyxl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('workbook')
    p.add_argument('--output-dir', default='data')
    a = p.parse_args()
    source = Path(a.workbook)
    w = openpyxl.load_workbook(source, read_only=True, data_only=True)
    raw = list(w['Tagged Sentences'].values)
    tagged = [dict(zip(raw[0], r), source_row=i) for i, r in enumerate(raw[1:], 2) if any(r)]
    lookup = defaultdict(list)
    for r in tagged:
        lookup[(r['text'], r['suggested_label'])].append(r)
    # Freeze by case before training. This is a provisional development allocation.
    validation = {'GR107112_19940224', 'GR88880_19910430'}
    test = {'GR104234_19950630', 'GR149420'}
    output, duplicates, seen = [], [], {}
    for rownum, row in enumerate(list(w['To be Fed'].values)[1:], 2):
        if not any(row):
            continue
        matches = lookup[(row[0], row[1])]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous provenance at To be Fed row {rownum}')
        r = matches[0]
        case = r['case_id'] or re.sub(r'_(?:R\d+_)?S\d+$', '', r['sentence_id'])
        split = 'validation' if case in validation else 'test' if case in test else 'train'
        text = r['text'].strip()
        key = re.sub(r'\s+', ' ', text).casefold()
        if key in seen:
            prior = seen[key]
            if prior['label'] != r['suggested_label']:
                raise ValueError('Conflicting labels on duplicate text')
            if prior['split'] != split:
                raise ValueError('Duplicate text crosses case splits; review allocation')
            duplicates.append({'source_row': r['source_row'], 'duplicate_of': prior['source_row']})
            continue
        item = {'case_id': case, 'split': split, 'text': text, 'label': r['suggested_label'],
                'sentence_id': r['sentence_id'], 'source_sheet': 'Tagged Sentences', 'source_row': r['source_row'],
                'feed_row': rownum, 'label_status': 'provisional_suggested', 'review_status': r['review_status'],
                'original_split': r['split'], 'case_id_inferred': not bool(r['case_id'])}
        seen[key] = item
        output.append(item)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    (out / 'sentences.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output), encoding='utf-8')
    cases = {r['case_id']: r['split'] for r in output}
    report = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'annotation_status': 'All final_label cells blank; selected suggested labels are provisional.',
              'source_tagged_rows': len(tagged), 'selected_rows': len(output), 'duplicates_removed': duplicates,
              'excluded_statuses': dict(Counter(r['review_status'] for r in tagged if (r['text'], r['suggested_label']) not in {(x['text'], x['label']) for x in output})),
              'splits': {s: {'sentences': sum(r['split'] == s for r in output),
                             'labels': dict(Counter(r['label'] for r in output if r['split'] == s)),
                             'cases': [c for c, part in cases.items() if part == s]} for s in ['train','validation','test']},
              'notes': ['Case IDs absent in the workbook recovered only from sentence_id prefixes.',
                        'Workbook TRAIN labels reallocated by whole case for validation. No source workbook cells changed.',
                        'Two validation and two test cases; development split differs from the proposal target of 4–5 validation cases.',
                        'Only three selected OTHER examples; test has no OTHER examples. Collect more before final claims.',
                        'No exemplar cases reserved by this classifier dataset; future generator exemplars must exclude test cases.']}
    (out / 'split_manifest.json').write_text(json.dumps(cases, indent=2), encoding='utf-8')
    (out / 'import_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
