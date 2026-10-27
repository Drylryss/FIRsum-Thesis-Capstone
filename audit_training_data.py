"""Inventory label provenance and produce a review queue without relabeling data."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from core import LABELS, read_jsonl, validate_splits


def audit(rows):
    validate_splits(rows)
    counts = {split: {'sentences': sum(r['split'] == split for r in rows),
                     'cases': len({r['case_id'] for r in rows if r['split'] == split}),
                     'labels': dict(Counter(r['label'] for r in rows if r['split'] == split))}
              for split in ('train', 'validation', 'test')}
    missing_review = [r for r in rows if not r.get('reviewer') or r.get('final_label') not in LABELS]
    text_groups = defaultdict(list)
    for row in rows:
        text_groups[' '.join(row['text'].casefold().split())].append(row)
    duplicates = [[{'case_id': r['case_id'], 'sentence_id': r.get('sentence_id'), 'split': r['split']}
                   for r in group] for group in text_groups.values()
                  if len({r['split'] for r in group}) > 1]
    queue = []
    for row in rows:
        # Do not expose test text for annotation triage or use it to tune a model.
        if row['split'] not in ('train', 'validation'):
            continue
        reasons = []
        if row.get('review_status') not in (None, '', 'PENDING'):
            reasons.append('Existing review flag: ' + row['review_status'])
        if row['label'] in ('ISSUE', 'OTHER'):
            reasons.append('Underrepresented role: ' + row['label'])
        if row in missing_review:
            reasons.append('No recorded reviewer and finalized label')
        if reasons:
            queue.append({'case_id': row['case_id'], 'sentence_id': row.get('sentence_id'),
                          'split': row['split'], 'text': row['text'], 'current_provisional_label': row['label'],
                          'review_priority': 1 if row.get('review_status') in ('NEEDS_DISCUSSION', 'SOURCE_CHECK') else 2 if row['label'] in ('ISSUE', 'OTHER') else 3,
                          'reasons': reasons, 'source_page_start': row.get('page_start'),
                          'source_page_end': row.get('page_end'), 'reviewer': None, 'final_label': None,
                          'source_text_sha256': hashlib.sha256(row['text'].encode('utf-8')).hexdigest()})
    queue.sort(key=lambda r: (r['review_priority'], r['split'], r['case_id'], str(r['sentence_id'])))
    return {'split_counts': counts, 'label_status_counts': dict(Counter(r.get('label_status', 'unspecified') for r in rows)),
            'missing_recorded_final_review': len(missing_review),
            'cross_split_exact_text_groups': duplicates,
            'review_queue_counts': dict(Counter(r['review_priority'] for r in queue)),
            'test_text_in_review_queue': False,
            'notes': ['Digest-level attorney references do not finalize sentence-level labels.',
                      'Exact cross-split text may be boilerplate; investigate before excluding it.',
                      'Validation has very few minority-role examples; do not treat its score as legal accuracy.']}, queue


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='data/revision_20261020_metadata/sentences.jsonl')
    p.add_argument('--output', default='runs/reliability_20261022')
    args = p.parse_args()
    report, queue = audit(read_jsonl(args.data))
    report['dataset_sha256'] = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    (folder / 'label_review_queue.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in queue) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
