"""Prepare a complete development review and import explicitly completed reviews.

Neither operation edits the input dataset or the deployed model. Test text is
excluded from review; imported labels require a complete development review.
"""
import argparse
from collections import Counter, defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path

from audit_training_data import audit
from core import LABELS, read_jsonl, validate_splits


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def index_rows(rows):
    validate_splits(rows)
    ids = [r.get('sentence_id') for r in rows]
    if any(not sid for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError('Missing or duplicate sentence IDs')
    return dict(zip(ids, rows))


def build_reviews(rows, proposals=()):
    index_rows(rows)
    proposals = {r['sentence_id']: r for r in proposals}
    cases = defaultdict(list)
    for row in rows:
        if row['split'] in ('train', 'validation'):
            cases[row['case_id']].append(row)
    result = []
    for case_rows in cases.values():
        for i, row in enumerate(case_rows):
            p = proposals.get(row['sentence_id'])
            if p and (p['text'] != row['text'] or p['current_provisional_label'] != row['label']):
                raise ValueError('Stale proposal: ' + row['sentence_id'])
            flags = []
            if row['label'] in ('ISSUE', 'OTHER'):
                flags.append('underrepresented_class')
            if row.get('review_status') in ('NEEDS_DISCUSSION', 'SOURCE_CHECK'):
                flags.append(row['review_status'])
            if p:
                flags.append('existing_AI_review_requires_verification')
            result.append({
                'sentence_id': row['sentence_id'], 'case_id': row['case_id'],
                'split': row['split'], 'text': row['text'],
                'source_text_sha256': text_sha(row['text']), 'original_label': row['label'],
                'source_sheet': row.get('source_sheet'), 'source_row': row.get('source_row'),
                'page_start': row.get('page_start'), 'page_end': row.get('page_end'),
                'nearby_annotated_sentences': [
                    {'sentence_id': n['sentence_id'], 'text': n['text']}
                    for j, n in enumerate(case_rows[max(0, i-2):i+3], max(0, i-2)) if j != i],
                'context_limit': 'Nearby annotated rows may omit intervening source text. Consult the full decision.',
                'triage_flags': flags,
                'AI_proposal': ({'label': p.get('proposed_label'), 'reason': p.get('reason'),
                                 'needs_full_source_check': p.get('needs_full_source_check'),
                                 'attorney_validated': False} if p else None),
                'review': {'status': 'PENDING', 'final_label': None, 'reviewer': None,
                           'reviewer_type': None, 'review_date': None, 'reason': None,
                           'source_checked': False, 'source_reference': None},
            })
    return result


def import_reviews(rows, reviews):
    originals = index_rows(rows)
    expected = {sid for sid, r in originals.items() if r['split'] in ('train', 'validation')}
    supplied = [r['sentence_id'] for r in reviews]
    if len(supplied) != len(set(supplied)) or set(supplied) != expected:
        raise ValueError('Review must cover every development row exactly once and exclude test rows')
    updates = {}
    for item in reviews:
        row = originals[item['sentence_id']]
        for key in ('case_id', 'split', 'text'):
            if item[key] != row[key]:
                raise ValueError('Source identity changed: ' + item['sentence_id'])
        if item['source_text_sha256'] != text_sha(row['text']) or item['original_label'] != row['label']:
            raise ValueError('Stale source or label: ' + item['sentence_id'])
        review = item['review']
        if review.get('status') != 'ACCEPTED' or review.get('final_label') not in LABELS:
            raise ValueError('Unresolved review: ' + item['sentence_id'])
        if review.get('reviewer_type') not in ('human_annotator', 'attorney'):
            raise ValueError('Explicit human review required; AI proposals are not accepted labels')
        if any(not isinstance(review.get(k), str) or not review[k].strip()
               for k in ('reviewer', 'review_date', 'reason', 'source_reference')):
            raise ValueError('Missing review provenance: ' + item['sentence_id'])
        if date.fromisoformat(review['review_date']) > date.today():
            raise ValueError('Review date cannot be in the future')
        if review.get('source_checked') is not True:
            raise ValueError('Full source check required')
        updates[item['sentence_id']] = review
    output = []
    for row in rows:
        revised = dict(row)
        if row['sentence_id'] in updates:
            review = updates[row['sentence_id']]
            revised.update(original_label=row['label'], label=review['final_label'],
                           final_label=review['final_label'], review_status='ACCEPTED',
                           label_status='human_reviewed', reviewer=review['reviewer'],
                           reviewer_type=review['reviewer_type'], review_date=review['review_date'],
                           review_reason=review['reason'], source_reference=review['source_reference'],
                           attorney_validated=review['reviewer_type'] == 'attorney', training_eligible=True)
        output.append(revised)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/revision_20261020_metadata/sentences.jsonl'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--completed-review', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output folder; existing review artifacts are preserved')
    rows = read_jsonl(args.data)
    if args.completed_review:
        packet = json.loads(args.completed_review.read_text(encoding='utf-8'))
        if packet['dataset_sha256'] != sha(args.data):
            raise ValueError('Dataset fingerprint changed')
        revised = import_reviews(rows, packet['records'])
        args.output.mkdir(parents=True)
        (args.output / 'sentences.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in revised)+'\n', encoding='utf-8')
        (args.output / 'import_provenance.json').write_text(json.dumps({
            'source_dataset_sha256': sha(args.data), 'review_packet_sha256': sha(args.completed_review),
            'test_rows_unchanged': True, 'deployment_performed': False,
            'notice': 'Review metadata is recorded, not independently authenticated.'}, indent=2), encoding='utf-8')
        return
    proposal_path = Path('data/private/ai_data_review_20261023/label_review.json')
    proposals = json.loads(proposal_path.read_text(encoding='utf-8')) if proposal_path.exists() else []
    reviews = build_reviews(rows, proposals)
    report, _ = audit(rows)
    args.output.mkdir(parents=True)
    packet = {'dataset_sha256': sha(args.data), 'records': reviews}
    (args.output / 'review.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for split in ('train', 'validation'):
        folder = args.output / split
        folder.mkdir()
        for cid in sorted({r['case_id'] for r in reviews if r['split'] == split}):
            lines = [f'# {cid} — {split}', '', 'Pending review. Context consists of selected annotation rows, not the complete decision.', '']
            for r in reviews:
                if r['case_id'] != cid:
                    continue
                lines += [f"## {r['sentence_id']} — provisional {r['original_label']}", '', r['text'], '']
                if r['AI_proposal']:
                    lines += ['AI proposal (unverified): '+json.dumps(r['AI_proposal'], ensure_ascii=False), '']
                lines += ['Reviewer / date / final label / source reference / reason: __________', '']
            (folder / (cid + '.md')).write_text('\n'.join(lines), encoding='utf-8')
    files = list(Path('.').glob('*.py')) + [Path('model_config.json'), args.data]
    config = json.loads(Path('model_config.json').read_text(encoding='utf-8'))
    files += [p for p in Path(config['legalbert_path']).glob('*') if p.is_file()]
    files += list(Path('web').glob('*'))
    manifest = {'purpose': 'Development baseline snapshot; not a final evaluation freeze',
                'co_worker_reserved_case_list_received': False,
                'files_sha256': {str(p): sha(p) for p in sorted(set(files)) if p.is_file()},
                'generator': 'nsi319/legal-pegasus (frozen); remote weights not fingerprinted by this snapshot',
                'dataset_sha256': sha(args.data)}
    (args.output / 'baseline_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'review_records': len(reviews), 'splits': dict(Counter(r['split'] for r in reviews)),
                      'AI_proposals': sum(r['AI_proposal'] is not None for r in reviews),
                      'test_text_exported': False, 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
