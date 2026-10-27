"""Build noncanonical, AI-proposed data repairs with reproducible provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


# Exact source anchors reviewed in context. End anchors are excluded.
SEGMENTS = {
    'GR146224_20070126_FACT_3': (9, 'Respondent alleged', ' [5] After trial'),
    'GR174012_20081114_FACT_3': (9, 'By letter of October', ' [1] As the demand'),
    'GR174012_20081114_FACT_4': (10, 'In its Answer with Counterclaim', ' [5] To prove'),
    'GR174012_20081114_FACT_5': (31, 'In contrast, in the case at bar', ' [29] Another witness'),
    'GR174012_20081114_FACT_6': (31, 'The MCIAA nevertheless urges', None),
    'GR174012_20081114_RULING_8': (38, 'While the MCIAA is obliged', ' [36] Respondents must likewise'),
}


def exact_span(text, start=None, end=None):
    if start is not None and text.count(start) != 1:
        raise ValueError('Start anchor must match exactly once')
    left = text.index(start) if start is not None else 0
    if end is not None and text.count(end) != 1:
        raise ValueError('End anchor must match exactly once')
    right = text.index(end) if end is not None else len(text)
    if right <= left:
        raise ValueError('Invalid source span')
    return {'start_char': left, 'end_char': right, 'text': text[left:right]}


def build(output):
    if output.exists():
        raise ValueError('Choose a new directory; review artifacts must not be overwritten')
    dataset = Path('data/revision_20261020_metadata/sentences.jsonl')
    rows = [json.loads(line) for line in dataset.read_text(encoding='utf-8').splitlines()]
    canonical_hash = digest(dataset)
    labels_path = Path('data/private/ai_data_review_20261023/label_review.json')
    alignment_path = Path('data/private/ai_data_review_20261023/alignment_review.json')
    labels, alignments = read(labels_path), read(alignment_path)
    proposed = {r['sentence_id']: r for r in labels if r['ai_review_status'] == 'proposed_change'}
    candidate, label_log = [], []
    for row in rows:
        changed = dict(row)
        proposal = proposed.get(row['sentence_id'])
        # Keep ambiguous/context-dependent changes in the queue, not the candidate.
        if proposal and not proposal['needs_full_source_check'] and row['split'] == 'train':
            if row['text'] != proposal['text'] or row['label'] != proposal['current_provisional_label']:
                raise ValueError('Stale label proposal: ' + row['sentence_id'])
            changed.update(label=proposal['proposed_label'], original_label=row['label'],
                           label_status='AI_proposed_not_final', review_status='PENDING',
                           final_label=None, reviewer=None, training_eligible=False,
                           revision_reason=proposal['reason'])
            label_log.append({'sentence_id': row['sentence_id'], 'old_label': row['label'],
                              'proposed_label': changed['label'], 'reason': proposal['reason']})
        candidate.append(changed)
    pairs, exclusions = [], []
    for review in alignments:
        source_path = Path(review['source_artifact'])
        if digest(source_path) != review['source_sha256']:
            raise ValueError('Source changed: ' + str(source_path))
        source = {r['id']: r for r in read(source_path)['sentences']}
        if review['split'] != 'train':
            raise ValueError('Target-conditioned repairs are restricted to development training cases')
        if review['decision'] == 'exclude':
            exclusions.append(review)
            continue
        spans = []
        for sid in review['proposed_source_sentence_ids']:
            text = source[sid]['text']
            segment = SEGMENTS.get(review['unit_id'])
            span = exact_span(text, segment[1], segment[2]) if segment and segment[0] == sid else exact_span(text)
            spans.append({'sentence_id': sid, **span})
        flags = []
        if review['decision'] in ('context', 'source_check'):
            flags.append(review['decision'])
        if review['unit_id'].endswith('FACT_5') and review['case_id'] == 'GR174012_20081114':
            flags.append('contrast_requires_precedent_context_or_target_revision')
        pairs.append({
            'unit_id': review['unit_id'], 'case_id': review['case_id'], 'split': 'train', 'role': review['role'],
            'input': '\n\n'.join(s['text'] for s in spans), 'target': review['original_target'],
            'source_spans': spans, 'original_source_sentence_ids': review['original_source_sentence_ids'],
            'source_artifact': str(source_path), 'source_sha256': review['source_sha256'],
            'repair': 'exact_span_segmentation' if review['unit_id'] in SEGMENTS else review['decision'],
            'reason': review['reason'], 'unresolved_flags': flags,
            'reviewer_type': 'AI_assistant', 'attorney_validated': False, 'training_eligible': False,
            'target_conditioned_alignment': True, 'evaluation_eligible': False,
        })
    registry = {}
    fingerprints = {}
    for row in rows:
        registry.setdefault(row['case_id'], set()).add('existing_sentence_dataset_' + row['split'])
    for item in read('data/private/alignment_20261023/six_case_inventory.json'):
        registry.setdefault(item['case_id'], set()).add('digest_or_source_inspected_during_development')
    # Include all saved replay cases; Bicol is not among the six supplied digests.
    audit = read('data/private/reliability_audit_20261023/report.json')
    for item in audit['cases']:
        saved = read(item['source_result'])
        registry.setdefault(saved['case_id'], set()).add('previously_seen_pipeline_development')
        known = fingerprints.setdefault(saved['case_id'], set())
        known.update(h for h in (saved.get('verification', {}).get('source_sha256'),
                     saved.get('preprocessing', {}).get('extraction_audit', {}).get('pdf_sha256')) if h)
    exposures = [{'case_id': cid, 'reasons': sorted(reasons), 'source_fingerprints': sorted(fingerprints.get(cid, set())),
                  'independent_final_test_eligible': False}
                 for cid, reasons in sorted(registry.items())]
    summary = {
        'canonical_dataset_sha256': canonical_hash, 'canonical_modified': False,
        'records_audited_structurally': len(rows), 'distinct_sentence_ids': len({r['sentence_id'] for r in rows}),
        'records_without_final_label': sum(r.get('final_label') is None for r in rows),
        'priority_labels_reviewed': len(labels), 'candidate_label_changes': len(label_log),
        'context_dependent_changes_deferred': len(proposed)-len(label_log),
        'candidate_pairs': len(pairs), 'segmented_pairs': len(SEGMENTS), 'excluded_units': len(exclusions),
        'pair_repairs': dict(Counter(r['repair'] for r in pairs)),
        'known_case_exclusions': len(exposures), 'eligible_independent_reference_cases': 0,
        'models_retrained': False, 'attorney_validation': False,
        'scope': 'Structural audit of 570 records; targeted label review only, not semantic review of all records.',
        'inputs': {str(p): digest(p) for p in (dataset, labels_path, alignment_path)},
    }
    output.mkdir(parents=True)
    def save(name, value):
        (output/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save('candidate_sentences_REVIEW_REQUIRED.json', candidate)
    save('label_changes.json', label_log)
    save('deferred_label_changes.json', [r for r in proposed.values() if r['needs_full_source_check']])
    save('aligned_pairs_REVIEW_REQUIRED.json', pairs)
    save('excluded_units.json', exclusions)
    save('case_exposure_registry.json', exposures)
    save('summary.json', summary)
    save('unseen_case_intake.json', {'case_id': '', 'source_sha256': '', 'source_file': '',
         'source_complete_verified': False, 'previously_used_for_development': None,
         'attorney_accepted': False, 'reviewer': None, 'review_date': None, 'split': 'unassigned',
         'blocks': {role: '' for role in ('FACT', 'ISSUE', 'RULING')}})
    lines = ['# Data repair review packet', '',
             'AI-proposed repairs only. Original data and model weights are unchanged. No attorney approval is implied.', '',
             '## Label changes', '']
    for record in label_log:
        lines += [f"- {record['sentence_id']}: {record['old_label']} → {record['proposed_label']}. {record['reason']}"]
    lines += ['', '## Repaired source–target pairs', '']
    for pair in pairs:
        lines += ['### '+pair['unit_id'], '', 'Repair: '+pair['repair']+'. '+pair['reason'], '',
                  'Input evidence:', '', pair['input'], '', 'Original digest target:', '', pair['target'], '',
                  'Unresolved flags: '+(', '.join(pair['unresolved_flags']) or 'No additional machine flag; legal review still pending.'), '',
                  'Reviewer/date: ______  Accept/edit/reject: ______', '']
    (output/'REVIEW_PACKET.md').write_text('\n'.join(lines), encoding='utf-8')
    assert digest(dataset) == canonical_hash
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    print(json.dumps(build(parser.parse_args().output), indent=2))
