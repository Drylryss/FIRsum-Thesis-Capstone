"""Triage existing unresolved annotations; never change canonical labels/splits."""
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    root = Path('data/private/label_audit_20261025')
    out = Path('data/private/label_review_20261026')
    out.mkdir(exist_ok=True)
    canonical = Path('data/revision_20261020_metadata/sentences.jsonl')
    before = hashlib.sha256(canonical.read_bytes()).hexdigest()
    assert before == '352ce1abf7a325b77ab0d6c3b6da89d734986efc6e169767ab74091e49a5db7f'
    queue = json.loads((root / 'unresolved_queue.json').read_text(encoding='utf-8'))
    diagnostics = {r.get('dataset_index'): r for r in json.loads((root / 'source_reconciliation.json').read_text(encoding='utf-8'))}
    groups = {
        'minor_wording_or_source_typo': {28, 38, 85, 157, 246, 354, 358, 362, 375, 392, 411, 416},
        'omitted_or_altered_substantive_wording': {78, 326, 348, 352, 378},
        'numeric_amount_discrepancy': {123},
        'citation_or_heading_reconciliation': {213, 282, 290, 467, 468, 469, 470, 529, 537},
        'quotation_ellipsis_policy': {116, 365, 368, 369, 539},
        'source_alignment_unresolved': {225, 333, 343, 363, 431},
        'mixed_rhetorical_role': {559},
    }
    actions = {
        'minor_wording_or_source_typo': 'Compare exact full-source span; propose a separate transcription correction. Do not infer a new rhetorical label from spelling differences.',
        'omitted_or_altered_substantive_wording': 'Review the complete source passage and omitted words before accepting this annotation; omission may change the proposition.',
        'numeric_amount_discrepancy': 'Resolve the currency/decimal discrepancy directly against the source before any reuse. Never repair money using similarity alone.',
        'citation_or_heading_reconciliation': 'Verify citation/heading markup or exhibit text. Retain substantive article and exhibit identifiers; do not strip all numbers.',
        'quotation_ellipsis_policy': 'Decide whether the unit is a contiguous source sentence or an edited quotation. Record exclusions explicitly instead of treating it as verbatim.',
        'source_alignment_unresolved': 'Find the exact source span first. The previous approximate paragraph is not evidence of a verified match.',
        'mixed_rhetorical_role': 'Review the existing exact-span split proposal: current assignment of error versus supporting party argument. Neither fragment is training eligible yet.',
    }
    records = []
    for row in queue:
        assert hashlib.sha256(Path(row['source_file']).read_bytes()).hexdigest() == row['source_sha256']
        idx = row['dataset_index']
        matches = [name for name, indices in groups.items() if idx in indices]
        assert len(matches) == 1, (idx, matches)
        category = matches[0]
        diagnostic = diagnostics.get(idx, {})
        records.append({'dataset_index': idx, 'sentence_id': row['sentence_id'], 'case_id': row['case_id'],
            'split': row['split'], 'original_label': row['original_label'], 'original_text': row['text'],
            'category': category, 'action': actions[category],
            'priority': 'high' if category in {'numeric_amount_discrepancy', 'omitted_or_altered_substantive_wording', 'mixed_rhetorical_role', 'source_alignment_unresolved'} else 'normal',
            'source_file': row['source_file'], 'source_sha256': row['source_sha256'],
            'diagnostic_differences': diagnostic.get('diagnostic_differences'),
            'candidate_source_excerpt': diagnostic.get('diagnostic_source_excerpt'),
            'candidate_is_verified': False, 'reviewer_type': 'AI_assistant',
            'attorney_validated': False, 'training_eligible': False, 'final_label': None,
            'protected_split': row['split'] != 'train'})
    assert len(records) == 38
    assert hashlib.sha256(canonical.read_bytes()).hexdigest() == before
    (out / 'review_queue.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'rows': len(records), 'categories': dict(Counter(r['category'] for r in records)),
               'splits': dict(Counter(r['split'] for r in records)), 'canonical_sha256': before,
               'canonical_changed': False, 'new_label_corrections': 0, 'training_performed': False}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Remaining annotation review', '',
        'All 38 unresolved rows were triaged. This is AI-assisted data-quality review, not attorney acceptance. No canonical text, label, split, or weight was changed. Prior 13 label-change proposals remain review-only.', '',
        'Do not retrain on this queue. Some differences involve omitted reasoning, a monetary amount, statutory headings or source alignment; automatic replacement would hide uncertainty. Existing validation rows remain protected.', '',
        '| Index | Split | Priority | Category |', '|---|---|---|---|']
    lines += [f"| {r['dataset_index']} | {r['split']} | {r['priority']} | {r['category']} |" for r in records]
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
