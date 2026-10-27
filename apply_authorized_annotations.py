"""Create a versioned user-authorized AI annotation revision, not human gold."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path('data/revision_20261020_metadata/sentences.jsonl')
    out = Path('data/revision_20261027_ai_annotations')
    adapter = Path('models/legalbert_balanced_v2/adapter_model.safetensors')
    before_hash = sha(source)
    adapter_hash = sha(adapter)
    assert before_hash == '352ce1abf7a325b77ab0d6c3b6da89d734986efc6e169767ab74091e49a5db7f'
    original = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    revised = copy.deepcopy(original)
    proposals = json.loads(Path('data/private/label_audit_20261025/changes.json').read_text(encoding='utf-8'))
    review = json.loads(Path('data/private/label_review_completion_20261026/reviewed_queue.json').read_text(encoding='utf-8'))
    audit = []
    for p in proposals:
        row = revised[p['dataset_index']]
        assert row['split'] == p['split'] == 'train'
        assert row['sentence_id'] == p['sentence_id'] and row['text'] == p['text']
        assert row['label'] == p['original_label'] == 'ISSUE' and p['proposed_label'] == 'FACT'
        assert sha(Path(p['source_file'])) == p['source_sha256']
        row.update(label='FACT', original_label='ISSUE', label_status='ai_revised_user_authorized',
                   review_status='AI_REVISED_NOT_HUMAN_VALIDATED', reviewer_type='AI_assistant',
                   attorney_validated=False, human_validated=False, annotation_revision_date='2026-10-27',
                   annotation_revision_reason=p['reason'], annotation_source_file=p['source_file'],
                   annotation_source_sha256=p['source_sha256'], revision_parent_dataset=str(source),
                   training_eligible=True)
        assert row.get('final_label') is None and row.get('reviewer') is None
        audit.append({**p, 'action': 'applied_to_versioned_dataset',
                      'authorization': 'User explicitly authorized AI sentence annotation adjustments.',
                      'applied_date': '2026-10-27', 'applied_label': 'FACT',
                      'human_validated': False, 'training_eligible': True,
                      'review_status': 'AI_REVISED_NOT_HUMAN_VALIDATED'})
    holds = [r for r in review if r['action'] in ('hold_text', 'split')]
    for r in holds:
        if r['split'] == 'train':
            row = revised[r['dataset_index']]
            assert row['sentence_id'] == r['sentence_id']
            row.update(training_eligible=False, annotation_hold=r['review_note'],
                       review_status='SOURCE_OR_SEGMENTATION_HOLD')
    assert len(original) == len(revised) == 570
    changed = [i for i, (old, new) in enumerate(zip(original, revised)) if old['label'] != new['label']]
    assert set(changed) == {p['dataset_index'] for p in proposals} and len(changed) == 13
    for old, new in zip(original, revised):
        for key in ('text', 'case_id', 'split', 'sentence_id'):
            assert old[key] == new[key]
        if old['split'] != 'train':
            assert old == new
    assert sha(source) == before_hash and sha(adapter) == adapter_hash
    out.mkdir(exist_ok=False)
    (out / 'sentences.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in revised), encoding='utf-8')
    for name, value in [('applied_changes', audit), ('unresolved_records', holds)]:
        (out / (name + '.json')).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'date': '2026-10-27', 'status': 'AI_annotation_revision_not_human_gold',
        'source_sha256': before_hash, 'revised_sha256': sha(out / 'sentences.jsonl'),
        'rows': len(revised), 'labels_changed': len(changed), 'sentence_texts_changed': 0,
        'validation_rows_unchanged': sum(r['split'] == 'validation' for r in revised),
        'test_rows_unchanged': sum(r['split'] == 'test' for r in revised),
        'training_counts_before': dict(Counter(r['label'] for r in original if r['split'] == 'train')),
        'training_counts_after': dict(Counter(r['label'] for r in revised if r['split'] == 'train')),
        'unresolved_records': len(holds), 'training_rows_on_hold': sum(r.get('training_eligible') is False for r in revised if r['split'] == 'train'),
        'training_performed': False, 'deployment_changed': False, 'adapter_sha256': adapter_hash,
        'human_validated': False, 'authorization': 'Explicit user authorization for AI annotation edits'}
    (out / 'manifest.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Applied AI annotation revision — 2026-10-27', '',
        'Applied 13 ISSUE-to-FACT corrections under explicit user authorization. These are AI-reviewed annotations, not attorney-validated gold labels. Original dataset preserved.', '',
        '| Case | Corrections |', '|---|---:|']
    lines += [f'| {case} | {count} |' for case, count in Counter(r['case_id'] for r in audit).items()]
    lines += ['', 'All 570 texts, sentence IDs and case splits remain unchanged. All 112 validation and 58 test records are unchanged. The 400 training rows now contain FACT 187, ISSUE 15, RULING 196 and OTHER 2.', '',
        'Four training records remain ineligible pending source/segmentation resolution (225, 378, 431, 559). Record 326 is a validation text conflict, tracked only in the sidecar. No source date, quantity or substantive wording was guessed. The mixed-unit segmentation proposal remains in the sidecar, not an orphaned training fragment.', '',
        'The training loader rejects this full revision until the holds are resolved or an explicitly audited training-only exclusion view is prepared. This is a data-quality hold, not a requirement for further user permission to annotate.', '',
        'No retraining or deployment change occurred. Annotation changes do not change model predictions until a separate candidate is trained and evaluated. Add consistent ISSUE/OTHER coverage before that experiment. AI-applied labels must not be described as independent attorney judgments.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
