"""Record explicit AI label decisions in a separate review-only dataset."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from core import read_jsonl

OUT = Path('data/private/label_audit_20261025')
DATA = Path('data/revision_20261020_metadata/sentences.jsonl')
EXPECTED_SHA = '352ce1abf7a325b77ab0d6c3b6da89d734986efc6e169767ab74091e49a5db7f'
# Exact dataset hash protects these positions; stable IDs are retained in exports.
CHANGES = {
    551: 'Supporting appellants\' due-process contention; formal assignments of error appear separately.',
    553: 'Reports the spouses\' request for procedural relaxation and medical justification.',
    554: 'Attributed allegation concerning the sufficiency of BPI\'s proof.',
    555: 'Reiterates the spouses\' supporting argument about default and illness.',
    556: 'Continues the spouses\' argument in the same source paragraph, not an independent court finding.',
    557: 'Party claim about insufficient proof and a prayer for dismissal.',
    560: 'Petitioner\'s supporting argument about summary proceedings; source footnote 18 interrupts the text.',
    562: 'Source context establishes that "It" refers to petitioner insisting on its position.',
    563: 'Petitioner distinguishes the Barican precedent to support its position.',
    566: 'Party account of conveyances and witnesses\' evidence.',
    567: 'Party argument concerning application of the Statute of Frauds.',
    568: 'Petitioners\' assertion about objections in the record.',
    569: 'Petitioners\' conclusion about waiver/admissibility; source footnotes 14 and 15 interrupt the text.',
}
RETAINS = {
    60:'Reports the contents of the board resolution, rather than a legal conclusion.',
    61:'Reports a direction in a board document; its location in reasoning does not change its factual function.',
    62:'Reports non-payment of the initial amount, a factual premise; the following legal conclusion is separately RULING.',
    63:'Reports rejection of a document, a factual occurrence distinct from the following legal conclusion.',
    205:'Procedural chronology inside Metrobank\'s supporting argument.',
    206:'Reports the date and contents of a repayment letter; the next sentence draws the legal inference.',
    209:'Reports amounts appearing in demand letters and an account statement, not an independent legal conclusion.',
}


def build_candidate(rows, audit, reconciled=()):
    reconciled = {r['sentence_id']:r for r in reconciled if r['status']=='matched_after_citation_cleanup'}
    by_index = {r['dataset_index']: r for r in audit}
    if len(by_index) != 512:
        raise ValueError('Expected complete 512-row development audit')
    candidate, decisions = [], []
    for i, row in enumerate(rows):
        revised = dict(row)
        if row['split'] == 'test':
            candidate.append(revised)
            continue
        ctx = by_index[i]
        if any(ctx[k] != row[k] for k in ('sentence_id', 'case_id', 'split', 'text')) or ctx['original_label'] != row['label']:
            raise ValueError('Stale audit identity: ' + row['sentence_id'])
        flags = list(ctx['text_quality_flags'])
        if ctx['source_match'] != 'normalized_substring' and i not in CHANGES and row['sentence_id'] not in reconciled:
            flags.append('source_reconciliation_needed')
        if i == 559:
            flags.append('mixed_ISSUE_argument')
        proposed = 'FACT' if i in CHANGES else row['label']
        if i in CHANGES and (row['split'] != 'train' or row['label'] != 'ISSUE'):
            raise ValueError('Correction would alter a protected split or unexpected label')
        decision = {**ctx, 'proposed_label': proposed,
            'action': 'change' if i in CHANGES else 'provisional_retain',
            'reason': CHANGES.get(i, RETAINS.get(i,'No clear change in sentence-level first pass; retention remains provisional.')),
            'source_context_manually_checked': i in CHANGES or i in RETAINS, 'review_flags': flags,
            'html_reconciliation': reconciled.get(row['sentence_id']),
            'review_status': 'AI_PROVISIONAL_NOT_HUMAN_ACCEPTED', 'training_eligible': False}
        decisions.append(decision)
        if row['split'] == 'train':
            revised.update(label=proposed, original_label=row['label'], final_label=None,
                label_status='AI_provisional_label_audit', reviewer_type='AI_assistant',
                review_status='AI_PROVISIONAL_NOT_HUMAN_ACCEPTED', attorney_validated=False,
                training_eligible=False, label_audit_reason=decision['reason'], label_audit_flags=flags)
        candidate.append(revised)
    return candidate, decisions


def main():
    if hashlib.sha256(DATA.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise ValueError('Dataset changed; re-audit first')
    rows = read_jsonl(DATA)
    audit = json.loads((OUT / 'context_audit.json').read_text(encoding='utf-8'))
    for r in audit:
        if hashlib.sha256(Path(r['source_file']).read_bytes()).hexdigest() != r['source_sha256']:
            raise ValueError('Source changed: ' + r['source_file'])
    reconciliation = json.loads((OUT/'source_reconciliation.json').read_text(encoding='utf-8'))
    for r in reconciliation:
        if r.get('html_source') and hashlib.sha256(Path(r['html_source']).read_bytes()).hexdigest() != r['html_sha256']:
            raise ValueError('HTML source changed')
    candidate, decisions = build_candidate(rows, audit, reconciliation)
    assert all(a == b for a,b in zip(rows,candidate) if a['split'] != 'train')
    assert all(all(a[k] == b[k] for k in ('text','case_id','split','sentence_id')) for a,b in zip(rows,candidate))
    changed = [r for r in decisions if r['action'] == 'change']
    assert len(changed) == 13
    unresolved = [r for r in decisions if r['review_flags']]
    (OUT / 'sentences_review_only.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in candidate)+'\n',encoding='utf-8')
    summary = {'first_pass_development_units':512, 'source_checked_label_corrections':13,
        'source_checked_retained_boundary_labels':len(RETAINS),
        'normalized_source_matches':sum(r['source_match']=='normalized_substring' for r in decisions),
        'additional_HTML_citation_matches':sum(bool(r['html_reconciliation']) for r in decisions),
        'source_reconciliation_remaining':sum('source_reconciliation_needed' in r['review_flags'] for r in decisions),
        'unresolved_rows':len(unresolved),
        'train_counts_after_proposals':dict(Counter(r['label'] for r in candidate if r['split']=='train')),
        'validation_test_unchanged':True, 'canonical_dataset_changed':False, 'model_weights_changed':False,
        'attorney_validated':False, 'scope':'First-pass review of 512 units; not full semantic verification of every retained label.'}
    for name, data in [('decisions.json',decisions),('changes.json',changed),('unresolved_queue.json',unresolved),('summary.json',summary)]:
        (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    lines = ['# Development label audit — 25 October 2026','',
        'Completed a first-pass reading of 512 development annotations and source-context checks for 13 ISSUE-to-FACT corrections. The corrected copy is AI-provisional and review-only, not attorney-approved.', '',
        f"432/512 units match normalized source paragraphs; a further {summary['additional_HTML_citation_matches']} match the complete source HTML after identified citation markers are removed. {summary['source_reconciliation_remaining']} still require source-text reconciliation. Source correspondence is not proof of the correct rhetorical label.",'',
        f"{len(unresolved)} distinct rows remain flagged: 37 source-text discrepancies and one mixed ISSUE/argument unit. Seven FACT/RULING boundary cases were source-checked and retained as FACT because they report events/document contents rather than legal conclusions.",'',
        'The original 112 validation and 58 test records are unchanged. The canonical dataset, new validation packet and deployed weights are unchanged. Training is disabled for this review copy through the existing training_eligible guard. No new performance improvement is claimed.', '',
        '| Sentence ID | Correction | Reason |','|---|---|---|']
    lines += [f"| {r['sentence_id']} | ISSUE → FACT | {r['reason']} |" for r in changed]
    lines += ['', 'See ANNOTATION_RUBRIC.md, changes.json, decisions.json, unresolved_queue.json, and context_audit.json for the rubric, all decisions, and source URLs/hashes/context. Resolve the flagged rows before treating the dataset as finalized. Quotation-boundary evidence selection is a separate failure; label corrections alone do not fix it.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
