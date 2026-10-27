"""Materialize explicit AI review decisions; never mutate training/evaluation labels."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path('data/private/label_audit_20261025')
OUT = Path('data/private/label_review_completion_20261026')

# Decisions follow source-context inspection, not similarity-based relabeling.
NOTES = {
28: ('retain', 'Earlier trial-court interpretation is procedural history: FACT. Source uses plural partnerships; propose transcription reconciliation only.'),
38: ('retain', 'Presentation of a named witness is a procedural event: FACT. Source includes the Human Resources Department; the previous context candidate pointed to later testimony and must not be used as this sentence location.'),
78: ('retain', 'Party and business background is FACT. Source includes long distance telephone service. Restore the omitted word only in a separately reviewed transcription.'),
85: ('retain', 'Attributed defense remains FACT. Source says posts, not post; no rhetorical change follows from this plural difference.'),
116: ('retain', 'The letter and proposed meeting are FACT. Record ellipsis-format differences explicitly; preserve time and attribution.'),
123: ('retain', 'Transaction is FACT. P3,029,250 and P3,029,250.00 have the same numeric value; this is decimal formatting, not a different amount. Preserve the previous dated event for On the same day.'),
157: ('retain', 'The bank letter is FACT, including its quoted reasons. Source uses reason singular; no label change.'),
213: ('retain', 'Filing an inheritance suit is FACT. Numeric citation links and one third/one-third formatting explain the alignment difference; retain ages and fractions.'),
225: ('hold_text', 'FACT is appropriate for the filing event, but dataset date 1938 conflicts with cached source 1983. Do not silently change either. Obtain another authoritative copy before settling the date.'),
246: ('retain', 'An explicit legal question in reasoning expressly adopted as our own remains ISSUE under the proposed function-based rubric. Not merely a historical appellate issue. Source spelling inequitous differs from dataset iniquitous.'),
282: ('retain', 'Article 1216 is used as the deciding Court legal basis: RULING. Dataset omits the repeated ART. 1216 heading; retain a documented exclusion or restore the source span.'),
290: ('retain', 'The Trust Receipts Law quotation is applied legal authority: RULING. Source contains Section of 7 and SECTION 7. Rights of the entruster; dataset is an edited composite, not a verbatim sentence.'),
326: ('hold_text', 'RULING fits adopted legal explanation, but cached source repeats a clause that the dataset omits. Do not certify the edited text as verbatim; verify another source or document the editorial repair.'),
333: ('retain', 'RULING: appellate reasoning about reformation and Article 1267 is endorsed later by the Supreme Court. Recovered actual passage; previous approximate source context was wrong. Source has inequitous spelling.'),
343: ('retain', 'RULING: commentator authority used in current analysis. Actual passage contains footnote 10 and unforseen, rather than unforeseen. Previous approximate source context was wrong.'),
348: ('retain', 'RULING: the Court explains practical consequences of its remedy. Dataset adds essential before service; source does not. Preserve original and propose source-faithful transcription separately.'),
352: ('retain', 'RULING: within the appellate resolution expressly quoted as our own. Source includes so highly inequitous; dataset omits so and changes spelling. Preserve the adoption cue with the example.'),
354: ('retain', 'RULING: legal authority within adopted reasoning. Source says appellant court, dataset appellate court; likely transcription/edition issue, not permission to silently repair the source.'),
358: ('retain', 'RULING: precedent used within adopted reasoning. Source has legal effect singular and different quotation formatting. Keep the attribution to precedent.'),
362: ('retain', 'RULING: adopted response rejecting a due-process contention, not the party contention itself. Source inequitous differs in spelling.'),
363: ('retain', 'RULING: prescription rule immediately applied by the Court. Source says right of action, not right of the action. Actual location recovered; previous approximate context was wrong.'),
365: ('retain', 'RULING: As correctly ruled explicitly adopts the finding. Ellipsis and parenthetical edits need a transcription record; preserve the accrual-date qualification.'),
368: ('retain', 'RULING: court application of the definition to a quoted contract term. The next sentence qualifies this conclusion; retain its context to avoid representing the whole contract as invalid.'),
369: ('retain', 'RULING: legal classification of the other conditions. Preserve both the quoted contract term and the Court explanation, including the mixed-condition conclusion that follows.'),
375: ('retain', 'RULING: doctrinal distinction used in current analysis. Source omits a before contract of sale; grammatical normalization must remain separate from label review.'),
378: ('hold_text', 'RULING fits the Court inference, but dataset inserts prior before to completion where cached source lacks it. This affects temporal meaning; keep on text hold pending source verification.'),
392: ('retain', 'RULING: definition used in contractual analysis. Dataset inserts between into the source phrase meeting of minds two persons; record editorial normalization separately.'),
411: ('retain', 'RULING: legal distinction between option and purchase. Dataset supplies a and for absent from cached source; no rhetorical change.'),
416: ('retain', 'RULING: enumerated legal distinctions. Source typo option money ids differs from is. Do not conflate spelling cleanup with new legal labels.'),
431: ('hold_text', 'RULING: application distinguishing performance from exercise of a right. Actual passage recovered; source says privilege of a right, dataset privilege or a right. Verify edition before accepting corrected text.'),
467: ('retain', 'RULING: current legal analysis of permitted rate increases. Exhibit identifiers were omitted; record exclusions rather than declaring the dataset verbatim.'),
468: ('retain', 'RULING: determines legal status of bank resolutions. Source exhibit identifiers omitted in dataset; retain actual resolution and circular numbers.'),
469: ('retain', 'RULING: legal limits on rate increases. Source is spread over adjacent paragraphs and includes Exh. 11 and quotation punctuation; treat dataset as an edited composite.'),
470: ('retain', 'RULING: mutuality applied to the bank conduct, with quoted Article 1308. Preserve or explicitly exclude the repeated statute heading.'),
529: ('retain', 'RULING: applies res judicata requisites to the case. Source citation 16, capitalization and dash spacing differ; no label change.'),
537: ('retain', 'RULING: explains scope of the preclusion rule. Citation 19 interrupts source wording; no substantive omission demonstrated.'),
539: ('retain', 'RULING: legal authority applied in the present analysis. Ellipsis marker and successors-in-interest hyphenation differ; preserve omissions transparently.'),
559: ('split', 'Mixed unit: formal assignment of error followed by supporting party argument. Propose exact parent-text spans ISSUE then FACT at , arguing that; retain the parent and context, do not rewrite the argument as a court finding.'),
}


def normalized(text):
    return re.sub(r'\s+', ' ', text.translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'}))).strip()


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Review artifacts already exist; preserve the previous review.')
    OUT.mkdir(exist_ok=True)
    canonical = Path('data/revision_20261020_metadata/sentences.jsonl')
    weights = Path('models/legalbert_balanced_v2/adapter_model.safetensors')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [canonical, weights]}
    queue = json.loads(Path('data/private/label_review_20261026/review_queue.json').read_text(encoding='utf-8'))
    assert {r['dataset_index'] for r in queue} == set(NOTES)
    sources, records = {}, []
    for row in queue:
        idx = row['dataset_index']
        source_path = Path(row['source_file'])
        assert hashlib.sha256(source_path.read_bytes()).hexdigest() == row['source_sha256']
        source = json.loads(source_path.read_text(encoding='utf-8'))
        text = normalized(source['raw_text'])
        sources[row['case_id']] = text
        anchor = ' '.join(normalized(row['original_text']).split()[:8])
        if idx == 343:
            anchor = 'According to Senator Arturo M. Tolentino,'
        if idx == 225:
            anchor = 'The complaint was filed on December 14,'
        if idx == 290:
            anchor = 'Thus, Section of 7 of PD No. 115 states:'
        if idx == 333:
            anchor = 'The fact that said contract has become inequitous'
        if idx == 469:
            anchor = 'CB Circular No. 905, Series of 1982'
        matches = list(re.finditer(re.escape(anchor), text, re.I))
        assert len(matches) == 1, (idx, anchor, len(matches))
        start = max(0, matches[0].start() - 500)
        end = min(len(text), matches[0].start() + len(row['original_text']) + 600)
        action, note = NOTES[idx]
        item = {**row, 'review_pass': 'source_context_review_20261026',
            'action': action, 'review_note': note, 'proposed_label': None if action == 'split' else row['original_label'],
            'final_label': None, 'human_reviewer': None, 'human_review_date': None,
            'reviewer_type': 'AI_assistant', 'attorney_validated': False, 'training_eligible': False,
            'source_url': source.get('decision_url'), 'source_context': text[start:end],
            'context_span': [start, end], 'span_basis': 'Whitespace and quotation-character normalized raw_text; exact anchor, no fuzzy matching.',
            'source_anchor': anchor, 'source_location_checked': True,
            'annotation_text_verbatim_verified': False, 'candidate_is_verified': False}
        if idx == 559:
            boundary = row['original_text'].index(', arguing that')
            item['proposed_segments'] = [dict(start=0, end=boundary, text=row['original_text'][:boundary], label='ISSUE'),
                                         dict(start=boundary, end=len(row['original_text']), text=row['original_text'][boundary:], label='FACT')]
            assert ''.join(s['text'] for s in item['proposed_segments']) == row['original_text']
        records.append(item)
    (OUT / 'normalized_sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'reviewed_queue.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    previous = json.loads((ROOT / 'changes.json').read_text(encoding='utf-8'))
    for r in previous:
        assert r['split'] == 'train'
    (OUT / 'prior_label_change_proposals.json').write_text(json.dumps(previous, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'reviewed_queue_rows': len(records), 'actions': dict(Counter(r['action'] for r in records)),
        'splits': dict(Counter(r['split'] for r in records)), 'prior_label_change_proposals': len(previous),
        'human_accepted_corrections': 0, 'training_performed': False, 'test_rows_reviewed': 0,
        'canonical_and_adapter_hashes': hashes, 'training_eligible_rows': 0}
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Source-context label review', '', 'AI-assisted proposals only. No human acceptance, training, canonical edits, or test-set review.', '',
             '| Index | Split | Existing label | Proposed action | Finding |', '|---|---|---|---|---|']
    lines += [f"| {r['dataset_index']} | {r['split']} | {r['original_label']} | {r['action']} | {r['review_note']} |" for r in records]
    lines += ['', '## Review sequence', '', 'Resolve four transcription holds (225, 326, 378, 431), adjudicate the mixed unit (559), then review the 13 earlier ISSUE-to-FACT proposals. The other 33 records have provisional retain recommendations with explicit text-reconciliation notes. None are automatically accepted.', '',
        'The current training set has only 28 ISSUE and 2 OTHER examples. Accepting all 13 prior ISSUE-to-FACT changes would leave 15 ISSUE examples before any segmentation/additions. Correctness must take priority over class counts, but reviewed ISSUE and OTHER examples are then needed from training/development cases. Do not move validation or test cases into training.', '',
        'Validation annotations are review-only proposals in a separate partition. Any human-approved revision must be versioned and both old and new models compared against the same fixed revision. Original benchmark results must retain their original labels and version.', '',
        'A source match verifies location, not truth or annotation quality. Source excerpts retain cached-source anomalies. No authoritative date or spelling correction is inferred from plausibility.']
    (OUT / 'REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
