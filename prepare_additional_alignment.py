"""Prepare source-linked development review; never turn digests into BERT labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from align_digest_pairs import coverage, numbers, neighboring_context, fragment_warning
from core import sentences
from preprocessing import prepare_decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/private/additional_digest_development_20261024_v3'))
    args = parser.parse_args()
    out = args.root / 'alignment'
    if out.exists():
        raise ValueError('Refusing to overwrite alignment review')
    out.mkdir()
    refs = json.loads((args.root / 'references.json').read_text(encoding='utf-8'))
    audit, candidates, source_inventory = [], [], []
    for ref in refs:
        if ref['split'] == 'test':
            continue
        source_path = args.root / 'sources' / (ref['docket'] + '.json')
        doc = json.loads(source_path.read_text(encoding='utf-8'))
        prepared = prepare_decision(doc['raw_text'])
        source = [{'id': i+1, 'text': t} for i, t in enumerate(sentences(prepared['text']))]
        (out / (ref['docket'] + '_source.json')).write_text(json.dumps({
            'case_id': ref['case_id'], 'decision_url': ref['decision_url'],
            'sentences': source, 'preprocessing': prepared,
            'labels': 'None assigned. Digest headings do not label source sentences.'}, ensure_ascii=False, indent=2), encoding='utf-8')
        source_inventory.append({'case_id': ref['case_id'], 'split': ref['split'],
                                 'source_sentence_count': len(source), 'warnings': prepared['warnings']})
        # Reference-conditioned alignment is permitted only for development training.
        if ref['split'] != 'train':
            continue
        for role, target in ref['blocks'].items():
            for i, unit in enumerate(sentences(target)):
                ranked = sorted(source, key=lambda s: -coverage(unit, s['text']))
                evidence = sorted([s for s in ranked[:3] if coverage(unit, s['text']) >= .15], key=lambda s:s['id'])
                joined = ' '.join(s['text'] for s in evidence)
                item = {'unit_id': f"{ref['case_id']}_{role}_{i+1}", 'case_id': ref['case_id'],
                        'split': 'train', 'role': role, 'target': unit,
                        'proposed_evidence': evidence, 'neighboring_context': neighboring_context(evidence, source),
                        'lexical_coverage': coverage(unit, joined),
                        'numbers_missing_from_suggested_evidence': sorted(numbers(unit)-numbers(joined)),
                        'fragment_requires_context': fragment_warning(unit),
                        'source_file': str(source_path),
                        'source_file_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        'approval_basis': ref['approval_basis'],
                        'semantic_support_verified': False, 'training_eligible': False,
                        'evaluation_eligible': False, 'review_status': 'AI_alignment_candidate'}
                candidates.append(item)
        audit.append({'case_id': ref['case_id'], 'target_units': sum(c['case_id']==ref['case_id'] for c in candidates)})
    (out/'candidates.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding='utf-8')
    summary = {'reference_allocation': dict(Counter(r['split'] for r in refs)),
               'source_cases': source_inventory, 'training_alignment_cases': audit,
               'candidate_units': len(candidates),
               'units_with_numeric_flags': sum(bool(c['numbers_missing_from_suggested_evidence']) for c in candidates),
               'fragment_flags': sum(c['fragment_requires_context'] for c in candidates),
               'validation_target_conditioned_inputs_created': False,
               'test_source_fetched': False, 'new_LegalBERT_labels_created': 0,
               'notice': 'Lexical matches and missing-number flags are triage only, not legal correctness findings.'}
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    lines = ['# Additional training-reference alignment review', '',
             'User-assumed approval permits development work. It is not a completed attorney alignment review.',
             'These are lexical evidence suggestions, not training-ready records or legal findings.', '']
    for c in sorted(candidates, key=lambda x:(not bool(x['numbers_missing_from_suggested_evidence']), x['lexical_coverage'])):
        lines += ['## '+c['unit_id'], '', c['target'], '',
                  'Numeric flags: '+str(c['numbers_missing_from_suggested_evidence']), '']
        lines += [f"- Source sentence {s['id']}: {s['text']}" for s in c['proposed_evidence']]
        lines += ['', 'Check speaker, outcome, negation, amounts and surrounding context.', '']
    (out/'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_cases'},indent=2))


if __name__ == '__main__':
    main()
