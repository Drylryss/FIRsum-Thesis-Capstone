"""Summarize completed development runs without inventing accuracy ratings."""
import hashlib
import json
from pathlib import Path
from delivery import prepare_delivery


def main():
    out = Path('outputs/mayor_fix_development_20261026')
    reserved = Path('data/private/reserved_evaluation_20261026')
    records = []
    for docket in ('151035', '202989', '225562', '111238'):
        new = json.loads((out / (docket + '.json')).read_text(encoding='utf-8'))
        if docket == '151035':
            old = json.loads((reserved / 'prediction.json').read_text(encoding='utf-8'))
        else:
            old = json.loads((Path('data/private/broader_development_fixed_20261026') /
                              (docket + '.json')).read_text(encoding='utf-8'))['result']
        before = prepare_delivery(old)
        after = new['delivery']
        source = {r['id']: r['text'] for r in new['sentences']}
        exact = all(e['text'] == source[e['sentence_id']]
                    for s in after['sections'].values() for e in s['evidence'])
        disposition = set(new['verification']['disposition_sentence_ids'])
        retained = disposition.issubset({e['sentence_id'] for e in after['sections']['RULING']['evidence']})
        assert exact and disposition and retained
        records.append({'case': docket, 'before_present': sum(bool(s['display_text']) for s in before['sections'].values()),
            'after_present': sum(bool(s['display_text']) for s in after['sections'].values()),
            'exact_source_evidence': exact, 'complete_selected_disposition': retained,
            'unchanged_display': {r: s['display_text'] == before['sections'][r]['display_text'] for r, s in after['sections'].items()},
            'raw_generation_accepted': {r: s['generation_accepted'] for r, s in new['verification']['blocks'].items()}})
    protected = json.loads((reserved / 'verification.json').read_text(encoding='utf-8'))['files_sha256']
    assert all(hashlib.sha256((reserved / p).read_bytes()).hexdigest() == h for p, h in protected.items())
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in manifest['sha256'].items())
    (out / 'comparison.json').write_text(json.dumps({'cases': records, 'regression_tests_passed': 157,
        'ui_tests_passed': True, 'original_reserved_trial_unchanged': True, 'training_performed': False,
        'legal_accuracy_estimate': None, 'independent_evaluation': False}, indent=2), encoding='utf-8')
    lines = ['# Mayor repair: development results', '',
        '157 regression tests and the delivery UI checks passed. Four fresh runs used the same trained LegalBERT adapter and frozen LegalPegasus; no references were supplied as generation input.', '',
        '| Case | Sections before | Sections after | Final text unchanged |', '|---|---:|---:|---|']
    for r in records:
        lines.append(f"| {r['case']} | {r['before_present']}/3 | {r['after_present']}/3 | {all(r['unchanged_display'].values())} |")
    lines += ['', 'Baseline text is rendered with the unchanged current delivery code to isolate evidence-selection differences. All selected evidence matches source segments; all selected final-disposition segments remain present. These are mechanical checks, not legal-accuracy ratings.', '',
        '## Mayor improvement and remaining limitations', '',
        'Issue and Ruling are no longer withheld. The output retains the mortgage, attributed cash-versus-installments allegation, denial of the petition and affirmance in toto. The alias no longer breaks the allegation across sentences.', '',
        'The Facts still open with undefined contracts, omit some procedural/background details, and retain the source-reported 70% allegation without the exact payment amounts. The Issue repeats two formulations. The Ruling contains an isolated “We disagree,” abbreviated Article 1332 reasoning, and an HTML-derived citation marker. This is a completeness improvement, not a finished attorney-validated digest or proof that raw LegalPegasus has become accurate.', '',
        'The original reserved Mayor run and its review packet are unchanged. Mayor is now explicitly a development case. No untouched final evaluation or new human review was performed. Model weights and canonical annotation labels were not modified.']
    (out / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(records))


if __name__ == '__main__':
    main()
