"""Keep raw generation, final delivery, and pending legal review separate."""
import argparse
import hashlib
import json
from pathlib import Path

from core import ROLES

ERROR_FIELDS = ('unsupported_statement', 'wrong_actor_or_speaker', 'outcome_or_negation_error',
                'missing_essential_fact', 'missing_reasoning', 'incoherent_or_unreadable')


def fingerprints(prediction):
    return {h for h in (prediction.get('verification', {}).get('source_sha256'),
                       prediction.get('preprocessing', {}).get('extraction_audit', {}).get('pdf_sha256')) if h}


def eligibility_errors(prediction, reference, registry):
    errors = []
    if not reference:
        return ['No accepted case-matched reference supplied.']
    cid = prediction.get('case_id')
    if not cid or cid != reference.get('case_id'):
        errors.append('Prediction/reference case identities differ or are missing.')
    if reference.get('split') != 'test':
        errors.append('Reference is not assigned to the final test split.')
    if reference.get('attorney_accepted') is not True or not reference.get('reviewer') or not reference.get('review_date'):
        errors.append('Documented reference acceptance is missing.')
    if reference.get('source_complete_verified') is not True:
        errors.append('Source completeness has not been verified.')
    if reference.get('previously_used_for_development') is not False:
        errors.append('No explicit untouched-case declaration.')
    hashes = fingerprints(prediction)
    if not reference.get('source_sha256') or reference['source_sha256'] not in hashes:
        errors.append('Reference source fingerprint does not match the prediction source.')
    if any(r.get('case_id') == cid or hashes.intersection(r.get('source_fingerprints', [])) for r in registry):
        errors.append('Case identity or source fingerprint is in the development exclusion registry.')
    if cid in prediction.get('exemplar_case_ids', []):
        errors.append('Case was supplied as an exemplar.')
    if any(not reference.get('blocks', {}).get(role, '').strip() for role in ROLES):
        errors.append('Reference has missing FIR sections.')
    return errors


def build_packet(prediction, registry, reference=None):
    errors = eligibility_errors(prediction, reference, registry)
    sections = []
    for role in ROLES:
        final = prediction.get('delivery', {}).get('sections', {}).get(role, {})
        # Never substitute fallback blocks for missing raw generation.
        for stage, text in (
                ('raw_legalpegasus', prediction.get('generated_blocks', {}).get(role, '')),
                ('final_delivery', final.get('display_text', final.get('text', '')) if final.get('status') != 'withheld' else '')):
            sections.append({'stage': stage, 'role': role, 'text': text,
                             'delivery_mode': final.get('status') if stage == 'final_delivery' else None,
                             'synthesis': final.get('synthesis') if stage == 'final_delivery' else None,
                             'missing_or_withheld': not bool(text.strip()),
                             'reference': reference.get('blocks', {}).get(role) if reference else None,
                             'evidence_ids': [e['sentence_id'] for e in final.get('evidence', [])] if stage == 'final_delivery' else [],
                             'human_review': {'reviewer': None, 'review_date': None, 'decision': None,
                                              'errors': {name: None for name in ERROR_FIELDS},
                                              'supporting_source_ids': [], 'notes': ''}})
    return {'case_id': prediction.get('case_id'), 'formal_evaluation_eligible': not errors,
            'eligibility_errors': errors, 'legal_accuracy_estimate': None,
            'review_complete': False, 'source_fingerprints': sorted(fingerprints(prediction)),
            'sections': sections, 'scoring_policy': 'Evaluate both stages separately; retain missing sections in denominators. Human edits are a separate artifact.',
            'textual_metrics': None}


def textual_scores(packet):
    if not packet['formal_evaluation_eligible']:
        raise ValueError('Formal scoring blocked: ' + '; '.join(packet['eligibility_errors']))
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    results = []
    for section in packet['sections']:
        text, reference = section['text'], section['reference']
        scores = {'rougeL': 0.0, 'bleu': 0.0, 'bertscore_f1': 0.0}
        if text.strip():
            _, _, f1 = score([text], [reference], model_type='roberta-large', lang='en', verbose=False)
            scores = {'rougeL': scorer.score(reference, text)['rougeL'].fmeasure,
                      'bleu': sacrebleu.sentence_bleu(text, [reference]).score / 100,
                      'bertscore_f1': float(f1[0])}
        results.append({'stage': section['stage'], 'role': section['role'],
                        'missing_or_withheld': section['missing_or_withheld'], **scores})
    return {'scale': '0–1, descriptive text similarity, not legal accuracy',
            'bertscore_model': 'roberta-large', 'empty_output_policy': 'zero scores; retain in denominator',
            'sections': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prediction', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score', action='store_true', help='Run textual metrics only after the eligibility gate passes')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite an earlier evaluation packet')
    read = lambda p: json.loads(p.read_text(encoding='utf-8'))
    packet = build_packet(read(args.prediction), read(args.registry), read(args.reference) if args.reference else None)
    packet['prediction_file_sha256'] = hashlib.sha256(args.prediction.read_bytes()).hexdigest()
    packet['registry_file_sha256'] = hashlib.sha256(args.registry.read_bytes()).hexdigest()
    if args.score:
        packet['textual_metrics'] = textual_scores(packet)
    args.output.mkdir(parents=True)
    (args.output/'packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Raw and final output evaluation', '', 'Formal evaluation eligible: '+str(packet['formal_evaluation_eligible']), '',
             *packet['eligibility_errors'], '', 'No legal-accuracy score has been inferred.', '']
    for section in packet['sections']:
        lines += ['## '+section['stage']+' / '+section['role'], '', section['text'] or '[MISSING OR WITHHELD]', '',
                  'Reviewer/date: ______  Accept/edit/reject: ______',
                  'Unsupported claim / wrong actor / outcome error / missing facts / missing reasoning / readability: ______',
                  'Supporting source and correction: ______', '']
    (args.output/'REVIEW.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'formal_evaluation_eligible': packet['formal_evaluation_eligible'],
                      'eligibility_errors': packet['eligibility_errors'], 'sections': len(packet['sections'])}, indent=2))


if __name__ == '__main__':
    main()
