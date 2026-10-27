"""One frozen-candidate reserved-case run; no tuning and no human-score claims."""
import hashlib
import json
import re
from pathlib import Path
from lxml import html
from preprocessing import prepare_decision
from pipeline import FIRPipeline
from delivery import prepare_delivery
from evaluation_protocol import build_packet

OUT = Path('data/private/reserved_evaluation_20261026')
URL = 'https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/1/46885'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if (OUT / 'prediction.json').exists():
        raise RuntimeError('Reserved prediction already exists; refusing a repeated evaluation run.')
    candidate = Path('data/private/broader_development_fixed_20261026/candidate_manifest.json')
    manifest = json.loads(candidate.read_text(encoding='utf-8'))
    expected = manifest['files_sha256']
    assert expected == {p: digest(p) for p in expected}, 'Frozen code has changed'
    adapter = 'models/legalbert_balanced_v2/adapter_model.safetensors'
    adapter_hash = digest(adapter)
    assert adapter_hash == 'cfa4333af59df77e19468df3702c50045034ad6a0c43c1dc94756f1c44b0e59a'
    tree = html.fromstring((OUT / 'mayor_source.html').read_bytes())
    content = tree.xpath('//div[contains(concat(" ", normalize-space(@class), " "), " single_content ")]')
    assert len(content) == 1
    raw = re.sub(r'\s+', ' ', ' '.join(content[0].itertext())).strip()
    assert '151035' in raw and 'SO ORDERED' in raw and 'YNARES' in raw
    prepared = prepare_decision(raw)
    assert prepared['text'].endswith('concur.') or 'SO ORDERED' in prepared['text']
    source = {'decision_url': URL, 'raw_text': raw, 'html_sha256': digest(OUT / 'mayor_source.html'),
              'extraction_method': 'Single official decision content container; whitespace join of text nodes; no content-specific correction or citation deletion.',
              'source_complete_verified': False, 'prepared': prepared}
    (OUT / 'source.json').write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    registry = json.loads(Path('data/private/additional_digest_development_20261024_v3/case_exposure_registry.json').read_text(encoding='utf-8'))
    registry += [{'case_id': 'GR151035', 'reasons': ['Prior sentence-classifier diagnostic on 2026-10-25; reserved from tuning. Not a pristine unseen case.'],
                  'source_fingerprints': [], 'training_allowed': False}]
    (OUT / 'exposure_registry.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
    run = {'case_id': 'GR151035', 'purpose': 'reserved-case trial with disclosed prior classifier exposure',
           'independent_final_evaluation': False, 'candidate_manifest_sha256': digest(candidate),
           'code_sha256': expected, 'adapter_sha256': adapter_hash, 'reference_used_as_input': False,
           'training_performed': False, 'tuning_after_result_permitted': False,
           'source_url': URL, 'source_html_sha256': source['html_sha256']}
    (OUT / 'run_manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print('Running frozen candidate once on reserved Mayor case', flush=True)
    engine = FIRPipeline()
    result = engine.run(prepared['text'], 'GR151035')
    result['preprocessing'] = prepared
    result['delivery'] = prepare_delivery(result)
    result['decision_url'] = URL
    (OUT / 'prediction.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    # References are read only after prediction has been saved.
    refs = json.loads(Path('data/private/validation_intake_20261024/references_deduplicated.json').read_text(encoding='utf-8'))
    reference = next(r for r in refs if r['case_id'] == 'GR151035')
    reference = {**reference, 'split': 'validation', 'reviewer': None, 'review_date': None,
                 'source_complete_verified': False, 'previously_used_for_development': None,
                 'source_sha256': result['verification']['source_sha256']}
    packet = build_packet(result, registry, reference)
    assert not packet['formal_evaluation_eligible'] and packet['legal_accuracy_estimate'] is None
    (OUT / 'review_packet.json').write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Reserved Mayor case: review packet', '',
             'Frozen-candidate trial, not a pristine unseen-case or attorney-validated final evaluation. Prior sentence-classifier diagnostic exposure is disclosed. No training or changes in response to this result are permitted within this evaluation.', '',
             'Official source: ' + URL, '', '## Eligibility limitations', '', *['- ' + e for e in packet['eligibility_errors']]]
    for section in packet['sections']:
        lines += ['', '## ' + section['stage'] + ' / ' + section['role'], '', section['text'] or '[WITHHELD]', '',
                  'Reference: ' + (section['reference'] or '[UNAVAILABLE]'), '',
                  'Reviewer: ______  Date: ______  Accept / edit / reject: ______',
                  'Unsupported claim / wrong actor / outcome error / missing facts / missing reasoning / readability: ______',
                  'Source evidence and correction: ______']
    (OUT / 'REVIEW.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    assert expected == {p: digest(p) for p in expected}
    assert digest(adapter) == adapter_hash
    print(json.dumps({'delivery_status': result['delivery']['status'],
                      'section_modes': {r: s['status'] for r, s in result['delivery']['sections'].items()},
                      'formal_evaluation_eligible': False, 'human_ratings_complete': False,
                      'weights_and_code_unchanged': True}), flush=True)


if __name__ == '__main__': main()
