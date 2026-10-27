"""Evaluate source preparation using only the two established development cases."""
import hashlib
import json
import re
from pathlib import Path
from core import ROLES
from preprocessing import prepare_decision
from pipeline import FIRPipeline
from delivery import prepare_delivery
from rouge_score import rouge_scorer


def main():
    out = Path('data/private/evidence_v10_20261026')
    out.mkdir(exist_ok=True)
    refs = {r['docket']: r for r in json.loads(Path('data/private/additional_digest_development_20261024_v3/references.json').read_text(encoding='utf-8'))}
    engine = FIRPipeline()
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    report, comparisons = ['# Source preparation v10: development results', '',
        'No training or reserved-case evaluation. Source preparation changes sentence boundaries, so sentence IDs and predictions may differ. References are scoring targets only. Human legal validation is pending.', ''], []
    for docket in ('127206', '176697'):
        print('Running', docket, flush=True)
        original = json.loads(Path('data/private/bart_comparison_20261025', docket + '_firm.json').read_text(encoding='utf-8'))['prepared_source']
        before = json.loads(Path('data/private/evidence_v9_20261026', docket + '.json').read_text(encoding='utf-8'))
        prepared = prepare_decision(original)
        assert re.findall(r'\d+', original) == re.findall(r'\d+', prepared['text']), 'Substantive numbers changed'
        result = engine.run(prepared['text'], before['result']['case_id'])
        result['preprocessing'] = prepared
        delivery = prepare_delivery(result)
        artifact = {'result': result, 'delivery': delivery, 'weights_changed': False,
                    'reference_used_as_input': False, 'numeric_sequence_preserved': True,
                    'original_source_sha256': hashlib.sha256(original.encode()).hexdigest()}
        (out / (docket + '.json')).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding='utf-8')
        comparison = {'docket': docket, 'before_sentences': len(before['result']['sentences']),
                      'after_sentences': len(result['sentences']), 'spacing_repairs': len(prepared['text_repairs']),
                      'roles': {}, 'warnings': delivery['warnings']}
        report += ['## ' + docket]
        for role in ROLES:
            old = before['delivery']['sections'][role]['display_text']
            new = delivery['sections'][role]['display_text']
            target = refs[docket]['blocks'][role]
            comparison['roles'][role] = {'before_words': len(old.split()), 'after_words': len(new.split()),
                'before_rougeL': scorer.score(target, old)['rougeL'].fmeasure if old else 0,
                'after_rougeL': scorer.score(target, new)['rougeL'].fmeasure if new else 0,
                'generation_accepted': result['verification']['blocks'][role]['generation_accepted']}
            report += ['', '### ' + role, '', new or '[WITHHELD]', '',
                       'Raw LegalPegasus: ' + (result['generated_blocks'][role] or '[NONE]')]
        comparisons.append(comparison)
        (out / 'comparison.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
        (out / 'RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        print(json.dumps(comparison), flush=True)


if __name__ == '__main__': main()
