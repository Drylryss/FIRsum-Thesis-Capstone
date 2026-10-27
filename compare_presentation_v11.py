"""Replay only final presentation; preserve the exact v10 model outputs."""
import json
from pathlib import Path
from delivery import prepare_delivery
from core import ROLES
from rouge_score import rouge_scorer


def main():
    source = Path('data/private/evidence_v10_20261026')
    out = Path('data/private/presentation_v11_20261026')
    out.mkdir(exist_ok=True)
    refs = {r['docket']: r for r in json.loads(Path('data/private/additional_digest_development_20261024_v3/references.json').read_text(encoding='utf-8'))}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    comparisons, report = [], ['# Presentation v11 development comparison', '',
        'Presentation-only replay of saved v10 outputs. No model rerun, training, evidence-selection change or reserved-case evaluation. Reference similarity is not legal accuracy.', '']
    for docket in ('127206', '176697'):
        old = json.loads((source / (docket + '.json')).read_text(encoding='utf-8'))
        new = prepare_delivery(old['result'])
        comparison = {'docket': docket, 'roles': {}}
        report += ['## ' + docket]
        for role in ROLES:
            previous = old['delivery']['sections'][role]
            current = new['sections'][role]
            before, after = previous['display_text'], current['display_text']
            target = refs[docket]['blocks'][role]
            comparison['roles'][role] = {'before_words': len(before.split()), 'after_words': len(after.split()),
                'before_rougeL': scorer.score(target, before)['rougeL'].fmeasure if before else 0,
                'after_rougeL': scorer.score(target, after)['rougeL'].fmeasure if after else 0,
                'omitted_ids': [e['sentence_id'] for e in current['omitted_evidence']]}
            report += ['', '### ' + role, '', after or '[WITHHELD]']
        assert set(old['result']['verification']['disposition_sentence_ids']) <= {e['sentence_id'] for e in new['sections']['RULING']['evidence']}
        (out / (docket + '.json')).write_text(json.dumps({'result': old['result'], 'delivery': new,
            'model_rerun': False, 'weights_changed': False}, ensure_ascii=False, indent=2), encoding='utf-8')
        comparisons.append(comparison)
    (out / 'comparison.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
    (out / 'RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__': main()
