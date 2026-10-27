"""Fresh two-case development comparison; reserved cases are never inputs."""
import hashlib
import json
import time
import sys
from pathlib import Path
from rouge_score import rouge_scorer
from core import ROLES
from pipeline import FIRPipeline
from delivery import prepare_delivery


def main():
    out = Path('data/private/evidence_v9_20261026')
    out.mkdir(exist_ok=True)
    previous = Path('data/private/evidence_v8_20261025')
    sources = Path('data/private/bart_comparison_20261025')
    refs = {r['docket']: r for r in json.loads(Path('data/private/additional_digest_development_20261024_v3/references.json').read_text(encoding='utf-8'))}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    redeliver = '--redeliver' in sys.argv
    engine = None if redeliver else FIRPipeline()
    comparisons = []
    report = ['# Evidence v9 development comparison', '',
              'Two development cases only. Model weights unchanged; no reserved evaluation case used. Reference digests are used for scoring only, never generation. Reference approval remains user-assumed.', '',
              'The current Issue in Palma Gil remains withheld. No inferred legal question is presented as an exact source excerpt. Source fidelity is not proof of legal accuracy.', '']
    for docket in ('127206', '176697'):
        old = json.loads((previous / (docket + '.json')).read_text(encoding='utf-8'))
        text = json.loads((sources / (docket + '_firm.json')).read_text(encoding='utf-8'))['prepared_source']
        print('Running', docket, flush=True)
        started = time.perf_counter()
        saved = json.loads((out / (docket + '.json')).read_text(encoding='utf-8')) if redeliver else None
        result = saved['result'] if saved else engine.run(text, old['result']['case_id'])
        delivery = prepare_delivery(result)
        artifact = {'result': result, 'delivery': delivery, 'elapsed_seconds': saved['elapsed_seconds'] if saved else time.perf_counter() - started,
                    'source_sha256': hashlib.sha256(text.encode()).hexdigest(), 'weights_changed': False,
                    'reference_used_as_input': False, 'presentation_replayed': redeliver}
        (out / (docket + '.json')).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding='utf-8')
        comparison = {'docket': docket, 'labels_identical': old['result']['sentences'] == result['sentences'],
                      'roles': {}, 'warnings': delivery['warnings']}
        assert comparison['labels_identical']
        report += ['## ' + docket]
        for role in ROLES:
            before = old['delivery']['sections'][role]['display_text']
            after = delivery['sections'][role]['display_text']
            ref = refs[docket]['blocks'][role]
            comparison['roles'][role] = {'before_words': len(before.split()), 'after_words': len(after.split()),
                                        'before_rougeL': scorer.score(ref, before)['rougeL'].fmeasure if before else 0,
                                        'after_rougeL': scorer.score(ref, after)['rougeL'].fmeasure if after else 0,
                                        'generation_accepted': result['verification']['blocks'][role]['generation_accepted']}
            report += ['', '### ' + role, '', after or '[WITHHELD]', '',
                       'Raw generation: ' + (result['generated_blocks'][role] or '[NONE]')]
        comparisons.append(comparison)
        (out / 'comparison.json').write_text(json.dumps(comparisons, indent=2), encoding='utf-8')
        (out / 'RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
        print(json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
