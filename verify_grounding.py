"""Run source-preservation checks on known development examples, not an accuracy benchmark.

Use --run-models to execute both real models. No training or reference exemplars
are created. The two previously inspected cases are development regressions and
must not be reported as an independent held-out evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path

from core import ROLES
from grounding import finalize, normalized, select_evidence
from preprocessing import prepare_decision

ROOT = Path(__file__).resolve().parent
CASES = [
    ('bicol', 'runs/bicol_cleaned_pipeline_result.json'),
    ('macasaet', 'runs/case_input_result.json'),
]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-models', action='store_true')
    parser.add_argument('--output', default='runs/grounding_20261022')
    args = parser.parse_args()
    folder = ROOT / args.output
    folder.mkdir(parents=True, exist_ok=True)
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in ('core.py', 'pipeline.py', 'grounding.py', 'preprocessing.py')}
    engine = None
    if args.run_models:
        from pipeline import FIRPipeline
        print('Loading configured LegalBERT and frozen LegalPegasus...', flush=True)
        engine = FIRPipeline()
    reports = []
    for name, filename in CASES:
        baseline = json.loads((ROOT / filename).read_text(encoding='utf-8'))
        source = baseline.get('preprocessing', {}).get('text') or ' '.join(r['text'] for r in baseline['sentences'])
        prepared = prepare_decision(source)
        print(f'Running {name}...', flush=True)
        if engine:
            result = engine.run(prepared['text'], baseline['case_id'])
            result['preprocessing'] = prepared
        else:
            tagged = baseline['sentences']
            result = {'case_id': baseline['case_id'], 'sentences': tagged,
                      'generated_blocks': baseline['blocks'],
                      **finalize(tagged, baseline['blocks'], select_evidence(tagged))}
        write(folder / f'{name}_result.json', result)
        by_id = {row['id']: row['text'] for row in result['sentences']}
        checks = {}
        for role in ROLES:
            audit = result['verification']['blocks'][role]
            evidence = audit['evidence']
            checks[role] = {
                'mode': audit['mode'],
                'all_evidence_matches_source': all(e['text'] == by_id[e['sentence_id']] for e in evidence),
                'final_matches_selected_source': normalized(result['blocks'][role]) == normalized(' '.join(e['text'] for e in evidence)),
                'baseline_matches_selected_source': normalized(baseline['blocks'][role]) == normalized(' '.join(e['text'] for e in evidence)),
                'selected_sentence_ids': audit['source_sentence_ids'],
                'generation_audit': result.get('generation_audit', {}).get(role, {}),
            }
        disposition = result['verification']['disposition_sentence_ids']
        disposition_complete = bool(disposition) and all(normalized(by_id[i]) in normalized(result['blocks']['RULING']) for i in disposition)
        reports.append({'case': name, 'case_id': baseline['case_id'],
                        'input_sha256': hashlib.sha256(prepared['text'].encode('utf-8')).hexdigest(),
                        'real_models_run': bool(engine), 'sentence_count': len(result['sentences']),
                        'complete_disposition_preserved': disposition_complete,
                        'label_conflicts': result['verification']['label_conflicts'], 'checks': checks})
        print(json.dumps(reports[-1], ensure_ascii=True, indent=2), flush=True)
        assert disposition_complete, f'{name}: missing final disposition'
        assert all(v['all_evidence_matches_source'] and v['final_matches_selected_source'] for v in checks.values())
        assert result['blocks']['ISSUE'], f'{name}: explicit issue was not recovered'
    write(folder / 'report.json', {'evaluation_type': 'development_source_preservation_regression',
                                 'independent_legal_accuracy_estimate': False,
                                 'implementation_sha256': implementation,
                                 'cases': reports})
    print(f'Saved results to {folder}', flush=True)


if __name__ == '__main__':
    main()
