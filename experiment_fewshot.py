"""Explicit development experiment with unapproved compact examples.

Does not change the live app's accepted-exemplar requirement, checkpoints, or
configuration. Reuses saved LegalBERT predictions to isolate generation changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from core import ROLES, generator_input


def validate_experiment(examples, target_ids):
    if len(examples) != 3 or len({e['case_id'] for e in examples}) != 3:
        raise ValueError('Experiment requires three distinct fixed examples')
    if set(target_ids) & {e['case_id'] for e in examples}:
        raise ValueError('Target case overlaps a prompt example')
    for e in examples:
        if e.get('split') != 'exemplar_candidate' or e.get('attorney_accepted') is not False:
            raise ValueError('Preserve candidate provenance; do not invent attorney acceptance')
        if any(not e.get('blocks', {}).get(role, '').strip() for role in ROLES):
            raise ValueError('Each candidate needs all FIR blocks')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/private/fewshot_experiment_20261022')
    args = parser.parse_args()
    out = Path(args.output)
    if (out / 'report.json').exists():
        raise ValueError('Use a new output directory to preserve completed experiments')
    out.mkdir(parents=True, exist_ok=True)
    candidate_path = Path('data/private/attorney_references_20261022/review/compact_exemplar_candidates.json')
    examples = json.loads(candidate_path.read_text(encoding='utf-8'))['examples']
    inputs = {name: json.loads(Path('runs/grounding_source_only_20261022', name + '_result.json').read_text(encoding='utf-8'))
              for name in ('bicol', 'macasaet')}
    validate_experiment(examples, [r['case_id'] for r in inputs.values()])
    manifest = {'type': 'development_generation_comparison', 'attorney_accepted_examples': False,
                'classifier_predictions_reused': True, 'model_weights_updated': False,
                'independent_accuracy_benchmark': False,
                'candidate_sha256': hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
                'implementation_sha256': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                          for p in ('core.py', 'pipeline.py', 'grounding.py')},
                'example_case_ids': [e['case_id'] for e in examples],
                'target_case_ids': [r['case_id'] for r in inputs.values()]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    from pipeline import FIRPipeline
    engine = FIRPipeline()
    budget = {role: 1024 - len(engine.gt.encode(generator_input(role, '', examples))) - 16 for role in ROLES}
    if min(budget.values()) < 128:
        raise ValueError('Candidate examples leave insufficient target context')
    report = []
    for name, source in inputs.items():
        for mode, fixed_examples in (('zero_shot', []), ('experimental_few_shot', examples)):
            print(f'Running {name}: {mode}', flush=True)
            engine.examples = fixed_examples
            start = time.monotonic()
            result = engine.run_tagged(source['sentences'], source['case_id'])
            result['experiment'] = {**manifest, 'mode': mode, 'elapsed_seconds': time.monotonic() - start,
                                    'source_token_budget': budget if fixed_examples else None}
            if fixed_examples:
                result['warnings'].insert(0, 'EXPERIMENT ONLY: shortened examples have not been accepted by the attorney.')
            (out / (name + '_' + mode + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            item = {'case': name, 'mode': mode, 'elapsed_seconds': result['experiment']['elapsed_seconds'],
                    'blocks': {role: {'generation_accepted': result['verification']['blocks'][role]['generation_accepted'],
                                     'final_mode': result['verification']['blocks'][role]['mode'],
                                     'generation': result['generation_audit'][role]}
                               for role in ROLES}}
            report.append(item)
            (out / 'progress.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(item), flush=True)
    (out / 'report.json').write_text(json.dumps({'manifest': manifest, 'runs': report}, indent=2), encoding='utf-8')
    print('Experiment completed: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
