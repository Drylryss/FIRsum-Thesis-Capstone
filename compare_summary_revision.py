"""Compare delivery on saved development cases; never rerun or score the models."""
import argparse
import hashlib
import json
from pathlib import Path

from delivery import prepare_delivery
from grounding import finalize, select_evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Use a new output directory to preserve earlier results.')
    paths = {
        'bicol': Path('data/private/sample_full_case_20261023/result_v4.json'),
        'macasaet': Path('runs/grounding_source_only_20261022/macasaet_result.json'),
    }
    folder = Path('data/private/attorney_references_20261022/pipeline_results')
    paths.update({n: folder / (n + '.json') for n in ('makati', 'real', 'philcomsat', 'mactan')})
    baseline = json.loads((args.baseline / 'results.json').read_text(encoding='utf-8')) if args.baseline else None
    results = {}
    text = ['# Source-backed summary comparison', '',
            'Previously seen development cases. No model rerun or independent legal-accuracy estimate.', '']
    for name, path in paths.items():
        old = json.loads(path.read_text(encoding='utf-8'))
        rows = old['sentences']
        plan = select_evidence(rows)
        replay = {**finalize(rows, {}, plan), 'sentences': rows,
                  'case_id': old['case_id'], 'preprocessing': old.get('preprocessing', {})}
        draft = prepare_delivery(replay)
        delivered = draft['sections']['RULING']['evidence']
        assert plan['disposition_ids'] and set(plan['disposition_ids']) <= {e['sentence_id'] for e in delivered}
        source = {r['id']: r['text'] for r in rows}
        assert all(e['text'] == source[e['sentence_id']]
                   for s in draft['sections'].values() for e in s['evidence'])
        results[name] = {'delivery': draft, 'source_result': str(path),
                         'source_result_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'disposition_ids': plan['disposition_ids']}
        text += ['## ' + name, '']
        for role, section in draft['sections'].items():
            text += ['### ' + role, '']
            if baseline:
                assert baseline[name]['source_result_sha256'] == results[name]['source_result_sha256']
                assert baseline[name]['disposition_ids'] == plan['disposition_ids']
                before = baseline[name]['delivery']['sections'][role]
                text += ['Before:', '', before['display_text'] or '[WITHHELD]', '', 'After:', '']
            text += [section['display_text'] or '[WITHHELD]', '',
                     'Source IDs: ' + ', '.join(str(e['sentence_id']) for e in section['evidence']), '']
        text += ['Warnings: ' + '; '.join(draft['warnings']), '']
    args.output.mkdir(parents=True)
    (args.output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output / 'COMPARISON.md').write_text('\n'.join(text), encoding='utf-8')
    manifest = {'models_rerun': False, 'independent_test': False, 'legal_accuracy_estimate': None,
                'implementation_hashes': {p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                          for p in ('grounding.py', 'delivery.py', 'compare_summary_revision.py')}}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    for name, result in results.items():
        print(name, {r: len(s['evidence']) for r, s in result['delivery']['sections'].items()})


if __name__ == '__main__':
    main()
