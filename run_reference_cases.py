"""Run full source decisions; attorney references are not model inputs."""
import json
from pathlib import Path
from pipeline import FIRPipeline
from preprocessing import prepare_decision


def main():
    folder = Path('data/private/attorney_references_20261022')
    cases = json.loads((folder / 'case_inputs.json').read_text(encoding='utf-8'))
    output = folder / 'pipeline_results'
    output.mkdir(exist_ok=True)
    engine = FIRPipeline()
    reports = []
    for case in cases:
        print('Running ' + case['name'], flush=True)
        source = json.loads(Path(case['source_file']).read_text(encoding='utf-8'))
        prepared = prepare_decision(source['raw_text'])
        result = engine.run(prepared['text'], case['case_id'])
        result['preprocessing'] = prepared
        result['decision_url'] = source['decision_url']
        (output / (case['name'] + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        reports.append({'name': case['name'], 'case_id': case['case_id'], 'reliability': result['reliability'],
                        'modes': {r: a['mode'] for r, a in result['verification']['blocks'].items()}})
        (output / 'report.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
        print(json.dumps(reports[-1]), flush=True)


if __name__ == '__main__':
    main()
