"""Fresh development runs after disclosed Mayor error analysis; no training."""
import hashlib
import json
import sys
import time
from pathlib import Path

from pipeline import FIRPipeline
from preprocessing import prepare_decision
from delivery import prepare_delivery


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    files = ['core.py', 'grounding.py', 'pipeline.py', 'delivery.py', 'model_config.json',
             'models/legalbert_balanced_v2/adapter_model.safetensors']
    hashes = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    save(out / 'manifest.json', {'purpose': 'development regression after Mayor review',
         'independent_evaluation': False, 'training_performed': False,
         'reference_used_as_input': False, 'sha256': hashes,
         'Mayor_status': 'Previously reserved trial preserved; now exposed development case.'})
    engine = FIRPipeline()
    cases = [('151035', Path('data/private/reserved_evaluation_20261026/source.json'))]
    cases += [(d, Path('data/private/additional_digest_development_20261024_v3/sources') / (d + '.json'))
              for d in ('202989', '225562', '111238')]
    for docket, path in cases:
        source = json.loads(path.read_text(encoding='utf-8'))
        prepared = prepare_decision(source['raw_text'])
        print('Running', docket, flush=True)
        start = time.monotonic()
        result = engine.run(prepared['text'], 'GR' + docket)
        result['preprocessing'] = prepared
        result['delivery'] = prepare_delivery(result)
        save(out / (docket + '.json'), result)
        text = ['# Development output: ' + docket, '', 'Not an independent evaluation. No human legal score.', '']
        for role, section in result['delivery']['sections'].items():
            text += ['## ' + role, '', section.get('display_text') or '[WITHHELD]', '']
        (out / (docket + '.md')).write_text('\n'.join(text), encoding='utf-8')
        print(json.dumps({'case': docket, 'seconds': round(time.monotonic() - start, 1),
              'sections': {k: s['status'] for k, s in result['delivery']['sections'].items()}}), flush=True)
    assert hashes == {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}


if __name__ == '__main__':
    main()
