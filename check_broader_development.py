"""Three already-exposed development cases; never an independent test score."""
import hashlib
import json
import sys
from pathlib import Path
from core import ROLES
from preprocessing import prepare_decision
from pipeline import FIRPipeline
from delivery import prepare_delivery
from evaluation_protocol import build_packet

DOCKETS = ('202989', '225562', '111238')
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('data/private/broader_development_20261026')
SOURCE = Path('data/private/additional_digest_development_20261024_v3')


def main():
    OUT.mkdir(exist_ok=True)
    refs = {r['docket']: r for r in json.loads((SOURCE / 'references.json').read_text(encoding='utf-8'))}
    files = ['core.py', 'pipeline.py', 'preprocessing.py', 'grounding.py', 'delivery.py', 'issue_synthesis.py',
             'model_config.json', 'models/legalbert_balanced_v2/adapter_model.safetensors']
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    assert not set(DOCKETS) & {'147324', '154630', '151035', '104234'}
    for docket in DOCKETS:
        assert refs[docket]['split'] == 'train' and refs[docket]['reference_use'] == 'development_reference'
    (OUT / 'run_manifest.json').write_text(json.dumps({'dockets': DOCKETS, 'purpose': 'broader development error analysis',
        'independent_evaluation': False, 'training_performed': False, 'reference_used_as_input': False,
        'reserved_cases_used': False, 'sha256': hashes}, indent=2), encoding='utf-8')
    engine = FIRPipeline()
    summaries = []
    for docket in DOCKETS:
        print('Running', docket, refs[docket]['title'], flush=True)
        source_path = SOURCE / 'sources' / (docket + '.json')
        source = json.loads(source_path.read_text(encoding='utf-8'))
        prepared = prepare_decision(source['raw_text'])
        result = engine.run(prepared['text'], refs[docket]['case_id'])
        result['preprocessing'] = prepared
        delivered = prepare_delivery(result)
        packet = build_packet({**result, 'delivery': delivered}, [])
        artifact = {'result': result, 'delivery': delivered, 'source_file': str(source_path),
                    'source_file_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    'source_url': source['decision_url'], 'reference_used_as_input': False}
        (OUT / (docket + '.json')).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding='utf-8')
        (OUT / (docket + '_review_packet.json')).write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
        summary = {'docket': docket, 'title': refs[docket]['title'],
            'sentences': len(result['sentences']), 'sections': {r: {'status': delivered['sections'][r]['status'],
                'words': len(delivered['sections'][r]['display_text'].split()),
                'raw_generation_accepted': result['verification']['blocks'][r]['generation_accepted']}
                for r in ROLES}, 'warnings': delivered['warnings']}
        summaries.append(summary)
        (OUT / 'summary.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
        text = ['# ' + refs[docket]['title'], '', 'Development case. Full source inspected separately; no human rating inferred.', '']
        for role in ROLES:
            text += ['## ' + role, '', delivered['sections'][role]['display_text'] or '[WITHHELD]', '',
                     'Raw LegalPegasus: ' + (result['generated_blocks'][role] or '[NONE]')]
        (OUT / (docket + '_output.md')).write_text('\n'.join(text) + '\n', encoding='utf-8')
        print(json.dumps(summary), flush=True)
    assert hashes == {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}, 'Code or weights changed during comparison'


if __name__ == '__main__': main()
