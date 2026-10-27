"""Snapshot tested code and create unfilled development review packets."""
import hashlib
import json
from pathlib import Path
from evaluation_protocol import build_packet


def main():
    root = Path('data/private/evidence_v10_20261026')
    snapshot = root / 'code_snapshot'
    snapshot.mkdir(exist_ok=True)
    paths = ['core.py', 'preprocessing.py', 'pipeline.py', 'grounding.py', 'delivery.py',
             'footnote_display.py', 'source_structure.py', 'extraction_quality.py', 'inference.py',
             'evaluation_protocol.py', 'model_config.json', 'web/index.html']
    files = {}
    for name in paths:
        data = Path(name).read_bytes()
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        files[name] = hashlib.sha256(data).hexdigest()
    registry = []
    for docket in ('127206', '176697'):
        artifact = json.loads((root / (docket + '.json')).read_text(encoding='utf-8'))
        result = artifact['result']
        result['delivery'] = artifact['delivery']
        registry.append({'case_id': result['case_id'], 'source_fingerprints': [result['verification']['source_sha256']],
                         'reason': 'Repeated pipeline development and diagnosis; not a final test case.'})
        packet = build_packet(result, registry)
        (root / (docket + '_review_packet.json')).write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
    (root / 'development_exclusions.json').write_text(json.dumps(registry, indent=2), encoding='utf-8')
    manifest = {'version': 'source-grounding-v10', 'date': '2026-10-26',
                'status': 'development_candidate_not_independently_validated', 'tests_passed': 133,
                'code_sha256': files,
                'legalbert_adapter_sha256': hashlib.sha256(Path('models/legalbert_balanced_v2/adapter_model.safetensors').read_bytes()).hexdigest(),
                'training_dataset_sha256': hashlib.sha256(Path('data/revision_20261020_metadata/sentences.jsonl').read_bytes()).hexdigest(),
                'training_performed': False, 'independent_evaluation_performed': False,
                'exclusion_registry_scope': 'These two development cases only; merge with existing exposure registry and training/validation/exemplar case lists before final evaluation.'}
    (root / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (root / 'NEXT_EVALUATION.md').write_text('''# Independent evaluation handoff

This is a frozen development candidate, not a validated release. The two attached review packets deliberately have no human ratings or legal-accuracy score. They document development outputs only.

Before final evaluation, supply full main opinions and case-matched FIR references with actual review provenance. Screen case identities and source hashes against all training, validation, exemplar and prior development exposure records, including the earlier validation-packet diagnostics. The local development_exclusions.json file is only a supplemental two-case list, not the complete study registry.

Run the frozen candidate once on the eligible cases. Assess raw LegalPegasus and final delivery separately. Record unsupported claims, wrong actors, outcome/negation errors, missing essential facts, missing reasoning and readability errors. Keep missing sections in the denominator. Report ROUGE-L, BLEU and BERTScore as text similarity, separately from human judgments; check long-reference truncation before BERTScore.

No completed ratings, reviewer acceptance or final evaluation findings may be inferred from these templates. If evaluation results are used to change this candidate, those cases become development cases and a new untouched set is needed for a final claim.
''', encoding='utf-8')
    print('Saved candidate snapshot and two unfilled development review packets.')


if __name__ == '__main__': main()
