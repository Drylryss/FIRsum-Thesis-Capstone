"""Descriptive comparison and reproducible candidate snapshot, not legal scoring."""
import hashlib
import json
from pathlib import Path
from rouge_score import rouge_scorer
from core import ROLES
from delivery import prepare_delivery
from evaluation_protocol import build_packet


def main():
    old = Path('data/private/broader_development_20261026')
    new = Path('data/private/broader_development_fixed_20261026')
    refs = {r['docket']: r for r in json.loads(Path('data/private/additional_digest_development_20261024_v3/references.json').read_text(encoding='utf-8'))}
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    metrics = []
    lines = ['# Broader development check', '',
        'Three already-exposed cases: Comglasco, Louh and Adelfa. Both models ran afresh before and after structural fixes. No training, reserved validation packet, independent test case or reference-digest prompting was used. Reference acceptance remains user-assumed. This is development error analysis, not an attorney rating or final reliability estimate.', '',
        '## Main finding', '',
        'All nine baseline FIR blocks were nonempty, but every baseline Ruling contained only the disposition and Adelfa Facts contained only caption material. Nonempty sections are therefore not evidence of adequate coverage. Raw LegalPegasus invented courts, actors or events; the literal-source gate rejected those generations.', '',
        '## Changes', '',
        'Recognize signed Resolution headings and older city/division captions. Recognize title-case current-court ruling headings and explicit application findings. Separate an explicitly identified current-petition issue list from an earlier appeal. Allow frontloaded Issues followed by an explicit Facts section. Stop reasoning context at named-party arguments or named precedent introductions. Recognize statutory reads blocks and narrowly bounded transitions back to the deciding Court. Flag ambiguous monetary characters rather than silently changing them.', '',
        '151 regression tests and UI safe copy/export checks passed. Saved-output regression replays preserve Palma Gil’s synthesized Issue and Areza’s full disposition and negligence qualification. Those two checks are replays, not additional fresh model runs. After fresh generation, one presentation-only check was added to omit a same-day sentence whose immediate source antecedent is missing. The final delivery was replayed without another model run; fresh_delivery_before_context_guard preserves the original delivery.', '',
        '| Case | Role | Before words | After words | Before ROUGE-L | After ROUGE-L |',
        '|---|---|---:|---:|---:|---:|']
    for docket in ('202989', '225562', '111238'):
        before = json.loads((old / (docket + '.json')).read_text(encoding='utf-8'))
        after = json.loads((new / (docket + '.json')).read_text(encoding='utf-8'))
        after.setdefault('fresh_delivery_before_context_guard', after['delivery'])
        after['delivery'] = prepare_delivery(after['result'])
        after['delivery_replayed_after_generation'] = True
        packet = build_packet({**after['result'], 'delivery': after['delivery']}, [])
        (new / (docket + '_final_review_packet.json')).write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
        (new / (docket + '.json')).write_text(json.dumps(after, ensure_ascii=False, indent=2), encoding='utf-8')
        final_text = ['# ' + refs[docket]['title'], '', 'Final presentation after context guard; source evidence and omitted excerpts remain in JSON.', '']
        for role in ROLES:
            final_text += ['## ' + role, '', after['delivery']['sections'][role]['display_text'] or '[WITHHELD]']
        (new / (docket + '_final.md')).write_text('\n'.join(final_text) + '\n', encoding='utf-8')
        for role in ROLES:
            a = before['delivery']['sections'][role]['display_text']
            b = after['delivery']['sections'][role]['display_text']
            target = refs[docket]['blocks'][role]
            record = {'docket': docket, 'role': role, 'before_words': len(a.split()), 'after_words': len(b.split()),
                      'before_rougeL': scorer.score(target, a)['rougeL'].fmeasure if a else 0,
                      'after_rougeL': scorer.score(target, b)['rougeL'].fmeasure if b else 0}
            metrics.append(record)
            lines.append(f"| {refs[docket]['title']} | {role} | {record['before_words']} | {record['after_words']} | {record['before_rougeL']:.4f} | {record['after_rougeL']:.4f} |")
        disposition = set(after['result']['verification']['disposition_sentence_ids'])
        delivered = {e['sentence_id'] for e in after['delivery']['sections']['RULING']['evidence']}
        assert disposition and disposition <= delivered
        assert not any(after['result']['verification']['blocks'][r]['generation_accepted'] for r in ROLES)
    (new / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    lines += ['', 'ROUGE-L is descriptive overlap with the reference; longer or more complete excerpts can change it without establishing legal accuracy. BERTScore and independent human scores were not computed.', '',
              '## Release status', '',
              'Development candidate only. Source ambiguity, missing supporting propositions and selection/attribution errors still require review. Full source files, raw generations, final outputs, hashes and unfilled review packets are retained. Do not use these cases as untouched final evaluation cases.']
    (new / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    snapshot = new / 'code_snapshot'
    files = ['core.py', 'pipeline.py', 'preprocessing.py', 'grounding.py', 'delivery.py', 'issue_synthesis.py',
             'source_structure.py', 'footnote_display.py', 'extraction_quality.py', 'inference.py',
             'evaluation_protocol.py', 'model_config.json', 'web/index.html']
    hashes = {}
    for name in files:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = Path(name).read_bytes()
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    manifest = {'date': '2026-10-26', 'status': 'development_candidate_not_independently_validated',
                'tests_passed': 151, 'ui_delivery_tests_passed': True, 'files_sha256': hashes,
                'training_performed': False, 'independent_evaluation_performed': False,
                'development_dockets': ['202989', '225562', '111238', '127206', '176697']}
    (new / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({key: sum(r[key] for r in metrics) / len(metrics) for key in ('before_rougeL', 'after_rougeL')}, indent=2))


if __name__ == '__main__': main()
