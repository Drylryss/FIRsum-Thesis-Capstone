"""Replay delivery only; never present a derived Issue as model generation."""
import json
from pathlib import Path
from delivery import prepare_delivery
from evaluation_protocol import build_packet


def main():
    root = Path('data/private/issue_fallback_v12_20261026')
    root.mkdir(exist_ok=True)
    report = ['# Source-supported Issue fallback', '',
              'Delivery-only replay of saved v10 model output. No retraining or new model inference. The fallback is a narrow template transformation, not verbatim extraction or a validated legal paraphrase.', '']
    for docket in ('127206', '176697'):
        saved = json.loads(Path('data/private/evidence_v10_20261026', docket + '.json').read_text(encoding='utf-8'))
        delivered = prepare_delivery(saved['result'])
        issue = delivered['sections']['ISSUE']
        if docket == '127206':
            assert issue['status'] == 'source_supported_synthesis'
            assert [e['sentence_id'] for e in issue['evidence']] == [241, 242]
            assert saved['result']['blocks']['ISSUE'] == ''
        else:
            assert issue['status'] == 'source_backed_draft'
            assert issue['display_text'] == saved['delivery']['sections']['ISSUE']['display_text']
        (root / (docket + '.json')).write_text(json.dumps({'result': saved['result'], 'delivery': delivered,
            'model_rerun': False, 'training_performed': False}, ensure_ascii=False, indent=2), encoding='utf-8')
        packet = build_packet({**saved['result'], 'delivery': delivered}, [])
        final_issue = next(s for s in packet['sections'] if s['stage'] == 'final_delivery' and s['role'] == 'ISSUE')
        assert final_issue['delivery_mode'] == issue['status']
        (root / (docket + '_review_packet.json')).write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding='utf-8')
        report += ['## ' + docket, '', 'Mode: ' + issue['status'], '', issue['display_text'], '',
                   'Evidence IDs: ' + ', '.join(str(e['sentence_id']) for e in issue['evidence']), '',
                   issue.get('synthesis', {}).get('scope', 'Existing explicit question retained.')]
        print(docket, issue['status'], ascii(issue['display_text']))
    report += ['', 'The underlying LegalBERT labels, raw LegalPegasus output and literal-source verification are unchanged. Palma Gil now has a substantive question in final delivery, but this does not prove complete coverage of its procedural issues or overall legal accuracy. The earlier blank raw Issue remains visible in the diagnostic audit.', '',
               'The fallback abstains without a clear current Court heading, a unique supported entitlement conclusion and adjacent selected explanation, or when source/delivery verification fails. It does not replace an explicit Issue.']
    (root / 'RESULTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
