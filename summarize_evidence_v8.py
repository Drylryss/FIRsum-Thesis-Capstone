"""Report development completeness and lexical similarity without claiming legal accuracy."""
import json
from pathlib import Path
from rouge_score import rouge_scorer
import sacrebleu
from core import ROLES


def main():
    old=Path('data/private/bart_comparison_20261025')
    out=Path('data/private/evidence_v8_20261025')
    refs={r['docket']:r for r in json.loads(Path('data/private/additional_digest_development_20261024_v3/references.json').read_text(encoding='utf-8'))}
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    metrics=[]
    lines=['# Evidence-selection revision v8 — development results','',
        'No model training, label override, reference-digest prompting, or reserved-case evaluation was performed. Both models ran afresh on the same prepared texts used in the v7 comparison. This is development testing, not independent final evaluation.','',
        'Changes: bounded exits from statutory and contract quotations, alternative terminal disposition wording, current Court ruling anchors, and preservation of an immediately following liability qualification. Generic quoted prior judgments remain excluded. All 121 regression tests passed.','',
        '| Case | Version | Facts | Issue | Ruling |','|---|---|---|---|---|']
    artifacts=[]
    for docket in ('127206','176697'):
        before=json.loads((old/(docket+'_firm.json')).read_text(encoding='utf-8'))
        after=json.loads((out/(docket+'.json')).read_text(encoding='utf-8'))
        assert before['result']['sentences']==after['result']['sentences'], 'Classifier output changed'
        artifacts.append((docket,after))
        for version,item in [('v7',before),('v8',after)]:
            sections=item['delivery']['sections']
            lines.append('| '+docket+' | '+version+' | '+' | '.join('Present' if sections[r]['display_text'] else 'Withheld' for r in ROLES)+' |')
            for role in ROLES:
                actual=sections[role]['display_text']
                target=refs[docket]['blocks'][role]
                metrics.append({'docket':docket,'version':version,'role':role,
                    'rougeL':scorer.score(target,actual)['rougeL'].fmeasure if actual else 0,
                    'bleu':sacrebleu.sentence_bleu(actual,[target]).score/100 if actual else 0,
                    'words':len(actual.split())})
    (out/'lexical_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    lines += ['', '## Interpretation','',
        'Presence improves from 2/6 to 5/6 blocks. Identical classifier outputs isolate this change to evidence selection and generation inputs, not improved LegalBERT weights. Presence is not correctness or completeness.','',
        'Palma Gil: the final denial and a non-joinder reason are recovered, but the current Issue remains withheld. Earlier Court of Appeals assignments of error were not substituted as the present Supreme Court question. The selected Ruling does not cover all contract/rescission reasoning in the reference digest.','',
        'Areza: the central debit question, final grant, monetary relief, deletion of moral damages/attorney fees, and the qualification about depositor negligence are selected. Source artifacts, compression, and attribution still need review.','',
        'ROUGE-L and sentence BLEU are saved per block on a 0–1 scale; missing blocks remain zero. BERTScore was not rerun. References retain their user-assumed approval status; no attorney review is fabricated.','',
        '| Mean across six blocks | ROUGE-L | BLEU |','|---|---:|---:|']
    for v in ('v7','v8'):
        subset=[r for r in metrics if r['version']==v]
        lines.append(f"| {v} | {sum(r['rougeL'] for r in subset)/6:.4f} | {sum(r['bleu'] for r in subset)/6:.4f} |")
    for docket,item in artifacts:
        lines += ['', '## '+docket]
        for role in ROLES:
            lines += ['', '### '+role+' — final source-backed text','',item['delivery']['sections'][role]['display_text'] or '[WITHHELD]',
                      '', 'Raw LegalPegasus: '+(item['result']['generated_blocks'][role] or '[NO OUTPUT]')]
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Report saved',out/'RESULTS.md')
    print(json.dumps({v:{m:sum(r[m] for r in metrics if r['version']==v)/6 for m in ('rougeL','bleu')} for v in ('v7','v8')},indent=2))


if __name__=='__main__': main()
