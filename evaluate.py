"""Evaluate one exported case against an attorney-accepted held-out reference."""
import argparse
import json
from pathlib import Path
from core import ROLES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--output', default='runs/metrics.json')
    a = p.parse_args()
    pred = json.loads(Path(a.prediction).read_text(encoding='utf-8'))
    ref = json.loads(Path(a.reference).read_text(encoding='utf-8'))
    if pred['case_id'] != ref['case_id'] or ref.get('split') != 'test' or ref.get('attorney_accepted') is not True:
        raise ValueError('Reference must be an accepted test digest for the same case')
    if ref['case_id'] in pred.get('exemplar_case_ids', []):
        raise ValueError('Exemplar leakage')
    if pred.get('review_status') not in {'accept', 'edit'}:
        raise ValueError('Only attorney-accepted or edited outputs proceed to evaluation')
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    report = {'case_id': pred['case_id'], 'scales': 'All metrics 0–1; descriptive, not legal accuracy', 'blocks': {}}
    for role in ROLES:
        target, actual = ref['blocks'][role], pred['blocks'][role]
        if not target.strip() or not actual.strip():
            raise ValueError(f'Empty {role} block: resolve before evaluation')
        _, _, f1 = score([actual], [target], model_type='roberta-large', lang='en', verbose=False)
        report['blocks'][role] = {'rougeL': scorer.score(target, actual)['rougeL'].fmeasure,
            'bleu': sacrebleu.sentence_bleu(actual, [target]).score / 100,
            'bertscore_f1': float(f1[0])}
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
