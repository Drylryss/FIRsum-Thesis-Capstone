"""Compare saved validation reports without loading or evaluating test examples."""
import argparse
import json
from pathlib import Path


def validation_signature(predictions):
    """Require identical labeled examples, not merely identical case names."""
    keys = [(r['case_id'], r['sentence_id'], r['text'], r['reference_label']) for r in predictions]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate validation examples in prediction report')
    return set(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='models/legalbert_issue35')
    parser.add_argument('--candidate', default='models/legalbert_balanced_v2')
    parser.add_argument('--output', default='runs/balanced_v2_comparison.json')
    args = parser.parse_args()
    results = []
    case_sets = []
    example_sets = []
    for folder in [args.baseline, args.candidate]:
        path = Path(folder)
        report = json.loads((path / 'validation_report.json').read_text())
        metadata = json.loads((path / 'training_metadata.json').read_text())
        if report['split'] != 'validation':
            raise ValueError('Only validation reports can be compared here')
        case_sets.append(set(metadata['validation_cases']))
        example_sets.append(validation_signature(json.loads((path / 'validation_predictions.json').read_text(encoding='utf-8'))))
        scores = report['classification_report']
        results.append({'model': folder, 'accuracy': scores['accuracy'],
                        'macro_f1': scores['macro avg']['f1-score'],
                        'per_label': {label: scores[label] for label in report['label_order']},
                        'confusion_matrix': report['confusion_matrix']})
    if case_sets[0] != case_sets[1]:
        raise ValueError('Validation case sets differ')
    if example_sets[0] != example_sets[1]:
        raise ValueError('Validation examples or reference labels differ')
    candidate = Path(args.candidate)
    predictions = json.loads((candidate / 'validation_predictions.json').read_text())
    issue_errors = [r for r in predictions if r['reference_label'] != r['predicted_label']
                    and 'ISSUE' in (r['reference_label'], r['predicted_label'])]
    output = {'split': 'validation', 'models': results, 'issue_errors': issue_errors,
              'note': 'Provisional labels; only three validation ISSUE examples. No test evaluation.'}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
