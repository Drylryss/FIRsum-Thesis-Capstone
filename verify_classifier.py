"""Verify the saved adapter through the same sentence-classification code as the app."""
import json
import argparse
from pathlib import Path
from core import LABELS, read_jsonl, classifier_path
from pipeline import FIRPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-validation', action='store_true', help='Check saved validation probabilities after reloading all adapter modules')
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel, PeftConfig
    torch.set_num_threads(4)
    folder = classifier_path()
    config = PeftConfig.from_pretrained(folder)
    engine = FIRPipeline.__new__(FIRPipeline)
    engine.torch, engine.device = torch, 'cpu'
    engine.bt = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    base = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
        num_labels=4, id2label=dict(enumerate(LABELS)), label2id={r:i for i,r in enumerate(LABELS)}, local_files_only=True)
    engine.bert = PeftModel.from_pretrained(base, folder).eval()
    if args.verify_validation:
        saved = json.loads(Path(folder, 'validation_predictions.json').read_text(encoding='utf-8'))
        max_difference = 0.0
        for start in range(0, len(saved), 8):
            batch_rows = saved[start:start + 8]
            inputs = engine.bt([r['text'] for r in batch_rows], padding=True, return_tensors='pt')
            with torch.inference_mode():
                probabilities = engine.bert(**inputs).logits.softmax(-1)
            expected = torch.tensor([[r['probabilities'][label] for label in LABELS] for r in batch_rows])
            max_difference = max(max_difference, float((probabilities - expected).abs().max()))
            if not torch.allclose(probabilities, expected, atol=1e-4, rtol=1e-4):
                raise AssertionError('Reloaded model differs from saved validation probabilities')
            if [LABELS[int(i)] for i in probabilities.argmax(-1)] != [r['predicted_label'] for r in batch_rows]:
                raise AssertionError('Reloaded validation labels differ')
        Path(folder, 'reload_validation_check.json').write_text(json.dumps({
            'examples': len(saved), 'maximum_probability_difference': max_difference,
            'passed': True}, indent=2), encoding='utf-8')
        print(f'All {len(saved)} validation predictions reproduced after reload.')
    rows = [r for r in read_jsonl('data/sentences.jsonl') if r['split'] == 'validation'][:3]
    output = []
    for row in rows:
        predictions = engine.classify(row['text'])
        assert predictions and all(p['role'] in LABELS and 0 <= p['confidence'] <= 1 for p in predictions)
        output.append({'sentence_id':row['sentence_id'], 'suggested_label':row['label'], 'predictions':predictions})
    Path(folder, 'reload_smoke_test.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print('Saved adapter reload and application classification passed for three held-out source rows.')


if __name__ == '__main__':
    main()
