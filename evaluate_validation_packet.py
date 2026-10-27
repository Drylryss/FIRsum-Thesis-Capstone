"""Measure classifier agreement with supplied provisional annotation units.

Does not train, relabel, split units, or claim final human validation.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

from core import LABELS, classifier_path, token_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    source = Path('data/private/validation_intake_20261024/workbook_rows.json')
    rows = json.loads(source.read_text(encoding='utf-8'))
    included = [r for r in rows if r['source_sheet'] != 'Sheet2']
    if any(r.get('proposed_label') not in LABELS or not r.get('text', '').strip() for r in included):
        raise ValueError('Invalid provisional label or empty unit')
    if len({r['sentence_id'] for r in included}) != len(included):
        raise ValueError('Duplicate annotation identifier')
    args.output.mkdir(parents=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftConfig, PeftModel
    from sklearn.metrics import classification_report, confusion_matrix
    torch.set_num_threads(4)
    adapter = classifier_path()
    config = PeftConfig.from_pretrained(adapter)
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    base = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path,
        num_labels=4, id2label=dict(enumerate(LABELS)), label2id={v:i for i,v in enumerate(LABELS)})
    model = PeftModel.from_pretrained(base, adapter).eval()
    predictions = []
    started = time.monotonic()
    for i, row in enumerate(included):
        ids = tokenizer.encode(row['text'], add_special_tokens=False)
        scores = []
        for window in token_windows(ids, 510):
            batch = tokenizer.prepare_for_model(window, return_tensors='pt')
            batch = {k:v.unsqueeze(0) if v.ndim == 1 else v for k,v in batch.items()}
            with torch.inference_mode():
                scores.append(model(**batch).logits.softmax(-1)[0])
        probs = torch.stack(scores).mean(0)
        label = LABELS[int(probs.argmax())]
        predictions.append({**row, 'predicted_label': label,
                            'confidence': float(probs.max()), 'token_windows': len(scores),
                            'agrees_with_provisional_label': label == row['proposed_label']})
        if (i+1) % 100 == 0:
            print(f'Classified {i+1}/{len(included)} units', flush=True)
    def metrics(group):
        y = [r['proposed_label'] for r in group]
        pred = [r['predicted_label'] for r in group]
        return {'units':len(group), 'classification_report':classification_report(y,pred,labels=LABELS,output_dict=True,zero_division=0),
                'confusion_matrix_rows_provisional_columns_predicted':confusion_matrix(y,pred,labels=LABELS).tolist()}
    report = {'interpretation':'Agreement with provisional labels; not verified legal accuracy or independent final pipeline evaluation.',
              'model_path':adapter, 'input_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'model_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(adapter).glob('*') if p.is_file() and p.suffix in ('.json','.safetensors')},
              'elapsed_seconds':round(time.monotonic()-started,2), 'label_order':LABELS,
              'units_preserved_without_resplitting':True,
              'excluded_unmapped_fragments':4, 'final_labels_filled':0,
              'training_performed':False, 'overall':metrics(predictions),
              'sentence_type_only':metrics([r for r in predictions if r.get('unit_type')=='sentence']),
              'by_case':{s:metrics([r for r in predictions if r['source_sheet']==s]) for s in sorted({r['source_sheet'] for r in predictions})}}
    (args.output/'predictions.json').write_text(json.dumps(predictions,ensure_ascii=False,indent=2),encoding='utf-8')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'units':len(predictions),'seconds':report['elapsed_seconds'],'overall':report['overall']},indent=2),flush=True)


if __name__ == '__main__':
    main()
