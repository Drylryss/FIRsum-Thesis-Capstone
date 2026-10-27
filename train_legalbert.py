"""Train a LoRA rhetorical classifier, preserving the classifier head in the adapter."""
import argparse
import json
from pathlib import Path
from core import LABELS, training_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', default='models/legalbert')
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--class-weighting', choices=['none', 'balanced'], default='none')
    p.add_argument('--max-class-weight', type=float, default=20.0)
    p.add_argument('--issue-weight-multiplier', type=float, default=1.0)
    p.add_argument('--issue-oversample-factor', type=int, default=1,
                   help='Number of presentations per training ISSUE example per epoch')
    p.add_argument('--evaluate-test', action='store_true', help='Explicit final evaluation only; omit during tuning')
    p.add_argument('--init-adapter', help='Continue training an existing adapter without modifying it')
    p.add_argument('--learning-rate', type=float, default=2e-4)
    p.add_argument('--warmup-ratio', type=float, default=0.0)
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--lora-rank', type=int, default=8)
    p.add_argument('--attention-targets', nargs='+', choices=['query', 'key', 'value'], default=['query', 'value'])
    p.add_argument('--train-pooler', action='store_true', help='Train and save the sentence pooling layer with the classifier')
    p.add_argument('--loss-normalization', choices=['batch', 'dataset'], default='batch')
    args = p.parse_args()
    if not 0 <= args.warmup_ratio < 1 or args.weight_decay < 0 or args.lora_rank < 1:
        p.error('Require 0 <= warmup ratio < 1, nonnegative weight decay, and positive LoRA rank')
    if args.init_adapter and (args.train_pooler or args.lora_rank != 8 or args.attention_targets != ['query', 'value']):
        p.error('Adapter architecture options apply only to training a new adapter')
    train, val = training_rows(args.data)
    if Path(args.output, 'adapter_config.json').exists():
        raise ValueError('Output already contains a trained model. Choose a new output directory to preserve it.')
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    from datasets import Dataset
    from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, set_seed
    from peft import LoraConfig, TaskType, get_peft_model, PeftModel
    from weighted_training import WeightedTrainer, class_weights, oversample_issues
    from collections import Counter
    sampled_train = oversample_issues(train, args.issue_oversample_factor)
    weights = class_weights(train, args.class_weighting, args.max_class_weight, args.issue_weight_multiplier)
    print('Training loss weights:', dict(zip(LABELS, weights)), flush=True)
    print('Training presentations per epoch:', dict(Counter(r['label'] for r in sampled_train)), flush=True)
    loss_normalizer = sum(weights[LABELS.index(r['label'])] for r in sampled_train) / len(sampled_train) if args.loss_normalization == 'dataset' else None
    set_seed(42)
    base = 'nlpaueb/legal-bert-base-uncased'
    tok = AutoTokenizer.from_pretrained(base)
    def dataset(rows):
        ds = Dataset.from_list([{'text': r['text'], 'labels': LABELS.index(r['label'])} for r in rows])
        # Reject long annotations rather than silently dropping their ending.
        if any(len(tok(r['text'])['input_ids']) > 512 for r in rows):
            raise ValueError('Annotation exceeds 512 tokens; split and re-annotate it before training')
        return ds.map(lambda b: tok(b['text']), batched=True, remove_columns=['text'])
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=4,
        id2label=dict(enumerate(LABELS)), label2id={v: i for i, v in enumerate(LABELS)})
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_rank, lora_alpha=2 * args.lora_rank,
            lora_dropout=0.1, target_modules=args.attention_targets,
            modules_to_save=['classifier', 'pooler'] if args.train_pooler else ['classifier']))
    model.print_trainable_parameters()
    trainable_lora = sum(p.numel() for name, p in model.named_parameters()
                         if p.requires_grad and 'lora_' in name)
    if not trainable_lora:
        raise ValueError('LoRA parameters are frozen; refuse a misleading adapter-training run')
    def metrics(pred):
        y = np.argmax(pred.predictions, axis=-1)
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(pred.label_ids, y, labels=list(range(4)), zero_division=0)
        return {'macro_f1': float(np.mean(f1)), 'accuracy': accuracy_score(pred.label_ids, y),
                'issue_precision': float(precision[1]), 'issue_recall': float(recall[1]), 'issue_f1': float(f1[1])}
    trainer = WeightedTrainer(class_weights=weights, loss_normalizer=loss_normalizer, model=model, tokenizer=tok, train_dataset=dataset(sampled_train), eval_dataset=dataset(val),
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics,
        args=TrainingArguments(output_dir=args.output, num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
            learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
            eval_strategy='epoch', save_strategy='epoch',
            load_best_model_at_end=True, metric_for_best_model='macro_f1', save_total_limit=2,
            seed=42, report_to='none', logging_steps=10, disable_tqdm=True,
            group_by_length=True, dataloader_pin_memory=torch.cuda.is_available()))
    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    Path(args.output, 'validation_metrics.json').write_text(json.dumps(trainer.evaluate(), indent=2))
    from core import read_jsonl
    evaluations = [('validation', val)]
    if args.evaluate_test:
        evaluations.append(('test', [r for r in read_jsonl(args.data) if r['split'] == 'test']))
    for split, rows in evaluations:
        if not rows:
            continue
        pred = trainer.predict(dataset(rows), metric_key_prefix=split)
        labels = np.argmax(pred.predictions, axis=-1)
        report = {'split': split, 'metrics': pred.metrics,
                  'classification_report': classification_report(pred.label_ids, labels, labels=list(range(4)), target_names=LABELS, output_dict=True, zero_division=0),
                  'confusion_matrix': confusion_matrix(pred.label_ids, labels, labels=list(range(4))).tolist(),
                  'label_order': LABELS,
                  'annotation_status': 'provisional' if any(r.get('label_status') == 'provisional_suggested' for r in rows) else 'see dataset'}
        Path(args.output, f'{split}_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        Path(args.output, f'{split}_predictions.json').write_text(json.dumps([
            {'sentence_id': r.get('sentence_id'), 'case_id': r['case_id'], 'text': r['text'],
             'reference_label': r['label'], 'predicted_label': LABELS[int(y)],
             'probabilities': dict(zip(LABELS, torch.softmax(torch.tensor(logits), dim=-1).tolist()))}
            for r, y, logits in zip(rows, labels, pred.predictions)], indent=2), encoding='utf-8')
    Path(args.output, 'training_metadata.json').write_text(json.dumps({
        'base_model': base, 'labels': LABELS, 'epochs': args.epochs, 'seed': 42,
        'trainable_lora_parameters': trainable_lora,
        'class_weighting': 'issue_focused' if args.class_weighting == 'none' and args.issue_weight_multiplier != 1 else args.class_weighting,
        'class_weights': dict(zip(LABELS, weights)),
        'issue_oversample_factor': args.issue_oversample_factor,
        'original_training_counts': dict(Counter(r['label'] for r in train)),
        'training_presentations_per_epoch': dict(Counter(r['label'] for r in sampled_train)),
        'init_adapter': args.init_adapter, 'learning_rate': args.learning_rate,
        'warmup_ratio': args.warmup_ratio, 'weight_decay': args.weight_decay,
        'adapter_config': model.peft_config['default'].to_dict(),
        'data_sha256': __import__('hashlib').sha256(Path(args.data).read_bytes()).hexdigest(),
        'loss_normalization': args.loss_normalization, 'training_weight_mean': loss_normalizer,
        'data_path': args.data,
        'max_class_weight': args.max_class_weight, 'issue_weight_multiplier': args.issue_weight_multiplier,
        'best_checkpoint': trainer.state.best_model_checkpoint, 'best_validation_macro_f1': trainer.state.best_metric,
        'test_evaluated': args.evaluate_test,
        'train_cases': sorted({r['case_id'] for r in train}), 'validation_cases': sorted({r['case_id'] for r in val}),
        'provisional_labels': any(r.get('label_status') == 'provisional_suggested' for r in train)}, indent=2, default=lambda value: sorted(value) if isinstance(value, set) else str(value)), encoding='utf-8')


if __name__ == '__main__':
    main()
