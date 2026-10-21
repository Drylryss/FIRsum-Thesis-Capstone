"""Optional supervised experiment: this departs from the proposal's frozen Pegasus method."""
import argparse
from core import training_rows, prompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--output', default='models/legalpegasus')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--experimental-finetune', action='store_true', required=True)
    args = p.parse_args()
    train, val = training_rows(args.data, generator=True)
    from datasets import Dataset
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer,
        Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, set_seed)
    set_seed(42)
    tok = AutoTokenizer.from_pretrained('nsi319/legal-pegasus')
    model = AutoModelForSeq2SeqLM.from_pretrained('nsi319/legal-pegasus')
    def dataset(rows):
        encoded = []
        for r in rows:
            x = tok(prompt(r['role'], r['text']))
            y = tok(text_target=r['summary'])['input_ids']
            if len(x['input_ids']) > 1024 or len(y) > 256:
                raise ValueError('Training pair exceeds 1024 input / 256 target tokens; curate shorter pairs')
            x['labels'] = y
            encoded.append(x)
        return Dataset.from_list(encoded)
    trainer = Seq2SeqTrainer(model=model, tokenizer=tok, train_dataset=dataset(train),
        eval_dataset=dataset(val), data_collator=DataCollatorForSeq2Seq(tok, model=model),
        args=Seq2SeqTrainingArguments(output_dir=args.output, num_train_epochs=args.epochs,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=8, learning_rate=2e-5, eval_strategy='epoch',
            save_strategy='epoch', load_best_model_at_end=True, save_total_limit=2,
            seed=42, report_to='none'))
    trainer.train()
    trainer.save_model(args.output)
    tok.save_pretrained(args.output)
    trainer.save_metrics('validation', trainer.evaluate())


if __name__ == '__main__':
    main()
