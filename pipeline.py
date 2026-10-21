import json
import os
from pathlib import Path
from core import LABELS, ROLES, sentences, token_windows, prompt, classifier_path


class FIRPipeline:
    def __init__(self):
        import torch
        torch.set_num_threads(4)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
        from peft import PeftModel, PeftConfig
        self.torch = torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        adapter = classifier_path()
        if not Path(adapter, 'adapter_config.json').exists():
            raise RuntimeError('Train LegalBERT first: python train_legalbert.py --data data/sentences.jsonl')
        metadata_path = Path(adapter, 'training_metadata.json')
        self.training_metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
        report_path = Path(adapter, 'validation_report.json')
        report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        self.unlearned_roles = [role for role in LABELS if report.get('classification_report', {}).get(role, {}).get('support', 0) > 0
                               and report['classification_report'][role]['recall'] == 0]
        cfg = PeftConfig.from_pretrained(adapter)
        self.bt = AutoTokenizer.from_pretrained(adapter)
        base = AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,
            num_labels=4, id2label=dict(enumerate(LABELS)), label2id={v: i for i, v in enumerate(LABELS)})
        self.bert = PeftModel.from_pretrained(base, adapter).to(self.device).eval()
        gen = os.getenv('LEGALPEGASUS_PATH', 'nsi319/legal-pegasus')
        self.gt = AutoTokenizer.from_pretrained(gen)
        self.gen = AutoModelForSeq2SeqLM.from_pretrained(gen).to(self.device).eval()
        exemplar_path = Path(os.getenv('EXEMPLARS_PATH', 'data/exemplars.json'))
        self.examples = json.loads(exemplar_path.read_text(encoding='utf-8')) if exemplar_path.exists() else []
        if self.examples and (len(self.examples) not in (2, 3) or len({e['case_id'] for e in self.examples}) != len(self.examples)):
            raise ValueError('Use 2–3 distinct fixed exemplar cases')
        for e in self.examples:
            if e.get('split') != 'exemplar' or e.get('attorney_accepted') is not True or any(not e.get('blocks', {}).get(r) for r in ROLES):
                raise ValueError('Exemplars need split=exemplar, attorney_accepted=true, and all FIR blocks')

    def classify(self, text):
        tagged = []
        for i, sentence in enumerate(sentences(text)):
            ids = self.bt.encode(sentence, add_special_tokens=False)
            scores = []
            for window in token_windows(ids, 510):
                batch = self.bt.prepare_for_model(window, return_tensors='pt')
                batch = {k: (v.unsqueeze(0) if v.ndim == 1 else v).to(self.device) for k, v in batch.items()}
                with self.torch.inference_mode():
                    scores.append(self.bert(**batch).logits.softmax(-1)[0])
            probs = self.torch.stack(scores).mean(0)
            label = int(probs.argmax())
            tagged.append({'id': i + 1, 'text': sentence, 'role': LABELS[label], 'confidence': round(float(probs[label]), 4)})
        return tagged

    def generate_block(self, role, source):
        if not source:
            return '', 0
        prefix_size = len(self.gt.encode(prompt(role, '', self.examples)))
        budget = min(1024, self.gen.config.max_position_embeddings) - prefix_size - 16
        if budget < 128:
            raise ValueError('Fixed exemplars leave fewer than 128 source tokens. Shorten all exemplar blocks.')
        ids = self.gt.encode(source, add_special_tokens=False)
        chunks = list(token_windows(ids, budget, min(64, budget // 4)))
        def generate(ids):
            content = self.gt.decode(ids, skip_special_tokens=True)
            batch = self.gt(prompt(role, content, self.examples), return_tensors='pt')
            if batch['input_ids'].shape[1] > 1024:
                raise ValueError('Prompt exceeds generator context; shorten exemplars')
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with self.torch.inference_mode():
                out = self.gen.generate(**batch, max_new_tokens=192, num_beams=4, no_repeat_ngram_size=3)
            return self.gt.decode(out[0], skip_special_tokens=True)
        summaries = [generate(c) for c in chunks]
        # Recursively reduce every chunk. Never truncate the tail of a role cluster.
        for _ in range(8):
            if len(summaries) == 1:
                return summaries[0], len(chunks)
            merged = self.gt.encode(' '.join(summaries), add_special_tokens=False)
            reduced = [generate(c) for c in token_windows(merged, budget, 0)]
            if len(reduced) >= len(summaries):
                raise RuntimeError('Chunk reduction did not converge; shorten exemplars or source')
            summaries = reduced
        raise RuntimeError('Too many chunk reduction stages')

    def run(self, text, case_id):
        if case_id in {e['case_id'] for e in self.examples}:
            raise ValueError('Exemplar cases cannot be evaluated as target cases')
        tagged = self.classify(text)
        return self.run_tagged(tagged, case_id)

    def run_tagged(self, tagged, case_id):
        if case_id in {e['case_id'] for e in self.examples}:
            raise ValueError('Exemplar cases cannot be evaluated as target cases')
        blocks, chunks = {}, {}
        warnings = ['Draft generation requires attorney review; classifier confidence is not factual accuracy.']
        if self.training_metadata.get('provisional_labels'):
            warnings.append('LegalBERT was trained on provisional suggested labels, not finalized attorney-reviewed labels.')
        if self.unlearned_roles:
            warnings.append('This checkpoint missed every validation example for: ' + ', '.join(self.unlearned_roles) + '. Corresponding output blocks may be unreliable or absent.')
        if not self.examples:
            warnings.append('No fixed exemplars configured: zero-shot generation, a departure from the proposal.')
        for role in ROLES:
            source = ' '.join(s['text'] for s in tagged if s['role'] == role)
            blocks[role], chunks[role] = self.generate_block(role, source)
            if not source:
                warnings.append(f'No {role} sentences detected; block left empty for review.')
        return {'case_id': case_id, 'blocks': blocks, 'sentences': tagged, 'chunks': chunks,
                'warnings': warnings, 'device': self.device, 'review_status': 'pending',
                'exemplar_case_ids': [e['case_id'] for e in self.examples]}
