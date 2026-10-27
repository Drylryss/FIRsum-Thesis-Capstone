"""Fixed two-case development comparison; never changes model weights or deployment."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from core import ROLES

OUT = Path('data/private/bart_comparison_20261025')
CORPUS = Path('data/private/additional_digest_development_20261024_v3')
DOCKETS = ('127206', '176697')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def cases():
    refs = {r['docket']: r for r in read(CORPUS / 'references.json')}
    for docket in DOCKETS:
        ref = refs[docket]
        assert ref['split'] == 'validation' and docket not in {'154630', '147324', '151035'}
        yield docket, ref


def pipeline_stage():
    from pipeline import FIRPipeline
    from preprocessing import prepare_decision
    from delivery import prepare_delivery
    engine = FIRPipeline()
    assert not engine.examples, 'Protocol specifies zero-shot inputs'
    for docket, ref in cases():
        target = OUT / f'{docket}_firm.json'
        if target.exists():
            continue
        print('FIRmSUM', docket, flush=True)
        source_path = CORPUS / 'sources' / f'{docket}.json'
        source = read(source_path)
        prepared = prepare_decision(source['raw_text'])
        started = time.perf_counter()
        result = engine.run(prepared['text'], ref['case_id'])
        save(target, {'result': result, 'delivery': prepare_delivery(result),
                      'prepared_source': prepared['text'],
                      'elapsed_seconds': time.perf_counter() - started,
                      'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                      'reference_used_as_model_input': False})
        print('Saved', target, flush=True)


def bart_stage():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from pipeline import FIRPipeline
    torch.set_num_threads(4)
    engine = FIRPipeline.__new__(FIRPipeline)
    engine.torch, engine.device, engine.examples = torch, 'cpu', []
    engine.gt = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', local_files_only=True)
    engine.gen = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn', local_files_only=True).eval()
    # Match the deployed LegalPegasus decoding settings for this controlled run.
    engine.gen.generation_config.min_length = 32
    engine.gen.generation_config.length_penalty = 0.8
    engine.gen.generation_config.early_stopping = False
    for docket, ref in cases():
        firm = read(OUT / f'{docket}_firm.json')
        for mode in ('standalone', 'shared_evidence'):
            target = OUT / f'{docket}_bart_{mode}.json'
            if target.exists():
                continue
            print('BART', docket, mode, flush=True)
            started = time.perf_counter()
            inputs = ({'FULL': firm['prepared_source']} if mode == 'standalone' else
                      {r: firm['result']['generation_audit'][r]['generator_source'] for r in ROLES})
            outputs, audits = {}, {}
            for role, source in inputs.items():
                text, chunks = engine.generate_block(role, source)
                outputs[role] = text
                audits[role] = {**engine.generation_audit[role], 'chunks': chunks,
                                'source_sha256': hashlib.sha256(source.encode()).hexdigest()}
            save(target, {'outputs': outputs, 'audit': audits, 'mode': mode,
                          'model': 'facebook/bart-large-cnn',
                          'revision': engine.gen.config._commit_hash,
                          'generation_config': engine.gen.generation_config.to_dict(),
                          'elapsed_seconds': time.perf_counter() - started})


def score_stage():
    import torch
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import BERTScorer
    from huggingface_hub import snapshot_download
    torch.set_num_threads(4)
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rows = []
    for docket, ref in cases():
        firm = read(OUT / f'{docket}_firm.json')
        shared = read(OUT / f'{docket}_bart_shared_evidence.json')['outputs']
        variants = {'legalpegasus_raw': firm['result']['generated_blocks'],
                    'firm_final': {r: firm['delivery']['sections'][r]['display_text'] for r in ROLES},
                    'bart_shared_evidence': shared}
        for name, blocks in variants.items():
            for role in [*ROLES, 'FULL']:
                actual = blocks[role] if role != 'FULL' else ' '.join(blocks[r] for r in ROLES)
                reference = ref['blocks'][role] if role != 'FULL' else ' '.join(ref['blocks'][r] for r in ROLES)
                rows.append({'docket': docket, 'system': name, 'role': role,
                             'prediction': actual, 'reference': reference})
        rows.append({'docket': docket, 'system': 'bart_standalone', 'role': 'FULL',
                     'prediction': read(OUT / f'{docket}_bart_standalone.json')['outputs']['FULL'],
                     'reference': ' '.join(ref['blocks'][r] for r in ROLES)})
    for row in rows:
        actual, ref = row['prediction'], row['reference']
        row.update(rougeL=rouge.score(ref, actual)['rougeL'].fmeasure,
                   bleu=sacrebleu.sentence_bleu(actual, [ref]).score / 100 if actual else 0.0)
    save(OUT / 'metrics_lexical.json', rows)
    model_path = snapshot_download('FacebookAI/roberta-large', local_files_only=True)
    scorer = BERTScorer(model_type=model_path, num_layers=17, lang='en',
                        device='cpu', batch_size=1, rescale_with_baseline=False, idf=False)
    for row in rows:
        row['bertscore_input_tokens'] = {k: len(scorer._tokenizer.encode(row[k], truncation=False))
                                       for k in ('prediction', 'reference')}
        row['bertscore_truncated'] = max(row['bertscore_input_tokens'].values()) > 512
        if row['prediction'].strip():
            _, _, f1 = scorer.score([row['prediction']], [row['reference']])
            row['bertscore_f1'] = float(f1[0])
        else:
            row['bertscore_f1'] = 0.0
    save(OUT / 'metrics.json', {'rows': rows,
         'bertscore': {'model': 'FacebookAI/roberta-large', 'layers': 17, 'idf': False,
                       'rescale_with_baseline': False, 'hash': scorer.hash,
                       'warning': 'Flagged long inputs are truncated by standard BERTScore; not full-text semantic coverage.'}})
    lines = ['# Preliminary BART-CNN comparison', '',
             'Two development cases: Palma Gil (127206) and Areza (176697). This is not held-out final evaluation. Reference approval was assumed by user instruction, not independently documented. No weights were trained or deployment changed.', '',
             'BART standalone uses the complete preprocessed decision with recursive chunk summarization. Shared-evidence BART uses exactly the FIRmSUM-selected role passages; it is a generator comparison and benefits from LegalBERT and evidence selection. Neither receives reference digest text. Raw LegalPegasus and final source-backed FIRmSUM are reported separately.', '',
             'Generation uses four beams, no sampling, no-repeat trigrams, minimum length 32, length penalty 0.8, early_stopping=False, 256 new tokens with one 384-token retry, and recursive reduction instead of input truncation. BART decoding settings were matched to LegalPegasus for this development comparison; these are not BART default settings. Tokenizers yield different chunk boundaries. Equal maximum output limits do not guarantee equal summary lengths.', '',
             '| Case | Output | Scope | ROUGE-L | BLEU | BERTScore F1 | BERTScore truncated |',
             '|---|---|---|---:|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['docket']} | {r['system']} | {r['role']} | {r['rougeL']:.4f} | {r['bleu']:.4f} | {r['bertscore_f1']:.4f} | {r['bertscore_truncated']} |")
    lines += ['', 'All scores use a 0–1 scale. Missing outputs remain in denominators with zero scores. Lexical metrics use complete text; BERTScore truncation is disclosed per row. Similarity is not legal correctness. A standalone prose summary has no defensible FIR block labels, so it receives FULL scores only.', '',
              'This run cannot prove superiority or isolate the contribution of legal-domain pretraining. Final evaluation still requires a fixed protocol and independent case-level test set. Model card: https://huggingface.co/facebook/bart-large-cnn', '',
              '## Outputs for review', '']
    for r in rows:
        if r['role'] == 'FULL':
            lines += [f"### {r['docket']} — {r['system']}", '', r['prediction'] or '[EMPTY OUTPUT]', '']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Comparison saved to', OUT, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pipeline', 'bart', 'score'))
    args = parser.parse_args()
    {'pipeline': pipeline_stage, 'bart': bart_stage, 'score': score_stage}[args.stage]()
