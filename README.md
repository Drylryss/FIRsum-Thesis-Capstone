# FIRmSUM

Local research implementation for Facts–Issue–Ruling summarization of Philippine Supreme Court decisions on Obligations and Contracts.

## Current status — 27 October 2026

The active classifier remains `models/legalbert_balanced_v2`; LegalPegasus remains frozen. Final FIR drafts use selected source excerpts when generated paraphrases cannot pass the deliberately strict literal-source gate. This gate does not establish legal accuracy or completeness. Recent annotation and generator experiments were not promoted because they did not demonstrate a reliable improvement. See [the current research status](docs/STATUS_20261027.md) for results and limitations; earlier reports below describe historical experiments.

Private source documents, new experimental datasets/checkpoints, and raw local outputs are excluded from new commits. Research scripts referencing `data/private/` require locally supplied materials; they are not standalone public benchmarks. Previously tracked model/data artifacts are preserved.

## Setup (Python 3.10 or 3.11)

PowerShell shortcuts: run `powershell -ExecutionPolicy Bypass -File .\setup.ps1` to create/repair the environment, then `powershell -ExecutionPolicy Bypass -File .\start.ps1` to launch the app. These apply the execution policy only to that PowerShell process. No activation is required.

This machine has an NVIDIA MX330 with 2 GB VRAM. The installed PyTorch build uses CPU; this is appropriate for local inference on this hardware. Full LegalPegasus training should use a machine or notebook with more GPU memory. Model checkpoints download separately on first use into `.cache/huggingface`. Run `.\.venv\Scripts\python.exe verify_environment.py` for an offline dependency and tiny-model smoke check. `requirements-lock.txt` records exact installed dependency versions for this environment.

Run these commands in this folder using PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe train_legalbert.py --data data/sentences.jsonl
.\.venv\Scripts\python.exe inference.py
```

Open http://127.0.0.1:8000 in your browser. Installation and first model use require internet and disk space for several GB of weights. CUDA-capable hardware is recommended; CPU inference is supported but slow. Install a matching PyTorch CUDA build if needed for your machine. A provisional LegalBERT checkpoint has now been trained from the supplied workbook. See TRAINING_RESULTS.md for results and class-level limitations. LegalPegasus downloads separately on first inference.

## Files and workflow

- `train_legalbert.py`: LegalBERT LoRA training for FACT / ISSUE / RULING / OTHER; validation macro F1, fixed seed, best-checkpoint selection, saved adapter plus classification head.
- `train_legalpegasus.py`: separate optional full supervised fine-tuning experiment. The proposal explicitly uses frozen LegalPegasus; this script requires `--experimental-finetune` to make the methodology change explicit.
- `pipeline.py`: learned role tagging, fixed exemplar prompts, bounded context windows, hierarchical chunk reduction, source evidence and warnings.
- `inference.py` and `web/index.html`: local FastAPI backend and responsive HTML interface; select a full-case PDF to start processing automatically; the filename identifies the output. Model labels and digest are read-only, with JSON downloads.
- The output workspace has three read-only stages: **LegalBERT** raw sentence labels and uncalibrated scores, **LegalPegasus** raw generated paragraphs, and **Final form** with source evidence and literal verification. Each output can be downloaded. Outputs appear when the full pipeline completes; no end-user label corrections or review confirmation are required.
- `evaluate.py`: per-block ROUGE-L, sentence BLEU, BERTScore for accepted/edited held-out outputs. Scores use 0–1 scales; save metrics separately from model validation.
- `data/README.md`: exact annotation, exemplar, generator and reference schemas.

The supplied proposal provides the design context. Its statements are research assumptions, not verified performance results. This implementation does not guarantee accurate FIR generation or factual grounding. PEGASUS is a summarization model rather than an instruction-tuned chatbot; few-shot formatting is experimental and needs empirical validation. The supplied workbook has now been imported and a provisional classifier is available locally in models/legalbert. Its suggested labels were not finalized; see TRAINING_RESULTS.md.

## Inference and verification

### Source grounding added October 22

The active pipeline still uses the saved LegalBERT adapter and frozen LegalPegasus. A separate evidence-selection layer now identifies explicit legal questions and a terminal WHEREFORE–SO ORDERED disposition, preserves attribution context, and supplies selected complete sentences to generation. Raw classifier labels are never overwritten; structural evidence disagreements are exported under `verification.label_conflicts`. No case-specific override file or reviewed test labels are used.

`generated_blocks` contains the actual Pegasus output. `blocks` contains the final output. The final gate accepts generation only when it reproduces the complete selected source wording after whitespace normalization. Other paraphrases are **unverified**, not automatically declared false: they are replaced with labeled source excerpts and sentence IDs. Missing explicit issues are withheld. Missing terminal dispositions produce a warning. This intentionally favors source fidelity over fluent compression; it is not a semantic entailment verifier and cannot establish legal correctness, source quality, relevance, or completeness. Context-selection heuristics can still fail on unusual decisions, quotations, and separate opinions.

This is a documented extension of the original model-only assembly step, not an improvement claimed for raw model accuracy. The UI and downloads retain the difference. The generator retries a length-capped output once at a larger limit and records every call; an unfinished or capped result cannot pass the final gate. Exact-case manual overrides remain disabled.

When no reviewed exemplars are configured, the generator receives only the selected source text, matching the [model card's source-text interface](https://huggingface.co/nsi319/legal-pegasus). The previous instruction prefix was observed leaking into generated summaries. The few-shot prompt format remains available when 2–3 qualifying exemplars are supplied; none are fabricated or marked attorney-approved by this implementation.

Run development checks with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_grounding.py --run-models
```

The two supplied example cases have already been inspected and are development regressions, not a new independent test set. No attorney-approved exemplars or new gold labels have been invented. Further classifier training requires consistently reviewed labels; independently reviewed decisions and accepted reference digests are needed before reporting legal-accuracy improvements. See `ACCURACY_IMPROVEMENTS.md` for the implementation and measured checks.

Use `requirements-lock.txt` for reproducible installation. `requirements.txt` also pins PyArrow to the compatible version because PyArrow 26 requires NumPy 2 whereas this project uses NumPy 1.

### Optional generator fine-tuning

```powershell
.\.venv\Scripts\python.exe train_legalpegasus.py --data data/generator.jsonl --experimental-finetune
$env:LEGALPEGASUS_PATH = 'models/legalpegasus'
.\.venv\Scripts\python.exe inference.py
```

Default inference uses `nsi319/legal-pegasus`. `LEGALBERT_PATH` overrides `models/legalbert`; `EXEMPLARS_PATH` overrides `data/exemplars.json`. With no exemplar file, output is explicitly marked zero-shot. Shorten fixed exemplars if they exhaust the 1024-token generator budget; the app never silently removes examples or source tails. Configure these paths before starting the server and run from this folder.

## ISSUE weighting experiments

`train_legalbert.py` supports training-only class weights. `--class-weighting balanced` computes capped inverse-frequency weights; `--issue-weight-multiplier 30 --max-class-weight 40` with the default `none` mode produces FACT=1, ISSUE=30, RULING=1, OTHER=1. This affects the training loss, not a forced inference label. `--loss-normalization dataset` uses the fixed average training weight so weight effects do not depend on the composition of each minibatch; the default `batch` uses PyTorch's weighted-mean loss.

Use a new output directory for each run. Existing trained directories are protected from overwrite. `--init-adapter models/legalbert` continues training from the original saved adapter while preserving it. For example:

```powershell
.\.venv\Scripts\python.exe train_legalbert.py --data data/revision_20261019/sentences.jsonl --output models/legalbert_issue_experiment --init-adapter models/legalbert --epochs 4 --learning-rate 0.0001 --issue-weight-multiplier 30 --max-class-weight 40 --loss-normalization dataset
```

Training now evaluates validation only by default. `--evaluate-test` is an explicit opt-in for a final frozen evaluation, not for selecting weights. The previously reported test set has already been inspected; do not use it to guide tuning. Weighting cannot replace additional reviewed ISSUE annotations.

The optional `model_config.json` key `legalbert_path` selects a saved classifier for the app. Without that file, the original `models/legalbert` remains active. `LEGALBERT_PATH` takes priority. Restart the server after changing the active checkpoint.

## Evaluation and verification

### Balanced LegalBERT candidate (October 21)

The training script supports learning-rate warm-up, weight decay, configurable LoRA rank/attention targets, and an optional trainable sentence pooler. Defaults preserve the earlier architecture. Adapter architecture options apply to new adapters; continued training uses the saved architecture. Validation predictions include class probabilities and training metadata records the dataset SHA-256 and adapter configuration.

Reproduce the expanded-data candidate with a fresh output directory:

```powershell
.\.venv\Scripts\python.exe train_legalbert.py --data data/revision_20261020_metadata/sentences.jsonl --output models/legalbert_balanced_v2_reproduction --epochs 16 --class-weighting balanced --max-class-weight 5 --loss-normalization dataset --warmup-ratio 0.1 --weight-decay 0.01 --lora-rank 16 --attention-targets query key value --train-pooler
```

This uses 400 training sentences, no oversampling, seed 42, batch size 8, and peak learning rate 0.0002. Best checkpoint selection uses validation macro F1 across all four labels. Weights come only from training counts and are capped at 5; no test evaluation is requested. See `LEGALBERT_BALANCED_V2_RESULTS.md` for the outcome when the run completes. Validation contains only three ISSUE sentences and one OTHER sentence, so gains require confirmation on additional independently annotated cases.

### ISSUE oversampling experiment

`--issue-oversample-factor 6` repeats each training ISSUE example six times per epoch. It does not create new annotations or change validation/test data. Class weights are computed from the original training counts; dataset loss normalization uses the resulting training presentations. The nine unique ISSUE examples become 54 presentations among 426 total per epoch.

```powershell
.\.venv\Scripts\python.exe train_legalbert.py --data data/revision_20261019/sentences.jsonl --output models/legalbert_issue_os6_w2 --epochs 8 --issue-oversample-factor 6 --issue-weight-multiplier 2 --loss-normalization dataset
```

This candidate starts from the pretrained base with a new LoRA adapter, uses seed 42 and learning rate 0.0002, and selects its checkpoint by validation macro F1. See `ISSUE_OVERSAMPLING_RESULTS.md` for the measured result and model-selection decision. Reproduction requires a new output directory if the saved candidate already exists.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe evaluate.py --prediction fir-digest.json --reference data/private/reference.json --output runs/metrics.json
```

Freeze disjoint case splits before training. The scripts enforce separation within each supplied training file; researchers must reconcile case IDs across files and exclude duplicates, related decisions and exemplars from test. No automatic split can replace this corpus audit.

The UI exports model outputs locally. Research evaluation against independently accepted reference digests remains separate from the end-user workflow. The prototype binds to localhost and does not store uploaded documents server-side. Use locally; add authentication, access controls and job scheduling before any shared deployment.

The PDF upload starts processing immediately; extraction details are available after generation. Conservative cleanup recognizes a decision heading with judicial author and a final SO ORDERED immediately followed by a judicial signature. It excludes the identified caption and signed trailing material, preserving both in the optional extraction details and exported preprocessing record. Quoted lower-court orders without a signature do not truncate the decision. Ambiguous boundaries, inline reference numbers and page-level footnotes are retained for review; this is not a universal legal-document parser. Testing input accepts only a PDF upload (10 MB maximum, 200,000 extracted characters maximum). No case identifier or source selection is required; the PDF filename identifies the output. Text-based PDFs are supported; scanned PDFs need OCR. Sentence splitting is heuristic; review boundaries before annotation. Long classification sentences use overlapping windows with mean probabilities. Empty predicted roles stay empty. A nonconverging generator reduction raises an error instead of discarding text.

The UI implements draft/review/export functionality. Automated BART baseline comparison, participant surveys, blinded two-week reliability rerating and full corpus-level statistical reporting remain research tasks; no results are fabricated here.

## Technical references

- [LegalBERT model card](https://huggingface.co/nlpaueb/legal-bert-base-uncased)
- [LegalPegasus model card and 1024-token limit](https://huggingface.co/nsi319/legal-pegasus)
- [Transformers 4.44.2 Trainer documentation](https://huggingface.co/docs/transformers/v4.44.2/trainer)

The new `/api/prepare` endpoint performs extraction and cleanup without loading the models. The end-user UI calls `/api/summarize` directly on file selection; extraction and cleanup run automatically. Sentence splitting protects Atty., SJ., Jr., Sr. and closing-quote boundaries. These changes do not retrain LegalBERT or guarantee factual Pegasus summaries.

## Automatic model-only workflow (October 21)

Select one full-case PDF and wait. There is no Generate button, preprocessing confirmation, role editor, final-text editor or review outcome form. The server runs extraction, conservative cleanup, raw LegalBERT classification and Pegasus generation. Optional extraction details appear after completion.

Case-specific reviewed label overrides are no longer applied by the pipeline. The old `label_review.py`, override JSON and review reports remain as historical research artifacts only; they do not change predictions. `/api/regenerate` is no longer exposed. A regression check verifies that `FIRPipeline.run` uses raw model labels even if an override file exists.

The automatic flow does not imply correct predictions. The model still needs independent evaluation and more reliable training labels; Pegasus factual errors are a separate unresolved limitation. See AUTOMATIC_PIPELINE_RESULTS.md for the new training comparison and deployment decision.
