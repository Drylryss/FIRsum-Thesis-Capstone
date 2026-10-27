# LegalPegasus supervised fine-tuning feasibility

October 23, 2026

## Scope and outcome

A separate supervised LoRA adapter was trained successfully on this machine. This changes the proposal's frozen-generator methodology, as authorized by the user under their stated assumption of attorney approval. It does not establish that an attorney reviewed these paired inputs or accepted generated outputs.

The experiment ran one epoch over nine role pairs with three optimizer updates. It updated 1,572,864 parameters (approximately 0.275% of the combined model) using rank-8 adapters on attention query/value projections, learning rate 0.0001, gradient accumulation of three, and seed 42. The original base weights and LegalBERT checkpoint were not modified. The adapter is not deployed.

## Data

- Training cases: Real, Mactan, and Macasaet (nine Facts/Issue/Ruling pairs).
- Development validation: Makati and Philcomsat (six pairs), with no case overlap with training.
- Picart excluded because the local sentence annotations do not supply a complete decision.
- All five cases were previously inspected during development. There is no independent final test set.

Targets preserve the full attorney-supplied FIR blocks. Inputs consist of saved LegalBERT-labeled source sentences, prefixed with FACT, ISSUE, or RULING. Whole source sentences are selected in order within a 960-token limit; sentences that do not fit are omitted and the selected IDs/counts are recorded. Target digests do not determine input selection. The longest target is 805 tokens, within the model's 1,024-position limit. No target is silently truncated.

This bounded selection can omit facts needed by the full target digest; those alignments are not expert-validated. It is a material limitation for training quality, and must be addressed before a larger run. Role-prefixed inputs also differ from the live pipeline; the adapter is not a drop-in deployment.

## Development loss

| Case | Role | Before | After |
|---|---|---:|---:|
| Makati | FACT | 1.062760 | 1.061291 |
| Makati | ISSUE | 5.990338 | 5.988082 |
| Makati | RULING | 2.685387 | 2.695275 |
| Philcomsat | FACT | 0.844841 | 0.844578 |
| Philcomsat | ISSUE | 1.382776 | 1.380093 |
| Philcomsat | RULING | 1.748239 | 1.747139 |
| Unweighted mean across six pairs | | 2.285723 | 2.286076 |

Loss is teacher-forced token cross-entropy, not legal accuracy. Lower is better. The small average increase provides no evidence of improved validation performance. A single tiny training run cannot establish that supervised adaptation will or will not work with a larger, better aligned dataset.

## Generation diagnostic

On both development ISSUE inputs, the adapter produced exactly the same decoded output as the base model under identical 4-beam, 128-new-token decoding. All four outputs reached the diagnostic token cap. Makati generation invented US securities-law allegations instead of stating the legal question. Philcomsat generation narrated purported rulings instead of producing an Issue question. This check does not evaluate complete FACT/RULING generation or longer decoding, but it provides no evidence of an improvement in the tested outputs.

Decision: retain the baseline and source safeguards in the local app. Preserve the new adapter as a reproducible feasibility artifact. Review source/target coverage and expand the dataset before considering a longer training run; do not interpret successful optimization as successful legal summarization.

## Reproduction and artifacts

- `pegasus_feasibility.py`: pair preparation, split enforcement, base loss evaluation, adapter training, and final loss evaluation. It refuses to overwrite the existing adapter/report.
- `compare_pegasus_adapter.py`: before/after ISSUE generation on both development cases, with identical inputs and decoding settings. This is a diagnostic, not a complete end-to-end evaluation.
- `models/legalpegasus_lora_feasibility_20261023`: approximately 6.3 MB adapter weights, adapter configuration, and tokenizer files. Load on top of `nsi319/legal-pegasus` using PEFT.
- `data/private/pegasus_feasibility_20261023`: complete paired data with source IDs, split manifest, training progress, loss report, and generation diagnostic.

No new accuracy score or attorney acceptance is claimed. Further work requires source/target coverage review and additional case-separated data before a meaningful training and evaluation study.
