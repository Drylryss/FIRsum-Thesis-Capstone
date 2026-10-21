# Expanded ISSUE dataset training — October 20, 2026

The supplied workbook was used to train a separate LegalBERT LoRA candidate at `models/legalbert_issue35`. The user explicitly assigned the new examples to training. The original model remains active because validation ISSUE detection did not improve and macro F1 was slightly lower.

## Dataset and experiment

- Workbook: 570 selected sentences, including 35 ISSUE labels overall.
- Training: 400 sentences (FACT 174, ISSUE 28, RULING 196, OTHER 2).
- Validation: unchanged 112 sentences, including three ISSUE examples.
- Test: unchanged 58 sentences; no test inference or evaluation performed.
- Nineteen new ISSUE-labeled rows: one from an existing training case and six each from three additional cases. Case provenance and source references are recorded in `data/revision_20261020_issue35/README.md`.
- Eight epochs from the pretrained base, learning rate 0.0002, seed 42, batch size 8, standard loss weights, no oversampling. Best checkpoint selected by validation macro F1.
- All source labels and wording were preserved. Eleven new rows were flagged NEEDS_DISCUSSION for possible supporting-argument or historical-assertion ambiguity, but included in this requested provisional experiment. No labels are represented as human-approved.

## Validation results

| Metric | Original | Expanded-data candidate |
| --- | ---: | ---: |
| Accuracy | 89.29% | 89.29% |
| Macro F1 | 0.4536 | 0.4532 |
| ISSUE precision | 0% | 0% |
| ISSUE recall | 0/3 | 0/3 |
| ISSUE F1 | 0 | 0 |
| FACT F1 | 0.9000 | 0.8974 |
| RULING F1 | 0.9143 | 0.9155 |

All three validation ISSUE examples were predicted as RULING. The candidate predicted no ISSUE or OTHER labels on validation. These results do not establish performance on additional unseen cases.

## Verification

Case separation and normalized exact-text uniqueness passed. Tokenization with LegalBERT revision 15b570cbf88259610b082a167dacc190124f60f6 found a maximum of 296 tokens, including special tokens. The saved candidate reloaded and passed the application's classification smoke check on three validation source rows.

Raw comparison: `runs/issue35_comparison.json`. Full validation predictions, metrics, and training metadata are in the candidate model directory. Log: `training_legalbert_issue35.log`.

## Next step

Review the additional labels against the agreed codebook and add representative FACT, RULING, and OTHER sentences from the newly introduced cases. Their current contributions are ISSUE-selected examples only. Broader independent validation is also needed: repeated model selection on only three validation ISSUE examples is a substantial limitation.

The workbook and active inference configuration were not modified by this training run. LegalPegasus was not retrained.
