# LegalBERT training improvement — October 21, 2026

Trained `models/legalbert_balanced_v2` on the latest imported dataset. The selected epoch-11 checkpoint improves validation macro F1 and ISSUE recall while matching the original model's accuracy. It is selected for provisional use through `model_config.json`; the original `models/legalbert` is preserved.

## Results

All scores below are on the same 112 validation sentences from two cases. No test inference or evaluation was performed.

| Metric | Original active model | Previous expanded-data model | New candidate |
| --- | ---: | ---: | ---: |
| Accuracy | 89.29% | 89.29% | 89.29% |
| Macro F1, all four labels | 0.4536 | 0.4532 | 0.5798 |
| ISSUE recall | 0/3 | 0/3 | 3/3 |
| ISSUE precision | 0% | 0% | 33.33% |
| ISSUE F1 | 0 | 0 | 0.5000 |
| FACT F1 | 0.9000 | 0.8974 | 0.8718 |
| RULING F1 | 0.9143 | 0.9155 | 0.9474 |
| OTHER F1 | 0 | 0 | 0 |

The candidate predicts nine ISSUE sentences: three correct, five actually labeled FACT, and one labeled RULING. It still misses the only validation OTHER sentence. Improved ISSUE recall comes with false positives and a lower FACT F1; unchanged overall accuracy does not mean identical errors. The two expanded-data models use identical training text, labels and splits, with the newest snapshot supplying additional provenance metadata. Their validation examples and reference labels were checked for exact agreement by the comparison script.

## Training recipe

- 400 training sentences from nine cases: FACT 174, ISSUE 28, RULING 196, OTHER 2. Validation 112; reserved test 58. No duplicate normalized sentence text or cross-split case overlap.
- Base `nlpaueb/legal-bert-base-uncased`, existing local cache; CPU training, four threads, seed 42, batch size 8.
- LoRA rank 16, alpha 32, dropout 0.1, query/key/value projections. Classifier and sentence pooler trained and saved with the adapter; 1,478,404 trainable parameters.
- Inverse-frequency class weights from training only: FACT 0.5747, ISSUE 3.5714, RULING 0.5102, OTHER capped at 5. Fixed dataset normalization; no oversampling.
- 16 epochs, peak learning rate 0.0002, 10% linear warm-up, linear decay, weight decay 0.01. Best checkpoint chosen by four-class validation macro F1, not by final epoch or ISSUE recall alone. Selected checkpoint: epoch 11 / step 550.
- Source annotations were not rewritten or promoted to approved labels. Eleven ISSUE rows remain marked NEEDS_DISCUSSION. No synthetic sentences were added.

Reproduction command is in README.md. Use a new output directory to preserve the saved model. Training metadata records the dataset hash, class weights, case IDs and adapter configuration. This is a combined configuration experiment, not an ablation proving which individual change caused the gain.

## Verification and artifacts

- All 112 saved validation labels and probabilities reproduced after reloading the adapter, including the trainable pooler. Application classification smoke check also passed.
- 17 regression checks passed across weighting, data contracts, comparison integrity and API behavior.
- `runs/balanced_v2_comparison.json`: comparison against the previous model trained on the expanded dataset, including every validation ISSUE error.
- `runs/balanced_v2_data_audit.json`: split and duplicate audit.
- `models/legalbert_balanced_v2/validation_report.json`, `validation_predictions.json`, `training_metadata.json`, `reload_validation_check.json`: detailed results and verification.
- `training_legalbert_balanced_v2.log`: complete training history.

## Limits and next work

These are provisional, repeatedly consulted validation results, not independent test accuracy. Three ISSUE examples and one OTHER example are too few to establish reliable generalization. Review the eleven disputed ISSUE annotations, add representative FACT/RULING/OTHER examples from the newly added cases, and obtain more independently annotated validation cases before final evaluation. Retain whole-case separation. The legal summarizer was not retrained or validated by this classifier experiment.

To restore the original model, set `legalbert_path` to `models/legalbert` in `model_config.json` and restart the server. An existing `LEGALBERT_PATH` environment variable overrides this file.
