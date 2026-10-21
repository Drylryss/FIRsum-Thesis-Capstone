# LegalBERT provisional training results

Completed October 18, 2026. The eight-epoch LoRA run took about 7.9 minutes of training time on CPU, excluding downloads and final evaluation. The best validation checkpoint was epoch 8. The adapter and tokenizer are saved in `models/legalbert`, the app's default classifier location. Adapter reload and inference through the application's classifier code pass.

## Data

Source: `C:\Users\drylr\Downloads\Consolidated Tagged Sentences File for LegalBERT.xlsx`.

The 551 rows selected in `To be Fed` were matched to `Tagged Sentences` to retain case and sentence IDs. All final labels were blank. Suggested labels are provisional and are not represented as attorney-approved annotations. The source workbook was not modified. The 122 NEEDS_DISCUSSION and 8 SOURCE_CHECK rows were excluded. No normalized duplicate text was found among selected rows.

Whole cases were allocated before training: 381 sentences / 6 cases for training, 112 / 2 for validation, and 58 / 2 for testing. Missing case IDs were recovered from sentence ID prefixes and recorded in the data. Original TRAIN assignments were reallocated by case for this development run; the proposal's larger validation-set target is not met. See `data/import_report.json` and `data/split_manifest.json` for the complete audit.

## Results against provisional suggested labels

| Metric | Validation | Test |
| --- | ---: | ---: |
| Accuracy | 89.29% (100/112) | 89.66% (52/58) |
| Macro F1 over four fixed labels | 0.4536 | 0.4666 |
| FACT F1 | 0.9000 | 0.9545 |
| ISSUE F1 | 0.0000 | 0.0000 |
| RULING F1 | 0.9143 | 0.9118 |
| OTHER F1 | 0.0000 | Not estimable: no test examples |

The four-label macro calculation assigns zero to absent OTHER in test; it should not be mistaken for an observed OTHER test result. Validation includes one OTHER example, which was misclassified. All three validation ISSUE examples and all four test ISSUE examples were misclassified as RULING. No ISSUE or OTHER predictions were made on either split.

**The model is not a reliable four-role classifier yet.** Overall accuracy is dominated by FACT and RULING. Training included only nine ISSUE and two OTHER examples. Small case counts and provisional labels further limit generalization claims. This classifier run does not measure LegalPegasus output accuracy or end-to-end summary quality.

## Next research steps

Finalize annotations with the reviewer, resolve excluded rows, and collect substantially more diverse ISSUE and OTHER examples across cases. Freeze a sufficiently broad independent test set before additional model development. Compare class-balanced training methods using validation data only. The current test set has now been evaluated and should not guide subsequent tuning decisions.

## Reproduction

`python train_legalbert.py --data data/sentences.jsonl --output models/legalbert --epochs 8 --batch-size 8 --threads 4`

Seed 42, base `nlpaueb/legal-bert-base-uncased`, LoRA rank 8 / alpha 16 / dropout 0.1 on query and value modules, saved classification head, learning rate 0.0002. The complete training log is `training_legalbert.log`; detailed classification reports and confusion matrices are in `models/legalbert/validation_report.json` and `models/legalbert/test_report.json`.
