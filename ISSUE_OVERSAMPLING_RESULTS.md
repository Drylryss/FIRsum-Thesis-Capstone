# ISSUE oversampling experiment — October 19, 2026

## Result and decision

Keep `models/legalbert` active. The separate candidate `models/legalbert_issue_os6_w2` did not improve ISSUE detection and reduced overall validation performance. No app model configuration was changed.

| Validation metric | Original | Oversampling candidate |
| --- | ---: | ---: |
| Accuracy | 89.29% | 81.25% |
| Macro F1 (four labels) | 0.4536 | 0.4081 |
| ISSUE precision | 0% | 0% |
| ISSUE recall | 0/3 (0%) | 0/3 (0%) |
| ISSUE F1 | 0 | 0 |
| FACT F1 | 0.9000 | 0.7792 |
| RULING F1 | 0.9143 | 0.8531 |

All three validation ISSUE sentences were predicted as RULING. Neither model predicted ISSUE or OTHER on this validation set. The candidate's best checkpoint was selected by validation macro F1, as in the original training workflow.

## Configuration

- Dataset: `data/revision_20261019/sentences.jsonl`, unchanged case splits and provisional annotations.
- Original training counts: FACT 174, ISSUE 9, RULING 196, OTHER 2.
- Each training ISSUE sentence was presented six times per epoch: 54 ISSUE presentations, 426 total training presentations. These are repeated examples, not 54 unique annotations.
- Loss weights: FACT 1, ISSUE 2, RULING 1, OTHER 1. Fixed dataset normalization uses the average weight over the oversampled training rows (480/426).
- Eight epochs, seed 42, learning rate 0.0002, batch size 8, LoRA rank 8, initialized from the pretrained LegalBERT base with a new adapter.
- Validation: 112 sentences from the same two cases, including only three ISSUE sentences. No test inference or test metrics were used in this experiment.

## Implementation and verification

`train_legalbert.py` now accepts `--issue-oversample-factor`. `weighted_training.py` rejects oversampling of validation, test, or exemplar rows and copies records without altering source annotations. Training metadata records original counts and actual presentations per epoch.

All 15 regression tests passed. The saved candidate also passed reload and classification through the application's classifier code on three validation source rows. These checks verify implementation, not legal accuracy.

Reproduce with the command in `README.md`, using a fresh output directory. Compare saved validation reports with `compare_issue_experiment.py`. Raw results and ISSUE error examples are in `runs/issue_oversampling_comparison.json`; the candidate folder contains full validation predictions, reports, and training metadata. Training logs are in `training_legalbert_issue_os6_w2.log`.

## Interpretation

This particular combination of repetition and moderate weights failed to improve validation results. It does not establish that every oversampling strategy fails. Repeated tuning on two validation cases also limits confidence in model selection.

The next substantive step is human review of provisional labels and collection of more diverse ISSUE examples from additional training cases. No annotations were automatically changed or invented. Preserve the test cases for final evaluation. LegalPegasus hallucinations remain a separate unresolved issue; this experiment only changed LegalBERT training.
