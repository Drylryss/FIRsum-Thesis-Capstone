# ISSUE weighting comparison — October 19, 2026

The newly supplied workbook was re-imported to `data/revision_20261019`. Its SHA-256 differs from the prior file, but matching by sentence ID shows no changes to selected texts, labels, cases or splits; source row positions changed. All selected labels remain provisional. The original workbook was preserved.

Models were compared using the same 112 validation sentences from two cases, including only three ISSUE examples. No new test-set evaluation was run for these tuning experiments. The historical baseline test results were not used to choose weights.

| Model | Accuracy | Macro F1 | ISSUE recall | ISSUE precision |
| --- | ---: | ---: | ---: | ---: |
| legalbert | 89.29% | 0.4536 | 0.00% | 0.00% |
| legalbert_weighted | 86.61% | 0.4395 | 0.00% | 0.00% |
| legalbert_issue30 | 89.29% | 0.4536 | 0.00% | 0.00% |
| legalbert_issue30_global | 89.29% | 0.4536 | 0.00% | 0.00% |

## Decision

Keep `models/legalbert` active. None of these weight adjustments improved ISSUE detection. All validation ISSUE examples were still assigned to RULING. The weighted models are saved separately; no original weights were overwritten. These experiments do not establish that every possible weighting method would fail.

## Experiments

- `legalbert_weighted`: inverse-frequency weights FACT=0.5474, ISSUE=10.5833, RULING=0.4860, OTHER=20; eight epochs from base.
- `legalbert_issue30`: FACT=1, ISSUE=30, RULING=1, OTHER=1; six more epochs initialized from the original adapter, learning rate 0.0001.
- `legalbert_issue30_global`: same 30x ISSUE weighting and initialization, four epochs, fixed dataset normalizer instead of a per-batch weight denominator.
- Each experiment selects the best validation macro-F1 checkpoint. The continued-training classifier head and 294,912 LoRA parameters were verified trainable.

The fixed dataset normalizer is the weighted average over training examples (642/381). It preserves the intended loss weighting across minibatches. Tests verify this behavior. Validation loss values between different loss functions are not directly comparable; the table compares classification metrics.

## Data limitation and next step

Training has only nine ISSUE sentences and two OTHER sentences. Review the existing provisional annotations, resolve flagged ISSUE rows, and collect more diverse reviewed ISSUE examples across cases before another model-development round. Keep the frozen test cases separate. This task adjusted and tested training loss weights; it did not fabricate labels or force inference outputs to contain an Issue block.

Raw results: `runs/issue_weight_comparison.json`. Per-sentence validation predictions and classification reports are saved in each weighted model folder.

## Follow-up: moderate weights plus oversampling

The subsequent `legalbert_issue_os6_w2` experiment repeated training ISSUE examples six times per epoch with ISSUE loss weight 2. Validation accuracy was 81.25%, macro F1 0.4081, and ISSUE recall remained 0/3. The original model remains active. See `ISSUE_OVERSAMPLING_RESULTS.md` for configuration, verification, and limitations.
