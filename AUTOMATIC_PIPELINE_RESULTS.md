# Automatic pipeline and model experiment — October 21, 2026

## End-user workflow

Selecting a full-case PDF now starts cleanup, LegalBERT classification and LegalPegasus generation automatically. There is no source confirmation, Generate button, role editor, final-text editor or review form. The output stages are read-only; users can view and download them. Extraction details remain available after completion. PDF extension/signature checks, 10 MB and 200,000 extracted-character limits, and scanned-PDF errors remain enforced.

FIRPipeline.run now uses raw classifier predictions only. Exact-case overrides are not applied, and the manual /api/regenerate endpoint has been removed. Historical correction files remain for research provenance only. No hard-coded sentence labels were converted into claimed model accuracy.

## Training attempt

Candidate: models/legalbert_precision_v3. Active retained checkpoint: models/legalbert_balanced_v2.

The candidate continued the balanced_v2 adapter for eight epochs with learning rate 0.00005, 10% warm-up, weight decay 0.01, inverse-frequency weights capped at 5, ISSUE multiplier 0.3 and fixed dataset loss normalization. Architecture was inherited from the saved adapter. Seed 42, batch size 8, four CPU threads. Best checkpoint selected by four-class validation macro F1.

A derived dataset excluded only 11 training rows already marked NEEDS_DISCUSSION. The source workbook and original dataset were not modified or relabeled. Training counts: FACT 174, ISSUE 17, RULING 196, OTHER 2 (389 total). Validation 112 and test 58 were unchanged. Bicol / G.R. No. 214073 was excluded from training. No test evaluation was run.

| Validation metric | Retained model | Candidate |
|---|---:|---:|
| Accuracy | 89.29% | 89.29% |
| Macro F1 | 0.5798 | 0.5660 |
| ISSUE precision | 33.33% | 33.33% |
| ISSUE recall | 3/3 | 2/3 |
| ISSUE F1 | 0.5000 | 0.4444 |

The candidate was not promoted. This experiment did not improve automatic prediction quality. The existing ISSUE/RULING errors remain unresolved, and removing manual controls does not make the model accurate. Only three ISSUE validation examples and repeated model selection are important limitations.

## Verification

25 regression tests passed, including a check that the application uses raw classifier labels and never calls the saved-override helper, and that /api/regenerate returns 404. Dataset audit: data/revision_20261021_filtered/audit.json. Full comparison with identical validation-example verification: runs/precision_v3_comparison.json. Candidate metrics and training metadata are saved under its model directory.

## Next research requirement

Build an independently reviewed annotation set that consistently distinguishes questions/issues, party arguments, prior-court procedural history, and the deciding court's reasons/disposition. Add representative full cases and OTHER examples; preserve case-level splits. Model and data changes need independent evaluation before claiming reliable automatic labels. Pegasus hallucinations remain a separate unresolved problem.

## Completed browser verification

The full Bicol PDF was selected in the HTML workspace. Processing started immediately, without clicking Generate or confirming extraction, and the final digest appeared automatically. The rendered interface had zero label selectors and zero preprocessing confirmation controls. Raw predictions were FACT 64, ISSUE 16, RULING 60, OTHER 0. Sentences 100, 131, 132 and 138 were still predicted ISSUE, confirming both that the historical overrides are inactive and that the known model errors remain unresolved. Screenshot: outputs/automatic-workspace.png.

The upload-only workflow is complete. Reliable automatic labeling has not been achieved; the failed candidate is preserved for research comparison, not deployed.
