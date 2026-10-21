# Reviewed label corrections — Bicol case

Corrected ISSUE to RULING for sentences 100, 131, 132 and 138 in the cleaned 140-segment case output. These sentences state judicial findings or the final disposition. The correction is an explicit assistant review requested by the user, pending expert confirmation, not a retrained LegalBERT prediction.

The underlying model still predicted ISSUE; original roles and probabilities are preserved. The reviewed role counts are FACT 64, ISSUE 12, RULING 64, OTHER 0. Counts do not measure accuracy. Other sentences have not been comprehensively adjudicated.

Exact-case overrides: data/reviewed_label_overrides.json.
Reviewed sentence record: runs/bicol_reviewed_labels.json.
The reviewed record contains no regenerated digest; use the workspace to regenerate from corrected roles.

24 tests passed. They cover matching only the exact case, preservation of raw predictions, regeneration without reclassification, and API input validation, alongside prior regression checks. No model training or test-score update was performed.

The corrected case was also regenerated successfully in the HTML workspace. The four reviewed RULING labels and preserved raw ISSUE predictions were verified in the rendered table. Pegasus still produced factual errors and an incomplete Issue paragraph; the corrected labels do not establish summary accuracy.
