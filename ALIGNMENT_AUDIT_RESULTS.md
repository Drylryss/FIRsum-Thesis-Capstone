# Source–digest alignment audit

Completed 23 October 2026. The six supplied attorney-created digests remain the entire reference collection. No additional reviewed cases were invented or added. Original digest wording and the deployed models are unchanged.

## Findings

The prior LegalPegasus feasibility dataset contained 15 role pairs across five cases. Splitting their targets for inspection produced 95 units; these are not 95 independent examples or cases. In 37 units, fewer than 80% of target word occurrences appeared in the old input. This is a lexical screening statistic, not a factual-support or legal-accuracy score.

Reference-assisted retrieval proposed whole source sentences from the same decision. Overlap improved for 32 units, worsened for 15, and stayed unchanged for 48. Five units had no candidate evidence. Twenty short fragments were flagged for contextual review. A high score cannot establish correct attribution, negation, disposition, or reasoning.

The export contains 40 provisional candidate units from the three existing training cases (Real, Mactan, Macasaet). Seven training units were not exported. The full audit retains these omitted targets; the export is not a complete replacement for the digests. Short answer/list fragments are excluded from standalone candidates. Remaining candidates can still need correction or rejection. Adjacent source sentences appear separately in the review file and are not silently added to model inputs.

Makati and Philcomsat remain previously seen development validation cases. Their reference-assisted evidence is audit-only: no validation model inputs were exported from it. The training candidates have not been used to train or promote a model.

## Sixth source recovered

The full [Picart decision](https://www.lawphil.net/judjuris/juri1918/mar1918/gr_l-12219_1918.html) was downloaded and preserved with its HTML hash. Extracted text separates the majority decision from the separate opinion. It remains machine-extracted, pending review. Picart already occurs in LegalBERT training, so it cannot become an independent test case. No Picart training pairs were added in this audit. The six-case inventory distinguishes downloaded decision text from Macasaet's saved pipeline sentence artifact.

## Review files

- `data/private/alignment_20261023/TRAINING_ALIGNMENT_REVIEW.md`: targets, candidate evidence, adjacent context, and blank review fields.
- `data/private/alignment_20261023/coverage_audit.json`: all 95 units, including omitted training targets and development validation audit.
- `data/private/alignment_20261023/training_candidates.json`: 40 machine-proposed units; semantic support is unverified.
- `data/private/alignment_20261023/six_case_inventory.json`: six references and source provenance.
- `data/private/alignment_20261023/new_case_intake_template.json`: empty intake record for future reviewed pairs.

The user's approval assumption is recorded as an assumption, not converted into an attorney's alignment review. The next useful step is resolving the candidate evidence and omitted targets, then running a case-disjoint development experiment. An independent reliability estimate needs additional untouched cases; the existing six cannot supply that retrospectively. No accuracy improvement is claimed from this audit.
