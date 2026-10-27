# Summary output update — 24 October 2026

The source-backed selector and delivery layer were improved. LegalBERT and LegalPegasus weights were not changed. The Chapter 4 draft retains its October 23 reporting cutoff and was not rewritten to include these later findings.

## Changes

- Reject retrospective discussions of an issue, including passages describing how another body overstepped its bounds, as issue statements.
- Recognize a limited set of current-case application passages as reasoning candidates, subject to quotation and attribution exclusions. This remains a heuristic, not semantic verification.
- Recognize party positions expressed through counsel or the Solicitor General, rather than promoting them to court findings.
- Keep preceding factual context for a repeated challenge or appeal.
- Omit the structural announcement “The facts follow” from the final Facts paragraph while retaining the original evidence in the audit.
- Keep a spaced “SO ORDERED .” closing after reasons, consistent with other closings.
- Surface warnings when selected passages may rely on omitted antecedents. Warnings do not rewrite or resolve the ambiguity.

## Comparison and validation

Saved inputs for Bicol, Macasaet, Makati, Real, Philcomsat and Mactan were replayed before and after the changes. The comparison script asserts unchanged input-file fingerprints and identified dispositions, preservation of every identified disposition, and exact correspondence of delivered evidence to source records.

Observed differences:

- Makati: one retrospective passage removed from Issue; Ruling now includes the source's explanation that the petition failed to identify the source of the asserted right, together with immediate supporting context. The preceding procedural event is preserved in Facts.
- Real: Ruling now includes source evidence concerning the LPG fire and the passage rejecting the fortuitous-event theory, rather than only the disposition. The passage starts with “In other words,” so its omitted preceding context is explicitly flagged for review.
- Macasaet: structural announcement removed from Facts; Ruling still contains only the disposition.
- Bicol: delivered text unchanged.
- Philcomsat: text unchanged; an omitted-antecedent warning is now visible. Ruling remains disposition-only.
- Mactan: Issue remains withheld; Ruling remains disposition-only.

All six identified dispositions remain intact. This is not a claim that all six digests are legally complete or accurate. These cases were already used in development; no model generation was rerun and no untouched benchmark or attorney assessment was performed.

Validation: 89 Python tests passed using the project virtual environment; delivery-interface tests passed using Node. The bundled Python runtime could not run the full suite because it lacks project ML dependencies and encountered a sandbox temporary-directory restriction; the project-environment run passed.

## Reproduction

Run `compare_summary_revision.py --output <new-directory> --baseline data/private/summary_revision_20261024/before` with the project Python environment. It refuses to overwrite an existing output directory.

Saved comparison: `data/private/summary_revision_20261024/final/COMPARISON.md`.
Machine-readable results and implementation fingerprints are in the same folder. Intermediate experiments and the baseline remain preserved.

## Remaining limitations

Some source records contain multiple sentences, quotations or extraction artifacts. The selector still cannot reliably recover every issue or supporting reason. Additional annotation and alignment review, unseen evaluation cases, and actual legal review remain necessary. No provisional labels were silently promoted to attorney-approved labels, and no unverified paraphrase was admitted into final delivery.
