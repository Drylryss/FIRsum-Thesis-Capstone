# PDF-backed display cleanup — 24 October 2026

The display layer now recognizes red, raised numeric references at the end of a PDF page, as well as isolated, centered Roman-numeral section headings. Removal still requires a unique contextual match against the complete extracted decision. Ordinary numeric values and inline Roman numerals remain untouched. Raw source evidence is preserved.

On the saved live Sample Full Case test, the new PDF-derived metadata removes the residual footnote 24 and the section heading II from the final paragraphs. The selected evidence for all three roles is identical to the live test. Dates, amounts, and Road Lot No. 3 remain unchanged.

Visual inspection of PDF page 8 confirmed that `[BJ\1C]` appears in the supplied document itself. The application therefore retains that text and displays a source-artifact warning instead of guessing a correction. It has not been independently corrected against an authoritative replacement source.

An eight-item Facts budget was also evaluated on six saved development cases. It shortened some outputs but omitted material background, procedural events, or qualifications. That experiment was rejected and the prior Facts selection restored. Its results are preserved under `data/private/summary_revision_20261024/compact_v6` for audit; they do not describe the deployed selection.

The accepted revision is `source-grounding-v6`. It does not retrain either model or claim improved abstractive generation. The sample was replayed with freshly extracted PDF display metadata; the five-minute model generation was not repeated for a presentation-only change.

Revised sample: `data/private/summary_revision_20261024/pdf_cleanup_v6/SUMMARY.md`.
Evidence-preserving delivery and detected PDF markers are saved alongside it.

Validation: all 93 Python tests and the delivery-interface checks passed. The sample replay asserts identical role evidence to the live test, removal of the two confirmed markers, retention of the suspicious source token, and preservation of Road Lot No. 3.

Remaining limitations: long Facts paragraphs, missing reasoning in some development cases, ambiguous source tokens, and unreliable raw generator outputs still require further work and legal review.
