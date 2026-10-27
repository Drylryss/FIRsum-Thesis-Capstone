# Reliability update October 22 2026

## Follow-up evidence revision

Version `source-grounding-v3` corrects missing question patterns and quotation state extending through current-case dispositions. It keeps unknown/earlier-court quotations excluded and rejects questions explicitly attributed to a named precedent. Sixty automated tests passed. Replaying the four saved classifier outputs recovered four of five missing blocks: Makati Issue, Philcomsat Issue, Real Ruling, and Mactan Ruling. Mactan Issue remains withheld. All four terminal dispositions are preserved exactly. This is an evidence-selection regression, not a fresh model run or an accuracy estimate; historical generation is not reused as if it had seen the new evidence. Bicol and Macasaet regressions also pass.

`verify_evidence_revision.py` reproduces the comparison. Results are under `data/private/attorney_references_20261022/evidence_revision_v3`.

`prepare_reliability_review.py` produces a focused 34-item review batch (11 previously flagged annotations plus 23 remaining ISSUE/OTHER annotations) with neighboring annotation records and source pages. Gaps in those records are explicitly disclosed. No labels are automatically accepted and no model is retrained. The readable batch and editable JSON are under `data/private/attorney_references_20261022/review`.

All ten eligible three-digest combinations exceed the generator's supported prompt budget in at least one role. `prepare_exemplar_candidates.py` prepares three compact verbatim candidates from Makati, Real, and Mactan, each with source-paragraph provenance. These candidates fit the input budget but remain pending attorney review of omissions. They are not activated or represented as independently accepted shortened digests. Their cases must be excluded from subsequent evaluation if activated. New sentence labels and compact-example acceptance are the remaining human inputs; neither is inferred from original digest authorship.

Both model checkpoints remain unchanged. The source safeguards now block summarization when a PDF contains pages with no readable text or substantial replacement-character corruption. Sparse pages are flagged for inspection. This does not certify that extraction is complete.

The evidence selector excludes text after an explicit separate-opinion heading, handles additional reported-speech and quotation boundaries, retains consecutive enumerated questions, and withholds the Ruling block when it cannot identify a terminal disposition. These conservative rules can withhold valid material; missing blocks remain explicit failures of coverage.

Final excerpts display sentence references and gaps. Copied drafts retain references and their review status. Source text and PDF hashes support reproducibility.

Validation: 56 automated tests passed; JavaScript syntax and Git whitespace checks passed. Existing Bicol and Macasaet outputs were rechecked with the new selector and still preserve their dispositions. These were re-verifications, not new model runs.

The attorney-supplied document contains six FIR digests. Exact imported wording and the original document are stored under gitignored `data/private/attorney_references_20261022`. Picart overlaps classifier training; Macasaet was previously inspected during development. All six are development references, not an independent legal-accuracy benchmark. The references are not fed into generation, used to relabel sentences, or used to train either model.

Four corresponding full decisions downloaded from the Supreme Court E-Library are processed by `run_reference_cases.py`. Per-case JSON and the run report are saved under `data/private/attorney_references_20261022/pipeline_results`. They expose missing roles, raw generation, final excerpts, and source provenance. Successful execution does not mean the digest is complete or attorney-approved.

The dataset audit found 570 provisional sentence labels with no recorded final reviewer/label. Training has 28 ISSUE and 2 OTHER examples; validation has 3 ISSUE and 1 OTHER examples. The review queue in `runs/reliability_20261022/label_review_queue.jsonl` prioritizes existing disputed labels and minority roles and excludes test text. Attorney-created digests do not finalize those sentence annotations.

The supplied DOCX text was extracted structurally. Visual rendering was unavailable because the bundled runtime had no LibreOffice executable; the document was not edited and no page-layout verification is claimed.
