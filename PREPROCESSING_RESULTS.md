# PDF preprocessing verification — October 21, 2026

Implemented conservative PDF cleanup, improved sentence boundaries, and a source-review preview before generation. The running API serves the updated interface with no-store caching.

## Changes

- Recognize a decision heading followed by a judicial author; preserve excluded caption text for inspection.
- Recognize SO ORDERED followed immediately by an uppercase judicial signature and judicial title; preserve that final disposition and exclude subsequent signatures, certifications, appended notes/opinions. A quoted lower-court SO ORDERED without such a signature does not end extraction.
- Retain ambiguous text rather than silently discarding it. This is a heuristic for supported formatting, not universal majority-opinion detection. Inline citation numbers and page-level footnotes may remain.
- Protect Atty., SJ., Jr., Sr., and similar abbreviations; split sentence endings followed by closing quotation marks.
- Add /api/prepare and a read-only retained-text preview with excluded material and warnings. Generation becomes available after the review checkbox is checked. Summarization repeats the same server-side cleanup. Exported reviewed JSON contains the preprocessing record.

## Case checked

Input: C:\Users\drylr\Downloads\Case Input  (1).pdf, Bicol Medical Center / G.R. No. 214073.

The previous UI run contained 256 segments. With the revised splitter, raw text has 248 segments; after structural cleanup, 140 remain. These are segmentation counts, not an accuracy metric. Two structural sections were excluded. The final WHEREFORE disposition and SO ORDERED were retained, including REVERSED and SET ASIDE and the permanent restraining order. The earlier quoted lower-court disposition remains as procedural history.

LegalBERT on the cleaned text: FACT 64, ISSUE 16, RULING 60, OTHER 0. The explicit main issue is segment 63 and is labeled ISSUE. No claim is made that all 16 ISSUE predictions are correct. Header/signature/footnote fragments previously seen in the classifier output no longer enter the model for this case.

## Verification

21 tests passed, including ambiguous-boundary preservation, quoted lower-court orders, monetary/statutory number preservation, names, closing quotes, extraction without model loading, and API validation. Browser verification confirmed that file selection shows the preview, checking the review box enables generation, and unchecking it disables generation.

The actual full /api/summarize run completed successfully with 140 segments. Saved result: runs/bicol_cleaned_pipeline_result.json. Cleanup/classifier audit: runs/bicol_preprocessing.json. Browser screenshot: outputs/source-review-workspace.png.

## Remaining generator defects

Cleanup did not resolve Pegasus factual reliability. In this run:

- The Facts block misattributes the intervenors' certiorari petition and mixes dates/courts.
- The Issue block describes purported holdings instead of stating the question and ends mid-sentence at the generation limit.
- The Ruling block describes the TRO as pending a hearing rather than accurately retaining the source's final permanent disposition.

Do not represent these summaries as accepted or accurate. LegalBERT was not retrained, and Pegasus was not modified by this change. Next work should evaluate source-grounded generation and sentence-complete output with independently reviewed reference digests. Longer output alone cannot fix invented facts.
