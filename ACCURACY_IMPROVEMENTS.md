# Source-grounded FIR pipeline — October 22, 2026

The observed Bicol digest confused the courts and petitioners, converted a party's argument into a holding, cut off the Issue paragraph, and omitted the Supreme Court's final disposition. The older Macasaet run invented a Boracay development and lost the payment order. Both examples are previously inspected development cases.

## Implemented behavior

- Preserve the active LegalBERT adapter, frozen LegalPegasus weights, automatic PDF workflow, and raw predictions. No training, manually overridden labels, or fabricated attorney-reviewed exemplars were introduced.
- Select evidence separately from classifier predictions: an explicit legal question, attributable court reasoning, and a terminal WHEREFORE–SO ORDERED disposition. Preserve the entire disposition regardless of its raw labels. Identify disagreements in the exported audit rather than rewriting model predictions.
- Keep raw Pegasus generation in `generated_blocks`. A strict whitespace-normalized literal check accepts only wording that matches all selected evidence. Other generation is retained for inspection and the final `blocks` use labeled source excerpts, with exact source sentence IDs. Unverified does not necessarily mean false: even a correct paraphrase is rejected by this conservative check.
- Withhold an Issue block when no explicit question can be selected. Warn when a final disposition is not located. Do not manufacture missing content.
- Retry an output that reaches the token cap once, from 256 to 384 new tokens. Record both attempts and reject still-capped or incomplete generation. Preserve source evidence and record a generation error if the decoder fails.
- For zero-shot generation, supply source text directly. The previous instruction prefix was observed copied into output. This follows the source-text interface in the [LegalPegasus model card](https://huggingface.co/nsi319/legal-pegasus). The existing few-shot format is retained for qualifying reviewed exemplars.
- Protect corporate abbreviations during sentence splitting. Recognize older judicial-author captions and concurrence endings conservatively. Retain ambiguous text and source artifacts for review rather than guessing corrections to names, amounts, or references.
- Display the raw model text separately from the final source excerpts. Provide evidence links and label classifier confidence as an uncalibrated model score.
- Install locked dependency versions by default. Pin the NumPy-compatible PyArrow release to avoid the incompatible fresh-install combination encountered on this machine.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe verify_grounding.py --run-models --output runs/grounding_source_only_20261022
```

The regression script executes both models using source text retained in the historical run artifacts. It does not upload or re-extract an original PDF. It saves raw generation, final blocks, source evidence, label conflicts, token counts, source hashes, and implementation hashes. Earlier instruction-prefixed diagnostics are retained in `runs/grounding_20261022` and should not be confused with the final source-only run.

## Measured checks

- **42 automated tests passed.** Coverage includes swapped actors/dates/places, changed negation, incomplete and capped output, generation failure, misclassified disposition sentences, quoted lower-court orders, quoted precedents, certificate boundaries, case separation, and existing API/data checks. JavaScript syntax and Git whitespace checks also passed.
- **Two real-model development runs passed** using the final source-only input: Bicol (139 segments) and Macasaet (45 segments). The corporate-abbreviation fix and conservative caption/trailer removal account for differences from historical segment counts; IDs in each output refer to that output's own source table.
- **Both complete terminal dispositions were retained.** Bicol preserves the granted petition, reversal and setting aside of the CA decision/resolution, and permanent TRO. Macasaet preserves the setting aside of the COA ruling and the order directing PTA to pay P219,302.47.
- **All six final blocks matched their selected source excerpts.** All six used fallback excerpts rather than accepting the raw generated paraphrase. This is source-integrity behavior, not a 100% legal-accuracy score.
- **No final source-only generation reached its token limit.** The final maximum lengths were 96/39/146 tokens for Bicol FACT/ISSUE/RULING and 239/41/41 for Macasaet. The earlier instruction-prefixed Macasaet FACT call exercised the retry path (256-token cap, then 268 tokens with the larger allowance).
- **Browser verification passed** for stage separation, fallback labels, evidence expansion, and navigation back to the raw sentence table. The browser preview used the saved real-model Bicol result; it did not pretend to be a fresh PDF upload. Screenshot: `outputs/grounded-final-ruling.png`.

Final results: `runs/grounding_source_only_20261022/report.json`, `bicol_result.json`, and `macasaet_result.json`. The regression artifacts include the raw inaccurate Pegasus output for audit; they must not be presented as expert-approved digests.

## Interpretation and remaining work

Literal source preservation is not legal-accuracy evaluation. The evidence selector is heuristic; unusual headings, quotations, multiple opinions, anaphora, imperfect extraction, and missing context can still produce incomplete or poorly selected excerpts. The fallback deliberately sacrifices concise paraphrasing to avoid publishing unverified generated statements. Raw classifier errors and Pegasus hallucinations can still be inspected in their respective stages.

These examples cannot establish generalization because they were already inspected. The retained classifier's prior validation accuracy of 89.29% and ISSUE precision of 33.33% are unchanged; the latter is based on only three true ISSUE examples. No new classifier accuracy or end-to-end legal-accuracy score is claimed.

Further model improvement requires consistently reviewed sentence labels that distinguish current-court reasoning from prior-court history, party arguments, legal questions, quotations, and administrative text. The two observed cases must remain development examples. Freeze new disjoint cases, obtain independently accepted FIR references, and measure attribution, factual consistency, disposition completeness, issue coverage, abstention, and compression separately. A manual review queue is available in each result's `verification.label_conflicts`; these suggestions are not gold labels.
