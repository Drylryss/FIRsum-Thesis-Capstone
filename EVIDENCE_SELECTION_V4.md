# Evidence-selection correction

23 October 2026. Both models are preserved; no weights were changed. This revision corrects source selection and keeps the literal verification gate. It does not claim to eliminate hallucinations inside LegalPegasus or correct LegalBERT's learned classification weights.

## Changes

- A reported finding before an embedded “question of whether” no longer qualifies as the deciding court's issue. This prevents the facts/reasoning boundary from moving into late reasoning.
- Fact selection reserves room for relevant early background, a triggering event, and the initial attributed positions of both sides, as well as procedural outcomes. These remain exact source sentences, not newly asserted facts.
- Mentioning a certification within a factual sentence no longer causes that sentence to be mistaken for a closing certification heading.
- Reasoning selection can recover explicit current-court findings regardless of their classifier label, while excluding adjacent reported lower-court text. Raw model labels remain unchanged and conflicts remain visible.
- The source-selection revision is reported as `source-grounding-v4`.

## Validation and limits

All 70 unit tests pass. The targeted Bicol regression checks the gate construction, competing public/private-road claims, single current issue, current-court reasoning, complete disposition, and absence of the previously fabricated California/securities material from final blocks.

Saved-prediction regression checks also pass for Makati, Real, Philcomsat and Mactan. All final dispositions remain present. Mactan's issue remains withheld rather than importing a question from an earlier precedent. These checks do not rerun model inference and are not attorney quality scores.

The source PDF has embedded footnote markers and extraction artifacts, retained to avoid silently altering quoted evidence. The final digest is longer because it preserves background, attribution and reasoning. This remains development work on previously inspected cases; an independent evaluation is still needed.

The original sample run is retained as `data/private/sample_full_case_20261023/result.json`. The saved-label replay is `evidence_v4.json`; a fresh live run, when completed, is saved separately as `result_v4.json` with its readable report beside it.
