# Reliability delivery update — 23 October 2026

The pipeline preserves LegalBERT and LegalPegasus. Their weights were not changed in this update. The generator still makes factual errors, and classifier labels remain provisional. This update prevents rejected/raw generation from being mistaken for the source-backed deliverable and makes missing support visible.

## Implemented

- A separate delivery contract checks each evidence sentence against its recorded source ID and exact text, checks final text against that evidence, and requires the complete identified disposition for RULING. Failed sections are withheld; missing audits fail closed.
- A ruling containing only the disposition now carries an explicit missing-reasoning warning. Empty sections mark the draft incomplete.
- Rejected/unverified model text is collapsed behind an inspection control and marked as unsuitable for a digest. Researchers can still inspect and download it in an explicitly labeled diagnostic artifact.
- Copy and digest download use only the delivery contract, excluding raw generation. Exports retain source references and pending review status.
- The supplied sample's source-backed export is saved at `data/private/sample_full_case_20261023/SOURCE_BACKED_DRAFT.json`. Its raw model run is preserved separately.

## Evidence

74 Python tests passed, including changed-source, substituted-draft, missing-audit and missing-disposition failures. The shipped page script also passed Node unit checks for fail-closed display, collapsed diagnostics, source-only copy/download, and diagnostic labeling. API tests verify an unverified legacy response cannot become an exportable source-backed draft. The updated app was started and its ready interface inspected.

Five saved cases were replayed through the new delivery contract without rerunning either model. Bicol has all three source-backed sections. Mactan remains incomplete because its current issue was not located reliably. Makati, Real, Philcomsat and Mactan carry warnings that their selected ruling lacks supporting reasoning. No numerical legal-accuracy claim follows from these software checks.

## What remains necessary

Source matching does not prove correct attribution, completeness, or the correctness of a quoted proposition. The six attorney-created digests do not constitute reviewed sentence labels, and an approval assumption is not recorded annotation. Existing alignment candidates and the priority label queue still require actual corrections/review before they can establish training quality. They have not been silently marked reviewed or used for a new training run.

Next training should use corrected, claim-supported inputs and targets, retain case-level splits, compare against the unchanged models, and be promoted only after held-out evaluation. The already inspected six cases cannot establish independent reliability. More untouched cases with reviewed digests are needed for that evaluation. Until then the system remains a drafting/review assistant, not an automatically reliable legal summarizer.
