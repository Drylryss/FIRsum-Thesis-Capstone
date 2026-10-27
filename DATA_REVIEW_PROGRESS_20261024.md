# Data review and evaluation preparation — 24 October 2026

The recommended data-first work has been carried out as a reviewable candidate revision. The canonical dataset, live models, and accepted-label fields have not been changed. This is not a completed attorney review or independent final evaluation.

## Data repairs prepared

The structural audit covers all 570 sentence records: all IDs are unique, and all 570 records still lack a final label. The semantic review remains targeted to the existing 34-item priority batch, which was reread for this revision; it is not a semantic review of the whole corpus.

Nine explicit party-position labels were changed from ISSUE to FACT in a separate candidate copy under the proposed annotation rubric. Four context-dependent changes remain deferred. Original sentence text, case identity and splits are preserved. Validation and test labels were not changed. These proposals are still marked pending review and are not attorney-approved training labels.

The 40-item alignment review produced 39 candidate pairs and one exclusion. The candidate pairs comprise eight previously supported alignments, 16 realignments, six exact-span segmentations, seven context-dependent pairs, and two unresolved source-check pairs. Thus, 39 candidates does not mean 39 cleared training examples.

Segmentation separates the Real party allegations and answer from a lower-court order; separates Mactan's repurchase demand from later proceedings; isolates MCIAA's position and witness material; and separates the current restitution application from a statutory passage. Each extracted span has offsets into a fingerprinted original source record. Original attorney-digest targets are preserved, including unresolved inconsistencies. No target-driven source selection was applied to validation or test inference.

The numerical inconsistency in Macasaet and the source's “Article 11 59” text remain flagged. They were not silently corrected.

## Evaluation safeguards implemented

`evaluation_protocol.py` builds a separate review record for raw LegalPegasus and final delivery for each FIR role. Missing or withheld sections remain represented, so unsuccessful outputs cannot silently disappear from the denominator. Human-review fields remain null until someone actually reviews them; null is not treated as “no error.”

Optional formal ROUGE-L, BLEU and BERTScore scoring requires a case-matched accepted reference, reviewer/date, verified source completeness, matching source fingerprint, an explicit untouched-case declaration, and no match in the development exclusion registry. It rejects exemplar leakage and known source fingerprints even when a case is renamed. These checks depend on truthful provenance records; they do not independently authenticate a reviewer's identity or prove legal correctness. Empty outputs receive zero textual scores and remain in the denominator; this policy must be recorded in the final methodology.

The existing `evaluate.py` remains unchanged. The new protocol provides the raw-versus-final distinction and prospective eligibility checks without misrepresenting previously accepted-output-only evaluation as a full-system result.

## Unseen evaluation set status

The registry conservatively excludes 19 known case identifiers represented in the sentence data, digest development, or saved pipeline development records. Some names may be aliases, so this is an identifier count, not a verified count of distinct legal decisions. All six supplied digests have been used in development. No eligible independent reference case is currently available in this reviewed collection.

An intake template is included for additional cases. A genuine unseen benchmark still requires additional decisions, accepted case-matched references, complete-source verification, and documented separation from training, prompting and development. It cannot be obtained by renaming or randomly re-splitting these six familiar cases.

## Files

- `data/private/data_revision_20261024_v2/summary.json`: audit counts and input hashes.
- `data/private/data_revision_20261024_v2/REVIEW_PACKET.md`: proposed labels and source–target pairs for review.
- `data/private/data_revision_20261024_v2/candidate_sentences_REVIEW_REQUIRED.json`: separate candidate sentence dataset.
- `data/private/data_revision_20261024_v2/aligned_pairs_REVIEW_REQUIRED.json`: source spans, offsets, reasons and unresolved flags.
- `data/private/data_revision_20261024_v2/case_exposure_registry.json`: case and source exclusions.
- `data/private/data_revision_20261024_v2/unseen_case_intake.json`: new-case intake fields.
- `data/private/evaluation_packets_20261024/sample/REVIEW.md`: raw-versus-final packet for the saved live sample run. Its final text is the version 5 live output, not a fresh version 6 generation.

The first revision directory is retained as an intermediate artifact. Use the `v2` package, which includes source fingerprints in the exposure registry.

## Verification and next step

All 100 project tests passed, including seven new regression tests for leakage prevention, missing-output retention, raw/final separation, scoring refusal, and exact-span safety. Artifact checks confirmed 570 unchanged texts and splits, exactly nine candidate training-label changes, and exact source correspondence for all 39 candidate pairs. The actual saved sample was also confirmed to fail the unseen-case gate even with synthetic acceptance fields supplied in a nonpersisted test fixture.

The next substantive step is to resolve the pending label/target/source decisions and obtain the new evaluation references. No model was retrained merely because a candidate file was produced, and no accuracy improvement is claimed from preparing it.
