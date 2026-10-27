# Reliability status — 23 October 2026

The system is a source-backed drafting assistant. It is not yet demonstrated to be an independently reliable legal summarizer. Both LegalBERT and LegalPegasus are preserved; this update does not change their weights.

## New safeguards

Delivery now checks the recorded decision fingerprint, rejects duplicate source/evidence IDs, and withholds evidence explicitly excluded as a separate opinion. These supplement exact evidence matching, complete identified-disposition retention, source-only final export, and PDF-backed footnote cleanup. A fingerprint detects accidental changes to records; it does not establish that the legal text is correct or defend against someone replacing both data and its fingerprint.

## Repeatable evaluation

Run `python audit_pipeline_reliability.py` from the project directory. It reads saved predictions for Bicol, Macasaet, Makati, Real, Philcomsat and Mactan, applies current source selection and delivery, and records input/code hashes. It never presents historical raw generation as if produced using the revised evidence selection. Results and a blank case-review packet are under `data/private/reliability_audit_20261023/`.

Observed results:

- 82 Python tests and the delivery interface unit checks passed.
- All six cases retained their identified final dispositions.
- One case (Mactan) still lacked an attributable current issue; it remains withheld.
- Zero of 18 historical raw model blocks were accepted by the strict literal-match gate. This is a gate-acceptance count, not an assertion that all 18 were factually wrong.
- The cases were previously inspected during development. Neither model was rerun for this replay. There is no independent legal-accuracy estimate.

## Remaining work requiring reviewed data

The actual attorney review fields remain blank. Source excerpts can still omit context, and model labels are provisional. Correct the sentence labels and claim-to-source alignments, recording who reviewed each correction. Keep the existing six digests in development. Obtain additional untouched decisions and reviewed references for the final evaluation. Compare raw zero-shot and few-shot LegalPegasus, and the fallback-assisted final output, separately on fixed case splits. Report wrong actors, negation/outcome reversals, unsupported statements, missing essential facts/reasons, and withheld sections, alongside the proposal's textual metrics.

Do not promote the earlier fine-tuned adapter: its small feasibility experiment did not demonstrate improvement. Do not call assumed approval actual review or report software-test counts as legal accuracy. The prepared review packet makes this next step concrete without fabricating annotations or repeatedly training on the same six examples.
