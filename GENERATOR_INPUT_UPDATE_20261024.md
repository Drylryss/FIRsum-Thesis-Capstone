# Generator input and data eligibility update

LegalPegasus already received structurally selected evidence rather than all raw classifier buckets. The remaining input-quality gap was that PDF-confirmed footnote markers and section headings were removed only from the displayed final output, not from generator inputs.

Version 7 applies the same conservative PDF-backed cleanup to the selected generator evidence. It removes only uniquely contextualized markers identified from PDF geometry. It preserves raw source sentences, classifier labels, complete disposition evidence and the existing final source-matching gate. It does not repair ambiguous words by guessing or introduce attorney-reference text into inference.

Each role's generation audit now records the exact source text supplied to generation, its SHA-256 hash, source sentence identifiers, and whether PDF cleanup changed it. Decoder calls record actual input token counts and the model context limit. Context checks respect a model limit smaller than 1,024 positions. Beam generation explicitly disables sampling. No model weights were changed.

The training loader now rejects training or validation records explicitly marked `training_eligible: false`, preventing review-only correction candidates from being silently consumed after format conversion. This guard does not imply that older provisional data is attorney-approved.

All 102 project tests passed before activation, including source-preservation and training-eligibility regressions. A fresh sample run is evaluated separately below; engineering test counts are not a legal-accuracy score.

The live application remains a source-backed research draft. A genuinely unseen benchmark and accepted reference digests remain unavailable in the six-digest development collection. No independent accuracy or attorney acceptance is inferred.

## Fresh live sample result

The version 7 API processed `Sample Full Case.pdf` in 192.7 seconds on CPU. Artifacts are saved in `data/private/live_pipeline_tests/20261024_173131/`. All six mechanical checks passed: version, all roles present, exact source grounding, disposition preservation, one paragraph per role, and no claim of legal validation. All three exact generator-source hashes were independently recomputed successfully. Input lengths were 533, 42 and 302 tokens for Facts, Issue and Ruling, respectively, below the 1,024-token context limit; no call reached its output-token cap.

AI inspection against the supplied source found:

- Raw Facts still incorrectly identifies Bicol Medical Center and the Department of Health as the parties whose petition the Court of Appeals granted. Source sentences 25–26 identify the Intervenors. The source distinguishes that proceeding from the petition for review described in sentence 2.
- Raw Issue remains consistent with the explicit source question. Failure of a literal-match gate is not proof that its paraphrase is factually wrong.
- Raw Ruling now includes reversal of the Court of Appeals decision and the permanent restraining order, unlike the earlier saved version 5 raw output. It still repeats a finding, contains awkward grammar and omits the explicit grant of the petition. This is an observation on one familiar development case, not an accuracy estimate or controlled model comparison.
- All three final sections still use source excerpts. Their displayed text and evidence match the earlier version 6 browser export. The final disposition includes the grant, reversal and permanent restraining order. The suspicious source token `[BJ\1C]` remains flagged rather than silently guessed.
- LegalBERT still labels source sentence 137 (the reversal) as ISSUE with high confidence. The structural disposition safeguard retains it in the final Ruling. Classifier confidence therefore must not be presented as factual accuracy.

No weights, adapters or attorney-approval records were changed. The fresh run verifies the input cleanup and continued fallback protection; it does not establish independent legal reliability. The remaining evaluation requirement is a genuinely unseen, case-matched reviewed reference set. Runtime is a single observation, not a speed benchmark.
