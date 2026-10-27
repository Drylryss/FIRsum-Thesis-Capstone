# Frozen LegalPegasus experiment findings

Analysis completed October 23, 2026. The run artifacts retain their original `fewshot_experiment_20261022` folder name.

## Decision

Do not promote the tested three-example configuration. It introduced unrelated example facts into both target cases and repeated prompt instructions in both Issue outputs. Zero-shot generation also made serious factual errors. Neither configuration is established as a reliable abstractive FIR generator.

## Experiment

`experiment_fewshot.py` ran the frozen generator twice for each of two previously inspected development cases, Bicol and Macasaet. The same saved LegalBERT predictions and current evidence selection were used in both conditions. No weights were updated. Example cases were Makati, Real, and Mactan; none was a target case. The compact verbatim examples were explicitly marked pending attorney acceptance and were not installed in the live app.

The comparison is between source-only zero-shot input and instructions plus three examples. It does not isolate the effect of examples from instructions. These two development cases are not an independent accuracy benchmark. The existing attorney digests are style material, not gold evaluation references.

## Observed failures

| Target | Zero-shot generation | Three-example generation |
|---|---|---|
| Bicol | Ruling invented a California court and US securities-law proceedings; it also misrepresented the TRO. | Facts combined Campos, a PWU food stall, and Tudtud's property with the target dispute. Ruling added IPO/SEC allegations and Lot 988. Issue repeated instructions. |
| Macasaet | Issue invented a New York federal court and a project cost above $1 million. Ruling attributed the payment order to COA rather than the deciding Court. | Facts imported the SEC and food-stall dispute. Ruling assigned the fee payment to Campos and added fire and Lot 988 material. Issue repeated instructions. |

These are findings from direct inspection of saved model outputs against their source text, not attorney ratings. The raw model output remains available for audit.

All 12 generated blocks (2 cases × 2 configurations × 3 roles) failed the deliberately strict full-source-match gate. All final blocks therefore used source excerpts. Failure of that literal gate alone does not establish factual error: even a correct paraphrase would fail it. The specific errors above independently demonstrate problems in this run.

Bicol few-shot FACT and RULING each reached the first 256-token cap and completed after retry with a 384-token allowance. No final attempt remained capped. Measured zero-shot run times were 174.5 s and 105.9 s; few-shot times were 331.7 s and 169.5 s. These are local observations, not a speed benchmark: the live app's duplicate model process was stopped during the experiment to free memory.

## Validation and next decisions

62 automated tests passed, including target/example separation and candidate-provenance checks. The live accepted-exemplar requirement was not weakened. The original configuration and checkpoints remain in place, with source grounding enabled.

The proposal's assumption that frozen LegalPegasus will follow three digest examples is not supported by this tested prompt. Report that result honestly. Additional prompt experiments would need a fresh evaluation set and independent legal review before any claim of improvement. A change to supervised generator fine-tuning or a different model would require a methodology decision; it has not been made here.

LegalBERT retraining still requires corrected sentence annotations. No human review has been fabricated. The prepared 34-item review batch remains pending.

## Artifacts

- `data/private/fewshot_experiment_20261022/manifest.json`: input identities, implementation hashes and experimental status.
- `data/private/fewshot_experiment_20261022/report.json`: four completed runs, token counts and gate results.
- Per-case `*_zero_shot.json` and `*_experimental_few_shot.json`: raw generation, final source excerpts, and evidence audits.

Reproduce with a new output directory to preserve the original run:

```powershell
.\.venv\Scripts\python.exe experiment_fewshot.py --output data/private/fewshot_repeat
```
