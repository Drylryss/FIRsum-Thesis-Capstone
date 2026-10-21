# Dataset contracts

The supplied consolidated workbook has been converted to `sentences.jsonl` for a provisional development run. `import_report.json` records the source hash, selected rows, exclusions, class counts and limitations; `split_manifest.json` freezes case assignments. The 551 selected rows come from `To be Fed`, matched back to `Tagged Sentences` for provenance. All final labels were blank, so suggested labels remain explicitly provisional. The workbook is unchanged.

The allocation is 381 train / 112 validation / 58 test sentences across 6 / 2 / 2 cases. This development split differs from the proposal's intended larger validation allocation. Only two OTHER examples occur in training, one in validation, and none in test. These data cannot establish reliable OTHER performance. Future generator exemplars must exclude test cases.

Do not treat the schema examples below as annotations or reference digests. Keep case IDs consistent across every dataset and maintain a master split manifest. Never reuse a case in train, validation, test, or exemplar splits. Review duplicates and related/consolidated decisions together before freezing splits.

`sentences.jsonl`: one manually annotated sentence per line:

```json
{"case_id":"CASE-001","split":"train","text":"An attorney-annotated sentence.","label":"FACT"}
```

Labels: FACT, ISSUE, RULING, OTHER. Provide all four in train; use separate validation cases. Test and exemplar rows are excluded by training scripts. The proposal targets 500–1,000 annotated sentences and 4–5 validation decisions. The proposal's split description does not explicitly allocate separate classifier training cases; finalize that allocation with the adviser before training.

`exemplars.json`: a JSON array of 2–3 fixed accepted style exemplars (three for the proposal), separate from test cases:

```json
[{"case_id":"EXEMPLAR-001","split":"exemplar","attorney_accepted":true,"blocks":{"FACT":"Reviewed facts paragraph","ISSUE":"Reviewed legal question?","RULING":"Reviewed ruling and legal basis"}}]
```

`generator.jsonl`: optional experimental paired supervision, one role cluster and matching accepted summary per line:

```json
{"case_id":"TRAIN-001","split":"train","role":"FACT","text":"Source sentences from the decision.","summary":"Matched attorney-reviewed facts summary."}
```

Exemplar-only digest materials must not become supervised targets without matched source data and an approved research-method revision. Generator pairs over 1024 prompt tokens or 256 target tokens are rejected, never truncated.

Reference JSON for evaluation uses the exemplar object schema with `split: "test"` and matching case ID. Keep reference digests out of inference prompts.
