# Verification

Implementation checks on October 18, 2026:

- All Python sources compile.
- Both training CLIs accept `--help` without loading ML packages or downloading models.
- Eight automated tests pass: case split leakage, full token-window coverage, legal abbreviation splitting, prompt source boundaries, HTML/status routes, empty/unsupported inputs, mocked inference response, and model-unavailable errors.
- API tests required execution outside the Windows sandbox because its local socket restriction prevented creation of the asyncio event loop.

Actual LegalBERT LoRA training has now completed for eight epochs on the imported provisional labels. Validation and test classification metrics were computed, and the trained adapter passed reload and inference through the application's classifier. That check exposed a missing batch dimension in single-sentence inference, which was corrected and rechecked. See `TRAINING_RESULTS.md` for results and limitations. The restarted app reports classifier_ready=true and classifier_provisional=true. All eight API/core tests pass after integration.

Still not verified: GPU execution, end-to-end summary metrics, or human usability/accuracy evaluation. API inference tests use a mocked engine and do not establish model quality.

Real end-to-end execution was subsequently tested with the user's `Case Input .pdf` (G.R. No. 83748). A missing protobuf dependency prevented Pegasus tokenizer conversion; protobuf 4.25.9 was installed and added to requirements. After downloading the 2.28 GB generator checkpoint, the PDF request returned HTTP 200 and its raw result was saved to `runs/case_input_result.json`. The input yielded 4 readable pages and 11,023 extracted characters. This verifies execution, not summary quality: ISSUE was empty and the generated ruling included an unsupported Boracay resort description. The source describes a Zamboanga golf project. Treat this output as an unsuccessful quality example requiring model/data improvements and expert review.

All nine API/core tests pass after the dependency fix. Browser checks also verify that the actual server error is displayed in output panels instead of a generic failure message.

The full `requirements.txt` is installed in `.venv` (Python 3.10.11, PyTorch 2.5.1 CPU). `pip check` reports no broken requirements. All dependency imports pass, and a tiny randomly initialized BERT LoRA model passes backward computation and adapter save/reload; a tiny random Pegasus model generates successfully. This validates software compatibility, not actual thesis weights or model quality. All eight automated tests pass in the completed environment. `requirements-lock.txt` records installed versions.

Model downloads use project-local `.cache/huggingface` by default. The reported classifier metrics are against provisional labels, not a completed legal-accuracy or usability study.

October 19, 2026: re-imported the latest workbook to `data/revision_20261019`; selected model inputs and case splits were unchanged, while source row positions differed. Balanced weighting, ISSUE=30 weighting, and fixed-dataset loss normalization were evaluated on validation only. None improved ISSUE recall; the original checkpoint remains the default. All 13 API/core/weighting tests pass, including a check that dataset normalization preserves weight effects across minibatch partitions. See `ISSUE_WEIGHT_RESULTS.md` for the comparison and reproduction settings.

The three-stage output interface was tested in headless Microsoft Edge using synthetic API responses: stage navigation, OTHER sentence visibility, preservation of Pegasus output after final edits, live final preview, JSON export of both original/edited text, and mobile width all pass. These browser checks verify UI behavior; they do not establish model accuracy.
