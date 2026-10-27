"""Record a negative reserved-case finding without tuning the frozen candidate."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path('data/private/reserved_evaluation_20261026')
    read = lambda name: json.loads((root / name).read_text(encoding='utf-8'))
    prediction, packet, run = read('prediction.json'), read('review_packet.json'), read('run_manifest.json')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert run['code_sha256'] == {p: sha(p) for p in run['code_sha256']}
    assert sha('models/legalbert_balanced_v2/adapter_model.safetensors') == run['adapter_sha256']
    modes = {role: section['status'] for role, section in prediction['delivery']['sections'].items()}
    assert modes == {'FACT': 'source_backed_draft', 'ISSUE': 'withheld', 'RULING': 'withheld'}
    assert not packet['formal_evaluation_eligible']
    assert all(s['human_review']['decision'] is None for s in packet['sections'])
    report = '''# Reserved-case evaluation result: Mayor v. Belen

## Outcome

The frozen candidate produced Facts but withheld Issue and Ruling. Only 1 of 3 final FIR sections is present. This is a completeness count, not an accuracy percentage. The result does not support claiming that the pipeline is reliably complete on other decisions.

The official Supreme Court E-Library decision was retrieved from https://elibrary.judiciary.gov.ph/thebookshelf/showdocs/1/46885. The run used text extracted from that HTML decision container, not an uploaded full-case PDF. Input-format differences matter: the bracketed footnote markers in this source expose a segmentation limitation. This run does not test the browser's PDF-upload path.

## Observed failures

- LegalBERT labeled the opening question as ISSUE (source sentence 1). The conservative Issue selector did not recognize its "crux of the controversy ... is whether" wording. The missing final Issue is therefore not explained by a missing ISSUE prediction on that sentence.
- The final disposition remains in prepared source sentence 81, joined to the preceding quoted passage after citation [19]. It is not a separate sentence beginning WHEREFORE. Quotation tracking also continues through that sentence and SO ORDERED. The terminal-order selector consequently withholds Ruling.
- No Issue or Ruling text was sent to LegalPegasus, and both raw outputs are empty. Do not describe those missing outputs as generated hallucinations. The raw Facts paraphrase failed the strict complete-verbatim gate; final Facts uses source excerpts. That rejection alone is not proof that every raw statement is false.
- Missing sections are retained in the packet. No fabricated Issue, Ruling, human rating or legal-accuracy score fills the gaps.

These observations are AI-assisted diagnostics, not attorney findings. The pipeline was not changed or rerun in response to them. They must be disclosed if this case informs a later development revision.

## Evaluation status

Mayor was reserved from tuning but had a prior sentence-classifier diagnostic on 25 October 2026. This is a disclosed frozen reserved-case trial, not a pristine unseen final evaluation. The supplied reference has pending reviewer acceptance, and final annotation cells remain unfilled. Formal evaluation eligibility is false for the reasons in review_packet.json.

Code hashes and the active adapter hash were verified unchanged before and after inference. Reference text was loaded only after prediction.json had been saved. No training, checkpoint selection or pipeline update occurred in this evaluation.

## What can happen now

The user confirmed there are no additional untouched decisions/digests available for now. Preserve this result, complete source/reference and human review using REVIEW.md, and finish the existing validation-label review without changing labels to match model predictions. Report the result and limitations honestly in the thesis. A later change informed by this case must not be advertised as independently validated on the same case.

EVALUATION_HANDOFF.md lists the required review steps and exposure histories. Human decisions and dates stay blank until supplied. The existing workspace remains on the same frozen candidate.
'''
    (root / 'RESULTS.md').write_text(report, encoding='utf-8')
    names = ['mayor_source.html', 'source.json', 'prediction.json', 'review_packet.json',
             'run_manifest.json', 'exposure_registry.json', 'REVIEW.md', 'RESULTS.md', 'EVALUATION_HANDOFF.md']
    verification = {'case_id': 'GR151035', 'final_sections_present': 1, 'final_sections_total': 3,
        'formal_evaluation_eligible': False, 'human_ratings_complete': False, 'legal_accuracy_estimate': None,
        'code_and_adapter_unchanged': True, 'pipeline_tuning_performed': False,
        'additional_untouched_cases_available': False, 'availability_basis': 'User reply in this task',
        'files_sha256': {name: sha(root / name) for name in names}}
    (root / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in verification.items() if k != 'files_sha256'}, indent=2))


if __name__ == '__main__': main()
