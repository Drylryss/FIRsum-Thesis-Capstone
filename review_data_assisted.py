"""Materialize the assistant's explicit review decisions; not attorney approval."""
import hashlib
import json
from pathlib import Path

ROOT=Path('data/private/ai_data_review_20261023')
LABEL_NOTES={
0:'Reports petitioners’ account of conveyances and evidence, not a question for determination.',
1:'A party’s argument about the Statute of Frauds; do not teach it as a court holding or standalone issue.',
2:'Reports the party’s assertion about objections in the record.',
3:'Petitioners’ conclusion about waiver/admissibility, not the deciding court’s conclusion.',
4:'Supporting legal argument attributed to petitioner; preserve the attribution.',
5:'The pronoun refers to petitioner’s argument in the preceding annotation; retain that context.',
6:'A party distinguishes a precedent. This is supporting argument rather than an issue heading.',
7:'Reports the requested procedural relaxation and its asserted medical justification.',
8:'Express allegation by the spouses about BPI’s proof.',
9:'Context between the spouses’ arguments suggests this is their assertion, not a court finding; full paragraph still needs confirmation.',
10:'Express party claim and prayer for dismissal.',
19:'Mixed issue formulation and supporting argument. Retain ISSUE provisionally because it expressly identifies the challenged consolidation ruling; split the supporting argument later.',
20:'Direct formulation of alleged appellate error. Retain ISSUE provisionally; distinguish it from the following supporting argument.',
24:'Reports the spouses’ procedural argument. Under the proposed rubric, this is attributed procedural history rather than an issue heading.',
25:'Attributed party contention; adjacent annotation is not enough to establish continuous full-decision context.',
27:'Structural transition introducing facts, without a factual proposition.',
28:'Structural transition introducing discussion, not substantive reasoning itself.',
31:'Structural transition announcing further disposition, without stating the disposition.',
32:'An explicit pivotal question. Available neighboring annotations are sparse; speaker/context should still be checked against the full decision.'}

# Replacement source IDs were read against saved source text. Some IDs contain
# several merged sentences and remain unsuitable for training without splitting.
ALIGNMENTS=[
([2,3],'supported','Parties, stalls and location match.'),
([4,5,6,7,8],'realign','Old evidence omitted the date, fire spread, demand and refusal; use the direct factual sequence.'),
([9],'segment','Claims are supported but source ID 9 also contains prior-court orders; isolate the attributed claim/answer span.'),
([43],'realign','Use the deciding court’s express issue, not only party arguments and prior proceedings.'),
([44],'supported','Doctrinal elements are supported; keep their attribution as jurisprudence.'),
([44,45],'supported','Article and exclusion-of-human-agency passage are present; source IDs include additional text.'),
([45,46],'realign','Include the court’s explicit rejection of the fortuitous-event theory; remove unrelated prior-court material.'),
([47,48],'realign','Include the missing bare-allegation sentence and avoid unrelated procedural history.'),
([1,2,3],'realign','Old evidence selected precedent discussion rather than the ownership and 1949 acquisition background.'),
([7,8],'realign','Transfer and airport closure are directly supported; later demand/complaint is unnecessary here.'),
([9],'segment','Demand is supported, but this source ID also merges the subsequent complaint.'),
([10],'segment','MCIAA’s position is supported; isolate it from respondents’ account and the precedent reference.'),
([31],'segment','Witness account is supported but mixed with a prior precedent and additional arguments. Remove the dangling contrast or retain its antecedent.'),
([31],'segment','Argument is supported; isolate the MCIAA sentence and retain party attribution.'),
([16,31,32],'context','This is a presented assignment of error, not a holding that the claim actually violates the statute. Preserve the question and opposing context.'),
([32],'context','Supported but ends with “as follows”; join the explanatory quotation or remove that lead-in in a proposed target revision.'),
([32,33,34],'realign','Old evidence omitted the middle sentence and introduced unrelated Fery material; include the complete local quotation.'),
([36,37],'realign','Old evidence omitted the current-case affirmation and substituted an earlier-case affirmation.'),
([37],'context','Statutory quotation is supported; attach Article 1190 and the current-case application.'),
([38],'context','Supported quotation needs its Article 1190/1189 antecedent, not a context-free training fragment.'),
([38],'segment','Reciprocal restitution and interest are supported; isolate the complete current-case application.'),
([38],'context','Supported, but “likewise” needs the preceding restitution obligation; keep together.'),
([39],'supported','Rights to fruits and interest match the source; retain both sides of the statement.'),
([40],'supported','Appreciation-in-value statement and statutory attribution match.'),
([3,4],'context','Supported, but the target ends with “as follows”; merge the fee provision or remove the dangling introduction.'),
([4],'context','Contract quotation is supported; identify the contract and parties in the input.'),
([5],'exclude','Introductory sentence alone is not a useful summary target; merge with the schedule or omit from standalone training.'),
([4,6],'realign','Use the fee-definition antecedent and the first installment; unrelated later installments are unnecessary.'),
([14],'realign','Use the completion-and-acceptance clause directly; avoid unrelated installment and reasoning fragments.'),
([16],'supported','Escalation payment and recipient match.'),
([17],'source_check','The requested P219,302.47 is transcribed from the source, but 7% of P3,148,198.26 is P220,373.8782. Do not silently “correct” the judicial text; flag the inconsistency.'),
([18],'supported','PTA’s denial and its stated reason match; preserve it as PTA’s position, not the court’s holding.'),
([22],'realign','The explicit issue alone supports the question; remove the answer from the ISSUE training input.'),
([22],'realign','The same explicit issue supports this subquestion; keep both questions together rather than duplicate the case.'),
([26,27,28],'realign','Old input omitted the periodic-payment basis; retain all three contractual bases and repair quotation-boundary artifacts.'),
([29,30,31,32,33],'realign','Old input omitted flexibility/equity and intermediate reasoning; use the complete local explanation.'),
([34,35],'realign','Old input substituted PTA’s denial for the court’s statement; add the actual court sentence. Remove the Lawphil watermark only in a documented target revision.'),
([36],'supported','Built-in escalation interpretation matches.'),
([39,40],'realign','Old input used the opposing party’s argument for the court’s concession; use both actual court sentences.'),
([41,42,43],'source_check','Substance matches; “Article 11 59” is a source/extraction artifact. Verify against the original page before normalizing to 1159.')]


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    labels=json.loads(Path('data/private/attorney_references_20261022/review/priority_labels.json').read_text(encoding='utf-8'))
    changes=set(range(11))|{24,25}
    reviewed=[]
    for i,row in enumerate(labels):
        proposed='FACT' if i in changes else row['current_provisional_label']
        reviewed.append({**row,'proposed_label':proposed,'ai_review_status':'proposed_change' if i in changes else 'provisional_retain',
            'reason':LABEL_NOTES.get(i,'Explicit legal question or formal assignment of error; retain ISSUE under the proposed rubric.'),
            'review_basis':'sentence_and_available_neighboring_annotations_only',
            'reviewer_type':'AI_assistant','attorney_validated':False,'final_label':None,
            'needs_full_source_check':i in {5,6,9,19,20,25,31,32},
            'training_eligible':False})
    candidates=json.loads(Path('data/private/alignment_20261023/training_candidates.json').read_text(encoding='utf-8'))
    assert len(candidates)==len(ALIGNMENTS)==40
    aligned=[]
    for row,(ids,decision,reason) in zip(candidates,ALIGNMENTS):
        path=Path(row['source_artifact']);doc=json.loads(path.read_text(encoding='utf-8'))
        byid={s['id']:s for s in doc['sentences']}
        aligned.append({'unit_id':row['unit_id'],'case_id':row['case_id'],'split':row['split'],'role':row['role'],
            'original_target':row['target'],'original_source_sentence_ids':row['source_sentence_ids'],
            'proposed_source_sentence_ids':ids,'proposed_evidence':[byid[i] for i in ids],
            'decision':decision,'reason':reason,'reviewer_type':'AI_assistant','attorney_validated':False,
            'training_eligible':False,'source_artifact':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for filename,records in [('label_review.json',reviewed),('alignment_review.json',aligned)]:
        (ROOT/filename).write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    summary={'priority_labels_reviewed':len(reviewed),'proposed_label_changes':len(changes),
        'alignment_candidates_reviewed':len(aligned),'alignment_decisions':{k:sum(r['decision']==k for r in aligned) for k in sorted({r['decision'] for r in aligned})},
        'canonical_data_modified':False,'models_retrained':False,'attorney_validation':False,
        'scope':'34 priority labels and 40 alignment candidates, not all 570 sentence labels or all six full digests'}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# AI-assisted data review','',summary['scope']+'.','',
        'Proposed rubric: FACT includes attributed party positions and procedural history, not an assertion that every allegation is true. ISSUE includes explicit questions and formal assignments of error; supporting arguments alone are FACT. RULING is the deciding court’s reasoning/disposition; OTHER is structural text. Mixed sentences require segmentation. This rubric needs project approval before retraining.','',
        'No original digest, source, canonical label, split or model was changed. AI review is not attorney validation. Source checks and segmentation remain open.','', '## Label decisions','']
    for r in reviewed:lines += ['### '+r['sentence_id'],'',r['text'],'',r['current_provisional_label']+' → '+r['proposed_label']+': '+r['reason'],'']
    lines+=['## Digest alignment decisions','']
    for r in aligned:lines+=['### '+r['unit_id'],'','Target: '+r['original_target'],'',r['decision']+': '+r['reason'],'','Proposed source IDs: '+', '.join(map(str,r['proposed_source_sentence_ids'])),'']
    (ROOT/'REVIEW_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
