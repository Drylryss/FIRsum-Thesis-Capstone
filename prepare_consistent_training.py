"""Build a separate, source-auditable AI training revision. Never edit evaluation rows."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

INPUT = Path('data/revision_20261027_experiment/sentences.jsonl')
OUT = Path('data/revision_20261027_consistency')
SOURCES = Path('data/private/additional_digest_development_20261024_v3/sources')

# Exact source spans inspected with the enclosing procedural/analysis context.
# Paragraph + prefix selects one sentence without guessing abbreviation boundaries.
SELECTIONS = {
 '225562': [
  (8, 'FACT', "The herein respondent, Bank of the Philippine Islands (BPI), issued a credit card in William's name, with Irene as the extension card holder.", 'Narrated transaction; not a judicial legal conclusion.'),
  (10, 'FACT', None, 'Complaint filing is procedural history.'),
  (18, 'FACT', None, 'Earlier appellate disposition is procedural history.'),
  (22, 'FACT', None, 'Failure to file a comment is a procedural event.'),
  (24, 'RULING', None, 'Present Court announces its disposition and modifications.'),
  (30, 'RULING', 'The Spouses Louh failed to show that they exerted due diligence in timely pursuing their cause so as to entitle them to a liberal construction of the rules, which can only be made in exceptional cases.', 'Present Court evaluates diligence; not the party request for relaxation.'),
  (34, 'RULING', "BPI cannot be made to pay for the Spouses Louh 's negligence, omission or belated actions.", 'Present Court rejects consequences sought by the party.'),
  (35, 'RULING', 'Be that as it may, the Court finds excessive the principal amount and attorneys fees awarded by the RTC and CA.', 'Present Court evaluates earlier relief.'),
  (48, 'RULING', None, 'Present Court changes the attorney fee award.'),
  (60, 'OTHER', None, 'Judicial attestation, not a substantive disposition.'),
  (63, 'OTHER', None, 'Constitutional certification of consultation, not legal analysis of this dispute.'),
  (66, 'OTHER', None, 'Justice assignment footnote; administrative metadata.'),
 ],
 '141877': [
  (5, 'FACT', None, 'Family relationship narrative; the word issues means children here.'),
  (6, 'FACT', None, 'Family history; issue does not mean a legal question.'),
  (8, 'FACT', None, 'Narrated partition transaction.'),
  (14, 'FACT', None, 'Attributed party assertion about a sale.'),
  (16, 'FACT', None, 'Narrated waiver during proceedings.'),
  (41, 'FACT', None, 'Earlier appellate remand, not the present Court disposition.'),
  (49, 'RULING', None, 'Present Court announces which party prevails.'),
  (50, 'RULING', 'Following Article 1405 of the Civil Code, 17 the contracts which infringed the Statute of Frauds were ratified by the failure to object to the presentation of parol evidence, hence, enforceable.', 'Present Court applies the rule to the evidence.'),
  (57, 'RULING', None, 'Present Court expressly disagrees with the appellate legal conclusion.'),
  (59, 'RULING', 'In proving the fact of partial or total performance, oral evidence may be received as what the trial court in the case at bar did.', 'Present Court adopts a legal evidentiary rule.'),
 ],
 '145441': [
  (51, 'RULING', 'Hence, the technical difference between Civil Case No. 53967 and LRC Case No. R-3951 must be disregarded in order to promote the ends of justice.', 'Court answer to consolidation argument; not the argument itself.'),
  (54, 'RULING', 'While indeed the two cases demonstrate palpable similarities, the Court of Appeals overlooked essential differences that would render the Barican doctrine inapplicable to the instant case.', 'Present Court rejects the earlier court reasoning.'),
  (63, 'RULING', 'The elements of novation are patently lacking in the instant case.', 'Present Court legal application, contrasted with party arguments.'),
  (64, 'RULING', 'Clearly, there is no mutual consent to replace the old mortgage contract with a new obligation.', 'Present Court conclusion on novation.'),
  (73, 'RULING', 'We however sustain the award of moral damages.', 'Present Court partially adopts earlier relief.'),
  (80, 'OTHER', None, 'Concurrence signature list, not substantive reasoning.'),
 ]
}

def key(text):
    return re.sub(r'\W+', '', text).lower()

def main():
    rows = [json.loads(s) for s in INPUT.read_text(encoding='utf8').splitlines()]
    protected = [r for r in rows if r['split'] != 'train']
    cases = {r['case_id'].split('_')[0][2:]: r['case_id'] for r in rows if r['split']=='train'}
    additions, holds = [], []
    # This supplementary contention is not explicitly introduced as an assignment
    # in its paragraph. Avoid teaching a disputed ISSUE/FACT distinction as certain.
    for r in rows:
        if r['split']=='train' and r['text'].startswith('Petitioner also contends that the Court of Appeals committed reversible error'):
            path = SOURCES/'145441.json'
            raw = json.loads(path.read_text(encoding='utf8'))['raw_text']
            holds.append({'record':r, 'reason':'Supplementary error contention outside the formal assignment list; ISSUE versus supporting FACT is ambiguous. Excluded, not forcibly relabeled.', 'source_file':str(path), 'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'source_context':raw.split('\n\n')[52], 'human_validated':False})
    result = [r for r in rows if r not in [h['record'] for h in holds]]
    for docket, specs in SELECTIONS.items():
        path = SOURCES/(docket+'.json')
        raw = json.loads(path.read_text(encoding='utf8'))['raw_text']
        paragraphs = raw.split('\n\n')
        for index, label, span, reason in specs:
            paragraph = paragraphs[index]
            text = paragraph if span is None else span
            assert paragraph.count(text)==1 and raw.count(text)==1, (docket,index)
            # Prevent adding a source spelling/citation variant of an existing unit.
            assert all(key(text) != key(r['text']) for r in result+additions)
            start = raw.index(text)
            additions.append(dict(case_id=cases[docket], split='train', text=text, label=label,
                sentence_id=f'{cases[docket]}_CONSISTENCY_{len(additions)+1:03}',
                label_status='ai_source_annotated_user_authorized', reviewer_type='AI_assistant',
                human_validated=False, attorney_validated=False, final_label=None, reviewer=None,
                training_eligible=True, source_file=str(path),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_span=[start,start+len(text)], source_paragraph=index,
                source_context='\n\n'.join(paragraphs[max(0,index-1):index+2]), annotation_reason=reason))
    result += additions
    assert [r for r in result if r['split']!='train']==protected
    assert {r['case_id'] for r in result if r['split']=='train'}==set(cases.values())
    groups = {}
    for r in result:
        if r['split']=='train': groups.setdefault(key(r['text']),set()).add(r['label'])
    assert all(len(labels)==1 for labels in groups.values()), 'Contradictory duplicate training labels'
    OUT.mkdir(exist_ok=False)
    (OUT/'sentences.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf8')
    manifest = {'input_sha256':hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        'output_sha256':hashlib.sha256((OUT/'sentences.jsonl').read_bytes()).hexdigest(),
        'counts':dict(Counter(r['label'] for r in result if r['split']=='train')),
        'added':dict(Counter(r['label'] for r in additions)), 'held':len(holds),
        'validation_unchanged':True, 'test_unchanged':True, 'new_training_cases':False,
        'human_validated':False, 'review_scope':'Targeted role-contrast additions and one ambiguous existing ISSUE; not full attorney review.'}
    for name, obj in [('additions',additions),('holds',holds),('manifest',manifest)]:
        (OUT/(name+'.json')).write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps(manifest))

if __name__=='__main__': main()
