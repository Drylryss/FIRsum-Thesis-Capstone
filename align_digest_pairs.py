"""Offline source/target alignment candidates, never evaluation-time retrieval.

Lexical alignment proposes evidence for human inspection; it is not entailment
or legal review. Only training cases produce candidate training records.
"""
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

from core import sentences, validate_splits


def words(text):
    return re.findall(r'[a-z]+|\d+(?:[,.]\d+)*', text.casefold())


def coverage(target, source):
    a,b=Counter(words(target)),Counter(words(source))
    return sum((a & b).values()) / max(1,sum(a.values()))


def numbers(text):
    return set(re.findall(r'(?<!\w)\d+(?:[,.]\d+)*(?:%)?',text))


def fragment_warning(text):
    """Short answers/list items need surrounding target context, not standalone training."""
    return len(words(text)) < 8 or bool(re.fullmatch(r'[xX\s.]+', text.strip()))


def neighboring_context(evidence, source):
    ids = {r['id'] for r in evidence}
    positions = {j for i, row in enumerate(source) if row['id'] in ids
                 for j in (i-1, i+1) if 0 <= j < len(source)}
    return [source[j] for j in sorted(positions) if source[j]['id'] not in ids]


def target_units(text, token_count, limit=256):
    """Keep exact paragraph/sentence text; oversized units remain review-only."""
    units=[]
    for paragraph in text.split('\n\n'):
        if not paragraph.strip():continue
        if token_count(paragraph)<=limit:
            units.append(paragraph)
        else:
            units.extend(sentences(paragraph))
    return units


def propose(unit, source_rows, token_count, budget=960):
    ranked=sorted(range(len(source_rows)),key=lambda i:(-coverage(unit,source_rows[i]['text']),i))
    selected=[]
    for i in ranked:
        if len(selected)>=3:break
        if coverage(unit,source_rows[i]['text'])<.15:continue
        trial=sorted(selected+[i])
        if token_count(' '.join(source_rows[j]['text'] for j in trial))<=budget:
            selected=trial
        if coverage(unit,' '.join(source_rows[j]['text'] for j in selected))>=.98:break
    return [source_rows[i] for i in selected]


def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('nsi319/legal-pegasus')
    count=lambda text:len(tok.encode(text))
    root=Path('data/private/alignment_20261023');root.mkdir(parents=True,exist_ok=True)
    old=json.loads(Path('data/private/pegasus_feasibility_20261023/paired_data.json').read_text(encoding='utf-8'))
    validate_splits(old)
    audits=[];candidates=[];review=[]
    for pair in old:
        name=pair['name']
        path=(Path('runs/grounding_source_only_20261022/macasaet_result.json') if name=='macasaet' else
              Path('data/private/attorney_references_20261022/pipeline_results')/(name+'.json'))
        document=json.loads(path.read_text(encoding='utf-8'))
        source=document['sentences']
        unit_results=[]
        for n,unit in enumerate(target_units(pair['summary'],count)):
            evidence=propose(unit,source,count)
            joined=' '.join(r['text'] for r in evidence)
            audit={'case_id':pair['case_id'],'name':name,'split':pair['split'],'role':pair['role'],
                   'unit_id':f"{pair['case_id']}_{pair['role']}_{n+1}",'target':unit,
                   'old_input_lexical_coverage':coverage(unit,pair['text']),
                   'candidate_lexical_coverage':coverage(unit,joined),
                   'target_numbers_missing_from_candidate':sorted(numbers(unit)-numbers(joined)),
                   'target_tokens':count(unit),'candidate_tokens':count(pair['role']+': '+joined),
                   'source_sentence_ids':[r['id'] for r in evidence],
                   'fragment_requires_context':fragment_warning(unit),
                   'neighboring_source_context':neighboring_context(evidence,source),
                   'proposed_evidence':evidence,'review_status':'pending','semantic_support_verified':False,
                   'approval_basis':'user_assumption_only_no_recorded_alignment_review'}
            unit_results.append(audit)
            if pair['split']=='train':
                review.append(audit)
                if evidence and not audit['fragment_requires_context'] and audit['candidate_tokens']<=960 and audit['target_tokens']<=256:
                    candidates.append({**audit,'text':pair['role']+': '+joined,'summary':unit,
                                       'usage':'training_alignment_candidate_only_not_approved',
                                       'source_artifact':str(path),
                                       'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        audits.append({'name':name,'case_id':pair['case_id'],'split':pair['split'],'role':pair['role'],
                       'old_selected_source_sentences':pair['source_selected'],
                       'old_available_role_sentences':pair['source_candidates'],
                       'units':unit_results})
    # Validation alignment is audit-only. Do not export an oracle validation input.
    assert all(r['split']=='train' for r in candidates)
    (root/'training_candidates.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'coverage_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2),encoding='utf-8')
    review.sort(key=lambda r:(r['old_input_lexical_coverage'],r['unit_id']))
    lines=['# Training alignment review','',
           'All alignments below are machine suggestions. Lexical overlap is not proof of factual or legal support.',
           'Confirm speaker, negation, facts, legal issue and context against the full decision. Retain case splits.',
           'Short target units are provisional training units, not newly attorney-authored digest paragraphs. Inspect list fragments and antecedents.', '']
    for row in review:
        lines+=['## '+row['unit_id'],'',f"Prior input token overlap: {row['old_input_lexical_coverage']:.1%}. Candidate overlap: {row['candidate_lexical_coverage']:.1%}. These are lexical diagnostics only.",'',
                'Target:',row['target'],'','Suggested source evidence:','']
        lines += [f"- Sentence {e['id']} (raw label {e['role']}): {e['text']}" for e in row['proposed_evidence']]
        lines += ['', 'Fragment needing target context: '+str(row['fragment_requires_context']), '',
                  'Adjacent source context (not included in candidate input):','']
        lines += [f"- Sentence {e['id']} ({e['role']}): {e['text']}" for e in row['neighboring_source_context']]
        lines += ['', 'Reviewer: ______  Supported / edit / reject: ______  Missing context or corrections: ______','']
    (root/'TRAINING_ALIGNMENT_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    units=[u for a in audits for u in a['units']]
    summary={'prior_role_pairs':len(old),'training_candidate_units':len(candidates),
             'training_units_not_exported':sum(u['split']=='train' for u in units)-len(candidates),
             'fragment_units_review_only':sum(u['fragment_requires_context'] for u in units),
             'units_without_candidate_evidence':sum(not u['proposed_evidence'] for u in units),
             'candidate_overlap_improved':sum(u['candidate_lexical_coverage']>u['old_input_lexical_coverage'] for u in units),
             'candidate_overlap_worsened':sum(u['candidate_lexical_coverage']<u['old_input_lexical_coverage'] for u in units),
             'unique_training_cases':len({r['case_id'] for r in candidates}),
             'new_reviewed_cases_added':0,'validation_candidate_inputs_exported':False,
             'unreviewed_candidates_not_used_for_training':True,
             'units_with_old_lexical_coverage_below_0_8':sum(u['old_input_lexical_coverage']<.8 for a in audits for u in a['units']),
             'total_audited_units':sum(len(a['units']) for a in audits),
             'note':'More units are not more independent cases. No semantic accuracy estimate.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
