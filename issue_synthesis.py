"""Narrow question synthesis from an attributable current-court conclusion.

This is an explicit template transformation, not verbatim extraction, model
generation, entailment verification, or a guarantee of complete issue coverage.
"""
import re
from grounding import CURRENT_HEADING


ENTITLEMENT = re.compile(
    r'^\s*(?:\d+\s+)?(?P<subject>The (?:petitioners|respondents)(?:, [^.;?]{1,100},)?) '
    r'are not (?:the injured parties )?entitled to (?P<remedy>[^.;?]{5,180})\.\s*$')


def synthesize_issue(rows, selected_ruling):
    heading = next((i for i, row in enumerate(rows) if CURRENT_HEADING.match(row['text'])), None)
    if heading is None:
        return None
    positions = {row['id']: i for i, row in enumerate(rows)}
    available = {e['sentence_id']: e for e in selected_ruling
                 if e.get('selection') == 'court_reasoning_context'}
    candidates = []
    for sid, item in available.items():
        position = positions.get(sid, -1)
        if position <= heading or position + 1 >= len(rows):
            continue
        match = ENTITLEMENT.fullmatch(item['text'])
        following = available.get(rows[position + 1]['id'])
        if not match or not following:
            continue
        subject = match['subject'][0].lower() + match['subject'][1:]
        candidates.append({'text': 'Whether ' + subject + ' are entitled to ' + match['remedy'] + '.',
            'evidence': [item, following], 'method': 'entitlement_conclusion_to_question_v1',
            'transformation': 'Recast the selected negative entitlement conclusion as an open question; preserve its subject and remedy. The adjacent selected sentence supplies context.',
            'scope': 'One substantive question inferred from the holding; other procedural or substantive issues may be omitted.',
            'legal_accuracy_verified': False})
    return candidates[0] if len(candidates) == 1 else None
