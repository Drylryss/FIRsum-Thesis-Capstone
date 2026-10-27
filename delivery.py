"""Fail-closed delivery contract for source-backed drafts, not legal validation."""
from grounding import normalized, select_evidence
from core import ROLES
import re
import hashlib

PRESENTATION_VERSION = 'source-presentation-v14'


def trim_procedural_clutter(evidence, issue_text=''):
    """Omit complete routine-motion groups only in substantive fact narratives.

    Procedural issues and short evidence sets remain untouched. Originals are
    always retained in omitted_evidence by the delivery contract.
    """
    if len(evidence) < 8 or re.search(
            r'\b(?:procedure|procedural|jurisdiction|motion|injunction|consolidat\w*|'
            r'dismissal|prescription|due process|certiorari|appeal)\b', issue_text, re.I):
        return evidence
    events = [e for e in evidence if re.search(
        r'\b(?:sold|paid|purchased|deposited|constructed|co-owners|executed a deed)\b', e['text'], re.I)]
    if len(events) < 3:
        return evidence
    groups = []
    for item in evidence:
        if groups and item['sentence_id'] == groups[-1][-1]['sentence_id'] + 1:
            groups[-1].append(item)
        else:
            groups.append([item])
    removed = set()
    for group in groups:
        if len(group) < 2:
            continue
        first = re.sub(r'^\s*\d+\s+', '', group[0]['text'])
        routine_start = re.search(r'\bfiled a motion (?:to intervene|for reconsideration)\b', first, re.I)
        routine_only = all(re.search(r'\b(?:motion|reconsideration)\b', e['text'], re.I) for e in group)
        if routine_start and routine_only:
            removed.update(e['sentence_id'] for e in group)
    # Remove only the initial review announcement, not the lower-court outcome.
    first = evidence[0]
    if re.match(r'^\s*(?:Before this Court is a Petition for Review|For review on appeal by certiorari are the Decision)\b', first['text'], re.I):
        removed.add(first['sentence_id'])
    return [e for e in evidence if e['sentence_id'] not in removed]


def readable_evidence(role, evidence, disposition, issue_text=''):
    """Reorder the outcome and omit a narrow, self-contained procedural aside.

    Never rewrite party claims or invent transitions. Keep removed sentences
    in the audit. Short drafts are left intact.
    """
    chosen=list(evidence)
    if role == 'FACT':
        chosen = trim_procedural_clutter(chosen, issue_text)
        # Structural announcements carry no case facts. Retain their originals
        # in omitted_evidence, rather than rewriting the source.
        chosen = [e for e in chosen if not re.fullmatch(
            r'\s*(?:(?:The facts follow|The facts are as follows)\s*[.:]|The records disclose the following antecedent facts[^.!?]*:\s*\d*\.?)\s*', e['text'], re.I)]
        # Do not juxtapose a personal pronoun or an unnamed vendor group with
        # an unrelated earlier transaction. Keep omissions in the source audit.
        kept, dropped = [], set()
        for item in chosen:
            text = re.sub(r'^\s*\d+\s+', '', item['text'])
            prior_present = bool(kept and kept[-1]['sentence_id'] == item['sentence_id'] - 1)
            dangling = re.match(r'^\s*(?:She|He|They|The spouses-vendors|On the same day|On that day)\b', text, re.I)
            follows_dropped = item['sentence_id'] - 1 in dropped and re.search(
                r'\b(?:the motion|said motion|the complaint|that petition)\b', text, re.I)
            if (dangling and not prior_present) or follows_dropped:
                dropped.add(item['sentence_id'])
            else:
                kept.append(item)
        chosen = kept
    if role=='FACT' and len(chosen)>10:
        # A dated announcement + extension-of-time + TRO pair is an incidental
        # procedural aside only when all three consecutive sentences are here.
        # Remove the complete unit so "first/second resolution" never dangles.
        drop=set()
        for i in range(len(chosen)-2):
            a,b,c=chosen[i:i+3]
            if (b['sentence_id']==a['sentence_id']+1 and c['sentence_id']==b['sentence_id']+1
                    and re.search(r'issued two \(2\) Resolutions',a['text'],re.I)
                    and re.search(r'first Resolution.*motion for extension',b['text'],re.I)
                    and re.search(r'second Resolution.*temporary restraining order',c['text'],re.I)):
                drop.update((i,i+1,i+2))
        chosen=[e for i,e in enumerate(chosen) if i not in drop]
    if role=='RULING':
        orders=[e for e in chosen if e['sentence_id'] in disposition]
        reasons=[e for e in chosen if e['sentence_id'] not in disposition]
        # With the actual outcome and consecutive explanation present, a bare
        # disagreement adds no meaning and otherwise has no visible antecedent.
        if orders:
            reason_ids = {e['sentence_id'] for e in reasons}
            reasons = [e for e in reasons if not (re.fullmatch(r'\s*We disagree\.\s*', e['text'], re.I)
                       and e['sentence_id'] + 1 in reason_ids)]
        # Omit a redundant standalone announcement only when the full order
        # explicitly contains the same result. Keep qualifications and context.
        order_text = ' '.join(e['text'] for e in orders)
        if re.search(r'\bREVERSED\b', order_text):
            reasons = [e for e in reasons if not re.fullmatch(r'\s*We reverse\.\s*', e['text'])]
        # The formal closing remains last, after both the outcome and reasons.
        closing=[e for e in orders if re.fullmatch(r'SO\s+ORDERED\s*\.',e['text'].strip(),re.I)]
        chosen=[e for e in orders if e not in closing]+reasons+closing
    return chosen


def context_warnings(evidence, rows):
    """Expose gaps in excerpt context without inventing an antecedent."""
    positions = {row['id']: i for i, row in enumerate(rows)}
    included = {e['sentence_id'] for e in evidence}
    warnings = []
    for item in evidence:
        if item.get('selection') == 'final_disposition':
            continue
        text = item['text']
        # Strip only a citation prefix for detection, never from delivered text.
        text = re.sub(r'^\s*(?:\[\d+\]|\d+)\s*', '', text)
        dependent = re.match(r'\s*(?:They|It|These|Those|Such|In other words|The latter|The former|'
                             r'The first|The second|This (?:finding|conclusion|argument|resolution))\b', text, re.I)
        if not dependent:
            continue
        # Impersonal constructions have no missing pronoun antecedent.
        if re.match(r'\s*It\s+(?:is|was|matters|does not matter)\b', text, re.I):
            continue
        index = positions[item['sentence_id']]
        if index == 0 or rows[index - 1]['id'] not in included:
            warnings.append('Source sentence '+str(item['sentence_id'])+
                            ' may depend on omitted preceding context; review its antecedent.')
    return warnings


def prepare_delivery(result):
    from footnote_display import display_sentences
    display = display_sentences(result.get('sentences', []),result.get('preprocessing', {}).get('display_footnote_markers', []))
    source = {s['id']: s['text'] for s in result.get('sentences', [])}
    verification = result.get('verification', {})
    disposition = set(verification.get('disposition_sentence_ids', []))
    rows=result.get('sentences', [])
    recomputed = select_evidence(rows) if rows and all('role' in row for row in rows) else None
    source_issues=[]
    if len(source)!=len(rows):
        source_issues.append('Duplicate source sentence IDs make provenance ambiguous.')
    actual_hash=hashlib.sha256(' '.join(s['text'] for s in rows).encode('utf-8')).hexdigest()
    if verification.get('source_sha256') != actual_hash:
        source_issues.append('The source fingerprint is missing or does not match the recorded decision.')
    excluded=set(verification.get('excluded_separate_opinion_sentence_ids', []))
    sections, warnings = {}, []
    for role in ROLES:
        audit = verification.get('blocks', {}).get(role, {})
        evidence = audit.get('evidence', [])
        issues = list(source_issues)
        if not evidence:
            issues.append('No source evidence is available.')
        if any(e.get('sentence_id') not in source or e.get('text') != source.get(e.get('sentence_id')) for e in evidence):
            issues.append('Evidence does not match the recorded source sentences.')
        expected = '\n\n'.join(e.get('text', '') for e in evidence)
        if normalized(result.get('blocks', {}).get(role, '')) != normalized(expected):
            issues.append('Draft text differs from its source evidence.')
        ids = {e.get('sentence_id') for e in evidence}
        if len(ids)!=len(evidence):
            issues.append('Repeated evidence sentence IDs would duplicate the summary.')
        if ids & excluded:
            issues.append('Evidence includes an excluded separate opinion.')
        if role == 'RULING' and (not disposition or not disposition <= ids):
            issues.append('The complete identified disposition is not present.')
        if role == 'RULING' and recomputed is not None and disposition != set(recomputed['disposition_ids']):
            issues.append('Recorded disposition does not match the terminal current-court order.')
        if role == 'ISSUE' and recomputed is not None and not ids <= {e['sentence_id'] for e in recomputed['evidence']['ISSUE']}:
            issues.append('Issue evidence is not an attributable current legal question.')
        presented=[] if issues else readable_evidence(role,evidence,disposition,result.get('blocks', {}).get('ISSUE', ''))
        if not issues and not presented:
            issues.append('No self-contained evidence remains after presentation checks.')
        context = context_warnings(presented, rows)
        artifacts = ['Source sentence '+str(e['sentence_id'])+
                     ' contains a suspicious bracketed token in the source PDF; retained rather than guessed.'
                     for e in presented if re.search(r'\[[^\]\n]*\\[^\]\n]*\]', e['text'])]
        artifacts.extend('Source sentence '+str(e['sentence_id'])+
                         ' contains ambiguous characters in a monetary amount; verify the amount against the original document.'
                         for e in presented if re.search(r'(?:\bP|₱)[lI]\s*\d', e['text']))
        sections[role] = {'text': ' '.join(normalized(e['text']) for e in presented),
                          'evidence': presented,
                          'display_text': ' '.join(display[e['sentence_id']] for e in presented),
                          'display_cleanup': 'PDF-backed citation markers, recognized heading prefixes and publisher watermarks removed; narrow disposition spacing repaired. Unverified numbers and original evidence retained.',
                          'display_edits': [{'sentence_id': e['sentence_id'], 'original': e['text'],
                                             'display': display[e['sentence_id']]}
                                            for e in presented if normalized(e['text']) != display[e['sentence_id']]],
                          'omitted_evidence': [] if issues else [e for e in evidence if e not in presented],
                          'presentation': 'outcome_then_reasons' if role=='RULING' else 'source_order',
                          'status': 'withheld' if issues else 'source_backed_draft',
                          'context_review_required': True,
                          'context_warnings': context,
                          'source_artifact_warnings': artifacts,
                          'problems': issues}
        warnings.extend(role+': '+issue for issue in issues)
        warnings.extend(role+': '+warning for warning in context)
        warnings.extend(role+': '+warning for warning in artifacts)
        if role == 'RULING' and not issues and not ids - disposition:
            warnings.append('RULING contains the disposition only; supporting reasoning was not selected.')
    # A synthesized question is a separate delivery mode. Never relax the
    # literal-source gate, overwrite model output, or recover a corrupted audit.
    issue_audit = verification.get('blocks', {}).get('ISSUE')
    if (not source_issues and recomputed is not None and isinstance(issue_audit, dict)
            and not issue_audit.get('evidence') and not result.get('blocks', {}).get('ISSUE')
            and not recomputed['evidence']['ISSUE']
            and sections['RULING']['status'] == 'source_backed_draft'):
        from issue_synthesis import synthesize_issue
        selected = recomputed['evidence']['RULING']
        eligible = [e for e in sections['RULING']['evidence'] if e in selected]
        synthesis = synthesize_issue(rows, eligible)
        if synthesis:
            sections['ISSUE'].update(text=synthesis['text'], display_text=synthesis['text'],
                evidence=synthesis['evidence'], status='source_supported_synthesis',
                presentation='synthesized_question', synthesis=synthesis,
                display_cleanup='No cleanup; explicitly synthesized from the cited ruling sentences.',
                problems=[], context_warnings=[synthesis['scope']])
            warnings = [w for w in warnings if w != 'ISSUE: No source evidence is available.']
            warnings.append('ISSUE is a source-supported synthesis, not a quotation or LegalPegasus output. Review its scope against the full decision.')
    return {'case_id': result.get('case_id', ''), 'artifact_type': 'source_backed_draft',
            'review_status': 'pending', 'legal_accuracy_verified': False,
            'status': 'incomplete' if any(s['status']=='withheld' for s in sections.values()) else 'review_required',
            'notice': 'Source-backed draft, not attorney-validated. Check attribution, omitted context and completeness against the full decision.',
            'sections': sections, 'warnings': warnings,
            'source_sha256': verification.get('source_sha256'),
            'grounding_version': verification.get('version'), 'presentation_version': PRESENTATION_VERSION}
