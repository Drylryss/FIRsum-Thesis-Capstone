"""Conservative evidence selection and literal grounding, separate from ML labels.

This is a source-excerpt fallback, not an entailment model or a claim that quoted
material is legally correct. It never changes a classifier prediction. A
paraphrase that cannot be verified literally is retained only as raw generation.
"""
import hashlib
import re
from collections import Counter

from core import ROLES
from source_structure import OPINION_HEADING

VERSION = 'source-grounding-v13'
STOP = set('a an the and or of to in on for by with as at is are was were be been '
           'this that these those it its their they from which whether not no '
           'court petition petitioner petitioners respondent respondents shall '
           'would could should may has have had issue case said also'.split())
ISSUE = re.compile(r'\b(?:the\s+)?(?:single\s+|sole\s+|main\s+|principal\s+)?'
                   r'(?:issues?|questions?)(?:\s*:|\s+(?:(?:of whether|to be|before|for|presented|raised|involved|submitted|is|are|here|now|to wit)\b))|'
                   r'\bThe crux of the controversy in this (?:petition|case)\b.{0,60}\bis whether\b', re.I)
DISPOSITION_START = re.compile(r'^\s*(?:\d+\s+)?(?:WHEREFORE\b|IN LIGHT OF ALL THE FOREGOING\b)', re.I)
CURRENT_HEADING = re.compile(r'^\s*(?:\d+\s+)?(?:THE RULING OF THE COURT|OUR RULING|THE COURT[\x27’]S RULING|Ruling of the Court(?=\s+(?:The Court|The petition|We\b)))\b')
STATUTORY = re.compile(r'\b(?:Civil Code|(?:Article|Art\.)\s+\d+|Jurisprudence defines|Sections?\s+\d+\s*(?:of\s+|,\s*)Rule\s+\d+|Section\s+\d+\s+of\s+the\s+[^:.]{0,80}Rules and Regulations)\b', re.I)
PARTY = re.compile(r'\b(?:petitioners?|respondents?|appellants?|appellees?|complainants?)'
                   r'(?:,?\s+through\s+[^,.;]{1,80},)?\s+'
                   r'(?:(?:also|likewise|further|thus)\s+)?'
                   r'(?:claim|argue|contend|assert|maintain|allege|submit|insist|emphasize|point out|ask|pray|aver)\w*\b', re.I)
QUOTED = re.compile(r'(?:as follows|(?:portion|order|decision|memorandum|certification).{0,100}\bread\b|'
                     r'\b(?:provides|provided|states|stated|reads|read)\s*:)', re.I)
SELF_FINDING = re.compile(r'\b(?:this Court\s+(?:finds|holds|rules|concludes|resolves|agrees)|'
                         r'we\s+(?:find|hold|rule|conclude|agree|disagree)|'
                         r'(?:convinces|persuades)\s+this Court|'
                         r'(?:petition|appeal)\s+(?:is|has)\s+(?:meritorious|without merit|no merit))\b', re.I)
TRAILER = re.compile(r'\b(?:concur|dissent|footnotes?|Associate Justice|Chief Justice|certification|attestation)\b', re.I)
OTHER_SPEAKER = re.compile(
    r'\b(?:trial court|lower court|Court of Appeals|RTC|CA|tribunal|'
    r'petitioners?|respondents?|appellants?|appellees?|earlier (?:case|decision))\b'
    r'.{0,120}\b(?:held|ruled|found|reasoned|said|declared|stated|concluded|'
    r'explained|wrote|argued|contended|submitted|asserted|maintained)\b', re.I)
QUESTION_START = re.compile(r'^\s*(?:(?:\(?[a-z0-9ivx]+[.)])\s*)?Whether\b', re.I)
APPLICATION_FINDING = re.compile(
    r'\b(?:It is established by (?:the )?evidence that|'
    r'No reversible error was committed|'
    r'(?:In|Thus,? in) the (?:present|instant) case|In the case at bar)\b', re.I)
APPLICATION_CONCLUSION = re.compile(
    r'\b(?:failed|insufficient|substantial compliance|no compelling need|'
    r'established by (?:the )?evidence|No reversible error)\b', re.I)


def current_finding(text):
    """Conservative textual anchors, not an entailment or legal accuracy test."""
    if PARTY.search(text) or attributed_elsewhere(text):
        return False
    if re.search(r'\bin\s+[A-Z][\w\x27-]*(?:\s+v\..{0,80})?,\s*this Court\b', text):
        return False
    return bool(SELF_FINDING.search(text) or CURRENT_HEADING.match(text) or
                re.match(r'^We have\b.{0,100}\b(?:record|evidence)\b.{0,100}\bnot\b.{0,80}\b(?:convincing evidence|proof)\b', text, re.I) or
                re.match(r'^\s*(?:\d+\s+)?(?:An analysis of the facts\b.{0,220}\bleads to the conclusion|Be that as it may, the Court finds\b)', text, re.I) or
                re.match(r'^\s*(?:\d+\s+)?(?:Based on the foregoing,\s*we\s+(?:affirm|reverse)|We\s+(?:reverse|delete the award)|To recap,)', text, re.I) or
                (APPLICATION_FINDING.search(text) and APPLICATION_CONCLUSION.search(text)))


def attributed_elsewhere(text):
    """Do not turn reported speech into a present-court finding."""
    finding = SELF_FINDING.search(text)
    prefix = text[:finding.start()] if finding else text
    return bool(OTHER_SPEAKER.search(prefix) or re.search(
        r'\baccording to\s+(?:the\s+)?(?:petitioners?|respondents?|trial court|Court of Appeals)\b', prefix, re.I))


def current_issue(text):
    match = ISSUE.search(text)
    if not match:
        return False
    prefix = text[:match.start()]
    # A finding about how another court resolved a question is reasoning, not
    # an issue heading. In particular, an embedded "question of whether" must
    # not move the facts/reasoning boundary to the end of the judgment.
    if re.search(r'\b(?:erred|misappreciated|disregarded|held|ruled|concluded|'
                 r'overstepped|not limiting|no (?:longer any|more) need)\b', prefix, re.I):
        return False
    return not (attributed_elsewhere(prefix) or re.search(
        r'\bin\s+[A-Z][\w\x27-]*(?:\s+v\..{0,80})?,\s*this Court\b', prefix))


def normalized(text):
    return re.sub(r'\s+', ' ', text).strip()


def terms(text):
    return Counter(w for w in re.findall(r'[a-z]{3,}', text.lower()) if w not in STOP)


def quote_ranges(rows):
    """Track explicit quoted blocks; uncertain blocks stay excluded from reasons."""
    result, start, statutory, contract_terms = set(), None, False, False
    testimony = False
    for i, row in enumerate(rows):
        text = row['text']
        if start is not None and re.match(
                r'^\s*(?:Dissatisfied|Aggrieved),\s+(?:the\s+)?(?:petitioners?|respondents?|appellants?)\s+(?:elevated|appealed|filed)\b', text, re.I):
            # Subsequent appellate history ends a quoted lower-court order.
            start, statutory, contract_terms, testimony = None, False, False, False
        if start is not None and CURRENT_HEADING.match(text) and re.match(r'^\s*(?:\d+\s+)?(?:THE|OUR)\b', text):
            start, statutory = None, False
        if start is not None and testimony and re.match(
                r'^\s*(?:\d+\s+)?Even if we were to\b', text, re.I):
            start, testimony = None, False
        if start is not None and contract_terms and (DISPOSITION_START.match(text) or
                re.match(r'^\s*(?:By the terms of the contract, the obligation|In sum, the decision)\b', text, re.I)):
            # Contract payment terms cannot pronounce the Court's judgment.
            start, contract_terms = None, False
        if start is not None and statutory and (DISPOSITION_START.match(text) or
                re.match(r'^\s*(?:In this case|In the present case|In the instant case),', text, re.I) or
                re.match(r'^\s*(?:\d+\s+|1âwphi1\s+)?This Court agrees\b', text, re.I) or
                re.match(r'^\s*(?:\d+\s+)?(?:\d+\.\s*)?Be that as it may,.{0,100}\bwe find and hold\b', text, re.I) or
                re.match(r'^\s*(?:\d+\s+)?(?:The (?:central|sole|main) issue is whether|Now to the substantive issues|To recap,|We (?:reverse|delete the award))', text, re.I)):
            # A statute/doctrinal enumeration cannot issue this case's order.
            # Do not apply this reset to a quoted court order or unknown speaker.
            start, statutory = None, False
        if start is not None and i > start and re.match(
                r'^\s*(?:\d+\s+)?(?:\([^)]*\)\s*)?On\s+(?:[A-Z][a-z]+\s+\d|\d+\s+[A-Z])', text):
            # A new dated narrative paragraph ends a quoted memorandum/order.
            start = None
        closing_inline_quote = (i > 0 and text.lstrip().startswith('"')
                                and rows[i - 1]['text'].count('"') % 2 == 1
                                and text.count('"') % 2 == 1)
        if start is None and (QUOTED.search(text) or (re.match(r'^\s*["“]', text) and not closing_inline_quote)
                              or (OTHER_SPEAKER.search(text) and re.search(r'["“]', text))):
            start = i
            statutory = bool(STATUTORY.search(text)) and not OTHER_SPEAKER.search(text)
            contract_terms = bool(re.search(r'\b(?:purchase price.{0,60}as follows|payable as follows)\b', text, re.I)) and not OTHER_SPEAKER.search(text)
            testimony = bool(re.search(r'\b(?:testified|ATTY\.)|\bQ\s+', text))
        if start is not None:
            result.add(i)
            # End markers are retained as quoted; the next sentence is eligible.
            if re.search(r'\bSO\s+ORDERED\s*\.|\((?:Emphasis|Underscoring|Citations?)[^)]*\)', text, re.I):
                start = None
            elif re.match(r'\s*(?:\d+\s+)?(?:Given|Issued|Signed)\s+this\s+\d', text, re.I):
                # A dated certification's closing line is not part of the
                # deciding court's reasoning that follows it.
                start = None
            elif re.search(r'[.!?]["”]\s*(?:\[\d+\]\s*)*(?:\([^)]*\))?\s*$', text):
                start = None
            elif i == start and '“' in text and text.count('“') == text.count('”'):
                # Inline scare quotes or a complete quotation do not open a
                # quotation spanning the rest of the decision.
                start = None
    return result


def disposition_indices(rows, quoted):
    """Use a terminal WHEREFORE ... SO ORDERED block, never a prior quoted order."""
    for start in reversed(range(len(rows))):
        if not DISPOSITION_START.match(rows[start]['text']):
            continue
        if start in quoted:
            continue
        ends = [i for i in range(start, len(rows))
                if re.search(r'\bSO\s+ORDERED\s*\.', rows[i]['text'], re.I)]
        if not ends:
            continue
        end = ends[0]
        after = rows[end + 1:]
        # A substantive continuation means this was not the final disposition.
        if after and not TRAILER.search(after[0]['text']):
            continue
        if start < len(rows) * .5 and len(rows) > 10:
            continue
        return list(range(start, end + 1))
    return []


def select_evidence(rows):
    """Select complete, attributable excerpts. Empty/ambiguous roles abstain."""
    # A dissent can have its own WHEREFORE and SO ORDERED. It must never replace
    # the main opinion merely because it appears last in the uploaded document.
    original_rows = rows
    boundary = next((i for i, row in enumerate(rows) if OPINION_HEADING.search(row['text'])), len(rows))
    rows = rows[:boundary]
    quoted = quote_ranges(rows)
    disposition = disposition_indices(rows, quoted)
    end = disposition[0] if disposition else len(rows)
    issues = [i for i, r in enumerate(rows[:end])
              if i not in quoted and ISSUE.search(r['text']) and not PARTY.search(r['text'])
              and current_issue(r['text'])
              and not re.search(r'\b(?:issue|question).{0,50}\bbefore\s+(?:the\s+)?(?:trial court|RTC|Court of Appeals|lower court)\b', r['text'], re.I)
              and re.search(r'\bwhether\b|\?', r['text'], re.I)]
    # An explicit question can occupy the next sentence after an issue heading.
    for i, r in enumerate(rows[:max(0, end - 1)]):
        if i not in quoted and current_issue(r['text']) and not PARTY.search(r['text']):
            j = i + 1
            while j < end and j not in quoted and QUESTION_START.match(rows[j]['text']):
                issues.extend([i, j])
                j += 1
    issues = sorted(set(issues))
    # An explicit current-petition issue list supersedes an earlier appeal's
    # list. Do not infer this boundary from case names or model predictions.
    current_lists = [i for i in issues if re.search(
        r'\bIn this petition\b.{0,100}\b(?:issues|questions)\b', rows[i]['text'], re.I)]
    if current_lists:
        issues = [i for i in issues if i >= current_lists[-1]]
    findings = [i for i, r in enumerate(rows[:end])
                if i not in quoted and current_finding(r['text'])]
    # Within the expressly marked current Court discussion, retain an applied
    # entitlement conclusion even without a first-person "we hold" formula.
    heading = next((i for i, r in enumerate(rows[:end]) if CURRENT_HEADING.match(r['text'])), None)
    if heading is not None:
        findings.extend(i for i in range(heading + 1, end)
                        if i not in quoted and rows[i]['role'] == 'RULING'
                        and not PARTY.search(rows[i]['text']) and not attributed_elsewhere(rows[i]['text'])
                        and re.search(r'\b(?:petitioners?|respondents?).{0,100}\b(?:are|is) not (?:the injured parties )?entitled\b', rows[i]['text'], re.I))
    findings = sorted(set(findings))
    reasoning_start = (max(issues) + 1 if issues else findings[0] if findings else end)
    facts_headings = [i for i, row in enumerate(rows[:end]) if i not in quoted and re.match(
        r'^\s*(?:The records disclose the following antecedent facts|The facts (?:are as follows|follow))\b', row['text'], re.I)]
    if issues and facts_headings and min(facts_headings) > max(issues):
        following_findings = [i for i in findings if i > min(facts_headings)]
        # Require an explicit current analysis boundary; otherwise abstain
        # rather than treating the entire decision as factual background.
        if following_findings:
            reasoning_start = min(following_findings)
    query = terms(' '.join(rows[i]['text'] for i in (issues or disposition)))

    def score(i):
        words = terms(rows[i]['text'])
        return sum(min(n, query[w]) for w, n in words.items()) / max(1, len(words)) ** .5

    fact_candidates = [i for i in range(reasoning_start)
                       if rows[i]['role'] == 'FACT' and i not in quoted and i not in issues
                       and not TRAILER.match(re.sub(r'^\s*\d+\s+', '', rows[i]['text']).lstrip())]
    complete_facts = [i for i in fact_candidates if not re.match(r'^\s*No\.\s*\d', rows[i]['text'])
                      and not re.search(r'\bPRAYER\b|\bPREMISES CONSIDERED\b', rows[i]['text'])
                      and not re.search(r':\s*(?:[a-z]|\d+)[.]\s*$', rows[i]['text'], re.I)]
    fact_candidates = complete_facts
    chosen_facts = fact_candidates[:1]
    # Reserve space for the underlying events, rather than letting repeated
    # procedural vocabulary crowd them out. Preserve the source's attribution
    # when selecting each side's position.
    first_procedure = next((i for i in fact_candidates if re.search(
        r'\b(?:filed|denied|dismissed|granted|appealed)\b', rows[i]['text'], re.I)
        and i != (fact_candidates[0] if fact_candidates else -1)), reasoning_start)
    background = [i for i in fact_candidates if i < first_procedure
                  and i not in chosen_facts and not PARTY.search(rows[i]['text'])]
    for i in sorted(background, key=lambda j: (-score(j), j))[:2]:
        if i not in chosen_facts:
            chosen_facts.append(i)
    # Reserve actual transactions/events before filling with procedural history.
    events = [i for i in fact_candidates if not PARTY.search(rows[i]['text'])
              and not re.search(r'\b(?:court|petition|motion|complaint|appeal|judgment)\b', rows[i]['text'], re.I)
              and re.search(r'\b(?:sold|purchased|paid|deposited|constructed|executed a (?:deed|real estate mortgage)|mortgaged|co-owners|borrowed|delivered)\b', rows[i]['text'], re.I)]
    for i in sorted(events, key=lambda j: (-score(j), j))[:4]:
        if i not in chosen_facts:
            chosen_facts.append(i)
    triggers = [i for i in background if re.search(
        r'\b(?:constructed|closed|terminated|refused|demolished|sold|seized|dismissed|cancelled|canceled)\b',
        rows[i]['text'], re.I)]
    if triggers and triggers[0] not in chosen_facts:
        chosen_facts.append(triggers[0])
    # Retain attributed inducement and its immediate explanatory context.
    allegations = [i for i in fact_candidates if re.search(
        r'\b(?:alleged|averred)\b.{0,220}\b(?:made .{0,20}believe|induced|misrepresented)\b', rows[i]['text'], re.I)]
    if allegations:
        i = allegations[0]
        chosen_facts.append(i)
        for j in range(i + 1, min(i + 4, reasoning_start)):
            if j not in fact_candidates or not re.match(
                    r'^(?:She|He|They|[A-Z][\w-]+ (?:further averred|readily agreed))\b', rows[j]['text']):
                break
            chosen_facts.append(j)
    for side in ('petitioners?|appellants?|complainants?', 'respondents?|appellees?'):
        positions = [i for i in fact_candidates if PARTY.search(rows[i]['text'])
                     and re.search(r'\b(?:'+side+r')\b', rows[i]['text'], re.I)]
        if positions:
            # The initial position supplies the premise; later arguments often
            # repeat procedural terms and win a relevance-only ranking.
            i = positions[0]
            if i not in chosen_facts:
                chosen_facts.append(i)
    # Keep the earliest denial and grant in the procedural history. These are
    # separate events; a relevance-only ranking can otherwise omit a reversal.
    for outcome in ('denied|dismissed', 'granted|reversed'):
        procedural = [i for i in fact_candidates if re.search(
            rf'\b(?:court|tribunal|commission|board)\b.{{0,100}}\b(?:{outcome})\b', rows[i]['text'], re.I)]
        if procedural and procedural[0] not in chosen_facts:
            chosen_facts.append(procedural[0])
    for i in sorted(fact_candidates, key=lambda j: (-score(j), j)):
        if i not in chosen_facts:
            chosen_facts.append(i)
        if len(chosen_facts) >= 10:
            break

    # A narrated agreement and its enumerated terms describe the transaction,
    # even when the classifier mistakes the contractual language for a ruling.
    # Keep this bounded and before the current Court's analysis.
    contract_facts = set()
    for i in range(reasoning_start):
        text = rows[i]['text']
        if (rows[i]['role'] == 'FACT' and not PARTY.search(text)
                and re.search(r'\b(?:executed|entered into)\b', text, re.I)
                and re.search(r'\bunder the following terms(?: and conditions)?\s*:', text, re.I)):
            contract_facts.add(i)
            for j in range(i + 1, min(i + 7, reasoning_start)):
                if not re.match(r'^\s*(?:\d+[.)]?\s*)?(?:The (?:selling price|sum|balance)|In case of default|All expenses)\b', rows[j]['text'], re.I):
                    break
                contract_facts.add(j)
    chosen_facts.extend(contract_facts)
    # Preserve the continuation that closes an inline document title split at
    # punctuation; otherwise the displayed excerpt retains a dangling quote.
    for i in list(chosen_facts):
        if (i + 1 in fact_candidates and rows[i]['text'].count('"') % 2 == 1
                and rows[i + 1]['text'].lstrip().startswith('"')
                and rows[i + 1]['text'].count('"') % 2 == 1):
            chosen_facts.append(i + 1)
    # Keep both the triggering dispute and the response to it, rather than
    # selecting a reply whose premise was removed by relevance ranking.
    for i in fact_candidates:
        if re.search(r'\b(?:hold|suspend|withhold)\w* payment\b', rows[i]['text'], re.I):
            chosen_facts.append(i)
            j = i - 1
            while j >= 0 and re.fullmatch(r'[\s\d.\[\]]+', rows[j]['text']):
                j -= 1
            if j in fact_candidates:
                chosen_facts.append(j)
    for i in fact_candidates:
        if re.search(r'\b(?:trial court|court a quo)\b.{0,80}\b(?:directed|ordered)\b', rows[i]['text'], re.I):
            chosen_facts.append(i)

    # Explicit self-attribution is stronger evidence than an isolated model label.
    reasons = [i for i in findings if i >= reasoning_start]
    reasons = sorted(reasons, key=lambda i: (-score(i), i))[:2]
    chosen_reasons = set()
    for i in reasons:
        chosen_reasons.add(i)
        # Preserve immediate explanatory context, even when its raw label is ISSUE.
        for j in range(i + 1, min(i + 3, end)):
            if (j in quoted or PARTY.search(rows[j]['text']) or ISSUE.search(rows[j]['text'])
                    or re.search(r'\b(?:maintains|reiterates?|argues?|contends?|alleges?)\s+that\b', rows[j]['text'], re.I)
                    or re.match(r'^\s*(?:\d+\s+)?In\s+[A-Z][\w.-]+(?:,|\s+\d+)', rows[j]['text'])
                    or attributed_elsewhere(rows[j]['text'])
                    or re.search(r'\b(?:court a quo|according to (?:said|the) court)\b', rows[j]['text'], re.I)
                    or re.search(r'\bv\.\s', rows[j]['text'])):
                break
            chosen_reasons.add(j)
        # Keep an immediately following limitation on the selected reasoning.
        # Otherwise a liability rule can lose its exception through the context cap.
        following = i + 3
        if (following < end and following not in quoted and not PARTY.search(rows[following]['text'])
                and not attributed_elsewhere(rows[following]['text'])
                and re.match(r'^\s*(?:It cannot|Unless\b|Provided that\b)', rows[following]['text'], re.I)
                and following - 1 in chosen_reasons):
            chosen_reasons.add(following)

        # A short disagreement needs its explanation, including a qualifying
        # concession. Do not cross into party speech or an unrelated quotation.
        if re.fullmatch(r'\s*We disagree\.\s*', rows[i]['text'], re.I):
            for j in range(i + 1, min(i + 5, end)):
                text = rows[j]['text']
                concession = re.match(r'^While indeed\b', text) and re.search(r'\b(?:scrutiny|examination).{0,50}\bshows\b', text, re.I)
                if j in quoted or (PARTY.search(text) and not concession) or attributed_elsewhere(text):
                    break
                chosen_reasons.add(j)

    # Explicit approval of an invoked statutory rule provides a narrow bridge
    # into the quoted rule and its burden/qualification explanation. A bare
    # earlier-court quotation does not qualify.
    for i in range(reasoning_start, end):
        text = rows[i]['text']
        if (i and i - 1 in quoted) or PARTY.search(text):
            continue
        if not re.search(r'\b(?:could not|cannot)\b.{0,25}\bbe faulted\b.{0,60}\binvoked Article\s+\d+', text, re.I):
            continue
        chosen_reasons.add(i)
        for j in range(i + 1, min(i + 7, end)):
            if PARTY.search(rows[j]['text']) or re.match(r'^In (?:this|the present|the instant) case\b', rows[j]['text'], re.I):
                break
            chosen_reasons.add(j)

    chosen = {'FACT': sorted(chosen_facts), 'ISSUE': issues,
              'RULING': sorted(chosen_reasons | set(disposition)) if disposition else []}
    evidence = {}
    for role in ROLES:
        indices = set(chosen[role])
        # Preserve antecedents for excerpts that otherwise begin with a dangling reference.
        for i in list(indices):
            while i and (re.match(r'\s*(?:\d+\s+)?(?:However|Thus|Therefore|She|He|They|It|This|These|Those|Such|The second|The first|The latter|The former)\b', rows[i]['text'], re.I)
                         or re.match(r'^In the complaint,', rows[i]['text'], re.I)
                         or re.match(r'\s*(?:\d+\s+)?(?:In|On)\s+[^,]{1,45},\s+the (?:latter|former)\b', rows[i]['text'], re.I)
                         or re.search(r'\b(?:this|such|said)\s+(?!Court\b)\w+|\b(?:granted|denied|dismissed)\s+the\s+(?:petition|appeal|motion|application)\b|\bagain\s+(?:challenged|appealed|sought)\b', rows[i]['text'], re.I)):
                if role != 'FACT' or i - 1 not in fact_candidates:
                    break
                indices.add(i - 1)
                i -= 1
        evidence[role] = [dict(sentence_id=rows[i]['id'], text=rows[i]['text'],
                               model_role=rows[i]['role'],
                               selection=('final_disposition' if i in disposition else
                                          'explicit_issue' if role == 'ISSUE' else
                                          'court_reasoning_context' if role == 'RULING' else
                                          'contract_terms_context' if i in contract_facts else
                                          'model_fact_context')) for i in sorted(indices)]
    warnings = []
    if not issues:
        warnings.append('No explicit legal question could be located reliably; the Issue block is withheld.')
    if not disposition:
        warnings.append('No unambiguous final disposition was located; the Ruling block is withheld. Review the complete decision.')
    if boundary < len(original_rows):
        warnings.append('A separate opinion heading was detected. Text from that heading onward was excluded from FIR evidence; check the main-opinion boundary.')
    if any(row['role'] == 'ISSUE' and row['id'] not in {rows[i]['id'] for i in issues} for row in rows):
        warnings.append('Some raw ISSUE predictions were not explicit legal questions; they were not used as the Issue block.')
    return {'evidence': evidence, 'disposition_ids': [rows[i]['id'] for i in disposition],
            'warnings': warnings, 'quoted_sentence_ids': [rows[i]['id'] for i in sorted(quoted)],
            'separate_opinion_sentence_ids': [row['id'] for row in original_rows[boundary:]]}


def verify_block(candidate, evidence, role, disposition_ids=(), generation=None):
    """Accept only the complete selected source wording; reject unverified paraphrase.

    Exact tokens/names or similarity scores do not establish who said what or
    preserve negation. This deliberately strict gate avoids claiming that they do.
    """
    fallback = '\n\n'.join(item['text'] for item in evidence)
    reasons = []
    if not evidence:
        reasons.append('No sufficiently attributable evidence selected.')
    elif normalized(candidate) != normalized(fallback):
        reasons.append('Generated wording is not a complete verbatim match to the selected evidence; paraphrase accuracy is unverified.')
    if candidate and not re.search(r'[.!?][\"”’\x27)]*\s*$', candidate):
        reasons.append('Generation ends without a complete sentence.')
    if generation and generation.get('hit_token_limit'):
        reasons.append('Generation reached its output token limit.')
    if role == 'RULING' and disposition_ids:
        by_id = {item['sentence_id']: item['text'] for item in evidence}
        if any(normalized(by_id[sid]) not in normalized(candidate) for sid in disposition_ids if sid in by_id):
            reasons.append('Generation does not preserve the complete final disposition.')
    accepted = bool(evidence) and not reasons
    return {'text': candidate if accepted else fallback,
            'mode': 'source_matched_generation' if accepted else 'source_excerpts' if evidence else 'withheld',
            'generation_accepted': accepted, 'reasons': reasons,
            'source_sentence_ids': [item['sentence_id'] for item in evidence],
            'evidence': evidence, 'legal_accuracy_verified': False}


def finalize(rows, generated, plan, generation=None):
    audits = {role: verify_block(generated.get(role, ''), plan['evidence'][role], role,
                                plan['disposition_ids'], (generation or {}).get(role)) for role in ROLES}
    source = ' '.join(row['text'] for row in rows)
    missing = [role for role in ROLES if not audits[role]['text']]
    return {'blocks': {role: audits[role]['text'] for role in ROLES},
            'reliability': {'status': 'incomplete' if missing else 'review_required',
                            'missing_roles': missing, 'legal_accuracy_verified': False,
                            'disposition_found': bool(plan['disposition_ids']),
                            'separate_opinion_detected': bool(plan.get('separate_opinion_sentence_ids')),
                            'message': ('Incomplete draft: ' + ', '.join(missing) + ' withheld.' if missing else
                                        'Source excerpts available. Attribution and completeness require review.')},
            'verification': {'version': VERSION, 'method': 'literal_source_match',
                             'legal_accuracy_verified': False,
                             'source_sha256': hashlib.sha256(source.encode('utf-8')).hexdigest(),
                             'disposition_sentence_ids': plan['disposition_ids'],
                             'excluded_separate_opinion_sentence_ids': plan.get('separate_opinion_sentence_ids', []),
                             'label_conflicts': [dict(sentence_id=item['sentence_id'], model_role=item['model_role'],
                                                       evidence_role=role, selection=item['selection'])
                                                for role in ROLES for item in plan['evidence'][role]
                                                if item['model_role'] != role],
                             'blocks': audits}}
