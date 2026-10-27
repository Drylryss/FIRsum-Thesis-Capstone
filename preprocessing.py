"""Conservative, auditable cleanup of text extracted from complete decisions.

Only explicit structural boundaries are excluded. Ambiguous material is retained.
"""
import re
import hashlib
from core import sentences
from source_structure import before_separate_opinion


def repair_spacing(text):
    """Narrow whitespace-only repairs with offsets into each intermediate text.

    No spelling guesses, number deletion, or semantic substitution is allowed.
    """
    edits = []
    patterns = [r'\b(informed|notified|ordered|directed)(petitioners|respondents)\b',
                r'\b(were|was)(honored|dishonored)\b', r'\b(that)(the)\b',
                r'\b(in)(turn)\b', r'(,)(?=[A-Za-zÑñ])',
                r'\b(REINSTATED|REVERSED|AFFIRMED|GRANTED|DENIED)(only)\b']
    for pattern in patterns:
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            replacement = match[1] + ' ' + (match[2] if match.lastindex == 2 else '')
            edits.append({'start': match.start(), 'end': match.end(), 'before': match[0],
                          'after': replacement, 'reason': 'recognized joined-word or comma spacing',
                          'offset_basis': 'intermediate text before this edit'})
            text = text[:match.start()] + replacement + text[match.end():]
    return text, edits


def prepare_decision(raw):
    normalized = re.sub(r'\s+', ' ', raw.replace('\u00ad', '')).strip()
    start, end = 0, len(normalized)
    removed, warnings = [], []
    # Require both a decision heading and a judicial author line.
    heading = re.search(r'\b(?:D\s*E\s*C\s*I\s*S\s*I\s*O\s*N|R\s*E\s*S\s*O\s*L\s*U\s*T\s*I\s*O\s*N)\s+([A-ZÑ][A-ZÑ .,’\x27-]{1,120}),\s*(?:J|C\.?\s*J)\s*\.:', normalized[:5000])
    if not heading and re.match(r'(?:(?:Manila|(?:FIRST|SECOND|THIRD)\s+DIVISION|EN BANC)\s+)*G\.\s*R\.', normalized):
        # Older decisions sometimes omit the DECISION heading. Require a case
        # caption plus an uppercase judicial author, not an ordinary name.
        heading = re.search(r'\b[A-ZÑ][A-ZÑ .,’\x27-]{2,100},\s*(?:J|C\.?\s*J)\s*\.:', normalized[:3000])
    if heading:
        start = heading.end()
        removed.append({'reason': 'Case caption, decision heading and author line', 'text': normalized[:start]})
    else:
        warnings.append('No clear decision heading and author line found; the opening text was retained.')
    # A quoted lower-court SO ORDERED is not a stopping point. Require a
    # following uppercase judicial signature immediately followed by its title.
    main_text = normalized[start:start + before_separate_opinion(normalized[start:])]
    endings = list(re.finditer(r'\bSO\s+ORDERED\s*\.(?=\s+[A-ZÑ][A-ZÑ .,’\x27-]{3,180}\s+(?:Associate|Chief)\s+Justice\b)', main_text))
    endings += list(re.finditer(r'\bSO\s+ORDERED\s*\.(?=\s+[A-ZÑ][^:;!?]{0,500}\b(?:J\.?J\.|C\.J\.)[^:;!?]{0,100}\bconcur\b)', main_text))
    ending = max(endings, key=lambda item: item.end()) if endings else None
    # A later main-opinion disposition means the signed quotation was historical.
    if ending and re.search(r'\bWHEREFORE\b', main_text[ending.end():], re.I):
        ending = None
    if ending:
        end = start + ending.end()
        removed.append({'reason': 'Judicial signatures and trailing material after the signed disposition (including appended footnotes/opinions)', 'text': normalized[end:]})
    else:
        warnings.append('No signed final disposition boundary found; trailing material was retained for review.')
    text, edits = repair_spacing(normalized[start:end].strip())
    warnings.append('Cleanup is heuristic. Inline footnote numbers and page-level footnotes may remain; review the extracted decision and excluded material.')
    return {'text': text, 'excluded_sections': removed, 'warnings': warnings, 'text_repairs': edits,
            'raw_text_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
            'cleaned_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'raw_characters': len(normalized), 'cleaned_characters': len(text),
            'raw_sentence_count': len(sentences(normalized)), 'sentence_count': len(sentences(text))}
