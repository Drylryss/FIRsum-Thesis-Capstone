"""Conservative, auditable cleanup of text extracted from complete decisions.

Only explicit structural boundaries are excluded. Ambiguous material is retained.
"""
import re
from core import sentences


def prepare_decision(raw):
    normalized = re.sub(r'\s+', ' ', raw.replace('\u00ad', '')).strip()
    start, end = 0, len(normalized)
    removed, warnings = [], []
    # Require both a decision heading and a judicial author line.
    heading = re.search(r'\bD\s*E\s*C\s*I\s*S\s*I\s*O\s*N\s+([A-ZÑ][A-ZÑ .,’\x27-]{1,120}),\s*(?:J|C\.?\s*J)\s*\.:', normalized[:5000])
    if heading:
        start = heading.end()
        removed.append({'reason': 'Case caption, decision heading and author line', 'text': normalized[:start]})
    else:
        warnings.append('No clear decision heading and author line found; the opening text was retained.')
    # A quoted lower-court SO ORDERED is not a stopping point. Require a
    # following uppercase judicial signature immediately followed by its title.
    ending = re.search(r'\bSO\s+ORDERED\.(?=\s+[A-ZÑ][A-ZÑ .,’\x27-]{3,180}\s+(?:Associate|Chief)\s+Justice\b)', normalized[start:])
    if ending:
        end = start + ending.end()
        removed.append({'reason': 'Judicial signatures and trailing material after the signed disposition (including appended footnotes/opinions)', 'text': normalized[end:]})
    else:
        warnings.append('No signed final disposition boundary found; trailing material was retained for review.')
    text = normalized[start:end].strip()
    warnings.append('Cleanup is heuristic. Inline footnote numbers and page-level footnotes may remain; review the extracted decision and excluded material.')
    return {'text': text, 'excluded_sections': removed, 'warnings': warnings,
            'raw_characters': len(normalized), 'cleaned_characters': len(text),
            'raw_sentence_count': len(sentences(normalized)), 'sentence_count': len(sentences(text))}
