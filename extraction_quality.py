"""Detect incomplete PDF extraction before either model sees the document."""
import hashlib


def audit_pages(pages, pdf_bytes):
    records = []
    for number, text in enumerate(pages, 1):
        readable = sum(char.isalnum() for char in text)
        replacements = text.count('\ufffd')
        records.append({'page': number, 'characters': len(text.strip()),
                        'alphanumeric_characters': readable, 'replacement_characters': replacements,
                        'unreadable': readable == 0,
                        'corrupt': replacements >= 3 and replacements / max(1, len(text)) >= .01,
                        'low_text': readable < 40})
    blocked = [row['page'] for row in records if row['unreadable'] or row['corrupt']]
    low_text = [row['page'] for row in records if row['low_text'] and row['page'] not in blocked]
    warnings = []
    if blocked:
        warnings.append('Pages with missing or corrupt text: ' + ', '.join(map(str, blocked)) +
                        '. Supply a complete searchable/OCR PDF, or remove confirmed blank pages, before summarizing.')
    if low_text:
        warnings.append('Very little text was extracted from pages ' + ', '.join(map(str, low_text)) +
                        '; inspect these pages against the PDF for missing content.')
    return {'pdf_sha256': hashlib.sha256(pdf_bytes).hexdigest(), 'page_count': len(pages),
            'pages': records, 'blocked_pages': blocked, 'low_text_pages': low_text,
            'summarization_allowed': bool(pages) and not blocked,
            'complete_extraction_verified': False, 'warnings': warnings}
