"""Explicit reviewed labels for exact case text; never used by training/evaluation."""
import hashlib
import json
from pathlib import Path
from core import LABELS


def apply_reviewed_labels(text, rows, path=None):
    path = Path(path) if path else Path(__file__).parent / 'data/reviewed_label_overrides.json'
    result = [dict(row) for row in rows]
    if not path.exists():
        return result
    record = json.loads(path.read_text(encoding='utf-8'))
    key = hashlib.sha256(text.encode('utf-8')).hexdigest()
    corrections = {r['text']: r for r in record['cases'].get(key, [])}
    for row in result:
        correction = corrections.get(row['text'])
        if correction and correction['role'] in LABELS and correction['role'] != row['role']:
            row.update(model_role=row['role'], model_confidence=row['confidence'],
                       role=correction['role'], confidence=None,
                       label_source='assistant_review', correction_reason=correction['reason'])
    return result
