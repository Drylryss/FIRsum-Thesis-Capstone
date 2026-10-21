"""Shared data contracts. Splits must be finalized by case before annotation/training."""
import json
import os
import re
from pathlib import Path

# Keep model downloads in the project, including under restricted Windows sessions.
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parent / '.cache' / 'huggingface'))

LABELS = ['FACT', 'ISSUE', 'RULING', 'OTHER']
ROLES = LABELS[:3]


def classifier_path():
    root = Path(__file__).resolve().parent
    config_path = root / 'model_config.json'
    config = json.loads(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    value = os.getenv('LEGALBERT_PATH') or config.get('legalbert_path', 'models/legalbert')
    path = Path(value)
    return str(path if path.is_absolute() else root / path)


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'Empty dataset: {path}')
    return rows


def validate_splits(rows):
    cases = {}
    for row in rows:
        case, split = row['case_id'], row['split']
        if not isinstance(case, str) or not case.strip():
            raise ValueError('Each row needs a nonempty case_id')
        if split not in {'train', 'validation', 'test', 'exemplar'}:
            raise ValueError(f'Invalid split: {split}')
        if case in cases and cases[case] != split:
            raise ValueError(f'Case leakage: {case}')
        cases[case] = split
    return cases


def training_rows(path, generator=False):
    rows = read_jsonl(path)
    validate_splits(rows)
    for row in rows:
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Each row needs nonempty text')
        if generator:
            if row.get('role') not in ROLES or not row.get('summary', '').strip():
                raise ValueError('Generator rows need role and summary')
        elif row.get('label') not in LABELS:
            raise ValueError('Classifier rows need a valid label')
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'validation']
    if not train or not val:
        raise ValueError('Provide train and validation cases; test and exemplar rows are excluded')
    if not generator and set(r['label'] for r in train) != set(LABELS):
        raise ValueError('Training data must cover all four labels')
    return train, val


def sentences(text):
    # Conservative splitter protects common legal abbreviations and initials.
    text = re.sub(r'\s+', ' ', text).strip()
    protected = re.sub(r'\b(?:G\.R\.|No\.|Nos\.|Art\.|art\.|Arts\.|Mr\.|Mrs\.|Dr\.|Atty\.|SJ\.|Jr\.|Sr\.|St\.|pp\.|p\.|Inc\.|v\.|vs\.)|\b[A-Z]\.(?=\s+[A-Z])',
                       lambda m: m[0].replace('.', '\uE000'), text)
    return [s.replace('\uE000', '.').strip() for s in re.split(r'(?:(?<=[.!?])|(?<=[.!?][\"”’)]))\s+(?=[A-Z0-9\"“])', protected) if s.strip()]


def token_windows(ids, size, overlap=64):
    if size <= overlap or overlap < 0:
        raise ValueError('Window size must exceed overlap')
    start = 0
    while start < len(ids):
        yield ids[start:start + size]
        if start + size >= len(ids):
            break
        start += size - overlap


def prompt(role, source, exemplars=()):
    examples = '\n'.join(f"Example {i+1} {role}: {e['blocks'][role]}" for i, e in enumerate(exemplars))
    return (f'Write one {role} paragraph for a Philippine Supreme Court case digest. '
            'Use only the target source; do not copy facts from examples. Preserve parties and legal bases.\n'
            f'{examples}\nTarget {role} source:\n{source}\n{role} summary:')
