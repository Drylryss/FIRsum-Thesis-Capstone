"""Exact-span proposals, not automatically accepted sentence-level training units."""
import json
import hashlib
from pathlib import Path
from audit_label_context import OUT


def main():
    audit = json.loads((OUT/'context_audit.json').read_text(encoding='utf-8'))
    record = next(r for r in audit if r['dataset_index']==559)
    parent = record['text']
    pivot = parent.index(', arguing that')
    segments = []

    def add(parent_text, start, end, label, parent_id, source, reason):
        assert 0 <= start < end <= len(parent_text)
        segments.append({'parent_id':parent_id, 'parent_text':parent_text,
            'parent_sha256':hashlib.sha256(parent_text.encode()).hexdigest(),
            'span':[start,end], 'text':parent_text[start:end], 'proposed_label':label,
            'source_file':str(source), 'source_sha256':hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'reason':reason, 'reviewer_type':'AI_assistant','final_label':None,
            'attorney_validated':False, 'training_eligible':False,
            'unit_policy':'Exact source span; fragments require explicit segmentation review before sentence-classifier training.'})
    add(parent,0,pivot,'ISSUE',record['sentence_id'],record['source_file'],'Current alleged appellate error.')
    add(parent,pivot,len(parent),'FACT',record['sentence_id'],record['source_file'],'Attributed supporting argument; needs parent context.')
    source = Path('data/private/attorney_references_20261022/picart_source.json')
    doc = json.loads(source.read_text(encoding='utf-8'))
    text = doc.get('majority_text') or doc['text']
    start = text.index('The question presented for decision is whether')
    end = text.index(' As the defendant',start)
    parent = text[start:end]
    pivot = parent.index('; and we are of the opinion')
    add(parent,0,pivot,'ISSUE','Picart_source_question_answer',source,'Current negligence question, omitted from existing training annotations.')
    add(parent,pivot,len(parent),'RULING','Picart_source_question_answer',source,'Court answers the question in the same source sentence.')
    source = Path('data/private/additional_digest_development_20261024_v3/sources/111238.json')
    text = json.loads(source.read_text(encoding='utf-8'))['raw_text']
    parent = next(p for p in text.split('\n\n') if p.startswith('The main issues presented'))
    add(parent,0,len(parent),'ISSUE','Adelfa_source_opening_issues',source,'Opening current-case questions include the first question missing from existing annotations. Preserve source typography and citation for review.')
    (OUT/'segmentation_and_coverage_proposals.json').write_text(json.dumps(segments,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Prepared',len(segments),'exact-span proposals; none inserted into training.')


if __name__ == '__main__':
    main()
