"""Training-only coverage additions with exact cached-source provenance."""
import json
import re
import hashlib
from collections import Counter
from pathlib import Path
from core import validate_splits, training_rows

OUT = Path('data/revision_20261027_experiment')


def main():
    src = Path('data/revision_20261027_ai_annotations/sentences.jsonl')
    rows = [json.loads(s) for s in src.read_text(encoding='utf-8').splitlines()]
    excluded = [r for r in rows if r['split'] == 'train' and r.get('training_eligible') is False]
    result = [r for r in rows if r not in excluded]
    cases = {r['case_id'].split('_')[0][2:]: r['case_id'] for r in rows if r['split'] == 'train'}
    additions = []
    texts = {re.sub(r'\W+', '', r['text']).lower() for r in rows}
    def add(docket, raw, text, label, reason, path):
        key = re.sub(r'\W+', '', text).lower()
        if key in texts:
            return
        assert docket in cases and raw.count(text) == 1
        start = raw.index(text)
        item = dict(case_id=cases[docket], split='train', text=text, label=label,
                    sentence_id=f'{cases[docket]}_AI20261027_{len(additions)+1:03}',
                    training_eligible=True, label_status='ai_source_annotated_user_authorized',
                    reviewer_type='AI_assistant', human_validated=False, attorney_validated=False,
                    final_label=None, reviewer=None, source_file=str(path),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    source_span=[start, start+len(text)], source_context=raw[max(0,start-250):start+len(text)+250],
                    annotation_reason=reason)
        additions.append(item)
        texts.add(key)
    for docket in ['142381','111238','145441','141877','225562']:
        path = Path('data/private/additional_digest_development_20261024_v3/sources')/(docket+'.json')
        raw = json.loads(path.read_text(encoding='utf-8'))['raw_text']
        paragraphs = raw.split('\n\n')
        if docket == '142381':
            for index in [101,108,117]:
                text = paragraphs[index]
                assert text.startswith('Whether Ching')
                add(docket,raw,text,'ISSUE','Current Court issue heading, distinct from a party supporting argument.',path)
        if docket == '111238':
            text = next(p for p in paragraphs if p.startswith('The main issues presented for resolution'))
            add(docket,raw,text,'ISSUE','Explicit current-review questions; source typo and citation retained.',path)
        if docket == '145441':
            text = next(p for p in paragraphs if p.startswith('Petitioner claims that the Court of Appeals erred'))
            text = text[:text.index(', arguing that')]
            add(docket,raw,text,'ISSUE','Exact first span of excluded mixed record 559; formal assignment separated from supporting argument. No paraphrase.',path)
        for p in paragraphs:
            if (p in ['FIRST DIVISION','SECOND DIVISION','THIRD DIVISION','D E C I S I O N','R E S O L U T I O N','Footnotes','The Case','Antecedent Facts']
                or re.fullmatch(r'[A-Z-]+, J\s*\.:',p)) and raw.count(p)==1:
                add(docket,raw,p,'OTHER','Standalone metadata, author attribution or non-substantive heading; not a legal finding.',path)
    result += additions
    validate_splits(result)
    assert [r for r in result if r['split']!='train']==[r for r in rows if r['split']!='train']
    assert all(r['case_id'] in set(cases.values()) for r in additions)
    OUT.mkdir(exist_ok=False)
    (OUT/'sentences.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf-8')
    training_rows(OUT/'sentences.jsonl')
    for name,value in [('additions',additions),('excluded_training_records',excluded)]:
        (OUT/(name+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'training_counts':dict(Counter(r['label'] for r in result if r['split']=='train')),
        'added_counts':dict(Counter(r['label'] for r in additions)),'excluded_train':len(excluded),
        'validation_unchanged':True,'test_unchanged':True,'new_cases_added':False,'human_validated':False,
        'method':'User-authorized AI annotation; unresolved text excluded, mixed parent replaced by exact issue span only.',
        'experiment':{'epochs':3,'learning_rate':0.00002,'init_adapter':'models/legalbert_balanced_v2',
        'deployment_rule':'No promotion unless validation macro-F1 improves without ISSUE or RULING recall regression; inspect development outputs.'}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest))


if __name__=='__main__': main()
