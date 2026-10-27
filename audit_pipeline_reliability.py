"""Reproducible development replay; separates raw generation from final delivery.

No model rerun, no reference-conditioned inference, no legal accuracy estimate.
"""
import hashlib
import json
from pathlib import Path
from grounding import select_evidence,finalize
from delivery import prepare_delivery


def main():
    cases={'bicol':Path('data/private/sample_full_case_20261023/result_v4.json'),
           'macasaet':Path('runs/grounding_source_only_20261022/macasaet_result.json')}
    folder=Path('data/private/attorney_references_20261022/pipeline_results')
    cases.update({name:folder/(name+'.json') for name in ('makati','real','philcomsat','mactan')})
    out=Path('data/private/reliability_audit_20261023');out.mkdir(parents=True,exist_ok=True)
    records=[]; review=['# Case-level review queue','',
        'Machine-prepared, pending actual review. Previously seen development cases only.',
        'Read the full decision. Source overlap is not legal accuracy. Do not replace blanks with assumed approval.','']
    for name,path in cases.items():
        old=json.loads(path.read_text(encoding='utf-8'))
        rows=old['sentences'];plan=select_evidence(rows)
        replay={**finalize(rows,{},plan),'sentences':rows,'case_id':old['case_id']}
        # Old raw generations belong to their original evidence selections.
        # Do not score them as if generated from the new replay inputs.
        replay['preprocessing']=old.get('preprocessing',{})
        draft=prepare_delivery(replay)
        ids=set(plan['disposition_ids'])
        delivered={e['sentence_id'] for e in draft['sections']['RULING']['evidence']}
        assert ids and ids<=delivered,name
        assert draft['sections']['FACT']['evidence'],name
        assert bool(draft['sections']['ISSUE']['evidence'])==(name!='mactan'),name
        original_modes={r:a['mode'] for r,a in old['verification']['blocks'].items()}
        record={'case':name,'source_result':str(path),'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'historical_generation_modes':original_modes,'current_delivery_status':draft['status'],
            'complete_identified_disposition_preserved':ids<=delivered,
            'missing_sections':[r for r,s in draft['sections'].items() if s['status']=='withheld'],
            'warnings':draft['warnings'],'label_conflicts':replay['verification']['label_conflicts'],
            'human_review_completed':False,'models_rerun':False}
        records.append(record)
        (out/(name+'.json')).write_text(json.dumps({'audit':record,'delivery':draft},ensure_ascii=False,indent=2),encoding='utf-8')
        review+=['## '+name,'','Source result: '+str(path),'']
        for role,section in draft['sections'].items():
            review += ['### '+role,'',section['display_text'] or '[WITHHELD]','',
                'Evidence IDs: '+', '.join(str(e['sentence_id']) for e in section['evidence']),'',
                'Reviewer/date: ______  Accept / edit / reject: ______',
                'Wrong party or speaker: ______  Negation/outcome error: ______',
                'Unsupported claim: ______  Missing essential context or legal basis: ______',
                'Correction and supporting source sentence(s): ______','']
    summary={'evaluation_type':'previously_seen_development_replay','case_count':len(records),
        'models_rerun':False,'independent_test':False,'legal_accuracy_estimate':None,
        'cases_preserving_identified_disposition':sum(r['complete_identified_disposition_preserved'] for r in records),
        'cases_with_missing_sections':sum(bool(r['missing_sections']) for r in records),
        'historical_raw_blocks_accepted':sum(m=='source_matched_generation' for r in records for m in r['historical_generation_modes'].values()),
        'historical_raw_blocks_total':sum(len(r['historical_generation_modes']) for r in records),
        'implementation_hashes':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['grounding.py','delivery.py','footnote_display.py']},
        'cases':records}
    (out/'report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'CASE_REVIEW.md').write_text('\n'.join(review),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cases','implementation_hashes')},indent=2))


if __name__=='__main__':main()
