"""Compare source selection using saved model predictions; no models are rerun."""
import hashlib
import json
from pathlib import Path
from grounding import select_evidence, finalize, normalized, VERSION


def main():
    source=Path('data/private/attorney_references_20261022/pipeline_results')
    output=source.parent/('evidence_revision_'+VERSION.rsplit('-',1)[-1])
    output.mkdir(exist_ok=True)
    report=[]
    for name in ('makati','real','philcomsat','mactan'):
        old=json.loads((source/(name+'.json')).read_text(encoding='utf-8'))
        plan=select_evidence(old['sentences'])
        # Historical generation used different evidence. Do not associate it
        # with this new selection or claim it was freshly generated.
        result=finalize(old['sentences'],{},plan)
        result.update(case_id=old['case_id'],sentences=old['sentences'],
                      generation_not_rerun=True, prior_result=str(source/(name+'.json')))
        by_id={r['id']:r['text'] for r in old['sentences']}
        for role,audit in result['verification']['blocks'].items():
            assert all(e['text']==by_id[e['sentence_id']] for e in audit['evidence'])
            assert normalized(result['blocks'][role])==normalized(' '.join(e['text'] for e in audit['evidence']))
        assert plan['disposition_ids'],name
        assert all(by_id[i] in result['blocks']['RULING'] for i in plan['disposition_ids'])
        if name=='mactan':assert not result['blocks']['ISSUE'], 'Do not import the earlier precedent question'
        else:assert result['blocks']['ISSUE'],name
        (output/(name+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        report.append({'case':name,'before_missing':old['reliability']['missing_roles'],
                       'after_missing':result['reliability']['missing_roles'],
                       'disposition_ids':plan['disposition_ids']})
    result={'type':'development_evidence_regression','models_rerun':False,
            'legal_accuracy_estimate':False,'implementation_sha256':hashlib.sha256(Path('grounding.py').read_bytes()).hexdigest(),'cases':report}
    (output/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
