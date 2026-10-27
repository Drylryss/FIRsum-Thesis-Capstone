"""Development regression on the supplied Bicol PDF, not an accuracy benchmark."""
import json
from pathlib import Path
from grounding import finalize, select_evidence


def check(result):
    blocks=result['blocks']
    assert 'constructed a steel gate' in blocks['FACT']
    assert 'never intended for use by the general public' in blocks['FACT']
    assert 'remains to be a public road' in blocks['FACT']
    assert 'single issue to be resolved' in blocks['ISSUE']
    assert 'completely disregarded' not in blocks['ISSUE']
    assert 'convinces this Court' in blocks['RULING']
    assert 'This Court finds' in blocks['RULING']
    for phrase in ('GRANT the Petition', 'REVERSED and SET ASIDE', 'made PERMANENT'):
        assert phrase in blocks['RULING']
    for text in blocks.values():
        assert 'California' not in text
        assert 'Securities Act' not in text


if __name__=='__main__':
    folder=Path('data/private/sample_full_case_20261023')
    old=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    result=finalize(old['sentences'],{},select_evidence(old['sentences']))
    check(result)
    result['models_rerun']=False
    result['prior_result']='result.json'
    (folder/'evidence_v4.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved-prediction Bicol evidence regression passed; no model rerun.')
