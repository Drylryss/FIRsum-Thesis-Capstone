"""Fresh development rerun against saved v7 outputs; no training or test-set use."""
import hashlib
import json
import time
from pathlib import Path
from pipeline import FIRPipeline
from delivery import prepare_delivery
from core import ROLES


def main():
    old=Path('data/private/bart_comparison_20261025')
    out=Path('data/private/evidence_v8_20261025')
    out.mkdir(exist_ok=True)
    engine=FIRPipeline()
    reports=[]
    for docket in ('127206','176697'):
        previous=json.loads((old/(docket+'_firm.json')).read_text(encoding='utf-8'))
        print('Running',docket,flush=True)
        started=time.perf_counter()
        result=engine.run(previous['prepared_source'],previous['result']['case_id'])
        delivery=prepare_delivery(result)
        artifact={'result':result,'delivery':delivery,'elapsed_seconds':time.perf_counter()-started,
                  'source_sha256':hashlib.sha256(previous['prepared_source'].encode()).hexdigest(),
                  'reference_used_as_input':False,'weights_changed':False}
        (out/(docket+'.json')).write_text(json.dumps(artifact,ensure_ascii=False,indent=2),encoding='utf-8')
        reports.append({'docket':docket,'before':{r:bool(previous['delivery']['sections'][r]['display_text']) for r in ROLES},
            'after':{r:bool(delivery['sections'][r]['display_text']) for r in ROLES},
            'labels_identical':result['sentences']==previous['result']['sentences'],
            'disposition_ids':result['verification']['disposition_sentence_ids'],
            'warnings':delivery['warnings'],'seconds':round(artifact['elapsed_seconds'],1)})
        (out/'comparison.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
        print(json.dumps(reports[-1]),flush=True)


if __name__=='__main__': main()
