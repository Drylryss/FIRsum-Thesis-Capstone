"""Two predeclared frozen-generator comparisons on previously exposed cases."""
import hashlib
import json
from pathlib import Path
from pipeline import FIRPipeline
from grounding import select_evidence, verify_block

OUT = Path('outputs/frozen_pegasus_20261027')
BASE = Path('outputs/coverage_selection_20261027')

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'summary.json').exists(), 'Preserve completed experiments'
    # Cached classifier results are sufficient; do not load a second BERT model.
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.set_num_threads(4)
    engine = FIRPipeline.__new__(FIRPipeline)
    engine.torch, engine.device, engine.examples = torch, 'cpu', []
    engine.gt = AutoTokenizer.from_pretrained('nsi319/legal-pegasus', local_files_only=True)
    engine.gen = AutoModelForSeq2SeqLM.from_pretrained('nsi319/legal-pegasus', local_files_only=True,
                                                    low_cpu_mem_usage=True).eval()
    print('Frozen generator loaded from local cache.', flush=True)
    manifest = {'cases':['151035','111238'], 'development_only':True,
        'classifier_predictions':'reused saved deployed-model predictions',
        'models_trained':False, 'baseline':'saved fresh generation from coverage_selection_20261027',
        'profiles':{'shorter':{'length_penalty':0.8},
                    'disposition_first':{'length_penalty':0.8,'input_change':'RULING disposition sentences moved first; all selected evidence retained'}},
        'code_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['pipeline.py','grounding.py','model_config.json']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary=[]
    for docket in manifest['cases']:
        baseline=json.loads((BASE/(docket+'_fresh.json')).read_text(encoding='utf8'))
        plan=select_evidence(baseline['sentences'])
        for profile in manifest['profiles']:
            result={}
            for role in ('FACT','ISSUE','RULING'):
                # Input change affects only RULING; reuse identical shorter runs.
                if profile=='disposition_first' and role!='RULING':
                    result[role]={**shorter[role], 'reused_identical_input':True}
                    continue
                source=baseline['generation_audit'][role]['generator_source']
                evidence=plan['evidence'][role]
                if profile=='disposition_first':
                    chosen=sorted(evidence,key=lambda r:r['sentence_id'] not in plan['disposition_ids'])
                    source=' '.join(' '.join(r['text'].split()) for r in chosen)
                    assert sorted(r['sentence_id'] for r in chosen)==sorted(r['sentence_id'] for r in evidence)
                engine.generation_overrides={'length_penalty':0.8}
                text,chunks=engine.generate_block(role,source)
                audit=engine.generation_audit[role]
                audit['generator_source']=source
                result[role]={'text':text,'chunks':chunks,'audit':audit,
                    'baseline_text':baseline['generated_blocks'][role],
                    'verification':verify_block(text,evidence,role,plan['disposition_ids'],audit)}
                print(json.dumps({'case':docket,'profile':profile,'role':role,'words':len(text.split()),'chunks':chunks}),flush=True)
            if profile=='shorter': shorter=result
            (OUT/(docket+'_'+profile+'.json')).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
            summary.append({'case':docket,'profile':profile,'accepted':{r:b['verification']['generation_accepted'] for r,b in result.items()},
                            'words':{r:len(b['text'].split()) for r,b in result.items()}})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))

if __name__=='__main__': main()
