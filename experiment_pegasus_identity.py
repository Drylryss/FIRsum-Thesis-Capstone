"""Follow-up to observed wrong-court/party errors: explicit source identity."""
import hashlib
import json
from pathlib import Path
from pipeline import FIRPipeline
from grounding import select_evidence, verify_block
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

OUT=Path('outputs/frozen_pegasus_20261027')

def main():
    assert not (OUT/'identity_results.json').exists()
    specs=[('151035',Path('data/private/reserved_evaluation_20261026/source.json')),
           ('111238',Path('data/private/additional_digest_development_20261024_v3/sources/111238.json'))]
    inputs=[]
    for docket,path in specs:
        source=json.loads(path.read_text(encoding='utf8'))
        raw=source['raw_text']
        if docket=='151035':
            start=raw.index('[ G.R. No.')
            end=raw.index(' D E C I S I O N')
            caption=raw[start:end]
        else:
            caption='\n\n'.join(raw.split('\n\n')[2:4])
        old=json.loads((OUT/(docket+'_disposition_first.json')).read_text(encoding='utf8'))
        prefix='Philippine Supreme Court decision\n'+caption+'\n\n'
        inputs.append({'docket':docket,'source_path':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'decision_url':source['decision_url'],'caption':caption,
            'court_identity':'Philippine Supreme Court; supplied as corpus metadata, not a verbatim source quote',
            'generator_source':prefix+old['RULING']['audit']['generator_source']})
    (OUT/'identity_inputs.json').write_text(json.dumps(inputs,indent=2,ensure_ascii=False),encoding='utf8')
    torch.set_num_threads(4)
    e=FIRPipeline.__new__(FIRPipeline)
    e.torch,e.device,e.examples=torch,'cpu',[]
    e.gt=AutoTokenizer.from_pretrained('nsi319/legal-pegasus',local_files_only=True)
    e.gen=AutoModelForSeq2SeqLM.from_pretrained('nsi319/legal-pegasus',local_files_only=True,low_cpu_mem_usage=True).eval()
    e.generation_overrides={'length_penalty':0.8}
    results=[]
    for item in inputs:
        docket=item['docket']
        text,count=e.generate_block('RULING',item['generator_source'])
        base=json.loads(Path('outputs/coverage_selection_20261027',docket+'_fresh.json').read_text(encoding='utf8'))
        plan=select_evidence(base['sentences'])
        results.append({'docket':docket,'text':text,'chunks':count,'audit':e.generation_audit['RULING'],
            'verification':verify_block(text,plan['evidence']['RULING'],'RULING',plan['disposition_ids'],e.generation_audit['RULING'])})
        print(json.dumps({'docket':docket,'words':len(text.split())}),flush=True)
        (OUT/(docket+'_identity.json')).write_text(json.dumps(results[-1],indent=2,ensure_ascii=False),encoding='utf8')
    (OUT/'identity_results.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf8')

if __name__=='__main__': main()
