"""Small before/after generation diagnostic on development ISSUE inputs."""
import json
from pathlib import Path
from contextlib import nullcontext
from core import generator_input


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from peft import PeftModel
    torch.set_num_threads(4)
    root=Path('data/private/pegasus_feasibility_20261023')
    report=json.loads((root/'report.json').read_text(encoding='utf-8'))
    rows=json.loads((root/'paired_data.json').read_text(encoding='utf-8'))
    tok=AutoTokenizer.from_pretrained(report['checkpoint'])
    base=AutoModelForSeq2SeqLM.from_pretrained('nsi319/legal-pegasus')
    model=PeftModel.from_pretrained(base,report['checkpoint']).eval()
    result=[]
    for row in rows:
        if row['split']!='validation' or row['role']!='ISSUE':continue
        for mode in ('base','adapter'):
            print('Generating',row['name'],mode,flush=True)
            with model.disable_adapter() if mode=='base' else nullcontext():
                with torch.inference_mode():
                    output=model.generate(**tok(row['text'],return_tensors='pt'),max_new_tokens=128,
                                          num_beams=4,no_repeat_ngram_size=3)
            result.append({'case_id':row['case_id'],'case':row['name'],'role':'ISSUE','mode':mode,
                           'generated':tok.decode(output[0],skip_special_tokens=True),
                           'target':row['summary'],'hit_token_limit':output.shape[-1]-1>=128})
            (root/'generation_diagnostic.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            print(result[-1]['generated'],flush=True)


if __name__=='__main__':main()
