import json
from pathlib import Path
from collections import Counter
from pypdf import PdfReader
from preprocessing import prepare_decision
from core import LABELS, classifier_path
from pipeline import FIRPipeline
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig

torch.set_num_threads(4)
source=Path(r'C:\Users\drylr\Downloads\Case Input  (1).pdf')
prepared=prepare_decision('\n'.join(p.extract_text() or '' for p in PdfReader(source).pages))
folder=classifier_path()
cfg=PeftConfig.from_pretrained(folder)
engine=FIRPipeline.__new__(FIRPipeline)
engine.torch,engine.device=torch,'cpu'
engine.bt=AutoTokenizer.from_pretrained(folder,local_files_only=True)
base=AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,num_labels=4,id2label=dict(enumerate(LABELS)),label2id={v:i for i,v in enumerate(LABELS)},local_files_only=True)
engine.bert=PeftModel.from_pretrained(base,folder).eval()
rows=engine.classify(prepared['text'])
assert prepared['text'].endswith('SO ORDERED.')
assert 'WHEREFORE, this Court resolves to GRANT the Petition.' in prepared['text']
assert 'Footnotes' not in prepared['text'] and 'WE CONCUR' not in prepared['text']
assert not any(r['text'] in ['Noe Botor (Atty.', '19 Atty.'] for r in rows)
result={'source':str(source),'preprocessing':prepared,'counts':dict(Counter(r['role'] for r in rows)),'sentences':rows}
Path('runs/bicol_preprocessing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Counts:',result['counts'])
print('Main issue:',[(r['id'],r['role']) for r in rows if 'The single issue to be resolved' in r['text']])
print('Cleanup and classification smoke check passed.')
