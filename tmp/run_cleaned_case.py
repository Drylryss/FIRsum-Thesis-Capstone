import json
from pathlib import Path
import requests
source=Path(r'C:\Users\drylr\Downloads\Case Input  (1).pdf')
with source.open('rb') as file:
    response=requests.post('http://127.0.0.1:8000/api/summarize',files={'file':(source.name,file,'application/pdf')},timeout=900)
response.raise_for_status()
result=response.json()
Path('runs/bicol_cleaned_pipeline_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print('Complete:',len(result['sentences']),'sentences')
print(json.dumps(result['blocks'],ensure_ascii=True,indent=2))
