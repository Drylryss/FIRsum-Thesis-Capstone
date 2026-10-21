"""Local HTML application. Run: python inference.py"""
import io
import threading
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from core import classifier_path
app = FastAPI(title='FIRmSUM')
ROOT = Path(__file__).parent
lock = threading.Lock()
engine = None


@app.get('/')
def home():
    return FileResponse(ROOT / 'web' / 'index.html', headers={'Cache-Control': 'no-store'})


@app.get('/api/status')
def status():
    import importlib.util
    import os
    import json
    metadata = Path(classifier_path(), 'training_metadata.json')
    settings = json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
    provisional = settings.get('provisional_labels', False)
    return {'classifier_ready': Path(classifier_path(), 'adapter_config.json').exists(),
            'classifier_provisional': provisional,
            'classifier_weighting': settings.get('class_weighting', 'none'),
            'ml_dependencies_ready': all(importlib.util.find_spec(x) is not None for x in ['torch', 'transformers', 'peft']),
            'loaded': engine is not None}


def execute(text, case_id):
    global engine
    with lock:
        if engine is None:
            from pipeline import FIRPipeline
            engine = FIRPipeline()
        return engine.run(text, case_id)


async def prepare_upload(file):
    filename = (file.filename or '').replace('\\', '/').rsplit('/', 1)[-1]
    if Path(filename).suffix.lower() != '.pdf':
        raise HTTPException(422, 'Upload a PDF file only.')
    case_id = Path(filename).stem.strip()[:200] or 'Uploaded case'
    data = await file.read(10 * 1024 * 1024 + 1)
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, 'PDF exceeds 10 MB')
    if not data.startswith(b'%PDF-'):
        raise HTTPException(422, 'The uploaded file is not a valid PDF.')
    try:
        from pypdf import PdfReader
        text = '\n'.join(p.extract_text() or '' for p in PdfReader(io.BytesIO(data)).pages)
    except Exception as exc:
        raise HTTPException(422, 'Cannot read this PDF. Upload a readable, unencrypted PDF.') from exc
    if not text.strip():
        raise HTTPException(422, 'No readable text. Scanned PDFs require OCR before upload.')
    if len(text) > 200000:
        raise HTTPException(413, 'Decision exceeds 200,000 characters')
    from preprocessing import prepare_decision
    prepared = prepare_decision(text)
    prepared['case_id'] = case_id
    return prepared


@app.post('/api/prepare')
async def prepare(file: UploadFile = File(...)):
    return await prepare_upload(file)


@app.post('/api/summarize')
async def summarize(file: UploadFile = File(...)):
    prepared = await prepare_upload(file)
    try:
        result = await run_in_threadpool(execute, prepared['text'], prepared['case_id'])
        result['preprocessing'] = prepared
        result.setdefault('warnings', []).extend(prepared['warnings'])
        return result
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        raise HTTPException(503, str(exc)) from exc



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
