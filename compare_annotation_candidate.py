"""Fixed-label validation comparison and development evidence diagnostics."""
import json
import hashlib
import argparse
from pathlib import Path
from core import LABELS
from grounding import select_evidence, finalize
from delivery import prepare_delivery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/legalbert_annotation_candidate_20261027')
    parser.add_argument('--output', default='outputs/annotation_candidate_20261027')
    args = parser.parse_args()
    root = Path(args.model)
    out = Path(args.output)
    out.mkdir(exist_ok=False)
    baseline = Path('models/legalbert_balanced_v2')
    old = json.loads((baseline/'validation_predictions.json').read_text(encoding='utf-8'))
    new = json.loads((root/'validation_predictions.json').read_text(encoding='utf-8'))
    assert [(r['sentence_id'],r['text'],r['reference_label']) for r in old] == [(r['sentence_id'],r['text'],r['reference_label']) for r in new]
    reports = [json.loads((p/'validation_report.json').read_text(encoding='utf-8')) for p in (baseline,root)]
    metrics = [r['classification_report'] for r in reports]
    passed = metrics[1]['macro avg']['f1-score'] > metrics[0]['macro avg']['f1-score'] and all(
        metrics[1][label]['recall'] >= metrics[0][label]['recall'] for label in ('ISSUE','RULING'))
    comparison = {'same_validation_rows':len(new),'validation_is_provisional':True,'independent_final_test':False,
        'baseline_report':reports[0],'candidate_report':reports[1], 'promotion_gate_passed':passed,
        'deployed':False, 'changed_predictions':[{'sentence_id':a['sentence_id'],'reference':a['reference_label'],
        'before':a['predicted_label'],'after':b['predicted_label']} for a,b in zip(old,new) if a['predicted_label']!=b['predicted_label']]}
    (out/'validation_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import PeftModel, PeftConfig
    from pipeline import FIRPipeline
    torch.set_num_threads(4)
    engine = FIRPipeline.__new__(FIRPipeline)
    engine.torch,engine.device = torch,'cpu'
    engine.bt = AutoTokenizer.from_pretrained(root,local_files_only=True)
    cfg = PeftConfig.from_pretrained(root)
    base = AutoModelForSequenceClassification.from_pretrained(cfg.base_model_name_or_path,num_labels=4,local_files_only=True)
    engine.bert = PeftModel.from_pretrained(base,root).eval()
    max_difference = 0.0
    for start in range(0,len(new),8):
        batch = new[start:start+8]
        inputs = engine.bt([r['text'] for r in batch],padding=True,return_tensors='pt')
        with torch.inference_mode(): probs = engine.bert(**inputs).logits.softmax(-1)
        expected = torch.tensor([[r['probabilities'][l] for l in LABELS] for r in batch])
        max_difference = max(max_difference,float((probs-expected).abs().max()))
        assert torch.allclose(probs,expected,atol=1e-4,rtol=1e-4)
    summaries=[]
    for docket in ('151035','202989','225562','111238'):
        previous=json.loads((Path('outputs/mayor_fix_development_20261026')/(docket+'.json')).read_text(encoding='utf-8'))
        tagged=engine.classify(previous['preprocessing']['text'])
        assert [r['text'] for r in tagged]==[r['text'] for r in previous['sentences']]
        plan=select_evidence(tagged)
        result={**finalize(tagged,{},plan),'sentences':tagged,'case_id':previous['case_id']}
        delivered=prepare_delivery(result)
        # Compare both classifiers under the same current evidence/presentation code.
        old_tagged=previous['sentences']
        old_plan=select_evidence(old_tagged)
        old_result={**finalize(old_tagged,{},old_plan),'sentences':old_tagged,'case_id':previous['case_id']}
        baseline_delivery=prepare_delivery(old_result)
        summary={'docket':docket,'mode':'fresh_classifier_and_source_fallback_no_Pegasus_generation',
            'changed_labels':sum(a['role']!=b['role'] for a,b in zip(tagged,previous['sentences'])),
            'section_statuses':{r:s['status'] for r,s in delivered['sections'].items()},
            'changed_final_sections':[r for r,s in delivered['sections'].items() if s['display_text']!=baseline_delivery['sections'][r]['display_text']]}
        summaries.append(summary)
        (out/(docket+'.json')).write_text(json.dumps({'summary':summary,'result':result,'delivery':delivered,'baseline_delivery':baseline_delivery},ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(summary),flush=True)
    comparison.update(reload_probability_max_difference=max_difference,development_checks=summaries)
    (out/'validation_comparison.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
    assert hashlib.sha256((baseline/'adapter_model.safetensors').read_bytes()).hexdigest()=='cfa4333af59df77e19468df3702c50045034ad6a0c43c1dc94756f1c44b0e59a'
    print('Promotion gate:',passed,flush=True)


if __name__=='__main__':main()
