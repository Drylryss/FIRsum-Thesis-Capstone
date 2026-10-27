"""Small supervised LoRA feasibility run, not an independent accuracy evaluation."""
import hashlib
import json
from pathlib import Path
import re
import time

from core import ROLES, sentences, validate_splits


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed
    from peft import LoraConfig, TaskType, get_peft_model
    torch.set_num_threads(4)
    set_seed(42)
    root=Path('data/private/pegasus_feasibility_20261023')
    model_dir=Path('models/legalpegasus_lora_feasibility_20261023')
    if model_dir.exists() or (root/'report.json').exists():
        raise ValueError('Experiment output exists; preserve it and choose another run')
    root.mkdir(parents=True,exist_ok=True)
    refs=json.loads(Path('data/private/attorney_references_20261022/references.json').read_text(encoding='utf-8'))['cases']
    tok=AutoTokenizer.from_pretrained('nsi319/legal-pegasus')
    rows=[]
    for name,split in [('real','train'),('mactan','train'),('macasaet','train'),('makati','validation'),('philcomsat','validation')]:
        ref=next(r for r in refs if r['name']==name)
        path=(Path('runs/grounding_source_only_20261022/macasaet_result.json') if name=='macasaet' else
              Path('data/private/attorney_references_20261022/pipeline_results')/(name+'.json'))
        saved=json.loads(path.read_text(encoding='utf-8'))
        for role in ROLES:
            # Input selection uses only frozen classifier labels, never the target digest.
            candidates=[r for r in saved['sentences'] if r['role']==role]
            selected=[]
            for r in candidates:
                if len(tok.encode(role+': '+' '.join(x['text'] for x in selected+[r])))<=960:
                    selected.append(r)
            if not selected:
                raise ValueError(f'{name}/{role}: no source evidence; curate instead of inventing input')
            source=role+': '+' '.join(r['text'] for r in selected)
            target=ref['blocks'][role]
            target_ids=tok(text_target=target)['input_ids']
            if len(target_ids)>1024:
                raise ValueError(f'{name}/{role}: target exceeds model positions; no silent truncation')
            rows.append({'case_id':ref['case_id'],'name':name,'split':split,'role':role,
                         'text':source,'summary':target,'source_sentence_ids':[r['id'] for r in selected],
                         'source_candidates':len(candidates),'source_selected':len(selected),
                         'target_tokens':len(target_ids),'input_tokens':len(tok.encode(source)),
                         'digest_source_sha256':ref['source_docx_sha256'],
                         'approval_basis':'user_authorized_assumption_not_recorded_attorney_review'})
    validate_splits(rows)
    write(root/'paired_data.json',rows)
    manifest={'method':'supervised_LoRA_fine_tuning','base':'nsi319/legal-pegasus',
              'train_cases':sorted({r['case_id'] for r in rows if r['split']=='train'}),
              'validation_cases':sorted({r['case_id'] for r in rows if r['split']=='validation'}),
              'independent_test_set':False,'epochs':1,'seed':42,'learning_rate':0.0001,
              'gradient_accumulation':3,'rank':8,'targets':['q_proj','v_proj'],
              'data_sha256':hashlib.sha256((root/'paired_data.json').read_bytes()).hexdigest(),
              'notes':['All cases were previously inspected; validation is developmental.',
                       'Picart excluded: available repository annotations are not a complete decision.',
                       'Full digest targets preserved. Inputs use a bounded selection of model-labeled source sentences; coverage is not expert-validated.',
                       'User authorized proceeding under assumed attorney approval. No reviewed labels or output ratings fabricated.',
                       'Role-prefixed input differs from live zero-shot input; adapter is not automatically deployed.']}
    write(root/'manifest.json',manifest)
    print('Prepared 9 training and 6 validation role pairs.',flush=True)
    model=AutoModelForSeq2SeqLM.from_pretrained(manifest['base'])
    model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=8,lora_alpha=16,
                                        lora_dropout=0.0,target_modules=['q_proj','v_proj']))
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    def batch(row):
        b=tok(row['text'],return_tensors='pt')
        b['labels']=tok(text_target=row['summary'],return_tensors='pt')['input_ids']
        return b
    val=[r for r in rows if r['split']=='validation']
    train=[r for r in rows if r['split']=='train']
    def evaluate():
        model.eval(); losses=[]
        with torch.inference_mode():
            for r in val:
                loss=float(model(**batch(r)).loss)
                losses.append({'case_id':r['case_id'],'role':r['role'],'loss':loss})
                print('Validation',r['name'],r['role'],loss,flush=True)
        return losses
    before=evaluate();write(root/'baseline_loss.json',before)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=0.0001)
    model.train();optimizer.zero_grad(); history=[]
    start=time.monotonic()
    # Fixed interleaving across cases; one epoch, no hyperparameter search.
    order=[train[i] for i in (0,3,6,1,4,7,2,5,8)]
    for i,r in enumerate(order):
        loss=model(**batch(r)).loss
        (loss/3).backward()
        if (i+1)%3==0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            optimizer.step();optimizer.zero_grad()
        item={'pair':i+1,'name':r['name'],'role':r['role'],'loss':float(loss.detach())}
        history.append(item);write(root/'training_progress.json',history)
        print('Training',json.dumps(item),flush=True)
    model.save_pretrained(model_dir);tok.save_pretrained(model_dir)
    after=evaluate()
    write(root/'report.json',{'manifest':manifest,'before':before,'after':after,'training':history,
                             'elapsed_training_and_final_eval_seconds':time.monotonic()-start,
                             'checkpoint':str(model_dir),'promoted':False,
                             'legal_accuracy_measured':False})
    print('Saved separate adapter and report.',flush=True)


if __name__=='__main__':main()
