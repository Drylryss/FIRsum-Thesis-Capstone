"""Build compact, verbatim candidates. They remain pending attorney acceptance."""
import json
from pathlib import Path
from core import ROLES, sentences, generator_input
from import_attorney_references import paragraphs


def main():
    folder=Path('data/private/attorney_references_20261022')
    paras=paragraphs(folder/'capstone-case-digests.docx')
    refs=json.loads((folder/'references.json').read_text(encoding='utf-8'))['cases']
    # Explicit source-paragraph selections; every selected sentence stays verbatim.
    selections={'makati':{'FACT':[(3,0),(4,None),(7,0)],'ISSUE':[(8,None)],'RULING':[(13,None),(14,None),(15,None),(16,None),(17,None),(18,None),(25,0)]},
                'real':{'FACT':[(72,0),(73,0),(74,0)],'ISSUE':[(75,None)],'RULING':[(78,0),(79,0),(79,1)]},
                'mactan':{'FACT':[(116,0),(117,0)],'ISSUE':[(119,'issue')],'RULING':[(122,0),(125,0)]}}
    result=[]
    for name,roles in selections.items():
        ref=next(r for r in refs if r['name']==name)
        blocks={};provenance={}
        for role,items in roles.items():
            excerpts=[]
            for p,index in items:
                text=paras[p].strip()
                if index=='issue':text=text.split('ISSUE:',1)[1].strip()
                elif index is not None:text=sentences(text)[index]
                elif text.startswith('ISSUE:'):text=text.split(':',1)[1].strip()
                assert text in paras[p], (name,role,p)
                excerpts.append(text)
            blocks[role]='\n\n'.join(excerpts)
            provenance[role]=[p+1 for p,_ in items]
        result.append({'case_id':ref['case_id'],'name':name,'split':'exemplar_candidate',
                       'attorney_accepted':False,'review_status':'pending_compact_excerpt_review',
                       'blocks':blocks,'source_paragraphs_1_based':provenance,
                       'source_docx_sha256':ref['source_docx_sha256']})
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('nsi319/legal-pegasus')
    budget={role:1024-len(tok.encode(generator_input(role,'',result)))-16 for role in ROLES}
    assert min(budget.values())>=128, budget
    payload={'status':'pending_attorney_review_not_active','source_token_budget':budget,'examples':result}
    (folder/'review/compact_exemplar_candidates.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# Compact example candidates for attorney review','',
        'These are selected verbatim excerpts, not complete digests. Confirm that the omissions preserve the facts, precise question, and legal reasoning before accepting them. They are not active model examples.','']
    for r in result:
        md += ['## '+r['name'],'']
        for role in ROLES:md += ['### '+role,'',r['blocks'][role],'']
        md += ['Attorney acceptance: ______  Reviewer: ______  Required edits: ______','']
    (folder/'review/COMPACT_EXAMPLES.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps(budget))


if __name__=='__main__':main()
