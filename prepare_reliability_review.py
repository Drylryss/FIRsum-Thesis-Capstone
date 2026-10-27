"""Prepare contextual annotation review and audit all three-example token budgets."""
import itertools
import json
from pathlib import Path
from core import ROLES, generator_input, read_jsonl


def main():
    folder = Path('data/private/attorney_references_20261022')
    out = folder / 'review'
    out.mkdir(exist_ok=True)
    rows = read_jsonl('data/revision_20261020_metadata/sentences.jsonl')
    queue = read_jsonl('runs/reliability_20261022/label_review_queue.jsonl')
    selected = [r for r in queue if r['review_priority'] <= 2]
    for item in selected:
        i = next(i for i,r in enumerate(rows) if r['sentence_id'] == item['sentence_id'])
        # These are neighboring ANNOTATED records; gaps are not represented as
        # adjacent original sentences. The reviewer must consult the full case.
        item['neighboring_annotations'] = [dict(sentence_id=r['sentence_id'], text=r['text'])
            for r in rows[max(0,i-1):i+2] if r['case_id']==item['case_id'] and r['sentence_id']!=item['sentence_id']]
    (out/'priority_labels.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Attorney sentence-label review', '',
           'Review these provisional labels against the complete decision. Neighboring annotations may have gaps and are not guaranteed consecutive source sentences.',
           'Keep validation cases in validation. Do not use the supplied digests as sentence labels.',
           'For each item, enter final_label (FACT, ISSUE, RULING, OTHER), reviewer, and a reason in priority_labels.json. Do not change IDs or source text.', '']
    for item in selected:
        lines += [f"## {item['sentence_id']} ({item['split']})", '',
                  f"Provisional label: {item['current_provisional_label']}. Source pages: {item.get('source_page_start')}–{item.get('source_page_end')}.",
                  '', item['text'], '', 'Neighboring annotations (source gaps possible):', '']
        lines += [f"- {r['sentence_id']}: {r['text']}" for r in item['neighboring_annotations']]
        lines += ['', 'Final label: ______  Reviewer: ______  Reason: ______', '']
    (out/'ATTORNEY_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained('nsi319/legal-pegasus')
    refs=json.loads((folder/'references.json').read_text(encoding='utf-8'))['cases']
    eligible=[r for r in refs if not r['classifier_split_overlap']]
    budgets=[]
    for combo in itertools.combinations(eligible,3):
        sizes={role:len(tok.encode(generator_input(role,'',combo))) for role in ROLES}
        budgets.append({'cases':[r['name'] for r in combo], 'prompt_tokens_without_target':sizes,
                        'source_budget':{r:1024-n-16 for r,n in sizes.items()},
                        'fits_minimum_128_source_tokens':all(1024-n-16>=128 for n in sizes.values())})
    (out/'exemplar_budget.json').write_text(json.dumps(budgets,indent=2),encoding='utf-8')
    print(json.dumps({'priority_review_count':len(selected),'eligible_triples':len(budgets),
                      'fitting_triples':sum(b['fits_minimum_128_source_tokens'] for b in budgets)},indent=2))


if __name__=='__main__':main()
