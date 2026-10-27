"""Preserve downloaded Picart text and inventory the six supplied references."""
import hashlib
import json
from pathlib import Path
from lxml import html


def main():
    root = Path('data/private/attorney_references_20261022')
    out = Path('data/private/alignment_20261023')
    raw = Path('.cache/reference_sources/picart.html').read_bytes()
    tree = html.fromstring(raw)
    paragraphs = [' '.join(p.itertext()).strip() for p in tree.xpath('//blockquote/p')]
    boundary = paragraphs.index('Separate Opinions')
    assert 'L-12219' in '\n'.join(paragraphs[:5])
    assert 'So ordered.' in '\n'.join(paragraphs[:boundary])
    record = {
        'name': 'picart', 'case_id': 'GRL12219_19180315',
        'source_url': 'https://www.lawphil.net/judjuris/juri1918/mar1918/gr_l-12219_1918.html',
        'source_html_sha256': hashlib.sha256(raw).hexdigest(),
        'raw_html_file': '.cache/reference_sources/picart.html',
        'text': '\n\n'.join(paragraphs),
        'majority_text': '\n\n'.join(paragraphs[:boundary]),
        'separate_opinion_text': '\n\n'.join(paragraphs[boundary:]),
        'extraction_method': 'Lawphil blockquote paragraph text; separate-opinion heading boundary',
        'review_status': 'machine_extracted_not_reviewed',
        'usage': 'existing_LegalBERT_training_case_not_independent_test',
        'training_pairs_created': False,
    }
    (root/'picart_source.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    refs=json.loads((root/'references.json').read_text(encoding='utf-8'))['cases']
    inventory=[]
    for ref in refs:
        name=ref['name']
        path=(Path('runs/grounding_source_only_20261022/macasaet_result.json') if name=='macasaet'
              else root/(name+'_source.json'))
        inventory.append({'name':name,'case_id':ref['case_id'],'source_artifact':str(path),
                          'source_exists':path.exists(),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                          'source_kind':'saved_pipeline_sentences' if name=='macasaet' else 'downloaded_decision_text',
                          'digest_origin':'attorney_created_as_reported_by_user',
                          'alignment_review_status':'not_recorded',
                          'experiment_split': 'training_eligible_existing_BERT_train' if name=='picart' else
                          ('validation_development_previously_seen' if name in ('makati','philcomsat') else 'train')})
    (out/'six_case_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    template={'case_id':'','decision_url':'','source_file':'','source_sha256':'',
              'attorney_digest_file':'','digest_sha256':'','split':'unassigned',
              'previously_used_for_training_or_prompting':None,
              'source_complete_verified':False,'reviewer':None,'review_date':None,
              'alignment_review_status':'pending','blocks':{role:{'target':'','supporting_sentence_ids':[],
              'speaker_and_negation_checked':False,'review_decision':None} for role in ('FACT','ISSUE','RULING')}}
    (out/'new_case_intake_template.json').write_text(json.dumps(template,indent=2),encoding='utf-8')
    print(json.dumps({'sources_in_inventory':len(inventory),'picart_majority_paragraphs':boundary,
                      'picart_separate_opinion_paragraphs':len(paragraphs)-boundary},indent=2))


if __name__=='__main__':
    main()
