import copy
import unittest
from delivery import prepare_delivery, readable_evidence, context_warnings
from grounding import select_evidence, finalize


class DeliveryTests(unittest.TestCase):
    def test_source_artifact_is_flagged_without_guessing_replacement(self):
        result=self.result()
        rows=result['sentences']
        rows[2]['text']='We hold that the title of [BJ\\1C] controls.'
        result={**result,**finalize(rows,{},select_evidence(rows))}
        draft=prepare_delivery(result)
        self.assertIn('[BJ\\1C]',draft['sections']['RULING']['display_text'])
        self.assertTrue(draft['sections']['RULING']['source_artifact_warnings'])

    def test_context_warning_does_not_invent_missing_antecedent(self):
        rows = [{'id': 1, 'text': 'The Board adopted a resolution.'},
                {'id': 2, 'text': 'This resolution was later withdrawn.'}]
        evidence = [{'sentence_id': 2, 'text': rows[1]['text']}]
        self.assertTrue(context_warnings(evidence, rows))
        evidence.insert(0, {'sentence_id': 1, 'text': rows[0]['text']})
        self.assertFalse(context_warnings(evidence, rows))

    def test_structural_fact_announcement_omitted_without_rewriting(self):
        evidence = [{'sentence_id': 1, 'text': 'The facts follow.'},
                    {'sentence_id': 2, 'text': 'The City closed the bridge.'}]
        self.assertEqual(readable_evidence('FACT', evidence, set()), evidence[1:])

    def test_spaced_order_closing_stays_after_reasons(self):
        evidence = [{'sentence_id': 1, 'text': 'We hold that payment is due.'},
                    {'sentence_id': 2, 'text': 'WHEREFORE, pay the debt.'},
                    {'sentence_id': 3, 'text': 'SO ORDERED .'}]
        self.assertEqual([e['sentence_id'] for e in readable_evidence('RULING', evidence, {2, 3})], [2, 1, 3])

    def test_source_and_evidence_tampered_together_still_fail_fingerprint(self):
        result=self.result()
        result['sentences'][0]['text']='The City opened the bridge.'
        result['blocks']['FACT']='The City opened the bridge.'
        result['verification']['blocks']['FACT']['evidence'][0]['text']='The City opened the bridge.'
        self.assertEqual(prepare_delivery(result)['sections']['FACT']['status'],'withheld')

    def test_duplicate_source_ids_fail_closed(self):
        result=self.result();result['sentences'].append(copy.deepcopy(result['sentences'][0]))
        self.assertEqual(prepare_delivery(result)['status'],'incomplete')

    def test_excluded_opinion_cannot_enter_delivered_ruling(self):
        result=self.result()
        result['verification']['excluded_separate_opinion_sentence_ids']=[3]
        self.assertEqual(prepare_delivery(result)['sections']['RULING']['status'],'withheld')

    def test_outcome_precedes_reasons_without_rewriting_or_losing_order(self):
        result=self.result(); draft=prepare_delivery(result)
        ruling=draft['sections']['RULING']
        self.assertTrue(ruling['text'].startswith('WHEREFORE'))
        self.assertLess(ruling['text'].index('GRANTED'),ruling['text'].index('We hold'))
        self.assertTrue(ruling['text'].endswith('SO ORDERED.'))
        source={s['id']:s['text'] for s in result['sentences']}
        self.assertTrue(all(source[e['sentence_id']]==e['text'] for e in ruling['evidence']))

    def test_incomplete_procedural_group_is_not_removed(self):
        evidence=[{'sentence_id':i,'text':'Background event.'} for i in range(12)]
        evidence[0]['text']='The first Resolution granted a motion for extension.'
        self.assertEqual(readable_evidence('FACT',evidence,set()),evidence)

    def result(self):
        rows=[{'id':i+1,'role':role,'text':text} for i,(role,text) in enumerate([
            ('FACT','The City closed the bridge.'),
            ('ISSUE','The issue is whether the closure was lawful.'),
            ('RULING','We hold that the closure was unlawful.'),
            ('RULING','WHEREFORE, the petition is GRANTED.'),
            ('RULING','SO ORDERED.')])]
        return {**finalize(rows,{},select_evidence(rows)), 'case_id':'test',
                'sentences':rows, 'generated_blocks':{'FACT':'Invented accusation.'}}

    def test_raw_generation_never_enters_delivery(self):
        result=self.result(); before=copy.deepcopy(result)
        draft=prepare_delivery(result)
        self.assertNotIn('Invented',str(draft))
        self.assertEqual(result,before)
        self.assertEqual(draft['status'],'review_required')
        self.assertFalse(draft['legal_accuracy_verified'])

    def test_modified_evidence_and_draft_are_withheld_even_if_they_agree(self):
        result=self.result()
        result['verification']['blocks']['FACT']['evidence'][0]['text']='The City opened the bridge.'
        result['blocks']['FACT']='The City opened the bridge.'
        draft=prepare_delivery(result)
        self.assertEqual(draft['sections']['FACT']['text'],'')
        self.assertEqual(draft['status'],'incomplete')

    def test_missing_disposition_and_missing_audit_fail_closed(self):
        result=self.result()
        result['verification']['disposition_sentence_ids'].append(999)
        result['verification']['blocks'].pop('ISSUE')
        draft=prepare_delivery(result)
        for role in ['ISSUE','RULING']:
            self.assertEqual(draft['sections'][role]['status'],'withheld')

    def test_complete_output_cannot_be_substituted_without_matching_evidence(self):
        result=self.result();result['blocks']['RULING']='The appeal was denied.'
        self.assertEqual(prepare_delivery(result)['sections']['RULING']['text'],'')
