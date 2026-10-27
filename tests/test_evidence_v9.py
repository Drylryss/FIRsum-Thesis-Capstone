import copy
import unittest

from grounding import select_evidence, finalize
from delivery import prepare_delivery, readable_evidence
from footnote_display import clean_display_artifacts


def rows(*texts):
    return [{'id': i + 1, 'text': text, 'role': 'RULING'} for i, text in enumerate(texts)]


class EvidenceV9Tests(unittest.TestCase):
    def test_all_dangling_facts_are_withheld(self):
        source = rows('She paid the debt.', 'The issue is whether payment discharged the debt.',
                      'WHEREFORE, the petition is GRANTED.', 'SO ORDERED.')
        source[0]['role'] = 'FACT'
        result = {**finalize(source, {}, select_evidence(source)), 'sentences': source}
        self.assertEqual(prepare_delivery(result)['sections']['FACT']['status'], 'withheld')

    def test_pronoun_cannot_attach_to_an_unrelated_selected_person(self):
        evidence = [dict(sentence_id=1, text='Maria sold the lot.'),
                    dict(sentence_id=8, text='She then filed a motion to dismiss the complaint.'),
                    dict(sentence_id=9, text='The court granted the motion.'),
                    dict(sentence_id=15, text='The spouses-vendors undertook to transfer title within ninety days.')]
        self.assertEqual(readable_evidence('FACT', evidence, set()), evidence[:1])
        evidence[1]['sentence_id'] = 2
        evidence[2]['sentence_id'] = 3
        self.assertEqual(readable_evidence('FACT', evidence[:3], set()), evidence[:3])

    def test_testimony_does_not_swallow_explicit_merits_transition(self):
        source = rows('THE RULING OF THE COURT We deny the petition.',
                      'The witness testified as follows: ATTY. Counsel:',
                      'Q Were you present?', 'A Yes.',
                      'Even if we were to disregard the procedural lapse, the claim fails.',
                      'The petitioners are not entitled to rescission.',
                      'WHEREFORE, the petition is DENIED.', 'SO ORDERED.')
        plan = select_evidence(source)
        self.assertIn(4, plan['quoted_sentence_ids'])
        self.assertNotIn(6, plan['quoted_sentence_ids'])
        self.assertIn(6, [e['sentence_id'] for e in plan['evidence']['RULING']])

    def test_previous_order_cannot_replace_terminal_disposition(self):
        source = rows('The trial court stated: WHEREFORE, the petition is DENIED.',
                      'SO ORDERED.', 'The issue is whether the trial court erred.',
                      'We hold that it erred.',
                      'WHEREFORE, the petition is GRANTED.', 'SO ORDERED.')
        result = {**finalize(source, {}, select_evidence(source)), 'sentences': source}
        evidence = [dict(sentence_id=r['id'], text=r['text'], selection='final_disposition') for r in source[:2]]
        result['verification']['blocks']['RULING']['evidence'] = evidence
        result['verification']['disposition_sentence_ids'] = [1, 2]
        result['blocks']['RULING'] = '\n\n'.join(e['text'] for e in evidence)
        self.assertEqual(prepare_delivery(result)['sections']['RULING']['status'], 'withheld')

    def test_contract_analysis_resumes_after_terms(self):
        source = rows('THE RULING OF THE COURT We deny the petition.',
                      'The purchase price is payable as follows: 1.',
                      'The buyer shall pay on delivery.',
                      'By the terms of the contract, the obligation arises upon delivery.',
                      'The petitioners are not entitled to rescission.',
                      'The vendor failed to deliver title within 120 days.',
                      'WHEREFORE, the petition is DENIED.', 'SO ORDERED.')
        plan = select_evidence(source)
        self.assertNotIn(5, plan['quoted_sentence_ids'])
        self.assertIn(5, [e['sentence_id'] for e in plan['evidence']['RULING']])

    def test_prior_court_entitlement_stays_quoted(self):
        source = rows('THE RULING OF THE COURT We deny the petition.',
                      'The trial court stated: "The petitioners are not entitled to relief.',
                      'By the terms of the contract, the obligation arises upon delivery.',
                      'The respondents are not entitled to rescission.',
                      'SO ORDERED.', 'WHEREFORE, the petition is DENIED.', 'SO ORDERED.')
        plan = select_evidence(source)
        self.assertIn(4, plan['quoted_sentence_ids'])
        self.assertNotIn(4, [e['sentence_id'] for e in plan['evidence']['RULING']])

    def test_display_preserves_substantive_numbers(self):
        text = '1âwphi1 REINSTATEDonly for ₱1,800,000.00 under Article 1191 within 120 days.'
        self.assertEqual(clean_display_artifacts(text),
                         'REINSTATED only for ₱1,800,000.00 under Article 1191 within 120 days.')
        self.assertEqual(clean_display_artifacts('34 The award is ₱34.00.'), '34 The award is ₱34.00.')

    def test_forged_role_assignment_cannot_pass_source_match(self):
        source = rows('The trial court denied the petition.',
                      'The issue is whether payment is due.',
                      'We hold that payment is due.',
                      'WHEREFORE, the petition is GRANTED.', 'SO ORDERED.')
        result = {**finalize(source, {}, select_evidence(source)), 'sentences': source}
        wrong = copy.deepcopy(result['verification']['blocks']['RULING']['evidence'][0])
        wrong.update(sentence_id=1, text=source[0]['text'])
        result['verification']['blocks']['ISSUE']['evidence'] = [wrong]
        result['blocks']['ISSUE'] = wrong['text']
        self.assertEqual(prepare_delivery(result)['sections']['ISSUE']['status'], 'withheld')


if __name__ == '__main__':
    unittest.main()
