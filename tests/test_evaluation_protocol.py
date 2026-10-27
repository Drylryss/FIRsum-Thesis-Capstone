import copy
import unittest

from evaluation_protocol import build_packet, eligibility_errors, textual_scores
from prepare_data_revision import exact_span


class EvaluationProtocolTests(unittest.TestCase):
    def fixtures(self):
        prediction = {'case_id': 'NEW_CASE', 'verification': {'source_sha256': 'source-hash'},
                      'generated_blocks': {'FACT': 'Unsupported raw claim.', 'ISSUE': 'A question?', 'RULING': 'Raw order.'},
                      'delivery': {'sections': {role: {'status': 'source_backed_draft', 'display_text': 'Source excerpt.', 'evidence': []}
                                               for role in ('FACT', 'ISSUE', 'RULING')}}}
        reference = {'case_id': 'NEW_CASE', 'split': 'test', 'attorney_accepted': True,
                     'reviewer': 'Test fixture reviewer', 'review_date': '2026-10-24',
                     'source_complete_verified': True, 'previously_used_for_development': False,
                     'source_sha256': 'source-hash', 'blocks': {r: 'Reference.' for r in ('FACT', 'ISSUE', 'RULING')}}
        return prediction, reference

    def test_known_case_cannot_be_declared_unseen(self):
        pred, ref = self.fixtures()
        self.assertTrue(eligibility_errors(pred, ref, [{'case_id': 'NEW_CASE'}]))

    def test_renaming_known_source_does_not_bypass_exclusion(self):
        pred, ref = self.fixtures()
        registry = [{'case_id': 'OTHER_ALIAS', 'source_fingerprints': ['source-hash']}]
        self.assertTrue(eligibility_errors(pred, ref, registry))

    def test_acceptance_flag_alone_is_insufficient(self):
        pred, ref = self.fixtures()
        ref['reviewer'] = None
        self.assertTrue(eligibility_errors(pred, ref, []))
        ref['reviewer'] = 'Test fixture reviewer'
        ref['source_sha256'] = 'wrong-source'
        self.assertTrue(eligibility_errors(pred, ref, []))

    def test_missing_raw_and_withheld_final_remain_in_denominator(self):
        pred, ref = self.fixtures()
        del pred['generated_blocks']['ISSUE']
        pred['delivery']['sections']['RULING']['status'] = 'withheld'
        before = copy.deepcopy(pred)
        packet = build_packet(pred, [], ref)
        self.assertEqual(len(packet['sections']), 6)
        self.assertEqual(sum(s['missing_or_withheld'] for s in packet['sections']), 2)
        self.assertEqual(pred, before)
        self.assertEqual(packet['sections'][0]['text'], 'Unsupported raw claim.')
        self.assertEqual(packet['sections'][1]['text'], 'Source excerpt.')
        self.assertIsNone(packet['sections'][0]['human_review']['errors']['unsupported_statement'])

    def test_unaccepted_reference_cannot_trigger_metric_downloads(self):
        pred, _ = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'Formal scoring blocked'):
            textual_scores(build_packet(pred, []))

    def test_exemplar_case_rejected(self):
        pred, ref = self.fixtures()
        pred['exemplar_case_ids'] = ['NEW_CASE']
        self.assertIn('Case was supplied as an exemplar.', eligibility_errors(pred, ref, []))

    def test_segmentation_preserves_exact_text_and_rejects_ambiguous_anchors(self):
        source = 'Prior order. Respondent alleged nonpayment. Later proceedings.'
        span = exact_span(source, 'Respondent alleged', ' Later proceedings.')
        self.assertEqual(span['text'], source[span['start_char']:span['end_char']])
        self.assertEqual(span['text'], 'Respondent alleged nonpayment.')
        with self.assertRaises(ValueError):
            exact_span('Claim. Claim.', 'Claim.')
        with self.assertRaises(ValueError):
            exact_span(source, 'Absent anchor')


if __name__ == '__main__':
    unittest.main()
