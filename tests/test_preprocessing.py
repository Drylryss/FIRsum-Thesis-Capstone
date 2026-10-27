import unittest
from core import sentences
from preprocessing import prepare_decision


class PreprocessingTests(unittest.TestCase):
    def test_caption_and_signed_tail_removed_but_quoted_order_retained(self):
        raw = ('G.R. No. 123 A v. B D E C I S I O N LEONEN, J.: '
               'The trial court said: SO ORDERED. The appeal followed. '
               'WHEREFORE, the petition is granted. SO ORDERED. '
               'MARVIC M.V.F. LEONEN Associate Justice WE CONCUR: Footnotes 1 Rollo.')
        result = prepare_decision(raw)
        self.assertTrue(result['text'].startswith('The trial court'))
        self.assertIn('The appeal followed.', result['text'])
        self.assertIn('WHEREFORE', result['text'])
        self.assertTrue(result['text'].endswith('SO ORDERED.'))
        self.assertNotIn('Footnotes', result['text'])
        self.assertEqual(len(result['excluded_sections']), 2)

    def test_ambiguous_text_and_numbers_retained(self):
        raw = 'The court ordered payment of 50,000. Article 1157 applies. SO ORDERED. Further text.'
        result = prepare_decision(raw)
        self.assertEqual(result['text'], raw)
        self.assertEqual(result['excluded_sections'], [])

    def test_older_caption_and_concurrence_boundary(self):
        raw = ('G.R. No. 123 May 12, 1989 A, petitioner, vs. B, respondent. '
               'MELENCIO-HERRERA, J.: The claim concerns a contract. '
               'WHEREFORE, the petition is granted. SO ORDERED . '
               'Fernan, C.J., Cruz and Paras, JJ., concur. Footnotes 1 A note.')
        prepared = prepare_decision(raw)
        self.assertTrue(prepared['text'].startswith('The claim'))
        self.assertTrue(prepared['text'].endswith('SO ORDERED .'))
        self.assertNotIn('Footnotes', prepared['text'])

    def test_names_and_quoted_sentence_boundaries(self):
        self.assertEqual(sentences('Dr. Efren SJ. Nerva met Atty. Noe Botor (Atty. Botor). The court agreed.'),
                         ['Dr. Efren SJ. Nerva met Atty. Noe Botor (Atty. Botor).', 'The court agreed.'])
        self.assertEqual(sentences('He said "Denied." The appeal followed.'),
                         ['He said "Denied."', 'The appeal followed.'])

    def test_signed_historical_quote_does_not_truncate_later_current_order(self):
        raw = ('The earlier decision read: WHEREFORE, the petition is denied. SO ORDERED. '
               'JUAN CRUZ Associate Justice The new petition raised a different ground. '
               'WHEREFORE, the petition is granted. SO ORDERED. '
               'PEDRO SANTOS Associate Justice WE CONCUR:')
        prepared = prepare_decision(raw)
        self.assertIn('The new petition', prepared['text'])
        self.assertIn('petition is granted.', prepared['text'])

    def test_separate_opinion_signature_is_not_used_to_extend_main_opinion(self):
        raw = ('WHEREFORE, the petition is granted. SO ORDERED. '
               'JUAN CRUZ Associate Justice DISSENTING OPINION '
               'WHEREFORE, I vote to deny. SO ORDERED. PEDRO SANTOS Associate Justice')
        prepared = prepare_decision(raw)
        self.assertNotIn('I vote to deny', prepared['text'])
        self.assertIn('I vote to deny', prepared['excluded_sections'][0]['text'])
