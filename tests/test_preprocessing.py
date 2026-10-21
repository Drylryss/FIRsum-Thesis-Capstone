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

    def test_names_and_quoted_sentence_boundaries(self):
        self.assertEqual(sentences('Dr. Efren SJ. Nerva met Atty. Noe Botor (Atty. Botor). The court agreed.'),
                         ['Dr. Efren SJ. Nerva met Atty. Noe Botor (Atty. Botor).', 'The court agreed.'])
        self.assertEqual(sentences('He said "Denied." The appeal followed.'),
                         ['He said "Denied."', 'The appeal followed.'])
