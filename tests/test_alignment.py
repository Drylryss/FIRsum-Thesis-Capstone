import unittest
from align_digest_pairs import coverage, propose, target_units, fragment_warning, neighboring_context


class AlignmentTests(unittest.TestCase):
    def test_answer_only_and_list_fragments_need_context(self):
        for text in ['NEGATIVE.', 'xxx', '(2) Contracts;']:
            self.assertTrue(fragment_warning(text))

    def test_context_uses_positions_not_consecutive_ids(self):
        rows=[{'id':2}, {'id':8}, {'id':20}]
        self.assertEqual(neighboring_context([rows[1]],rows),[rows[0],rows[2]])

    def test_lexical_match_does_not_hide_negation_loss(self):
        self.assertLess(coverage('The Court did not grant the appeal.', 'The Court granted the appeal.'),1)

    def test_budget_keeps_whole_source_sentences(self):
        rows=[{'id':1,'text':'The claim was denied.'},{'id':2,'text':'A '*100}]
        result=propose('The claim was denied.',rows,len,budget=40)
        self.assertEqual(result,[rows[0]])

    def test_oversized_target_not_silently_truncated(self):
        text='A '*100+'.'
        self.assertEqual(target_units(text,len,20),[text])

    def test_unrelated_source_not_forced_into_candidate(self):
        self.assertEqual(propose('Payment of fees.',[{'id':1,'text':'Storm damaged buildings.'}],len),[])
