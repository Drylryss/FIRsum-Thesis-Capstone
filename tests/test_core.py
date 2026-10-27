import unittest
from core import validate_splits, sentences, token_windows, prompt, generator_input, training_rows
from unittest.mock import patch


class DataTests(unittest.TestCase):
    def test_review_only_candidate_cannot_be_used_for_training(self):
        records = [{'case_id': 'a', 'split': 'train', 'text': 'A claim.', 'label': 'FACT', 'training_eligible': False}]
        with patch('core.read_jsonl', return_value=records):
            with self.assertRaisesRegex(ValueError, 'Review-only'):
                training_rows('not-read.jsonl')

    def test_case_leakage(self):
        with self.assertRaises(ValueError):
            validate_splits([{'case_id': 'a', 'split': 'train'}, {'case_id': 'a', 'split': 'test'}])

    def test_window_tail_and_coverage(self):
        ids = list(range(3021))
        chunks = list(token_windows(ids, 900))
        self.assertEqual(set(ids), {i for c in chunks for i in c})
        self.assertEqual(chunks[-1][-1], 3020)
        self.assertTrue(all(len(c) <= 900 for c in chunks))

    def test_legal_abbreviations(self):
        self.assertEqual(len(sentences('G.R. No. 123 concerns Art. 1159. The Court ruled.')), 2)

    def test_corporate_citation_not_split_from_its_predicate(self):
        self.assertEqual(len(sentences('The Court in A v. Sample Corp. explained the rule. It applies here.')), 2)

    def test_example_delimiting(self):
        text = prompt('FACT', 'Target facts', [{'blocks': {'FACT': 'Style only'}}])
        self.assertIn('Target FACT source:\nTarget facts', text)

    def test_zero_shot_input_contains_only_the_source(self):
        self.assertEqual(generator_input('ISSUE', 'Whether the order is valid?'), 'Whether the order is valid?')
        self.assertIn('Example 1', generator_input('FACT', 'Target facts', [{'blocks': {'FACT': 'Style only'}}]))


if __name__ == '__main__':
    unittest.main()
