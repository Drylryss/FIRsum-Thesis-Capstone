import unittest
from core import validate_splits, sentences, token_windows, prompt


class DataTests(unittest.TestCase):
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

    def test_example_delimiting(self):
        text = prompt('FACT', 'Target facts', [{'blocks': {'FACT': 'Style only'}}])
        self.assertIn('Target FACT source:\nTarget facts', text)


if __name__ == '__main__':
    unittest.main()
