import unittest
from compare_issue_experiment import validation_signature


class ComparisonTests(unittest.TestCase):
    def test_changed_reference_is_not_comparable(self):
        row = dict(case_id='case', sentence_id='s1', text='sentence', reference_label='ISSUE')
        changed = dict(row, reference_label='RULING')
        self.assertNotEqual(validation_signature([row]), validation_signature([changed]))

    def test_duplicate_examples_rejected(self):
        row = dict(case_id='case', sentence_id='s1', text='sentence', reference_label='ISSUE')
        with self.assertRaises(ValueError):
            validation_signature([row, row])
