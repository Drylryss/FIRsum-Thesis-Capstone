import unittest
from experiment_fewshot import validate_experiment


class ExperimentTests(unittest.TestCase):
    def examples(self):
        return [dict(case_id=str(i), split='exemplar_candidate', attorney_accepted=False,
                     blocks=dict(FACT='Source fact.', ISSUE='Whether payment is due?', RULING='Payment is due.'))
                for i in range(3)]

    def test_target_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            validate_experiment(self.examples(), ['1'])

    def test_pending_status_is_preserved(self):
        examples = self.examples()
        validate_experiment(examples, ['new'])
        self.assertTrue(all(e['attorney_accepted'] is False for e in examples))
        examples[0]['attorney_accepted'] = True
        with self.assertRaisesRegex(ValueError, 'provenance'):
            validate_experiment(examples, ['new'])
