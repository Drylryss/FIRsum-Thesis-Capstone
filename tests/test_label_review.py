import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from label_review import apply_reviewed_labels
from pipeline import FIRPipeline
from unittest.mock import patch


class LabelReviewTests(unittest.TestCase):
    def test_automatic_pipeline_uses_raw_model_labels(self):
        engine = FIRPipeline.__new__(FIRPipeline)
        engine.examples, engine.training_metadata, engine.unlearned_roles, engine.device = [], {}, [], 'cpu'
        predicted = [dict(id=1, text='A sentence.', role='ISSUE', confidence=0.9)]
        engine.classify = lambda text: predicted
        engine.generate_block = lambda role, source: (source, int(bool(source)))
        with patch('label_review.apply_reviewed_labels', side_effect=AssertionError('Overrides must not run')):
            result = engine.run('A sentence.', 'case')
        self.assertEqual(result['sentences'], predicted)
        self.assertEqual(result['blocks']['ISSUE'], 'A sentence.')

    def test_exact_case_only_and_model_prediction_preserved(self):
        rows = [dict(id=1, text='REVERSED and SET ASIDE.', role='ISSUE', confidence=0.98)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, 'review.json')
            key = hashlib.sha256(b'exact case').hexdigest()
            path.write_text(json.dumps({'cases': {key: [dict(text=rows[0]['text'], role='RULING', reason='Disposition')]}}))
            fixed = apply_reviewed_labels('exact case', rows, path)
            self.assertEqual(fixed[0]['role'], 'RULING')
            self.assertEqual(fixed[0]['model_role'], 'ISSUE')
            self.assertIsNone(fixed[0]['confidence'])
            self.assertEqual(rows[0]['role'], 'ISSUE')
            self.assertEqual(apply_reviewed_labels('different case', rows, path), rows)

    def test_generation_uses_reviewed_role_without_reclassification(self):
        engine = FIRPipeline.__new__(FIRPipeline)
        engine.examples, engine.training_metadata, engine.unlearned_roles, engine.device = [], {}, [], 'cpu'
        engine.classify = lambda text: self.fail('Must not overwrite reviewed labels')
        engine.generate_block = lambda role, source: (source, int(bool(source)))
        result = engine.run_tagged([dict(id=1, text='Petition granted.', role='RULING', label_source='user_review')], 'case')
        self.assertEqual(result['blocks']['RULING'], 'Petition granted.')
        self.assertEqual(result['blocks']['ISSUE'], '')
