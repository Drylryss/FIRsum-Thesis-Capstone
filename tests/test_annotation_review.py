import copy
from datetime import date
import unittest

from prepare_annotation_review import build_reviews, import_reviews


class AnnotationReviewTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'sentence_id': 't1', 'case_id': 'a', 'split': 'train', 'text': 'Party claims ownership.', 'label': 'ISSUE'},
            {'sentence_id': 'v1', 'case_id': 'b', 'split': 'validation', 'text': 'Whether relief is due.', 'label': 'ISSUE'},
            {'sentence_id': 'x1', 'case_id': 'c', 'split': 'test', 'text': 'Held-out text.', 'label': 'RULING'},
        ]

    def completed(self):
        records = build_reviews(self.rows)
        for record in records:
            record['review'].update(status='ACCEPTED', final_label='FACT', reviewer='Test reviewer',
                                    reviewer_type='human_annotator', review_date=date.today().isoformat(),
                                    reason='Fixture decision', source_checked=True, source_reference='Fixture paragraph 1')
        return records

    def test_test_text_and_context_never_exported(self):
        records = build_reviews(self.rows)
        self.assertEqual(len(records), 2)
        self.assertNotIn('Held-out text.', str(records))
        self.assertTrue(all(r['review']['final_label'] is None for r in records))

    def test_pending_and_ai_review_cannot_be_promoted(self):
        with self.assertRaisesRegex(ValueError, 'Unresolved'):
            import_reviews(self.rows, build_reviews(self.rows))
        records = self.completed()
        records[0]['review']['reviewer_type'] = 'AI_assistant'
        with self.assertRaisesRegex(ValueError, 'human review'):
            import_reviews(self.rows, records)

    def test_rejects_partial_duplicate_and_test_reviews(self):
        records = self.completed()
        for invalid in [records[:1], records + records[:1], records + [{'sentence_id': 'x1'}]]:
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, 'every development row'):
                    import_reviews(self.rows, invalid)

    def test_rejects_changed_source_split_or_label(self):
        for field, value in [('text', 'Edited source'), ('split', 'test'), ('original_label', 'OTHER'),
                             ('source_text_sha256', 'wrong')]:
            records = self.completed()
            records[0][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    import_reviews(self.rows, records)

    def test_requires_source_check_and_provenance(self):
        for field, value in [('source_checked', False), ('reviewer', ' '), ('source_reference', ''),
                             ('review_date', '2999-01-01')]:
            records = self.completed()
            records[0]['review'][field] = value
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    import_reviews(self.rows, records)

    def test_import_preserves_originals_test_rows_and_review_type(self):
        original = copy.deepcopy(self.rows)
        revised = import_reviews(self.rows, self.completed())
        self.assertEqual(self.rows, original)
        self.assertEqual(revised[-1], original[-1])
        self.assertEqual(revised[0]['label'], 'FACT')
        self.assertFalse(revised[0]['attorney_validated'])
        self.assertEqual(revised[0]['original_label'], 'ISSUE')

    def test_stale_ai_proposal_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Stale proposal'):
            build_reviews(self.rows, [{'sentence_id': 't1', 'text': 'wrong', 'current_provisional_label': 'ISSUE'}])


if __name__ == '__main__':
    unittest.main()
