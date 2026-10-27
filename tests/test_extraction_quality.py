import unittest
from extraction_quality import audit_pages


class ExtractionTests(unittest.TestCase):
    def test_partly_scanned_pdf_blocks_summarization(self):
        result = audit_pages(['The petition concerns a contract.', '', 'SO ORDERED.'], b'pdf')
        self.assertFalse(result['summarization_allowed'])
        self.assertEqual(result['blocked_pages'], [2])
        self.assertFalse(result['complete_extraction_verified'])

    def test_sparse_signature_page_is_visible_but_not_silently_deleted(self):
        result = audit_pages(['SO ORDERED.'], b'pdf')
        self.assertTrue(result['summarization_allowed'])
        self.assertEqual(result['low_text_pages'], [1])
        self.assertEqual(len(result['pages']), 1)

    def test_corrupt_decoding_blocks_summarization(self):
        result = audit_pages(['The court \ufffd\ufffd\ufffd\ufffd ordered payment.'], b'pdf')
        self.assertFalse(result['summarization_allowed'])

    def test_hash_tracks_input_bytes_not_filename(self):
        self.assertNotEqual(audit_pages(['Text'], b'first')['pdf_sha256'],
                            audit_pages(['Text'], b'second')['pdf_sha256'])
