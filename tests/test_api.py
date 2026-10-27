import unittest
import io
from unittest.mock import patch
from fastapi.testclient import TestClient
from inference import app
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject


def decision_pdf(blank_page=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
        NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 50 700 Td (A decision.) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    if blank_page:
        writer.add_blank_page(width=612, height=792)
    output = io.BytesIO()
    writer.write(output)
    return ('decision.pdf', output.getvalue(), 'application/pdf')


class APITests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_home_and_status(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        self.assertIn('classifier_ready', self.client.get('/api/status').json())

    def test_empty_and_unsupported(self):
        self.assertEqual(self.client.post('/api/summarize', data={'case_id': 'a'}).status_code, 422)
        self.assertEqual(self.client.post('/api/summarize', data={'case_id': 'a'}, files={'file': ('x.exe', b'abc')}).status_code, 422)
        self.assertEqual(self.client.post('/api/summarize', data={'case_id': 'a', 'text': '   '}).status_code, 422)
        self.assertEqual(self.client.post('/api/summarize', data={'case_id': 'a'}, files={'file': ('x.txt', b'A decision.')}).status_code, 422)
        self.assertEqual(self.client.post('/api/summarize', data={'case_id': 'a'}, files={'file': ('fake.pdf', b'not a pdf')}).status_code, 422)

    def test_paste_without_pdf_rejected(self):
        with patch('inference.execute') as execute:
            response = self.client.post('/api/summarize', data={'case_id': 'a', 'text': 'Copied decision.'})
            self.assertEqual(response.status_code, 422)
            execute.assert_not_called()

    def test_inference_contract(self):
        with patch('inference.execute', return_value={'case_id': 'decision', 'blocks': {'FACT': 'x'}}) as execute:
            response = self.client.post('/api/summarize', files={'file': decision_pdf()})
            execute.assert_called_once_with('A decision.', 'decision', [])
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['blocks']['FACT'], 'x')
        # Legacy/unverified model responses must not become exportable drafts.
        self.assertEqual(response.json()['delivery']['status'], 'incomplete')
        self.assertEqual(response.json()['delivery']['sections']['FACT']['text'], '')

    def test_missing_model_error(self):
        with patch('inference.execute', side_effect=RuntimeError('Train LegalBERT first')):
            response = self.client.post('/api/summarize', data={'case_id': 'a'}, files={'file': decision_pdf()})
        self.assertEqual(response.status_code, 503)

    def test_prepare_does_not_run_models(self):
        with patch('inference.execute') as execute:
            response = self.client.post('/api/prepare', files={'file': decision_pdf()})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['text'], 'A decision.')
            self.assertIn('excluded_sections', response.json())
            execute.assert_not_called()

    def test_manual_regeneration_not_exposed(self):
        self.assertEqual(self.client.post('/api/regenerate', json={}).status_code, 404)

    def test_partial_extraction_stops_before_models_run(self):
        with patch('inference.execute') as execute:
            response = self.client.post('/api/summarize', files={'file': decision_pdf(blank_page=True)})
            self.assertEqual(response.status_code, 422)
            self.assertIn('Pages with missing or corrupt text: 2', response.json()['detail'])
            execute.assert_not_called()

    def test_preparation_keeps_partial_extraction_audit_for_inspection(self):
        with patch('inference.execute') as execute:
            response = self.client.post('/api/prepare', files={'file': decision_pdf(blank_page=True)})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['extraction_audit']['blocked_pages'], [2])
            execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
