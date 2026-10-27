import unittest
from import_additional_digests import extract_source


class SourceImportTests(unittest.TestCase):
    def test_layout_table_does_not_erase_decision(self):
        raw = b'<table><tr><td><blockquote><p>G.R. No. 12345</p><p>Facts.</p><table><tr><td><p>100 pesos</p></td></tr></table><p>SO ORDERED.</p></blockquote></td></tr></table>'
        text = extract_source(raw, '12345')
        self.assertIn('Facts.', text)
        self.assertEqual(text.count('100 pesos'), 1)

    def test_premature_blockquote_close_retains_disposition(self):
        raw = b'<table><tr><td><blockquote><p>G.R. No. 12345</p><p>Facts.</p></blockquote><p>Final ruling.</p><p>SO ORDERED.</p></td></tr></table>'
        self.assertIn('Final ruling.', extract_source(raw, '12345'))

    def test_wrong_docket_or_incomplete_source_rejected(self):
        for raw in [b'<blockquote><p>G.R. No. 999</p><p>SO ORDERED.</p></blockquote>',
                    b'<blockquote><p>G.R. No. 12345</p><p>Incomplete.</p></blockquote>']:
            with self.assertRaises(ValueError):
                extract_source(raw, '12345')


if __name__ == '__main__':
    unittest.main()
