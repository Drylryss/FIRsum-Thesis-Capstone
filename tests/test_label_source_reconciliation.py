import unittest
from lxml import html
from reconcile_label_sources import strip_citation_links, normalize_offsets


class CitationReconciliationTests(unittest.TestCase):
    def test_only_identified_citations_removed(self):
        tree = html.fromstring('<p>Article 1216<a href="#fnt3">3</a> applies in 1983. <a href="/case">42</a></p>')
        self.assertEqual(len(strip_citation_links(tree)), 1)
        self.assertEqual(tree.text_content(), 'Article 1216 applies in 1983. 42')

    def test_legacy_marker_does_not_delete_amount(self):
        tree = html.fromstring('<p>P150,000<a class="nt" name="fnt2.2">2</a> due. <a name="fnt3.3">3</a></p>')
        strip_citation_links(tree)
        self.assertEqual(tree.text_content(), 'P150,000 due. 3')

    def test_offsets_preserve_original_text(self):
        text = 'At ₱150,000, paid.'
        clean, offsets = normalize_offsets(text)
        start = clean.index('p150000')
        self.assertEqual(text[offsets[start]:offsets[start+6]+1], '₱150,000')


if __name__ == '__main__':
    unittest.main()
