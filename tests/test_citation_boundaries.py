import unittest

from core import sentences
from grounding import select_evidence


def tagged(text, role='RULING'):
    return [dict(id=i + 1, text=s, role=role) for i, s in enumerate(sentences(text))]


class CitationBoundaryTests(unittest.TestCase):
    def test_citation_closes_quote_before_current_disposition(self):
        rows = tagged('The precedent stated: “The findings are conclusive. '
                      'No exception applies.” [19] WHEREFORE, the petition is DENIED. SO ORDERED.')
        plan = select_evidence(rows)
        selected = plan['evidence']['RULING']
        self.assertTrue(plan['disposition_ids'])
        self.assertTrue(selected[0]['text'].startswith('WHEREFORE'))
        self.assertNotIn('No exception', ' '.join(e['text'] for e in selected))
        self.assertIn('[19]', rows[1]['text'])

    def test_quoted_lower_court_order_not_selected(self):
        rows = tagged('The trial court stated: “WHEREFORE, the claim is GRANTED. '
                      'SO ORDERED.” [3] Dissatisfied, petitioners appealed the judgment. '
                      'The issue is whether fraud occurred. WHEREFORE, the petition is DENIED. SO ORDERED.')
        plan = select_evidence(rows)
        self.assertNotIn('GRANTED', ' '.join(e['text'] for e in plan['evidence']['RULING']))
        self.assertTrue(plan['evidence']['ISSUE'])

    def test_unclosed_order_ends_at_explicit_appeal_history(self):
        rows = tagged('The trial court decision reads: WHEREFORE, the sale is void. '
                      'Dissatisfied, petitioners elevated their cause to the Court of Appeals. '
                      'The issue for resolution is whether fraud occurred. '
                      'WHEREFORE, the petition is DENIED. SO ORDERED.')
        self.assertEqual(len(select_evidence(rows)['evidence']['ISSUE']), 1)

    def test_alias_and_footnote_do_not_lose_attribution(self):
        parts = sentences('The buyer alleged that A a.k.a. B induced her to sign. [4] She sued.')
        self.assertEqual(len(parts), 2)
        self.assertIn('a.k.a. B', parts[0])
        self.assertIn('[4]', parts[0])

    def test_crux_issue_excludes_reported_lower_court_question(self):
        rows = tagged('The crux of the controversy in this petition is whether fraud occurred. '
                      'WHEREFORE, the petition is DENIED. SO ORDERED.')
        self.assertEqual(len(select_evidence(rows)['evidence']['ISSUE']), 1)
        rows[0]['text'] = 'The trial court stated that the crux of the controversy in this case is whether fraud occurred.'
        self.assertFalse(select_evidence(rows)['evidence']['ISSUE'])

    def test_mortgage_and_inducement_survive_fact_selection(self):
        rows = tagged('The buyer purchased land. The seller executed a real estate mortgage to secure a loan. '
                      'The buyer alleged that the seller induced her to sign by misrepresenting the terms. '
                      'She agreed believing it protected her payments. The issue is whether fraud occurred. '
                      'WHEREFORE, the petition is DENIED. SO ORDERED.', 'FACT')
        facts = ' '.join(e['text'] for e in select_evidence(rows)['evidence']['FACT'])
        self.assertIn('real estate mortgage', facts)
        self.assertIn('alleged', facts)
        self.assertIn('protected her payments', facts)


if __name__ == '__main__':
    unittest.main()
