import re
import unittest
from core import sentences
from preprocessing import repair_spacing, prepare_decision
from grounding import select_evidence


class SourcePreparationTests(unittest.TestCase):
    def test_docket_prefix_is_not_a_sentence_boundary(self):
        for docket in ('CA-G.R. CV. No. 43188', 'G. R. No. 123', 'CA-G.R. SP. No. 456'):
            text = 'The decision in ' + docket + ' was affirmed. The petition failed.'
            self.assertEqual(sentences(text), [text.rsplit(' The petition', 1)[0], 'The petition failed.'])
        self.assertEqual(sentences('The Court ruled. No. 123 identifies the case.'),
                         ['The Court ruled.', 'No. 123 identifies the case.'])

    def test_spacing_repairs_are_reversible_and_preserve_numbers(self):
        original = 'The Bank,returning ₱1,800,000.00, informedpetitioners thatthe checks werehonored inturn under Article 1191.'
        fixed, edits = repair_spacing(original)
        self.assertIn('informed petitioners that the checks were honored in turn', fixed)
        self.assertEqual(re.findall(r'\d+', original), re.findall(r'\d+', fixed))
        self.assertEqual(re.sub(r'\s', '', original), re.sub(r'\s', '', fixed))
        restored = fixed
        for edit in reversed(edits):
            start = edit['start']
            self.assertEqual(restored[start:start + len(edit['after'])], edit['after'])
            restored = restored[:start] + edit['before'] + restored[start + len(edit['after']):]
        self.assertEqual(restored, original)

    def test_no_unverified_spelling_or_citation_replacement(self):
        text = '34 Ruic 45 applies to Mambuay and ₱200,000.00.'
        result = prepare_decision(text)
        self.assertEqual(result['text'], text)
        self.assertEqual(result['text_repairs'], [])

    def test_dated_reference_keeps_antecedent(self):
        source = [dict(id=i + 1, role=role, text=text) for i, (role, text) in enumerate([
            ('FACT', 'A buyer paid with checks.'),
            ('FACT', 'The dispute went to the arbitration committee.'),
            ('FACT', 'In February 2001, the latter ruled for the bank.'),
            ('ISSUE', 'The issue is whether the bank may debit the account.'),
            ('RULING', 'WHEREFORE, the petition is GRANTED.'), ('RULING', 'SO ORDERED.')])]
        ids = [e['sentence_id'] for e in select_evidence(source)['evidence']['FACT']]
        if 3 in ids:
            self.assertIn(2, ids)


if __name__ == '__main__': unittest.main()
