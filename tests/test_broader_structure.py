import unittest
from preprocessing import prepare_decision
from grounding import select_evidence, finalize
from delivery import prepare_delivery, readable_evidence


def rows(*items):
    return [dict(id=i+1, role=role, text=text) for i, (role, text) in enumerate(items)]


class BroaderStructureTests(unittest.TestCase):
    def test_same_day_does_not_attach_to_an_unrelated_date(self):
        evidence = [dict(sentence_id=1, text='On February 23, 1990, the court dismissed the case.'),
                    dict(sentence_id=4, text='On the same day, February 28, 1990, the owners sold the land.')]
        self.assertEqual(readable_evidence('FACT', evidence, set()), evidence[:1])
        evidence.insert(1, dict(sentence_id=3, text='On February 28, 1990, the buyer annotated the title.'))
        self.assertEqual(readable_evidence('FACT', evidence, set()), evidence)

    def test_resolution_heading_requires_judicial_author(self):
        text = 'THIRD DIVISION May 1, 2017 G.R. No. 123 A v. B R E S O L U T I O N REYES, J.: The bank sued.'
        self.assertEqual(prepare_decision(text)['text'], 'The bank sued.')
        ambiguous = 'The board passed a RESOLUTION. The bank sued.'
        self.assertEqual(prepare_decision(ambiguous)['text'], ambiguous)

    def test_older_caption_with_city_and_division(self):
        text = 'Manila SECOND DIVISION G.R. No. 123 A v. B REGALADO, J.: The buyer paid.'
        self.assertEqual(prepare_decision(text)['text'], 'The buyer paid.')

    def test_current_petition_questions_supersede_earlier_appeal(self):
        source = rows(('ISSUE', 'In its appeal, the buyer raised these issues: Whether the court erred.'),
                      ('FACT', 'The Court of Appeals affirmed.'),
                      ('ISSUE', 'In this petition, the buyer raises the following issues: Whether payment was due.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        plan = select_evidence(source)
        self.assertEqual([e['sentence_id'] for e in plan['evidence']['ISSUE']], [3])

    def test_frontloaded_issues_do_not_hide_following_facts(self):
        source = rows(('ISSUE', 'The issue is whether the sale was valid.'),
                      ('FACT', 'The records disclose the following antecedent facts: 1.'),
                      ('FACT', 'The buyer paid a deposit.'),
                      ('FACT', 'The seller sold the land to another buyer.'),
                      ('RULING', 'An analysis of the facts leads to the conclusion that the agreement is a contract to sell.'),
                      ('RULING', 'WHEREFORE, the judgment is AFFIRMED.'), ('RULING', 'SO ORDERED.'))
        plan = select_evidence(source)
        self.assertIn(3, [e['sentence_id'] for e in plan['evidence']['FACT']])
        self.assertNotIn(5, [e['sentence_id'] for e in plan['evidence']['FACT']])

    def test_title_case_heading_does_not_adopt_named_party_argument(self):
        source = rows(('ISSUE', 'The issue is whether payment is due.'),
                      ('RULING', 'Ruling of the Court The petition is denied.'),
                      ('FACT', 'Acme maintains that its debt should be cancelled.'),
                      ('RULING', 'WHEREFORE, the petition is DENIED.'), ('RULING', 'SO ORDERED.'))
        ruling = finalize(source, {}, select_evidence(source))['blocks']['RULING']
        self.assertIn('Ruling of the Court', ruling)
        self.assertNotIn('Acme maintains', ruling)

    def test_quoted_title_case_heading_stays_excluded(self):
        source = rows(('FACT', 'The trial court stated:'),
                      ('RULING', 'Ruling of the Court The petition is granted.'),
                      ('RULING', 'SO ORDERED.'), ('RULING', 'WHEREFORE, the petition is DENIED.'),
                      ('RULING', 'SO ORDERED.'))
        self.assertNotIn('petition is granted', finalize(source, {}, select_evidence(source))['blocks']['RULING'])

    def test_statutory_reads_block_ends_at_explicit_current_agreement(self):
        source = rows(('ISSUE', 'The issue is whether the bank must pay.'),
                      ('RULING', 'Section 1, Rule 34 reads: Sec.'),
                      ('RULING', 'The party may seek judgment.'),
                      ('RULING', 'This Court agrees that the bank must pay.'),
                      ('RULING', 'WHEREFORE, the petition is GRANTED.'), ('RULING', 'SO ORDERED.'))
        ruling = finalize(source, {}, select_evidence(source))['blocks']['RULING']
        self.assertIn('This Court agrees', ruling)
        self.assertNotIn('party may seek', ruling)

    def test_ambiguous_amount_is_flagged_without_guessing(self):
        source = rows(('ISSUE', 'The issue is whether the bank must pay.'),
                      ('RULING', 'WHEREFORE, pay Pl 13,756.83.'), ('RULING', 'SO ORDERED.'))
        item = {**finalize(source, {}, select_evidence(source)), 'sentences': source}
        section = prepare_delivery(item)['sections']['RULING']
        self.assertIn('Pl 13,756.83', section['display_text'])
        self.assertTrue(section['source_artifact_warnings'])


if __name__ == '__main__': unittest.main()
