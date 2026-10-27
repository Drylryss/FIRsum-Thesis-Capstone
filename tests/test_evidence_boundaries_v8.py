import unittest
from grounding import select_evidence


def rows(*text):
    return [{'id':i+1,'text':t,'role':'RULING','confidence':0.99} for i,t in enumerate(text)]


class BoundaryTests(unittest.TestCase):
    def test_procedural_rule_does_not_swallow_current_issue(self):
        p=select_evidence(rows('Sections 5, Rule 15 states: Section 5.',
            'The hearing shall be scheduled.',
            'The central issue is whether the bank may debit the account.',
            'We reverse.', 'WHEREFORE, the petition is GRANTED.', 'SO ORDERED.'))
        self.assertEqual([e['sentence_id'] for e in p['evidence']['ISSUE']], [3])
        self.assertEqual(p['disposition_ids'],[5,6])

    def test_contract_enumeration_allows_alternative_terminal_order(self):
        p=select_evidence(rows('The purchase price shall be paid as follows: 1.',
            'The balance is due upon delivery.',
            'IN LIGHT OF ALL THE FOREGOING, the petition is DENIED.', 'SO ORDERED.'))
        self.assertEqual(p['disposition_ids'],[3,4])

    def test_alternative_order_inside_prior_judgment_stays_excluded(self):
        p=select_evidence(rows('The trial court stated: “The claim fails.',
            'IN LIGHT OF ALL THE FOREGOING, the petition is DENIED.', 'SO ORDERED.'))
        self.assertEqual(p['disposition_ids'],[])

    def test_recap_preserves_immediate_liability_qualification(self):
        p=select_evidence(rows('The issue is whether the bank bears the loss.',
            'Section 21 of the Philippine Clearing House Rules and Regulations provides:',
            'The presenting bank guarantees endorsements.',
            'To recap, the collecting bank bears the loss.',
            'The drawee may recover from that bank.', 'The collecting bank is liable.',
            'It cannot pass the loss to depositors absent their contributing negligence.',
            'WHEREFORE, the petition is GRANTED.', 'SO ORDERED.'))
        self.assertIn(7,[e['sentence_id'] for e in p['evidence']['RULING']])


if __name__=='__main__': unittest.main()
