import unittest
from grounding import select_evidence, finalize
from delivery import prepare_delivery


def rows(items):
    return [dict(id=i+1, role=role, text=text) for i,(role,text) in enumerate(items)]


class CoverageSelectionTests(unittest.TestCase):
    def test_complaint_allegation_keeps_filing_context(self):
        source=rows([('FACT','The buyer filed a complaint to annul the sale.'),
            ('FACT','In the complaint, the buyer alleged that the seller induced her to sign.'),
            ('ISSUE','The issue is whether fraud occurred.'),
            ('RULING','WHEREFORE, the petition is DENIED.'),('RULING','SO ORDERED.')])
        ids={e['sentence_id'] for e in select_evidence(source)['evidence']['FACT']}
        self.assertTrue({1,2}.issubset(ids))

    def test_closing_document_quote_does_not_hide_later_letter(self):
        source=rows([('FACT','The buyer signed a "Deed of Sale.'),
            ('FACT','" Subsequently, the property was partitioned.'),
            ('FACT','Before payment, the buyer received summons in an ownership suit.'),
            ('FACT','As a consequence, in a letter dated May 1, the buyer would hold payment.'),
            ('ISSUE','The issue is whether payment could be suspended.'),
            ('RULING','WHEREFORE, the petition is DENIED.'),('RULING','SO ORDERED.')])
        plan=select_evidence(source)
        ids={e['sentence_id'] for e in plan['evidence']['FACT']}
        self.assertTrue({1,2,3,4}.issubset(ids))

    def test_narrated_contract_terms_are_facts_not_judicial_orders(self):
        source=rows([('FACT','An agreement was executed under the following terms and conditions: 1.'),
            ('RULING','The selling price is P200.00.'),
            ('RULING','The sum of P20.00 is credited as partial payment.'),
            ('RULING','In case of default, the option shall be cancelled.'),
            ('FACT','The buyer filed suit.'),('ISSUE','The issue is whether the contract is valid.'),
            ('RULING','WHEREFORE, the petition is DENIED.'),('RULING','SO ORDERED.')])
        p=select_evidence(source)
        self.assertTrue({1,2,3,4}.issubset({e['sentence_id'] for e in p['evidence']['FACT']}))
        self.assertEqual(source[1]['role'],'RULING')
        self.assertNotIn(4,p['disposition_ids'])

    def test_adopted_statute_keeps_burden_and_exception(self):
        source=rows([('ISSUE','The issue is whether consent was valid.'),
            ('RULING','We disagree.'),('RULING','The claim was not substantiated.'),
            ('RULING','The appellate court could not be faulted when it invoked Article 123 which states: ART.'),
            ('RULING','123.'),('RULING','The enforcing party must explain the terms.'),
            ('RULING','The burden rests on that party.'),('RULING','If that burden is not met, the presumption remains.'),
            ('FACT','In this case, petitioners alleged consent.'),
            ('RULING','WHEREFORE, the petition is DENIED.'),('RULING','SO ORDERED.')])
        result={**finalize(source,{},select_evidence(source)),'sentences':source}
        section=prepare_delivery(result)['sections']['RULING']
        self.assertIn('presumption remains',section['display_text'])
        self.assertNotIn('petitioners alleged',section['display_text'])
        self.assertNotIn('We disagree.',section['display_text'])

    def test_unadopted_lower_court_statute_not_promoted(self):
        source=rows([('ISSUE','The issue is whether consent was valid.'),
            ('FACT','The trial court stated:'),
            ('RULING','The appellate court could not be faulted when it invoked Article 123.'),
            ('RULING','The enforcing party must explain the terms.'),
            ('RULING','SO ORDERED.'),('RULING','WHEREFORE, the petition is DENIED.'),('RULING','SO ORDERED.')])
        text=finalize(source,{},select_evidence(source))['blocks']['RULING']
        self.assertNotIn('Article 123',text)


if __name__=='__main__':unittest.main()
